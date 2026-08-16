"""Manual-collection scraper for the Hindi and Nepali corpora.

This produces the ``source_type="manual"`` portion of each corpus — the share the
assignment requires us to gather ourselves rather than download. It runs in two phases:

1. **Discover.** For each configured source, locate its sitemap(s), walk them
   recursively, and write every candidate article URL to a frontier file. This is cheap
   (a few dozen requests yields tens of thousands of URLs) and can be inspected before
   any content is fetched.
2. **Fetch.** Pull each URL from the frontier, extract the main text with trafilatura,
   apply cheap quality filters, and append to compressed shards.

Both phases are resumable. The frontier and the set of already-fetched URLs live on disk,
so an interrupted run — or a Kaggle session hitting its 12-hour limit — is restarted by
re-running the same command.

Examples:
    # Build the URL frontier only, so it can be reviewed before fetching anything
    python -m scripts.scrape --lang ne --out data/manual/ne --discover-only

    # Small trial run against one site
    python -m scripts.scrape --lang ne --out data/manual/ne --sources onlinekhabar --limit 50

    # Full run
    python -m scripts.scrape --lang hi --out data/manual/hi --limit 200000
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

import trafilatura

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.clean import clean_text, devanagari_ratio, is_good_document
from common.fetch import PoliteFetcher
from common.schema import MANUAL, Document, ShardWriter
from common.sources import CANDIDATE_SITEMAP_PATHS, Source, get_sources
from common.wordpress import collect_wp_sharded

# Matches <loc>...</loc> in any sitemap flavour, namespaced or not.
LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.IGNORECASE | re.DOTALL)

# Devanagari Unicode block, used for a cheap "is this the right script?" check.
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


# --------------------------------------------------------------------- discovery

def parse_sitemap(xml: str) -> tuple[list[str], list[str]]:
    """Extract ``<loc>`` URLs from a sitemap, split into sub-sitemaps and page URLs.

    A well-formed sitemap index uses a ``<sitemapindex>`` root, but plenty of real sites
    (Setopati among them) list their sub-sitemaps inside an ordinary ``<urlset>``.
    Classifying by URL suffix rather than by root element handles both, and stops
    ``category.xml`` from being queued as if it were an article.

    Args:
        xml: Raw sitemap XML.

    Returns:
        A ``(sub_sitemaps, page_urls)`` pair.
    """
    urls = [u.strip() for u in LOC_RE.findall(xml) if u.strip()]
    subs = [u for u in urls if u.lower().split("?")[0].endswith((".xml", ".xml.gz"))]
    pages = [u for u in urls if u not in set(subs)]
    return subs, pages


async def discover_sitemaps(fetcher: PoliteFetcher, source: Source) -> list[str]:
    """Locate a source's sitemap URLs.

    Tries the explicitly configured sitemaps first, then any ``Sitemap:`` lines in
    robots.txt, then the conventional paths. Returns whatever responds with usable XML.
    """
    if source.sitemaps:
        return list(source.sitemaps)

    found: list[str] = []

    robots = await fetcher.get_text(f"{source.home}/robots.txt", check_robots=False)
    if robots:
        for line in robots.splitlines():
            if line.lower().startswith("sitemap:"):
                found.append(line.split(":", 1)[1].strip())

    if not found:
        for path in CANDIDATE_SITEMAP_PATHS:
            candidate = f"{source.home}{path}"
            body = await fetcher.get_text(candidate)
            if body and "<loc>" in body.lower():
                found.append(candidate)
                break

    return found


async def build_frontier(
    fetcher: PoliteFetcher,
    sources: list[Source],
    max_sitemaps_per_source: int,
    max_urls_per_source: int,
) -> list[tuple[str, str]]:
    """Walk every source's sitemaps and collect candidate article URLs.

    Sitemap indexes are expanded one level deep, which is enough for every site in the
    registry and keeps discovery bounded.

    Args:
        fetcher: The shared HTTP client.
        sources: Sources to discover.
        max_sitemaps_per_source: Cap on sub-sitemaps expanded per source.
        max_urls_per_source: Cap on URLs kept per source.

    Returns:
        A list of ``(source_name, url)`` pairs, deduplicated by URL.
    """
    frontier: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    for source in sources:
        sitemaps = await discover_sitemaps(fetcher, source)
        if not sitemaps:
            print(f"  [{source.name}] no sitemap found — skipping", flush=True)
            continue

        print(f"  [{source.name}] {len(sitemaps)} sitemap(s) advertised", flush=True)

        # Walk sitemaps breadth-first, expanding nested indexes as they are found.
        # A visited set guards against sitemaps that reference each other in a cycle.
        queue: list[str] = list(sitemaps)
        visited: set[str] = set()
        kept_for_source = 0
        expanded = 0

        while queue and kept_for_source < max_urls_per_source and expanded < max_sitemaps_per_source:
            sitemap_url = queue.pop(0)
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            expanded += 1

            body = await fetcher.get_text(sitemap_url)
            if not body:
                continue

            subs, pages = parse_sitemap(body)
            queue.extend(s for s in subs if s not in visited)

            for url in pages:
                if kept_for_source >= max_urls_per_source:
                    break
                if url in seen_urls or not source.wants(url):
                    continue
                seen_urls.add(url)
                frontier.append((source.name, url))
                kept_for_source += 1

        print(f"  [{source.name}] {kept_for_source:,} URLs queued", flush=True)

    return frontier


# ---------------------------------------------------------------------- fetching

def extract_document(
    html: str, url: str, source: Source, min_prose_words: int, min_script_ratio: float
) -> Document | None:
    """Turn a raw HTML page into a Document, or None if it fails quality checks.

    trafilatura strips navigation, ads, comments and related-article boxes, leaving the
    article body. ``favor_precision`` biases it towards dropping uncertain blocks, which
    costs a little volume and buys noticeably cleaner text.

    The output is then passed through :func:`clean_text`, and every quality check is
    applied to the *cleaned* result. This ordering is the important part: news pages
    routinely extract to thousands of characters that reduce to a few hundred once
    whitespace scaffolding, leaked CMS JSON and app-download boilerplate are removed.
    Measuring before cleaning would wave those pages straight through.
    """
    raw = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not raw:
        return None

    text = clean_text(raw)
    if not is_good_document(text, min_prose_words=min_prose_words):
        return None
    if devanagari_ratio(text) < min_script_ratio:
        return None

    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title or "") if metadata else ""

    return Document(
        text=text,
        lang=source.lang,
        source_type=MANUAL,
        source=source.name,
        url=url,
        title=title,
    )


async def fetch_worker(
    queue: asyncio.Queue,
    fetcher: PoliteFetcher,
    by_name: dict[str, Source],
    writer: ShardWriter,
    seen_fh,
    stats: dict,
    min_prose_words: int,
    min_script_ratio: float,
) -> None:
    """Consume URLs from the queue, extract text, and write successful documents.

    Runs as one of several concurrent workers. Writing is safe without a lock because
    asyncio is cooperative and neither the shard write nor the seen-file append awaits.
    """
    while True:
        item = await queue.get()
        if item is None:  # shutdown sentinel
            queue.task_done()
            return

        source_name, url = item
        try:
            body = await fetcher.get(url)
            stats["attempted"] += 1

            if body is None:
                stats["failed"] += 1
            else:
                doc = extract_document(
                    body.decode("utf-8", errors="replace"),
                    url,
                    by_name[source_name],
                    min_prose_words,
                    min_script_ratio,
                )
                if doc is None:
                    stats["rejected"] += 1
                else:
                    writer.write(doc)
                    stats["kept"] += 1

            # Record the URL as visited whatever the outcome, so a resumed run does not
            # retry pages that are permanently broken or permanently off-language.
            seen_fh.write(url + "\n")
            if stats["attempted"] % 200 == 0:
                seen_fh.flush()
                report_progress(stats, writer)

        except Exception as exc:  # one bad page must not kill the worker
            stats["failed"] += 1
            print(f"    error on {url}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            queue.task_done()


def report_progress(stats: dict, writer: ShardWriter) -> None:
    """Print a one-line progress summary with a rough token estimate."""
    elapsed = max(time.monotonic() - stats["started"], 1e-6)
    rate = stats["attempted"] / elapsed
    # ~1.8 subword tokens per whitespace word is typical for Devanagari at a 32k vocab;
    # this is only an in-flight estimate, the real count comes from the tokenizer.
    est_tokens = writer.chars_written / 5.0 * 1.8
    print(
        f"    {stats['kept']:,} kept / {stats['attempted']:,} tried "
        f"({stats['rejected']:,} rejected, {stats['failed']:,} failed) "
        f"| {rate:.1f} pages/s | ~{est_tokens/1e6:.1f}M tokens",
        flush=True,
    )


# -------------------------------------------------------------------------- main

async def collect_wordpress(
    fetcher: PoliteFetcher, sources: list[Source], out_dir: Path, args: argparse.Namespace
) -> None:
    """Collect from every WordPress-backed source via its REST API.

    These sources bypass the frontier entirely — the API is itself an ordered, resumable
    index of the archive, so there is nothing to discover first.
    """
    for source in sources:
        print(f"Collecting {source.name} via WordPress API...", flush=True)
        await collect_wp_sharded(
            fetcher,
            source,
            out_dir=out_dir,
            workers=args.workers,
            min_prose_words=args.min_prose_words,
            min_script_ratio=args.min_script_ratio,
            max_docs=args.limit or None,
        )


async def run(args: argparse.Namespace) -> None:
    """Execute the discover and fetch phases according to parsed CLI arguments."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier_path = out_dir / "frontier.tsv"
    seen_path = out_dir / "seen.txt"

    all_sources = get_sources(args.lang, args.sources)
    wp_sources = [s for s in all_sources if s.api == "wordpress"]
    sources = [s for s in all_sources if s.api is None]
    by_name = {s.name: s for s in sources}

    async with PoliteFetcher(
        concurrency=args.concurrency,
        per_domain_delay=args.delay,
    ) as fetcher:

        # ---- WordPress API sources (no frontier needed) ----
        if wp_sources and not args.discover_only:
            await collect_wordpress(fetcher, wp_sources, out_dir, args)

        if not sources:
            return

        # ---- Phase 1: discovery (skipped if a frontier already exists) ----
        if frontier_path.exists() and not args.rediscover:
            print(f"Reusing existing frontier: {frontier_path}", flush=True)
        else:
            print(f"Discovering URLs for lang={args.lang}...", flush=True)
            frontier = await build_frontier(
                fetcher, sources, args.max_sitemaps, args.max_urls_per_source
            )
            with open(frontier_path, "w", encoding="utf-8") as fh:
                for source_name, url in frontier:
                    fh.write(f"{source_name}\t{url}\n")
            print(f"Frontier written: {len(frontier):,} URLs -> {frontier_path}", flush=True)

        if args.discover_only:
            print("--discover-only set; stopping before fetch.", flush=True)
            return

        # ---- Phase 2: fetch ----
        seen: set[str] = set()
        if seen_path.exists():
            seen = {line.strip() for line in open(seen_path, encoding="utf-8") if line.strip()}
            print(f"Resuming: {len(seen):,} URLs already visited", flush=True)

        pending: list[tuple[str, str]] = []
        with open(frontier_path, encoding="utf-8") as fh:
            for line in fh:
                source_name, _, url = line.rstrip("\n").partition("\t")
                if url and url not in seen and source_name in by_name:
                    pending.append((source_name, url))
                if args.limit and len(pending) >= args.limit:
                    break

        if not pending:
            print("Nothing left to fetch.", flush=True)
            return

        print(f"Fetching {len(pending):,} URLs with {args.concurrency} workers...", flush=True)

        queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
        stats = {
            "attempted": 0, "kept": 0, "rejected": 0, "failed": 0,
            "started": time.monotonic(),
        }

        with ShardWriter(out_dir, f"{args.lang}-manual") as writer, \
             open(seen_path, "a", encoding="utf-8") as seen_fh:

            workers = [
                asyncio.create_task(
                    fetch_worker(queue, fetcher, by_name, writer, seen_fh, stats,
                                 args.min_prose_words, args.min_script_ratio)
                )
                for _ in range(args.concurrency)
            ]

            for item in pending:
                await queue.put(item)
            for _ in workers:
                await queue.put(None)

            await asyncio.gather(*workers)
            seen_fh.flush()

            print("\nDone.", flush=True)
            report_progress(stats, writer)
            print(f"Shards written to {out_dir}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", required=True, choices=["hi", "ne"],
                        help="Target language.")
    parser.add_argument("--out", required=True,
                        help="Output directory for shards, frontier and resume state.")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Restrict to named sources (default: all for the language).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Fetch at most this many URLs this run. 0 = no limit.")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Concurrent requests across all domains.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Monthly windows fetched in parallel per WordPress source. "
                             "Each window writes its own shard file, so restarts skip "
                             "windows that already completed.")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Minimum seconds between requests to the same domain.")
    parser.add_argument("--min-prose-words", type=int, default=120,
                        help="Discard documents with fewer prose words than this, "
                             "measured after cleaning.")
    parser.add_argument("--min-script-ratio", type=float, default=0.5,
                        help="Discard text with less than this fraction of Devanagari.")
    parser.add_argument("--max-sitemaps", type=int, default=200,
                        help="Cap on sub-sitemaps expanded per source.")
    parser.add_argument("--max-urls-per-source", type=int, default=150_000,
                        help="Cap on URLs queued per source.")
    parser.add_argument("--discover-only", action="store_true",
                        help="Build the frontier and stop, for review before fetching.")
    parser.add_argument("--rediscover", action="store_true",
                        help="Rebuild the frontier even if one already exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
