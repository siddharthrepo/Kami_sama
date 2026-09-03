"""Collection via the WordPress REST API.

Sites running WordPress expose ``/wp-json/wp/v2/posts``, which returns article bodies as
structured JSON. Where it is available this beats HTML scraping on every axis:

* **100 articles per request** instead of one, so a 150,000-article archive costs ~1,500
  requests rather than 150,000.
* **The body arrives already isolated** — no navigation, no ads, no related-story boxes
  to guess at.
* **Far gentler on the site**, which matters when we are collecting at volume.

Pagination walks *backwards through time* using the ``before`` parameter rather than
``page=N``. Deep page offsets get slow and many installations refuse them past a certain
depth, whereas a date cursor walks an archive of any size and resumes exactly where it
stopped.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator

import trafilatura

from common.clean import clean_text, devanagari_ratio, is_good_document
from common.fetch import PoliteFetcher
from common.schema import MANUAL, AtomicJsonlWriter, Document
from common.sources import Source

# Fields we ask for. Requesting only these keeps responses several times smaller.
WP_FIELDS = "id,link,title,content,date"

# Fallback tag stripper, used when trafilatura declines a short HTML fragment.
TAG_RE = re.compile(r"<[^>]+>")


def html_fragment_to_text(html: str) -> str:
    """Convert a WordPress ``content.rendered`` HTML fragment to plain text.

    trafilatura expects a whole page, so the fragment is wrapped in a minimal document
    before extraction. Very short fragments are sometimes rejected outright, in which
    case a plain tag strip is good enough — the fragment is already only article body.
    """
    wrapped = f"<html><body><article>{html}</article></body></html>"
    text = trafilatura.extract(
        wrapped, include_comments=False, include_tables=False, favor_precision=True
    )
    if text:
        return text
    stripped = TAG_RE.sub("\n", html)
    return re.sub(r"\n{3,}", "\n\n", stripped)


async def iter_wp_posts(
    fetcher: PoliteFetcher,
    source: Source,
    max_posts: int,
    state_dir: Path,
    per_page: int = 100,
    min_prose_words: int = 120,
) -> AsyncIterator[Document]:
    """Yield cleaned Documents from a WordPress site, newest first.

    Args:
        fetcher: Shared polite HTTP client.
        source: The source definition; ``source.home`` must host ``/wp-json``.
        max_posts: Stop after yielding this many accepted documents.
        state_dir: Directory holding the resume cursor for this source.
        per_page: Posts per request. WordPress caps this at 100.
        min_prose_words: Quality gate applied to the cleaned body.

    Yields:
        Documents that passed cleaning and quality checks.
    """
    cursor_path = Path(state_dir) / f"wp-cursor-{source.name}.txt"
    before = cursor_path.read_text().strip() if cursor_path.exists() else None
    if before:
        print(f"  [{source.name}] resuming from {before}", flush=True)

    yielded = 0
    empty_responses = 0

    while yielded < max_posts:
        url = (
            f"{source.home}/wp-json/wp/v2/posts"
            f"?per_page={per_page}&orderby=date&order=desc&_fields={WP_FIELDS}"
        )
        if before:
            url += f"&before={before}"

        body = await fetcher.get(url, check_robots=False)
        if body is None:
            print(f"  [{source.name}] request failed, stopping", flush=True)
            break

        try:
            posts = json.loads(body)
        except json.JSONDecodeError:
            print(f"  [{source.name}] non-JSON response, stopping", flush=True)
            break

        if not isinstance(posts, list) or not posts:
            empty_responses += 1
            if empty_responses >= 2:  # archive exhausted
                print(f"  [{source.name}] archive exhausted", flush=True)
                break
            continue
        empty_responses = 0

        for post in posts:
            raw_html = (post.get("content") or {}).get("rendered", "")
            if not raw_html:
                continue

            text = clean_text(html_fragment_to_text(raw_html))
            if not is_good_document(text, min_prose_words=min_prose_words):
                continue

            title_html = (post.get("title") or {}).get("rendered", "")
            yield Document(
                text=text,
                lang=source.lang,
                source_type=MANUAL,
                source=source.name,
                url=post.get("link", ""),
                title=TAG_RE.sub("", title_html).strip(),
            )
            yielded += 1
            if yielded >= max_posts:
                break

        # Advance the cursor to the oldest post seen, so the next request continues
        # further back in time. Persisting it makes the whole walk resumable.
        oldest = posts[-1].get("date")
        if not oldest or oldest == before:
            print(f"  [{source.name}] cursor stalled, stopping", flush=True)
            break
        before = oldest
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(before)


# ------------------------------------------------------- sharded parallel collection

async def archive_start_date(fetcher: PoliteFetcher, source: Source) -> datetime | None:
    """Return the publication date of the site's oldest post.

    Asking for one post in ascending date order costs a single request and tells us how
    far back the archive reaches, which is what bounds the set of windows to collect.
    """
    url = (
        f"{source.home}/wp-json/wp/v2/posts"
        f"?per_page=1&orderby=date&order=asc&_fields=date"
    )
    body = await fetcher.get(url, check_robots=False)
    if body is None:
        return None
    try:
        posts = json.loads(body)
        return datetime.fromisoformat(posts[0]["date"])
    except (json.JSONDecodeError, IndexError, KeyError, ValueError):
        return None


def month_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime, str]]:
    """Split a date range into one window per calendar month, newest first.

    Monthly granularity is deliberate. Whole-year windows would be badly unbalanced —
    a busy news site publishes far more in 2025 than in 2012 — so workers assigned the
    heavy years would still be grinding while everyone else sat idle. Many small windows
    let each worker take more work as it frees up, which keeps them all busy until the
    end. Newest-first ordering means an interrupted run still yields recent, and
    generally more useful, text.

    Args:
        start: Earliest date to cover.
        end: Latest date to cover.

    Returns:
        ``(after, before, label)`` triples, where label looks like ``"2015-03"``.
    """
    windows: list[tuple[datetime, datetime, str]] = []
    cursor = datetime(start.year, start.month, 1)

    while cursor < end:
        if cursor.month == 12:
            nxt = datetime(cursor.year + 1, 1, 1)
        else:
            nxt = datetime(cursor.year, cursor.month + 1, 1)
        windows.append((cursor, min(nxt, end), f"{cursor.year:04d}-{cursor.month:02d}"))
        cursor = nxt

    return list(reversed(windows))


async def fetch_window(
    fetcher: PoliteFetcher,
    source: Source,
    after: datetime,
    before: datetime,
    label: str,
    out_dir: Path,
    min_prose_words: int = 120,
    min_script_ratio: float = 0.5,
    per_page: int = 100,
) -> tuple[int, str]:
    """Collect every post in one date window into a single shard file.

    The window is walked newest-to-oldest with a local cursor. Because ``after`` stays
    fixed, this worker can never wander into another worker's window, so the shards stay
    disjoint and no coordination is needed between workers.

    Args:
        fetcher: Shared polite HTTP client.
        source: The WordPress source.
        after: Window start (exclusive lower bound).
        before: Window end (exclusive upper bound).
        label: Window label used in the filename, e.g. ``"2015-03"``.
        out_dir: Directory to write the shard into.
        min_prose_words: Quality gate applied to the cleaned body.
        min_script_ratio: Minimum Devanagari fraction.
        per_page: Posts per request; WordPress caps this at 100.

    Returns:
        A ``(documents_kept, status)`` pair, where status is ``"done"`` or ``"skipped"``.
    """
    path = Path(out_dir) / f"shard-{source.name}-{label}.jsonl.zst"
    if path.exists():
        return 0, "skipped"  # completed by an earlier run

    cursor = before.isoformat()
    after_iso = after.isoformat()
    seen_ids: set[int] = set()
    kept = 0

    with AtomicJsonlWriter(path) as writer:
        while True:
            url = (
                f"{source.home}/wp-json/wp/v2/posts"
                f"?per_page={per_page}&orderby=date&order=desc&_fields={WP_FIELDS}"
                f"&after={after_iso}&before={cursor}"
            )
            body = await fetcher.get(url, check_robots=False)
            if body is None:
                break

            try:
                posts = json.loads(body)
            except json.JSONDecodeError:
                break
            if not isinstance(posts, list) or not posts:
                break

            for post in posts:
                post_id = post.get("id")
                if post_id in seen_ids:  # posts sharing a timestamp can repeat
                    continue
                seen_ids.add(post_id)

                raw_html = (post.get("content") or {}).get("rendered", "")
                if not raw_html:
                    continue

                text = clean_text(html_fragment_to_text(raw_html))
                if not is_good_document(text, min_prose_words=min_prose_words):
                    continue
                if devanagari_ratio(text) < min_script_ratio:
                    continue

                title_html = (post.get("title") or {}).get("rendered", "")
                writer.write(
                    Document(
                        text=text,
                        lang=source.lang,
                        source_type=MANUAL,
                        source=source.name,
                        url=post.get("link", ""),
                        title=TAG_RE.sub("", title_html).strip(),
                    )
                )
                kept += 1

            # A short page means the window is exhausted.
            if len(posts) < per_page:
                break

            oldest = posts[-1].get("date")
            if not oldest or oldest == cursor:
                break
            cursor = oldest

    return kept, "done"


async def collect_wp_sharded(
    fetcher: PoliteFetcher,
    source: Source,
    out_dir: Path,
    workers: int = 8,
    min_prose_words: int = 120,
    min_script_ratio: float = 0.5,
    max_docs: int | None = None,
) -> int:
    """Collect a WordPress archive in parallel, one shard file per month.

    Windows are handed out through a queue, so a worker that finishes a quiet month
    immediately picks up the next unclaimed one instead of idling.

    Cursor-chained collection is serial by construction — each request needs the
    previous request's date — so the whole archive takes as long as its request count.
    Splitting the timeline first removes that dependency: the chains become independent
    and run concurrently.

    Args:
        fetcher: Shared polite HTTP client. Its per-domain delay still applies, so
            raising ``workers`` alone will not exceed the configured request rate.
        source: The WordPress source to collect.
        out_dir: Directory for shard files.
        workers: Number of windows fetched concurrently.
        min_prose_words: Quality gate applied to the cleaned body.
        min_script_ratio: Minimum Devanagari fraction.
        max_docs: Stop once this many documents have been kept. Approximate — in-flight
            windows are allowed to finish so their shards stay complete.

    Returns:
        Total documents kept across all windows this run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A pinned archive_start always wins. Probing is only a fallback, because some
    # installations answer order=asc with a recent post — which would quietly shrink the
    # window list to the last few months and cap the corpus without any error.
    if source.archive_start:
        start = datetime.fromisoformat(source.archive_start)
        print(f"  [{source.name}] using pinned archive start {start:%Y-%m-%d}", flush=True)
    else:
        start = await archive_start_date(fetcher, source)
        if start is None:
            print(f"  [{source.name}] could not read archive start date", flush=True)
            return 0

    windows = month_windows(start, datetime.now())
    done_already = sum(
        1 for _, _, lbl in windows
        if (out_dir / f"shard-{source.name}-{lbl}.jsonl.zst").exists()
    )
    print(
        f"  [{source.name}] archive starts {start:%Y-%m-%d}; "
        f"{len(windows)} monthly windows, {done_already} already complete",
        flush=True,
    )

    queue: asyncio.Queue = asyncio.Queue()
    for window in windows:
        queue.put_nowait(window)

    totals = {"kept": 0, "done": 0, "skipped": 0}

    async def worker() -> None:
        """Take windows off the queue until it is empty."""
        while True:
            if max_docs is not None and totals["kept"] >= max_docs:
                return
            try:
                after, before, label = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                kept, status = await fetch_window(
                    fetcher, source, after, before, label, out_dir,
                    min_prose_words, min_script_ratio,
                )
                totals["kept"] += kept
                totals[status] += 1
                if status == "done":
                    print(
                        f"    [{source.name}] {label}: {kept:,} docs "
                        f"({totals['done']}/{len(windows) - done_already} windows, "
                        f"{totals['kept']:,} total)",
                        flush=True,
                    )
            except Exception as exc:  # a bad window must not kill the run
                print(f"    [{source.name}] {label}: FAILED {type(exc).__name__}: {exc}",
                      flush=True)
            finally:
                queue.task_done()

    await asyncio.gather(*[worker() for _ in range(workers)])

    print(
        f"  [{source.name}] finished: {totals['kept']:,} documents kept across "
        f"{totals['done']} windows ({totals['skipped']} skipped)",
        flush=True,
    )
    return totals["kept"]
