"""Registry of manual-collection sources for Hindi and Nepali.

Each source is a website we enumerate through its **sitemap** rather than by following
links. Sitemaps are the cheap path: one XML file lists thousands of article URLs
directly, so we skip the crawl-and-discover phase entirely and go straight to fetching
content.

Sitemap locations are auto-discovered at run time (robots.txt first, then the handful of
conventional paths) so this file does not go stale when a site reorganises. Explicit
``sitemaps`` entries below are used when a site advertises a better one than the default.

Every source here is a public news, government, or wiki site. Nothing behind a paywall or
a login.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Conventional sitemap locations, tried in order when robots.txt names none.
CANDIDATE_SITEMAP_PATHS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/wp-sitemap.xml",          # WordPress 5.5+
    "/sitemap-index.xml",
    "/sitemap/sitemap-index.xml",
)

# URLs that are listed in sitemaps but carry no usable prose: galleries, video pages,
# tag/category listings, author pages, static assets.
DEFAULT_EXCLUDE = re.compile(
    r"(/photos?/|/photo-gallery|/video[s]?/|/web-stories/|/tag[s]?/|/topic/|"
    r"/author/|/category/|/live-|/cricket-score|\.(jpg|jpeg|png|gif|pdf|mp4)$)",
    re.IGNORECASE,
)


@dataclass
class Source:
    """One website to collect from.

    Attributes:
        name: Short stable identifier, stored on every document as ``source``.
        lang: ``"hi"`` or ``"ne"``.
        home: Site root, used for sitemap auto-discovery.
        api: Collection strategy. ``"wordpress"`` uses the WordPress REST API, which
            returns 100 article bodies per request; ``None`` falls back to walking
            sitemaps and extracting each HTML page individually.
        archive_start: ``"YYYY-MM-DD"`` date of the oldest post, when the site's API
            cannot be trusted to report it. Some installations ignore
            ``orderby=date&order=asc`` and answer with a recent post instead, which
            would silently truncate collection to the last few months.
        sitemaps: Explicit sitemap URLs. Empty means auto-discover.
        include: Optional regex an article URL must match to be queued.
        exclude: Regex of URLs to skip. Defaults to the shared junk filter.
        notes: Why this source is here — quoted in the dataset statistics report.
    """

    name: str
    lang: str
    home: str
    api: str | None = None
    archive_start: str | None = None
    sitemaps: tuple[str, ...] = ()
    include: re.Pattern | None = None
    exclude: re.Pattern = field(default=DEFAULT_EXCLUDE)
    notes: str = ""

    def wants(self, url: str) -> bool:
        """Return True if this URL looks like a content page worth fetching."""
        if self.exclude and self.exclude.search(url):
            return False
        if self.include and not self.include.search(url):
            return False
        return url.startswith("http")


# --------------------------------------------------------------------------- Hindi

HINDI_SOURCES = [
    Source(
        name="jansatta",
        lang="hi",
        home="https://www.jansatta.com",
        api="wordpress",
        # Its API answers order=asc with a 2025 post, but posts dated 2014 are plainly
        # present in the archive, so the start date is pinned rather than probed.
        archive_start="2014-01-01",
        notes="Indian Express group's Hindi daily. WordPress REST API with a deep "
              "archive — the workhorse source for Hindi.",
    ),
    Source(
        name="thewirehindi",
        lang="hi",
        home="https://thewirehindi.com",
        api="wordpress",
        notes="Long-form Hindi journalism via WordPress API. Smaller archive than "
              "Jansatta but editorially careful prose.",
    ),
    Source(
        name="amarujala",
        lang="hi",
        home="https://www.amarujala.com",
        sitemaps=("https://www.amarujala.com/sitemap/sitemap-index.xml",),
        notes="National Hindi daily. Sitemap reaches deep archive pages; roughly a "
              "third survive cleaning, the rest being paywalled stubs.",
    ),
    Source(
        name="navbharattimes",
        lang="hi",
        home="https://navbharattimes.indiatimes.com",
        # The site-wide sitemapxml.cms lists only section landing pages, which carry no
        # article prose. The 48-hour news sitemap is the one that lists real articles;
        # it is a rolling window, so this source rewards being re-run periodically
        # rather than once.
        sitemaps=("https://navbharattimes.indiatimes.com/staticsitemap/nbt/news/sitemap-48hours.xml",),
        notes="Times group Hindi edition. Rolling 48-hour news sitemap only.",
    ),
    Source(
        name="bbc-hindi",
        lang="hi",
        home="https://www.bbc.com",
        include=re.compile(r"/hindi/"),
        notes="Editorially clean Hindi prose; useful counterweight to tabloid style.",
    ),
    Source(
        name="jagran",
        lang="hi",
        home="https://www.jagran.com",
        notes="Large-circulation Hindi daily.",
    ),
    Source(
        name="hi-wikisource",
        lang="hi",
        home="https://hi.wikisource.org",
        notes="Public-domain Hindi literature; long-form, edited text.",
    ),
]

# -------------------------------------------------------------------------- Nepali

# Note on eKantipur: deliberately excluded. Its robots.txt carries the Cloudflare
# "content signals" preamble, whose stated purpose is reserving rights against the
# ai-train use case. No explicit signal value is set, so nothing is formally forbidden,
# but the intent is clear enough that collecting from it for LM training is not worth
# the ambiguity. It also publishes no usable sitemap.

NEPALI_SOURCES = [
    Source(
        name="onlinekhabar",
        lang="ne",
        home="https://www.onlinekhabar.com",
        api="wordpress",
        notes="High-volume Nepali news portal. WordPress REST API returns 100 full "
              "articles per request, making it the workhorse source for Nepali.",
    ),
    Source(
        name="setopati",
        lang="ne",
        home="https://www.setopati.com",
        notes="Major Nepali online daily.",
    ),
    Source(
        name="ratopati",
        lang="ne",
        home="https://www.ratopati.com",
        notes="Nepali news portal; adds topical breadth.",
    ),
    Source(
        name="nagariknews",
        lang="ne",
        home="https://www.nagariknews.com.np",
        notes="Nepali daily, print-derived content.",
    ),
    Source(
        name="bbc-nepali",
        lang="ne",
        home="https://www.bbc.com",
        include=re.compile(r"/nepali/"),
        notes="Edited Nepali prose.",
    ),
    Source(
        name="ne-wikisource",
        lang="ne",
        home="https://ne.wikisource.org",
        notes="Public-domain Nepali texts; scarce but high quality.",
    ),
]

SOURCES: dict[str, list[Source]] = {
    "hi": HINDI_SOURCES,
    "ne": NEPALI_SOURCES,
}


def get_sources(lang: str, only: list[str] | None = None) -> list[Source]:
    """Return the configured sources for a language.

    Args:
        lang: ``"hi"`` or ``"ne"``.
        only: Optional list of source names to restrict to — useful for testing a
            single site before launching a full run.

    Raises:
        ValueError: If the language is unknown or a requested source name is not found.
    """
    if lang not in SOURCES:
        raise ValueError(f"Unknown language {lang!r}; expected one of {list(SOURCES)}")

    sources = SOURCES[lang]
    if only:
        by_name = {s.name: s for s in sources}
        missing = set(only) - set(by_name)
        if missing:
            raise ValueError(
                f"Unknown source(s) for {lang}: {sorted(missing)}. "
                f"Available: {sorted(by_name)}"
            )
        sources = [by_name[n] for n in only]
    return sources
