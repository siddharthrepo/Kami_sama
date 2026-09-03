"""Polite asynchronous HTTP fetching.

The scraper hits public news sites and wikis at volume, so it has to behave. This module
enforces three things:

1. **robots.txt is obeyed** — every URL is checked against the origin's rules before the
   request is made, and the result is cached per domain.
2. **Per-domain rate limiting** — requests to any one domain are serialised with a
   minimum delay between them, so we never hammer a single server regardless of how high
   global concurrency is set.
3. **Bounded retries with backoff** — transient failures are retried a couple of times;
   permanent ones are given up on quickly rather than blocking the queue.

Provenance matters for the report, so the User-Agent identifies the crawler honestly
rather than impersonating a browser.
"""

from __future__ import annotations

import asyncio
import gzip
import time
import urllib.robotparser
from collections import defaultdict
from urllib.parse import urlparse

import httpx

USER_AGENT = (
    "LMA-Course-Crawler/1.0 (academic coursework; monolingual corpus collection; "
    "contact: jainvidhi2702@gmail.com)"
)


class PoliteFetcher:
    """An async HTTP client that respects robots.txt and per-domain rate limits."""

    def __init__(
        self,
        concurrency: int = 16,
        per_domain_delay: float = 1.0,
        timeout: float = 20.0,
        max_retries: int = 2,
        obey_robots: bool = True,
    ) -> None:
        """Configure the fetcher.

        Args:
            concurrency: Maximum in-flight requests across all domains.
            per_domain_delay: Minimum seconds between two requests to the same domain.
            timeout: Per-request timeout in seconds.
            max_retries: Retry attempts for transient errors (timeouts, 5xx, 429).
            obey_robots: Check robots.txt before fetching. Leave enabled.
        """
        self.per_domain_delay = per_domain_delay
        self.max_retries = max_retries
        self.obey_robots = obey_robots

        self._semaphore = asyncio.Semaphore(concurrency)
        self._domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_request: dict[str, float] = defaultdict(float)
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._robots_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "hi,ne,en;q=0.5"},
            limits=httpx.Limits(max_connections=concurrency * 2),
        )

    # ---------------------------------------------------------------- robots.txt

    async def _get_robots(self, domain: str):
        """Fetch and cache the robots.txt parser for one domain.

        A domain whose robots.txt is missing or unreachable is treated as permissive,
        which matches standard crawler behaviour.
        """
        async with self._robots_locks[domain]:
            if domain in self._robots:
                return self._robots[domain]

            parser = urllib.robotparser.RobotFileParser()
            try:
                response = await self._client.get(f"https://{domain}/robots.txt")
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser = None  # no usable rules -> allow
            except Exception:
                parser = None

            self._robots[domain] = parser
            return parser

    async def allowed(self, url: str) -> bool:
        """Return True if robots.txt permits fetching this URL."""
        if not self.obey_robots:
            return True
        domain = urlparse(url).netloc
        parser = await self._get_robots(domain)
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)

    # ------------------------------------------------------------------ fetching

    async def _throttle(self, domain: str) -> None:
        """Block until enough time has passed since the last hit on this domain."""
        async with self._domain_locks[domain]:
            elapsed = time.monotonic() - self._last_request[domain]
            if elapsed < self.per_domain_delay:
                await asyncio.sleep(self.per_domain_delay - elapsed)
            self._last_request[domain] = time.monotonic()

    async def get(self, url: str, check_robots: bool = True) -> bytes | None:
        """Fetch a URL and return its raw body, or None if it could not be retrieved.

        Returns bytes rather than text so that gzipped sitemaps and HTML with declared
        encodings are both handled correctly by the caller.

        Args:
            url: Absolute URL to fetch.
            check_robots: Whether to consult robots.txt first.

        Returns:
            The response body, or None on disallowed / failed / non-200 responses.
        """
        if check_robots and not await self.allowed(url):
            return None

        domain = urlparse(url).netloc

        for attempt in range(self.max_retries + 1):
            async with self._semaphore:
                await self._throttle(domain)
                try:
                    response = await self._client.get(url)
                except (httpx.TimeoutException, httpx.TransportError):
                    if attempt == self.max_retries:
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue

            if response.status_code == 200:
                return response.content

            # 429 and 5xx are worth another try; everything else is permanent.
            if response.status_code in (429,) or response.status_code >= 500:
                if attempt == self.max_retries:
                    return None
                await asyncio.sleep(2 ** attempt * 2)
                continue

            return None

        return None

    async def get_text(self, url: str, check_robots: bool = True) -> str | None:
        """Fetch a URL and decode it as UTF-8 text, transparently gunzipping if needed.

        Sitemaps are frequently served as ``.xml.gz``, so gzip is detected by magic
        number rather than by file extension.
        """
        body = await self.get(url, check_robots=check_robots)
        if body is None:
            return None
        if body[:2] == b"\x1f\x8b":  # gzip magic number
            try:
                body = gzip.decompress(body)
            except Exception:
                return None
        return body.decode("utf-8", errors="replace")

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> "PoliteFetcher":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
