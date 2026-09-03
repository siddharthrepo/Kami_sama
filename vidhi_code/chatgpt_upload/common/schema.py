"""Corpus document schema and sharded, compressed I/O.

Every document collected in Phase 1 — whether scraped by us or downloaded from a public
dataset — is stored as one JSON object per line inside zstd-compressed shards
(``.jsonl.zst``). Compression matters: Devanagari is 3 bytes per character in UTF-8, and
zstd gets roughly 4:1 on this kind of text, which is the difference between the corpus
fitting on disk and not.

The ``source_type`` field is mandatory on every record. It is what lets us report the
manual-vs-downloaded token split that the assignment requires, and it must be set at
write time — it cannot be reconstructed later.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import zstandard as zstd

# The two provenance classes. At least 20% of final training tokens must be MANUAL.
MANUAL = "manual"
DOWNLOADED = "downloaded"
SOURCE_TYPES = (MANUAL, DOWNLOADED)


@dataclass
class Document:
    """A single collected document.

    Attributes:
        text: The extracted plain text body. No HTML, no navigation boilerplate.
        lang: Target language code, ``"hi"`` or ``"ne"``. Set by the collector, and
            re-verified by language identification in the cleaning stage.
        source_type: ``"manual"`` (we scraped it) or ``"downloaded"`` (public dataset).
        source: Short stable name of the origin, e.g. ``"amarujala"`` or ``"cc100"``.
        url: Origin URL for scraped documents; empty for dataset documents.
        title: Page title where the extractor could find one.
        fetched_at: Unix timestamp of collection.
        doc_id: Stable content hash, used for exact deduplication and for joining
            statistics back to documents. Derived automatically when not supplied.
    """

    text: str
    lang: str
    source_type: str
    source: str
    url: str = ""
    title: str = ""
    fetched_at: float = field(default_factory=time.time)
    doc_id: str = ""

    def __post_init__(self) -> None:
        """Validate provenance and derive a stable document id."""
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(
                f"source_type must be one of {SOURCE_TYPES}, got {self.source_type!r}"
            )
        if not self.doc_id:
            self.doc_id = self.content_hash(self.text)

    @staticmethod
    def content_hash(text: str) -> str:
        """Return a short stable hash of document text, used as the document id.

        Hashing the *text* rather than the URL means the same article published at two
        different URLs collapses to one id, which is what exact dedup needs.
        """
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def to_json(self) -> str:
        """Serialise to a single JSON line, preserving Indic characters verbatim."""
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Document":
        """Reconstruct a Document from one JSON line."""
        return cls(**json.loads(line))


class ShardWriter:
    """Writes Documents to rotating zstd-compressed JSONL shards.

    Sharding keeps any single file small enough to reprocess or re-upload
    independently, and means an interrupted run loses at most one partial shard rather
    than the whole corpus.

    Use as a context manager::

        with ShardWriter(Path("data/manual/hi"), "hi-manual") as w:
            w.write(doc)
    """

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        max_docs_per_shard: int = 25_000,
        compression_level: int = 10,
    ) -> None:
        """Prepare an output directory for sharded writing.

        Args:
            out_dir: Directory to write shards into. Created if absent.
            prefix: Filename prefix, e.g. ``"hi-manual"``.
            max_docs_per_shard: Rotate to a new shard after this many documents.
            compression_level: zstd level. 10 is a good speed/ratio balance for text.
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_docs_per_shard = max_docs_per_shard
        self.compressor = zstd.ZstdCompressor(level=compression_level)

        self.docs_written = 0        # total across all shards
        self.chars_written = 0       # total characters, for statistics
        self._shard_docs = 0         # documents in the current shard
        self._shard_index = self._next_shard_index()
        self._fh = None
        self._writer = None

    def _next_shard_index(self) -> int:
        """Return the next unused shard number, so reruns append instead of clobber."""
        existing = sorted(self.out_dir.glob(f"{self.prefix}-*.jsonl.zst"))
        if not existing:
            return 0
        last = existing[-1].name.replace(f"{self.prefix}-", "").split(".")[0]
        return int(last) + 1

    def _open_shard(self) -> None:
        """Open a fresh shard file and its streaming compressor."""
        path = self.out_dir / f"{self.prefix}-{self._shard_index:05d}.jsonl.zst"
        self._fh = open(path, "wb")
        self._writer = self.compressor.stream_writer(self._fh)
        self._shard_docs = 0

    def _close_shard(self) -> None:
        """Flush and close the current shard, if one is open."""
        if self._writer is not None:
            self._writer.close()
            self._fh.close()
            self._writer = None
            self._fh = None

    def write(self, doc: Document) -> None:
        """Append one document, rotating to a new shard when the current one is full."""
        if self._writer is None:
            self._open_shard()
        elif self._shard_docs >= self.max_docs_per_shard:
            self._close_shard()
            self._shard_index += 1
            self._open_shard()

        self._writer.write((doc.to_json() + "\n").encode("utf-8"))
        self._shard_docs += 1
        self.docs_written += 1
        self.chars_written += len(doc.text)

    def close(self) -> None:
        """Close the writer. Safe to call more than once."""
        self._close_shard()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AtomicJsonlWriter:
    """Writes Documents to one ``.jsonl.zst`` file that appears only when complete.

    Output goes to a ``.tmp`` path and is renamed into place on clean exit. Because
    rename is atomic on POSIX filesystems, a file under its final name is guaranteed to
    be whole — which lets the *existence of the file* serve as the completion marker for
    a work unit. A crash leaves only a ``.tmp``, which the next run discards and redoes.

    This is what makes sharded collection restartable without a separate bookkeeping
    file, and without the race a separate marker would introduce (dying after writing
    the data but before writing the flag).
    """

    def __init__(self, path: Path, compression_level: int = 10) -> None:
        """Prepare to write one shard.

        Args:
            path: Final destination path, ending in ``.jsonl.zst``.
            compression_level: zstd level.
        """
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.docs_written = 0
        self.chars_written = 0
        self._fh = None
        self._writer = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self._fh = open(self.tmp_path, "wb")
        self._writer = self.compressor.stream_writer(self._fh)
        return self

    def write(self, doc: Document) -> None:
        """Append one document to the in-progress shard."""
        self._writer.write((doc.to_json() + "\n").encode("utf-8"))
        self.docs_written += 1
        self.chars_written += len(doc.text)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Finalise the shard: rename into place on success, discard on failure."""
        if self._writer is not None:
            self._writer.close()
            self._fh.close()

        if exc_type is None:
            self.tmp_path.rename(self.path)
        else:
            self.tmp_path.unlink(missing_ok=True)


def read_shards(path: Path, pattern: str = "*.jsonl.zst") -> Iterator[Document]:
    """Stream every Document from a directory of shards (or a single shard file).

    Streaming rather than loading into memory is deliberate — the full corpus is far
    larger than available RAM.

    Args:
        path: A shard file, or a directory containing shards.
        pattern: Glob used when ``path`` is a directory.

    Yields:
        Each Document in filename order.
    """
    path = Path(path)
    files = sorted(path.glob(pattern)) if path.is_dir() else [path]
    decompressor = zstd.ZstdDecompressor()

    for shard in files:
        with open(shard, "rb") as fh:
            with decompressor.stream_reader(fh) as reader:
                buffer = b""
                while True:
                    chunk = reader.read(1 << 20)  # 1 MiB at a time
                    if not chunk:
                        break
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    for line in lines:
                        if line.strip():
                            yield Document.from_json(line.decode("utf-8"))
                if buffer.strip():
                    yield Document.from_json(buffer.decode("utf-8"))
