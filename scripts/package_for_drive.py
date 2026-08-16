"""Package the large Phase 1 artifacts for upload to Google Drive.

The assignment requires datasets and tokenized corpora to live on Google Drive with
shareable links in the README, because they are far too large to commit to git. This
script bundles them into a small number of archives, records checksums, and generates the
README table rows to paste once the links exist.

Two profiles are available, and ``full`` is the default because a complete bundle is worth
more than a compact one when the corpus cannot be rebuilt from scratch.

* ``full`` (~11.1 GB) — every artifact: raw scraped shards, downloaded public shards, the
  cleaned corpus, the splits, both tokenizers with all four candidate vocabulary sizes and
  their training samples, plus reports and logs. Anyone with this can reproduce Phase 2
  and Phase 3 without re-running collection.
* ``minimal`` (~6.1 GB) — drops ``data/downloaded`` and ``<lang>/data/clean``. Those are
  the two recoverable pieces: the downloaded shards can be re-streamed from HuggingFace
  using the record counts in ``download-progress.json``, and the cleaned corpus is exactly
  what the splits contain, merely unpartitioned.

What is irreplaceable either way is ``data/manual``. News sites change, articles are
removed, and walking the same archives tomorrow will not return the same set. It is also
the evidence behind the >=20% manual-collection requirement.

Uploading itself needs a tool with Drive credentials. If ``rclone`` is configured, pass
``--rclone <remote>:<path>`` and the archives are pushed automatically. Otherwise the
archives are written locally for a browser upload.

Examples:
    python -m scripts.package_for_drive --out drive_upload
    python -m scripts.package_for_drive --out drive_upload --profile minimal
    python -m scripts.package_for_drive --out drive_upload --rclone gdrive:LMA/phase1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Each entry: archive name -> (paths to include, profile, description for the README).
# Paths that do not exist are skipped rather than failing the run, so this works before
# every stage has been executed. Nothing inside a listed path is filtered out: the Drive
# copy is a complete bundle, including the SentencePiece training samples, so the
# tokenizers can be reproduced byte-for-byte rather than merely re-derived.
ARTIFACT_GROUPS: dict[str, dict] = {
    "hindi-manual-corpus": {
        "paths": ["data/manual/hi"],
        "profile": "minimal",
        "description": "Hindi manually scraped shards (Jansatta, The Wire Hindi)",
    },
    "nepali-manual-corpus": {
        "paths": ["data/manual/ne"],
        "profile": "minimal",
        "description": "Nepali manually scraped shards (Onlinekhabar)",
    },
    "hindi-splits": {
        "paths": ["hindi/data/splits"],
        "profile": "minimal",
        "description": "Hindi train/validation/test splits (input to Phase 2)",
    },
    "nepali-splits": {
        "paths": ["nepali/data/splits"],
        "profile": "minimal",
        "description": "Nepali train/validation/test splits (input to Phase 2)",
    },
    "tokenizers": {
        "paths": ["hindi/tokenizer", "nepali/tokenizer"],
        "profile": "minimal",
        "description": "Trained SentencePiece models, vocabularies, all four candidate "
                       "vocabulary sizes, and the training samples they were built from",
    },
    "reports-and-logs": {
        "paths": ["report", "logs"],
        "profile": "minimal",
        "description": "Statistics JSON, figures, phase report and all run logs",
    },
    "hindi-downloaded-corpus": {
        "paths": ["data/downloaded/hi"],
        "profile": "full",
        "description": "Hindi public-corpus shards (FineWeb-2, Wikipedia) — regenerable",
    },
    "nepali-downloaded-corpus": {
        "paths": ["data/downloaded/ne"],
        "profile": "full",
        "description": "Nepali public-corpus shards — regenerable",
    },
    "hindi-clean-corpus": {
        "paths": ["hindi/data/clean"],
        "profile": "full",
        "description": "Hindi cleaned corpus before splitting — redundant with splits",
    },
    "nepali-clean-corpus": {
        "paths": ["nepali/data/clean"],
        "profile": "full",
        "description": "Nepali cleaned corpus before splitting — redundant with splits",
    },
}


def directory_size(path: Path) -> int:
    """Return the total size in bytes of every file under a path."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in chunks so memory stays bounded.

    Checksums let whoever downloads an archive confirm it arrived intact — worth having
    for multi-gigabyte files moved through a browser.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(
    name: str, paths: list[Path], out_dir: Path, exclude_suffixes: tuple[str, ...] = ()
) -> Path | None:
    """Bundle one artifact group into a single ``.tar`` file.

    No compression is applied. Every shard is already zstd-compressed, so compressing
    again costs CPU and saves essentially nothing.

    Args:
        name: Archive base name.
        paths: Directories or files to include.
        out_dir: Where to write the archive.
        exclude_suffixes: Skip files ending with any of these.

    Returns:
        Path to the archive, or None if none of the inputs existed.
    """
    present = [p for p in paths if p.exists()]
    if not present:
        print(f"  [{name}] no inputs present, skipping", flush=True)
        return None

    archive_path = out_dir / f"{name}.tar"
    started = time.monotonic()

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        """Filter callback dropping excluded files from the archive."""
        if any(info.name.endswith(suffix) for suffix in exclude_suffixes):
            return None
        return info

    with tarfile.open(archive_path, "w") as tar:
        for path in present:
            tar.add(path, arcname=path.as_posix(), filter=keep)

    size = archive_path.stat().st_size
    print(f"  [{name}] {size/1e9:.2f} GB in {time.monotonic()-started:.0f}s", flush=True)
    return archive_path


def upload_with_rclone(archive: Path, remote: str) -> bool:
    """Copy one archive to a configured rclone remote.

    Returns:
        True if rclone reported success.
    """
    print(f"    uploading {archive.name} -> {remote} ...", flush=True)
    result = subprocess.run(
        ["rclone", "copy", "--progress", str(archive), remote],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    rclone failed: {result.stderr[:300]}", flush=True)
        return False
    return True


def run(args: argparse.Namespace) -> None:
    """Package the selected artifact groups, and optionally upload them."""
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rclone and not shutil.which("rclone"):
        print("rclone requested but not installed. Install it and run "
              "`rclone config` once to authorise Google Drive, or omit --rclone "
              "and upload the archives through the browser.", flush=True)
        return

    selected = {
        name: spec for name, spec in ARTIFACT_GROUPS.items()
        if args.profile == "full" or spec["profile"] == "minimal"
    }

    print(f"Packaging profile={args.profile} -> {out_dir}", flush=True)
    total_source = sum(
        directory_size(root / p) for spec in selected.values()
        for p in spec["paths"] if (root / p).exists()
    )
    print(f"  {len(selected)} groups, {total_source/1e9:.2f} GB of source data\n", flush=True)

    manifest = {
        "profile": args.profile,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archives": [],
    }

    for name, spec in selected.items():
        archive = build_archive(
            name,
            [root / p for p in spec["paths"]],
            out_dir,
            spec.get("exclude_suffixes", ()),
        )
        if archive is None:
            continue

        entry = {
            "archive": archive.name,
            "description": spec["description"],
            "bytes": archive.stat().st_size,
            "size_human": f"{archive.stat().st_size/1e9:.2f} GB",
            "sha256": file_sha256(archive) if args.checksums else None,
            "drive_link": "PASTE_LINK_HERE",
        }
        if args.rclone:
            entry["uploaded"] = upload_with_rclone(archive, args.rclone)
        manifest["archives"].append(entry)

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # A ready-to-paste README fragment, so the links table does not have to be
    # reconstructed by hand once the uploads finish.
    rows = ["| Artifact | Size | SHA-256 | Link |", "|---|---|---|---|"]
    for entry in manifest["archives"]:
        checksum = (entry["sha256"][:16] + "…") if entry["sha256"] else "—"
        rows.append(
            f"| {entry['description']} | {entry['size_human']} | `{checksum}` | "
            f"_paste link_ |"
        )
    (out_dir / "README-rows.md").write_text("\n".join(rows) + "\n")

    total = sum(e["bytes"] for e in manifest["archives"])
    print(f"\n=== packaged {len(manifest['archives'])} archives, "
          f"{total/1e9:.2f} GB ===", flush=True)
    print(f"  manifest    : {manifest_path}", flush=True)
    print(f"  README rows : {out_dir/'README-rows.md'}", flush=True)
    if not args.rclone:
        print("\n  Next: upload the .tar files in this directory to Google Drive,", flush=True)
        print("  set each to 'Anyone with the link can view', and paste the links", flush=True)
        print("  into the table in README.md.", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="drive_upload",
                        help="Directory to write archives into.")
    parser.add_argument("--profile", choices=["minimal", "full"], default="full",
                        help="full (default): every artifact, ~11.1 GB. minimal: drops "
                             "the downloaded shards and the pre-split clean corpus, "
                             "both of which are recoverable, ~6.1 GB.")
    parser.add_argument("--rclone", default=None,
                        help="rclone destination, e.g. gdrive:LMA/phase1. Requires "
                             "rclone installed and configured.")
    parser.add_argument("--no-checksums", dest="checksums", action="store_false",
                        help="Skip SHA-256 computation (faster on large archives).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
