#!/usr/bin/env python3
"""
scripts/backfill_paper_metadata.py

Applies real title/authors/publication_year/venue from
experiments/datasets/paper_manifest.json onto papers that are ALREADY
ingested (matched by filename) but are missing that metadata -- e.g. the
5 real papers ingested via `main.py --dir ... ` before the --manifest
option existed (see main.py, pipeline/orchestrator.py).

Does NOT re-ingest, re-chunk, re-run Phase 2/3, or touch anything except
the `papers` table's title/authors/publication_year/venue columns, via the
real Repository.update_paper_metadata() (only sets fields that are
non-None and, for already-populated fields, ingest_paper's backfill path
additionally only fills genuinely-missing ones -- see that function's
docstring). Every value written comes directly from the manifest file,
never typed into this script -- run `git diff`/inspect the manifest
yourself to audit exactly what will be written before trusting the output.

Usage:
    python scripts/backfill_paper_metadata.py
    python scripts/backfill_paper_metadata.py --db experiments/runs/experiment.db
    python scripts/backfill_paper_metadata.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.repository import Repository  # noqa: E402
from main import load_metadata_by_filename  # noqa: E402

MANIFEST_PATH = "experiments/datasets/paper_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="experiments/runs/experiment.db")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would change without writing anything.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist.")

    metadata_by_filename = load_metadata_by_filename(args.manifest)
    repo = Repository(db_path)

    updated, unchanged, not_found = [], [], []
    for filename, meta in metadata_by_filename.items():
        paper = repo.get_paper_by_filename(filename)
        if paper is None:
            not_found.append(filename)
            continue
        missing = {k: v for k, v in meta.items() if v is not None and getattr(paper, k, None) is None}
        if not missing:
            unchanged.append((filename, paper.paper_id))
            continue
        print(f"{filename} (paper_id={paper.paper_id}): filling {sorted(missing)}")
        for k, v in missing.items():
            print(f"    {k}: {v!r}")
        if not args.dry_run:
            repo.update_paper_metadata(paper.paper_id, **missing)
        updated.append((filename, paper.paper_id, missing))

    print()
    print(f"{'Would update' if args.dry_run else 'Updated'}: {len(updated)}")
    print(f"Already had all available metadata: {len(unchanged)}")
    if not_found:
        print(f"In manifest but not found in {db_path} (not ingested under that filename): {not_found}")

    if not args.dry_run and updated:
        print("\nVerifying persisted values:")
        for filename, paper_id, _ in updated:
            p = repo.get_paper_by_id(paper_id)
            print(f"  {filename}: title={p.title!r} authors={p.authors!r} "
                  f"publication_year={p.publication_year!r} venue={p.venue!r}")

    return updated, unchanged, not_found


if __name__ == "__main__":
    main()
