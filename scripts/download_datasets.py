#!/usr/bin/env python3
"""Auto-download the RUGD and ORFD off-road datasets.

Skips already-downloaded files (resumed via ``.part`` semantics in the
underlying downloader). Verifies SHA-256 when available. Extracts zip
archives once, marked by a ``<archive>.extracted`` sentinel file.

Examples
--------

    python scripts/download_datasets.py                 # both datasets
    python scripts/download_datasets.py --dataset rugd
    python scripts/download_datasets.py --out /data
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parent / "src").resolve()))

from perception.datasets.orfd import download_orfd  # noqa: E402
from perception.datasets.rugd import download_rugd  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", choices=["rugd", "orfd", "all"], default="all")
    p.add_argument("--out", default="./datasets", help="Download root directory")
    p.add_argument("--no-extract", action="store_true",
                   help="Skip extracting downloaded archives")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extract = not args.no_extract

    if args.dataset in ("rugd", "all"):
        download_rugd(out / "rugd", extract=extract)
    if args.dataset in ("orfd", "all"):
        download_orfd(out / "orfd", extract=extract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
