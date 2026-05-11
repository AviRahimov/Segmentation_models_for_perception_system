#!/usr/bin/env python3
"""Auto-download the RUGD and ORFD off-road datasets.

For RUGD, ``--sequence`` (repeatable) selects which per-sequence folder
to fetch. The default ``creek`` keeps the download to ~620 MB instead
of pulling the full 5.3 GB archive.

For ORFD, the Google Drive file id is read from
``PERCEPTION_ORFD_GDRIVE_ID``; if unset the ORFD step logs a warning
and is skipped (no crash).

Examples
--------

    python scripts/download_datasets.py --sequence creek
    python scripts/download_datasets.py --dataset rugd --sequence trail-11
    python scripts/download_datasets.py --dataset orfd
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parent / "src").resolve()))

from perception.datasets.orfd import download_orfd  # noqa: E402
from perception.datasets.rugd import RUGD_SEQUENCES, download_rugd  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--dataset", choices=["rugd", "orfd", "all"], default="all")
    p.add_argument("--out", default="./datasets", help="Download root directory")
    p.add_argument(
        "--sequence",
        action="append",
        default=None,
        choices=list(RUGD_SEQUENCES),
        help=(
            "RUGD sequence to download (repeatable). Default: 'creek'. "
            "Pass multiple flags to fetch several sequences."
        ),
    )
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

    sequences = args.sequence if args.sequence else ["creek"]

    if args.dataset in ("rugd", "all"):
        download_rugd(out / "rugd", extract=extract, sequences=sequences)
    if args.dataset in ("orfd", "all"):
        download_orfd(out / "orfd", extract=extract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
