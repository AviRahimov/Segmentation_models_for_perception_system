"""RUGD off-road dataset downloader (per-sequence, budget-friendly).

The official RUGD distribution at http://rugd.vision/ ships a single
5.3 GB ``RUGD_frames-with-annotations.zip`` covering all 18 sequences;
the maintainers do not publish per-sequence ZIPs at predictable URLs.
This module first probes the rugd.vision per-sequence URL pattern (in
case it is ever provisioned) and falls back to the community
HuggingFace mirror at ``WilliamBonilla62/RUGD``, which preserves the
upstream layout and exposes one folder per sequence — letting us pull a
single ~tens-to-hundreds-of-MB sequence instead of the full archive.

The 18 published sequences are hardcoded in :data:`RUGD_SEQUENCES`.
The caller (e.g. the eval script) typically passes a single sequence to
keep the download tiny.
"""
from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import requests

from .downloader import http_download

logger = logging.getLogger(__name__)


# Hardcoded RUGD sequence names. Order is informational only.
RUGD_SEQUENCES: tuple[str, ...] = (
    "creek",
    "park-1",
    "park-2",
    "park-8",
    "trail",
    "trail-3",
    "trail-4",
    "trail-5",
    "trail-6",
    "trail-7",
    "trail-9",
    "trail-10",
    "trail-11",
    "trail-12",
    "trail-13",
    "trail-14",
    "trail-15",
    "village",
)

# Legacy alias retained so tests / callers that imported the old name
# from the package keep working (``from perception.datasets import RUGD_FILES``).
RUGD_FILES: tuple[str, ...] = RUGD_SEQUENCES

# Primary (rugd.vision) per-sequence URL pattern. As of 2026-05 this
# returns 404 — the maintainers only host the full archive — but we
# probe it anyway so that if they ever publish per-sequence ZIPs the
# loader uses them automatically.
_RUGD_PRIMARY_URL = (
    "http://rugd.vision/data/RUGD_frames-with-annotations/{seq}.zip"
)

# HuggingFace mirror that re-hosts the upstream tree per-sequence.
# License-wise the mirror defers to RUGD's original terms.
_HF_MIRROR_REPO = "WilliamBonilla62/RUGD"
_HF_MIRROR_PREFIX = "RUGD_frames-with-annotations"


def download_rugd(
    out_dir: str | Path,
    *,
    extract: bool = True,
    sequences: list[str] | None = None,
) -> Path:
    """Download one or more RUGD sequences into ``out_dir``.

    Parameters
    ----------
    out_dir:
        Root output directory. Each sequence lands under
        ``out_dir/<seq>/`` (regardless of which source served it), so
        downstream code only has to walk one path layout.
    extract:
        If True (default), extracted ZIPs are unpacked in place and
        a ``<archive>.extracted`` sentinel file is written so we
        skip extraction on subsequent runs.
    sequences:
        Subset of :data:`RUGD_SEQUENCES` to fetch. ``None`` means *all*
        sequences (~5.3 GB total — almost always too big; callers
        typically pass exactly one name).

    Returns
    -------
    Path
        ``out_dir`` (so it composes cleanly with the existing CLI).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seqs = _validate_sequences(sequences)
    if not seqs:
        logger.warning("download_rugd: no sequences requested.")
        return out

    for seq in seqs:
        seq_dir = out / seq
        if _sequence_already_present(seq_dir):
            logger.info("RUGD %r already on disk at %s; skipping.", seq, seq_dir)
            continue

        if _try_primary_http(seq, out, extract=extract):
            continue

        if _try_hf_mirror(seq, out):
            continue

        logger.error(
            "Failed to download RUGD sequence %r from any source. "
            "The official rugd.vision host only ships the full 5.3 GB archive, "
            "and the HuggingFace mirror %r could not be reached. "
            "Manually download from http://rugd.vision/ if needed.",
            seq, _HF_MIRROR_REPO,
        )

    return out


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _validate_sequences(seqs: Iterable[str] | None) -> list[str]:
    if seqs is None:
        return list(RUGD_SEQUENCES)
    valid = set(RUGD_SEQUENCES)
    out: list[str] = []
    for s in seqs:
        if s not in valid:
            raise ValueError(
                f"Unknown RUGD sequence {s!r}. Valid: {sorted(valid)}"
            )
        out.append(s)
    return out


def _sequence_already_present(seq_dir: Path) -> bool:
    """A sequence is 'present' if its folder has at least one image."""
    if not seq_dir.is_dir():
        return False
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        if any(seq_dir.rglob(ext)):
            return True
    return False


def _try_primary_http(seq: str, out: Path, *, extract: bool) -> bool:
    """Try the rugd.vision per-sequence ZIP URL.

    Returns True on success. Returns False (without raising) if the
    upstream returns 404 or any non-fatal HTTP error, so the caller can
    fall back to the mirror.
    """
    url = _RUGD_PRIMARY_URL.format(seq=seq)
    try:
        head = requests.head(url, timeout=30, allow_redirects=True)
    except requests.RequestException as e:
        logger.info("RUGD primary HEAD %s failed: %s", url, e)
        return False

    if head.status_code == 404:
        logger.info(
            "RUGD primary URL not provisioned for %r (HTTP 404); "
            "falling back to HuggingFace mirror.", seq,
        )
        return False
    if head.status_code >= 400:
        logger.warning(
            "RUGD primary URL returned HTTP %d for %r; falling back.",
            head.status_code, seq,
        )
        return False

    archive = out / f"{seq}.zip"
    try:
        http_download(url, archive)
    except Exception as e:  # noqa: BLE001
        logger.warning("RUGD primary download failed for %r: %s", seq, e)
        return False

    if extract:
        _extract_zip(archive, out)
    return True


def _try_hf_mirror(seq: str, out: Path) -> bool:
    """Pull a single sequence folder from the HuggingFace mirror.

    Uses :func:`huggingface_hub.snapshot_download` with an
    ``allow_patterns`` filter so only the requested sequence is
    materialised. After the snapshot lands, files are moved into
    ``out/<seq>/`` to match the layout produced by the primary path.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed; cannot fall back to the "
            "RUGD mirror at %s. Install it (already in transformers' "
            "deps) or provide the sequence files manually.",
            _HF_MIRROR_REPO,
        )
        return False

    cache_dir = out / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{_HF_MIRROR_PREFIX}/{seq}/*"
    logger.info("Fetching RUGD %r from HuggingFace mirror %s ...", seq, _HF_MIRROR_REPO)
    try:
        snapshot = snapshot_download(
            repo_id=_HF_MIRROR_REPO,
            repo_type="dataset",
            allow_patterns=[pattern],
            cache_dir=str(cache_dir),
        )
    except Exception as e:  # noqa: BLE001
        logger.error("HF mirror snapshot_download failed for %r: %s", seq, e)
        return False

    src_dir = Path(snapshot) / _HF_MIRROR_PREFIX / seq
    if not src_dir.is_dir():
        logger.error(
            "HF snapshot returned %s but expected sequence dir %s is missing.",
            snapshot, src_dir,
        )
        return False

    dst_dir = out / seq
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for f in src_dir.iterdir():
        if f.is_file():
            target = dst_dir / f.name
            if target.exists():
                continue
            # Symlinks from HF cache point at the blob store; copy by
            # following them so the eval script never reaches into
            # ``.hf-cache`` at runtime.
            shutil.copy2(f, target, follow_symlinks=True)
            n_copied += 1
    logger.info(
        "RUGD %r ready: %d new files in %s (%d total).",
        seq, n_copied, dst_dir, sum(1 for _ in dst_dir.iterdir()),
    )
    return True


def _extract_zip(archive: Path, out: Path) -> None:
    marker = out / (archive.name + ".extracted")
    if marker.exists():
        logger.info("Already extracted: %s", archive.name)
        return
    try:
        with zipfile.ZipFile(archive) as z:
            logger.info("Extracting %s ...", archive.name)
            z.extractall(out)
        marker.touch()
    except zipfile.BadZipFile as e:
        logger.warning("Skipping bad zip %s: %s", archive.name, e)
