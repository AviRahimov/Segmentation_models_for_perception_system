"""ORFD off-road freespace-detection dataset downloader (best-effort).

ORFD is published from
https://github.com/chaytonmin/Off-Road-Freespace-Detection . The
dataset itself sits on Google Drive, and the maintainers occasionally
rotate the file id. Because Google Drive also enforces per-file daily
quota that easily blocks unauthenticated ``gdown`` clients, this
module is *strictly best-effort*:

- The file id is read from the ``PERCEPTION_ORFD_GDRIVE_ID`` env var.
- If the env var is unset, we log a clear instruction block and return
  ``out_dir`` without raising.
- If gdown fails (quota, link rotated, network blocked), we log the
  underlying error and again return ``out_dir`` without raising.

Callers that need a stricter contract can check whether the returned
directory actually contains data after calling.
"""
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

from .downloader import gdrive_download

logger = logging.getLogger(__name__)

#: Name of the environment variable that supplies the Google Drive file id.
ORFD_GDRIVE_ENV_VAR = "PERCEPTION_ORFD_GDRIVE_ID"

#: Filename used for the downloaded archive.
ORFD_ARCHIVE_NAME = "ORFD.zip"


def download_orfd(out_dir: str | Path, *, extract: bool = True) -> Path:
    """Best-effort download of the ORFD dataset.

    Parameters
    ----------
    out_dir:
        Output directory. Created if missing.
    extract:
        Extract the zip after a successful download (idempotent via a
        ``<archive>.extracted`` sentinel file).

    Returns
    -------
    Path
        ``out_dir`` — possibly empty if no file id was provided or if
        the Google Drive download failed. This function NEVER raises.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_id = os.environ.get(ORFD_GDRIVE_ENV_VAR, "").strip()
    if not file_id:
        logger.warning(
            "ORFD: %s is not set; skipping ORFD download. To fetch ORFD, "
            "look up the current Google Drive file id at "
            "https://github.com/chaytonmin/Off-Road-Freespace-Detection "
            "and re-run with %s=<id>.",
            ORFD_GDRIVE_ENV_VAR, ORFD_GDRIVE_ENV_VAR,
        )
        return out

    archive = out / ORFD_ARCHIVE_NAME
    try:
        archive = gdrive_download(file_id, archive)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "ORFD download failed (id=%s): %s. ORFD remains best-effort; "
            "the eval will continue without it.",
            file_id, e,
        )
        return out

    if extract:
        _extract_zip(archive, out)
    return out


def _extract_zip(archive: Path, out: Path) -> None:
    marker = out / (archive.name + ".extracted")
    if marker.exists():
        logger.info("ORFD already extracted.")
        return
    try:
        with zipfile.ZipFile(archive) as z:
            logger.info("Extracting %s ...", archive.name)
            z.extractall(out)
        marker.touch()
    except zipfile.BadZipFile as e:
        logger.warning("ORFD archive is not a valid zip: %s", e)
