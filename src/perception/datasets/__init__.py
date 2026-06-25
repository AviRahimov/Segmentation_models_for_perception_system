"""Off-road dataset downloaders (RUGD, ORFD).

The two public entry points are :func:`download_rugd` and
:func:`download_orfd`; both are also exposed as a small CLI in
``scripts/download_datasets.py``.
"""
from .downloader import gdrive_download, http_download, sha256_of
from .orfd import ORFD_GDRIVE_ENV_VAR, download_orfd
from .rugd import RUGD_FILES, RUGD_SEQUENCES, download_rugd

__all__ = [
    "ORFD_GDRIVE_ENV_VAR",
    "RUGD_FILES",
    "RUGD_SEQUENCES",
    "download_orfd",
    "download_rugd",
    "gdrive_download",
    "http_download",
    "sha256_of",
]
