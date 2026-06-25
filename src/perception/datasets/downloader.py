"""Generic resumable HTTP / Google Drive downloader with checksum + tqdm.

The :func:`http_download` helper is intentionally minimal: it streams to a
``.part`` companion file, atomically renames on success, and verifies a
SHA-256 digest when the caller supplies one. It does NOT attempt HTTP
range resumes — the upstream off-road dataset hosts (rugd.vision,
Google Drive) are mostly hostile to byte-range requests, and a partial
file with a wrong checksum is more dangerous than a clean re-download.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def sha256_of(path: Path | str, chunk: int = 1 << 20) -> str:
    """Return the hex-encoded SHA-256 digest of ``path``."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def http_download(
    url: str,
    dest: Path | str,
    sha256: str | None = None,
    *,
    chunk: int = 1 << 20,
    timeout: float = 60.0,
) -> Path:
    """Stream ``url`` into ``dest`` with progress + optional checksum.

    Parameters
    ----------
    url:
        HTTP(S) URL to GET.
    dest:
        Final on-disk path. The parent directory is created if missing.
    sha256:
        Optional hex-encoded expected SHA-256. When set, an existing
        ``dest`` is verified before any network call; on mismatch after
        download the file is deleted and ``RuntimeError`` is raised.
    chunk:
        Streaming chunk size in bytes.
    timeout:
        Per-connection timeout passed to ``requests.get``.

    Returns
    -------
    Path
        ``dest`` on success.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx status.
    RuntimeError
        If a SHA-256 mismatch is detected after download.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        if sha256:
            digest = sha256_of(dest)
            if digest == sha256:
                logger.info("Already downloaded (checksum ok): %s", dest.name)
                return dest
            logger.warning(
                "Existing %s has unexpected checksum (%s != %s); re-downloading.",
                dest.name, digest, sha256,
            )
        else:
            logger.info("Already on disk (no checksum): %s", dest.name)
            return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=os.path.basename(str(dest)),
        ) as pbar:
            for c in r.iter_content(chunk_size=chunk):
                if not c:
                    continue
                f.write(c)
                pbar.update(len(c))

    if sha256:
        digest = sha256_of(tmp)
        if digest != sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {url}: expected {sha256}, got {digest}"
            )

    tmp.replace(dest)
    return dest


def gdrive_download(file_id: str, dest: Path | str) -> Path:
    """Download a single Google Drive file by id, best-effort.

    Uses :mod:`gdown` (lazy import). Raises ``RuntimeError`` on failure
    so the caller can choose to log-and-continue. Already-downloaded
    files are skipped.
    """
    import gdown  # imported lazily so unit tests don't need gdown

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Already downloaded: %s", dest.name)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    out = gdown.download(id=file_id, output=str(tmp), quiet=False)
    if out is None or not Path(out).exists() or Path(out).stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"gdown failed for Google Drive id={file_id}. "
            "The file may be private, deleted, quota-exceeded, or moved."
        )
    Path(out).replace(dest)
    return dest
