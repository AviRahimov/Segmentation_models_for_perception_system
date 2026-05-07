"""Model-weight resolver and downloader.

Given a checkpoint *name* (e.g. ``"yoloe-26l-seg.pt"``) and an ordered list
of mirrors, returns a local path inside ``./weights/`` after downloading
the file if it isn't already cached. Used by the model wrappers so that
adding a new instance/semantic model to the project requires zero manual
weight management - the wrapper only declares the mirrors and the
resolver does the rest.

Why a separate resolver instead of relying on Ultralytics' built-in
auto-download? Ultralytics only auto-downloads files whose names appear
in its hard-coded ``GITHUB_ASSETS_NAMES`` list, which is updated
release-by-release. ``yoloe-26l-seg.pt`` was published to
``ultralytics/assets`` v8.4.0 but is not yet on the auto-download list of
older Ultralytics versions, so a generic resolver is more robust.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_DIR = Path("weights")


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeightSource:
    """A single mirror for a checkpoint.

    Use exactly one of ``url`` (HTTP) or (``repo_id``, ``filename``) (HF Hub).
    """

    kind: str  # "http" | "hf_hub"
    url: str = ""
    repo_id: str = ""
    filename: str = ""


# --------------------------------------------------------------------------- #
def resolve_weights(
    name: str,
    sources: Sequence[WeightSource],
    cache_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> Path:
    """Return a local path to ``name``, downloading from ``sources`` if needed.

    Behaviour:

    1. If ``name`` is an existing path (relative or absolute), return it.
    2. Otherwise check ``cache_dir / basename(name)``; if present, return it.
    3. Otherwise try each :class:`WeightSource` in order until one succeeds,
       writing to ``cache_dir / basename(name)`` atomically (``.part`` then
       rename).
    4. Raise :class:`RuntimeError` only if all sources fail.
    """
    p = Path(name)
    if p.exists():
        return p.resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / p.name
    if target.exists() and target.stat().st_size > 0:
        return target.resolve()

    if not sources:
        raise FileNotFoundError(
            f"Weights {name!r} not found locally (looked in cwd and {cache_dir}/) "
            "and no download mirrors are configured for this checkpoint."
        )

    last_exc: Exception | None = None
    for i, src in enumerate(sources):
        try:
            logger.info(
                "Downloading %s (%d/%d) from %s ...",
                p.name, i + 1, len(sources), _describe(src),
            )
            if src.kind == "http":
                _download_http(src.url, target)
            elif src.kind == "hf_hub":
                _download_hf_hub(src.repo_id, src.filename, target)
            else:
                raise ValueError(f"Unknown WeightSource.kind: {src.kind!r}")
            if target.exists() and target.stat().st_size > 0:
                logger.info("Resolved weights %s -> %s", name, target)
                return target.resolve()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning(
                "Mirror %d/%d failed for %s: %s",
                i + 1, len(sources), p.name, e,
            )

    raise RuntimeError(
        f"Failed to download {name!r} from any of {len(sources)} mirrors. "
        f"Last error: {last_exc!r}"
    )


def _describe(src: WeightSource) -> str:
    if src.kind == "http":
        return src.url
    if src.kind == "hf_hub":
        return f"hf://{src.repo_id}/{src.filename}"
    return f"<{src.kind}>"


def _download_http(
    url: str,
    dst: Path,
    chunk: int = 1 << 20,
    timeout: float = 60.0,
) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name
        ) as pbar:
            for c in r.iter_content(chunk_size=chunk):
                if not c:
                    continue
                f.write(c)
                pbar.update(len(c))
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Empty download from {url}")
    tmp.replace(dst)


def _download_hf_hub(repo_id: str, filename: str, dst: Path) -> None:
    """Download via ``huggingface_hub.hf_hub_download`` and copy into ``dst``.

    The HF cache lives at ``~/.cache/huggingface/hub`` by default. We copy
    out into the project-local weights directory so the layout is
    predictable and self-contained.
    """
    from huggingface_hub import hf_hub_download  # transitive dep of transformers

    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    cached_path = Path(cached)
    if not cached_path.exists() or cached_path.stat().st_size == 0:
        raise RuntimeError(
            f"hf_hub_download returned empty/missing file for "
            f"{repo_id}/{filename} -> {cached!r}"
        )
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(cached_path, tmp)
    tmp.replace(dst)


# --------------------------------------------------------------------------- #
# Known-good registry of mirrors for the project's model weights.            #
#                                                                            #
# Add a new model = add an entry here + register the wrapper in              #
# perception.models.factory. No other code changes required.                 #
# --------------------------------------------------------------------------- #
INSTANCE_WEIGHT_SOURCES: dict[str, list[WeightSource]] = {
    "yoloe-26l-seg.pt": [
        # Primary: official Ultralytics asset release v8.4.0.
        WeightSource(
            kind="http",
            url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg.pt",
        ),
        # Fallback: Hugging Face mirror by the OpenVision org.
        # The file in that repo is named "model.pt"; we save it locally
        # under the canonical "yoloe-26l-seg.pt" so Ultralytics' filename-
        # based dispatch still recognizes it.
        WeightSource(
            kind="hf_hub",
            repo_id="openvision/yoloe26-l-seg",
            filename="model.pt",
        ),
    ],
}


def resolve_instance_weights(name_or_path: str) -> Path:
    """Convenience wrapper used by instance-model wrappers."""
    sources = INSTANCE_WEIGHT_SOURCES.get(Path(name_or_path).name, [])
    return resolve_weights(name_or_path, sources)
