"""Cache HuggingFace processor/model configs locally for offline startup.

Downloads the tiny JSON configs (preprocessor_config.json + config.json,
a few KB each) for the SegFormer variants and saves them under
``weights/hf_assets/<model>/``.  Once cached, loading a fine-tuned local
.pth checkpoint makes zero network calls — no HF Hub freshness checks and
no ADE base-weight download.

Run once per machine (dev PC and Jetson), or commit the JSONs to git.

Usage
-----
    # Cache the default production model (segformer-b2):
    python scripts/tools/cache_hf_assets.py

    # Cache specific variants:
    python scripts/tools/cache_hf_assets.py --models segformer-b2 segformer-b4

    # Cache everything in the registry:
    python scripts/tools/cache_hf_assets.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cache_hf_assets")


def main() -> int:
    from perception.models.semantic.segformer import _HF_ASSETS_DIR, _HF_BASES

    # Deduplicate: dash/underscore aliases map to the same hub ID.
    canonical = sorted(set(_HF_BASES.values()))
    name_choices = sorted(k for k in _HF_BASES if "-" in k)

    p = argparse.ArgumentParser(
        description="Cache SegFormer HF configs locally for offline startup"
    )
    p.add_argument("--models", nargs="+", default=["segformer-b2"],
                   choices=name_choices,
                   help="Model variants to cache (default: segformer-b2)")
    p.add_argument("--all", action="store_true",
                   help="Cache all registered variants")
    args = p.parse_args()

    from transformers import SegformerConfig, SegformerImageProcessor

    hub_ids = canonical if args.all else sorted(
        {_HF_BASES[m] for m in args.models}
    )

    assets_root = _ROOT / _HF_ASSETS_DIR
    for hub_id in hub_ids:
        dest = assets_root / hub_id.split("/")[-1]
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Caching %s -> %s", hub_id, dest)
        SegformerImageProcessor.from_pretrained(hub_id).save_pretrained(dest)
        SegformerConfig.from_pretrained(hub_id).save_pretrained(dest)
        saved = sorted(f.name for f in dest.iterdir())
        logger.info("  saved: %s", ", ".join(saved))

    logger.info("Done. SegFormer startup is now fully offline for cached models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
