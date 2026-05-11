#!/usr/bin/env python3
"""Derive DDRNet 12-way head → canonical GOOSE-12 permutation (no training).

Fits a bijection **raw_output_channel → semantic_GOOSE_category** by greedy
maximum-p agreement on GOOSE-Ex val labels (fine → 12-way category), using
per-pixel argmax over the 12 raw logits.

Writes the tuple into ``src/perception/models/semantic/ddrnet_goose_perm.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def _load_compare_module():
    sys.path.insert(0, str(_REPO / "src"))
    name = "_calibrate_csm_cmp"
    spec = importlib.util.spec_from_file_location(
        name,
        _HERE / "compare_semantic_models.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gather_pairs(root: Path, max_frames: int | None) -> list[tuple[Path, Path]]:
    pairs = _load_compare_module().goose_ex_val_pairs(root)
    pairs = sorted(pairs, key=lambda x: str(x[0]))
    if max_frames is not None:
        pairs = pairs[: int(max_frames)]
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--goose-root",
        default="datasets/goose/gooseEx_2d_val/gooseEx_2d_val",
    )
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--write-perm",
        action="store_true",
        help="Overwrite ddrnet_goose_perm.py",
    )
    args = p.parse_args()

    sys.path.insert(0, str(_REPO / "src"))
    from perception.config.loader import load_config
    from perception.config.schema import SemanticModelCfg
    from perception.models.backends.pytorch import PyTorchBackend
    from perception.models.factory import SEMANTIC_DEFAULT_WEIGHTS, build_semantic_model
    from perception.models.semantic.ddrnet import DDRNetSemanticModel

    g_root = Path(args.goose_root).resolve()
    csv_p = g_root / "goose_label_mapping.csv"
    if not csv_p.is_file():
        print("Missing", csv_p)
        return 2

    pairs = gather_pairs(g_root, args.max_frames)
    if not pairs:
        print("No GOOSE pairs under", g_root)
        return 2

    cfg = load_config(_REPO / args.config)
    hw = cfg.hardware
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; use --device cpu")
        return 2

    w_path = SEMANTIC_DEFAULT_WEIGHTS.get("ddrnet", "")
    ddr = build_semantic_model(
        SemanticModelCfg(name="ddrnet", weights=w_path),
        hw,
        PyTorchBackend(),
    )
    assert isinstance(ddr, DDRNetSemanticModel)
    ddr.warmup(cfg.classes)

    mod = _load_compare_module()
    fine_lut = mod.build_fine_to_category_lut(csv_p)

    M = np.zeros((12, 12), dtype=np.int64)
    n_pix = 0

    for img_p, lbl_p in pairs:
        bgr = cv2.imread(str(img_p))
        raw = cv2.imread(str(lbl_p), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw is None:
            continue
        if raw.ndim == 3:
            lbl = raw[..., 0].astype(np.int32, copy=False)
        else:
            lbl = raw.astype(np.int32, copy=False)
        if lbl.shape[:2] != bgr.shape[:2]:
            lbl = cv2.resize(
                lbl,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        max_f = int(fine_lut.shape[0])
        oob = (lbl < 0) | (lbl >= max_f)
        gt = np.full(lbl.shape, -1, dtype=np.int32)
        gt[~oob] = fine_lut[lbl[~oob]].astype(np.int32, copy=False)
        if (gt < 0).all():
            continue

        with torch.inference_mode():
            raw_logits = ddr.raw_logits_hw(bgr)
        arg = raw_logits.argmax(dim=0).cpu().numpy().astype(np.int32)
        del raw_logits

        valid = (gt >= 0) & (gt < 12)
        if not valid.any():
            continue
        rav = arg[valid]
        gtv = gt[valid]
        np.add.at(M, (rav, gtv), 1)
        n_pix += int(valid.sum())

    if n_pix < 1000:
        print("Too few labelled pixels:", n_pix)
        return 2

    MM = M.astype(np.float64)
    slot_to_raw = np.full(12, -1, dtype=np.int64)
    raw_to_slot = np.full(12, -1, dtype=np.int64)

    for _ in range(12):
        k, c = np.unravel_index(int(np.argmax(MM)), MM.shape)
        if MM[k, c] < 0:
            break
        slot_to_raw[int(c)] = int(k)
        raw_to_slot[int(k)] = int(c)
        MM[k, :] = -1.0
        MM[:, c] = -1.0

    for c in range(12):
        if slot_to_raw[c] < 0:
            for k in range(12):
                if raw_to_slot[k] < 0:
                    slot_to_raw[c] = k
                    raw_to_slot[k] = c
                    break

    perm = tuple(int(x) for x in slot_to_raw.tolist())
    print("pixels used:", n_pix)
    print("DOC_SLOT_TO_RAW_CHANNEL =", perm)
    print("(canonical slot c reads raw channel DOC_SLOT_TO_RAW_CHANNEL[c])")

    if args.write_perm:
        out = _REPO / "src/perception/models/semantic/ddrnet_goose_perm.py"
        body = (
            '"""GOOSE-12 channel alignment for ``ddrnet_category_512.pth``.\n\n'
            "The published 12-way head does not list classes in the same order as\n"
            "``GOOSE_12_NAMES`` / ``goose_label_mapping.csv``. Before softmax, we gather\n"
            "logits so axis ``c`` matches that canonical ordering.\n\n"
            "``DOC_SLOT_TO_RAW_CHANNEL[c]`` is the **raw checkpoint channel index** that\n"
            "feeds **canonical** GOOSE class ``c`` (vegetation=0 … void=11).\n\n"
            "Regenerate with::\n\n"
            "    PYTHONPATH=src python scripts/calibrate_ddrnet_goose_channels.py \\\\\n"
            "        --max-frames 200 --write-perm\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            "# Greedy match on GOOSE-Ex val (calibrate_ddrnet_goose_channels.py).\n"
            f"DOC_SLOT_TO_RAW_CHANNEL: tuple[int, ...] = {perm!r}\n"
        )
        out.write_text(body, encoding="utf-8")
        print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
