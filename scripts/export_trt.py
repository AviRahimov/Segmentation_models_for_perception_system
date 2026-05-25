#!/usr/bin/env python3
"""One-time TensorRT FP16 engine export for YOLOE and SegFormer.

Run this script ONCE on the Jetson to produce optimised ``.engine`` files.
Engines are tied to the exact GPU, driver, and TensorRT version — rebuild
after any JetPack upgrade or if you change ``models.semantic.processor_size``.

Usage
-----
Build both models (recommended):
    python scripts/export_trt.py --config config/config.yaml

Build one model only:
    python scripts/export_trt.py --config config/config.yaml --model yoloe
    python scripts/export_trt.py --config config/config.yaml --model segformer

After the script prints the engine paths, edit config.yaml:
    models.instance.weights:         "<yoloe engine path>"
    models.semantic.trt_engine_path: "<segformer engine path>"
    hardware.use_tensorrt:            true

Then rebuild Docker and run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parent / "src").resolve()))

logger = logging.getLogger("export_trt")


# --------------------------------------------------------------------------- #
# YOLOE export (via Ultralytics native TRT path)                               #
# --------------------------------------------------------------------------- #

def export_yoloe(cfg) -> Path:
    """Export YOLOE .pt → .engine using Ultralytics' built-in TRT exporter."""
    from perception.models.instance._ultralytics_compat import apply_patches
    from perception.models._weights import resolve_instance_weights

    apply_patches()

    weights_path = str(resolve_instance_weights(cfg.models.instance.weights or "yoloe-26l-seg.pt"))
    if weights_path.endswith(".engine"):
        raise ValueError(
            f"models.instance.weights already points to an engine file: {weights_path}\n"
            "Point it back to the .pt file before exporting."
        )

    logger.info("Exporting YOLOE: %s → TRT FP16 (imgsz=%d, workspace=%dGB)",
                weights_path, cfg.models.instance.imgsz, cfg.hardware.trt_workspace_gb)

    try:
        from ultralytics import YOLOE as _Cls  # type: ignore
    except ImportError:
        try:
            from ultralytics.models.yoloe import YOLOE as _Cls  # type: ignore
        except ImportError:
            from ultralytics import YOLO as _Cls  # type: ignore

    model = _Cls(weights_path)
    model.export(
        format="engine",
        imgsz=cfg.models.instance.imgsz,
        half=True,
        workspace=cfg.hardware.trt_workspace_gb,
        dynamic=False,
    )
    # Ultralytics saves the engine next to the .pt file, same stem + .engine
    engine_path = Path(weights_path).with_suffix(".engine")
    logger.info("YOLOE engine: %s", engine_path)
    return engine_path


# --------------------------------------------------------------------------- #
# SegFormer export (PyTorch → ONNX → TRT)                                     #
# --------------------------------------------------------------------------- #

def export_segformer(cfg) -> Path:
    """Export SegFormer .pth → .onnx → .engine.

    Works for any SegFormer variant (B2, B4, local .pth, or HF model) because
    it loads the exact same model that the inference stack uses, respecting
    num_classes and processor_size from config.
    """
    import torch
    import tensorrt as trt  # type: ignore

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # type: ignore

    from perception.models.semantic.segformer import _HF_BASES

    sem_cfg = cfg.models.semantic
    hw_cfg  = cfg.hardware

    name   = sem_cfg.name.lower().strip()
    weights = sem_cfg.weights
    proc_size = sem_cfg.processor_size or 512
    fp16      = bool(hw_cfg.fp16)
    device    = hw_cfg.device

    # ---- load model (same logic as SegFormerSemanticModel.__init__) ----
    _is_local = Path(weights).suffix == ".pth"
    hf_base   = _HF_BASES.get(name, _HF_BASES["segformer-b2"])
    processor_source = hf_base if _is_local else (weights or hf_base)

    processor = SegformerImageProcessor.from_pretrained(processor_source)
    processor.size = {"height": proc_size, "width": proc_size}

    if _is_local:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
        n_labels = state_dict["decode_head.classifier.weight"].shape[0]
        model = SegformerForSemanticSegmentation.from_pretrained(
            hf_base, num_labels=n_labels, ignore_mismatched_sizes=True
        )
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded local SegFormer checkpoint: %s (%d classes)", weights, n_labels)
    elif sem_cfg.num_classes is not None and sem_cfg.num_classes != 150:
        model = SegformerForSemanticSegmentation.from_pretrained(
            weights or hf_base, num_labels=sem_cfg.num_classes, ignore_mismatched_sizes=True
        )
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(weights or hf_base)

    model.eval().to(device)
    if fp16:
        model = model.half()

    # ---- determine output engine path ----
    if _is_local:
        stem = Path(weights).stem
        engine_dir = Path(weights).parent
    else:
        stem = name.replace("/", "_")
        engine_dir = Path("weights")
    engine_dir.mkdir(parents=True, exist_ok=True)

    variant = f"{stem}-{proc_size}x{proc_size}"
    onnx_path   = engine_dir / f"{variant}.onnx"
    engine_path = engine_dir / f"{variant}.engine"

    # ---- 1. Export to ONNX ----
    dtype  = torch.float16 if fp16 else torch.float32
    sample = torch.zeros(1, 3, proc_size, proc_size, device=device, dtype=dtype)
    logger.info("Exporting ONNX → %s  (input %dx%d, fp16=%s)", onnx_path, proc_size, proc_size, fp16)

    torch.onnx.export(
        model,
        {"pixel_values": sample},
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info("ONNX export complete: %s", onnx_path)

    # ---- 2. Build TRT FP16 engine ----
    logger.info("Building TRT engine → %s  (workspace=%dGB) — this may take 5-15 min …",
                engine_path, hw_cfg.trt_workspace_gb)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(trt_logger)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(str(e) for e in errors))

    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, hw_cfg.trt_workspace_gb << 30
    )
    if fp16:
        trt_config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, trt_config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed — check TRT logs above.")
    engine_path.write_bytes(engine_bytes)
    logger.info("TRT engine saved: %s  (%d KB)", engine_path, engine_bytes.nbytes // 1024)

    return engine_path


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(description="Export TRT FP16 engines for YOLOE and SegFormer")
    p.add_argument("--config",  default="config/config.yaml")
    p.add_argument("--model",   choices=["yoloe", "segformer", "both"], default="both")
    args = p.parse_args()

    from perception.config.loader import load_config
    cfg = load_config(args.config)

    instructions: list[str] = [
        "",
        "=" * 70,
        "ENGINE EXPORT COMPLETE — update config.yaml then rebuild Docker:",
        "=" * 70,
    ]

    if args.model in ("yoloe", "both"):
        ep = export_yoloe(cfg)
        instructions.append(f"  models.instance.weights:         \"{ep}\"")

    if args.model in ("segformer", "both"):
        ep = export_segformer(cfg)
        instructions.append(f"  models.semantic.trt_engine_path: \"{ep}\"")

    instructions += [
        "  hardware.use_tensorrt:           true",
        "=" * 70,
        "",
    ]
    print("\n".join(instructions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
