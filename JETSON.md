# Jetson Deployment Guide

Jetson AGX Orin 64GB · JetPack 6.x · CUDA 12.2 · TensorRT 10

Working directory on device: `/mnt/nvme/avi_ws/Segmentation_models_for_perception_system`

---

## Prerequisites

- JetPack 6.x installed (ships CUDA 12.2, cuDNN 9, TensorRT 10)
- Docker with `nvidia-container-runtime`:
  ```bash
  sudo apt install nvidia-container-runtime
  sudo systemctl restart docker
  ```
- SSH access to the device (all commands below run over SSH unless stated otherwise)

---

## Repo Setup

```bash
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Weights are mounted at runtime — not baked into the image.
# Make sure your trained checkpoints are in the weights/ directory:
ls weights/orfd/frozen_backbone/segformer-b2/best.pth   # expected

# The CLIP model used by YOLOE must be present in the repo root:
ls mobileclip2_b.ts   # ~240 MB — downloaded during Docker build or copy manually

# HuggingFace models (SegFormer base) are cached in a Docker volume (hf_cache)
# and downloaded automatically on first run.
```

---

## Local Setup (without Docker)

```bash
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Use the system Python 3 — torch/torchvision come from the JetPack L4T image
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install only the packages NOT already in the base image
pip install -r requirements-jetson.txt

# Install the perception package in editable mode
pip install --no-deps -e .

# Verify CUDA is visible
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Docker: Build

Two build targets — `headless` (inference-only, default) and `gui` (with PyQt5 player).

```bash
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Headless image (recommended for benchmarking and deployment)
docker build --target headless -t perception:headless -f Dockerfile.jetson .

# GUI image (only if you have a display connected or X11 forwarding)
docker build --target gui -t perception:gui -f Dockerfile.jetson .
```

Build takes ~10–15 minutes on first run (pip installs + mobileclip2 download).
Subsequent builds use the layer cache and finish in < 1 minute if only source files changed.

---

## Docker: Run

### Headless (default — logs FPS to terminal)

```bash
# Run on the default video from config.yaml (source.path)
docker compose -f docker-compose.jetson.yml up perception

# Run on a specific video file
docker compose -f docker-compose.jetson.yml run --rm perception \
  scripts/run_headless.py --source samples/off_road_vid1.mp4

# Save annotated output video
docker compose -f docker-compose.jetson.yml run --rm perception \
  scripts/run_headless.py \
    --source samples/off_road_vid1.mp4 \
    --output /app/samples/annotated/off_road_vid1_annotated.mp4

# Limit to first 200 frames (quick smoke test)
docker compose -f docker-compose.jetson.yml run --rm perception \
  scripts/run_headless.py --source samples/off_road_vid1.mp4 --max-frames 200
```

### GUI player (requires X11 forwarding over SSH)

On your laptop:
```bash
ssh -X simulation-jetson@<ip>
```

On the Jetson:
```bash
xhost +local:docker
docker compose -f docker-compose.jetson.yml --profile gui up perception-gui
```

---

## Local Run (without Docker)

```bash
source venv/bin/activate
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Headless — logs FPS at end
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4

# With output video
python3 scripts/run_headless.py \
  --source samples/off_road_vid1.mp4 \
  --output samples/annotated/out.mp4

# GUI player (requires display)
python3 scripts/run_player.py --source samples/off_road_vid1.mp4

# Override config file
python3 scripts/run_headless.py \
  --config config/config.yaml \
  --source samples/off_road_vid1.mp4
```

---

## TensorRT Export

Run **once** on the Jetson to build `.engine` files. Engines are tied to the
exact GPU + TRT version — rebuild after any JetPack upgrade.

```bash
source venv/bin/activate
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Export both YOLOE and SegFormer (recommended)
python3 scripts/export_trt.py --config config/config.yaml

# Export only one model
python3 scripts/export_trt.py --config config/config.yaml --model yoloe
python3 scripts/export_trt.py --config config/config.yaml --model segformer
```

Expected build times on Jetson AGX Orin:
- YOLOE-26L: ~3–5 minutes
- SegFormer-B2 (512px): ~10–15 minutes
- SegFormer-B1 (512px): ~8–12 minutes

After the script prints the engine paths, update `config/config.yaml`:

```yaml
models:
  instance:
    weights: "weights/yoloe-26l-seg.engine"   # printed by export_trt.py

  semantic:
    trt_engine_path: "weights/orfd/frozen_backbone/segformer-b2/best-512x512.engine"

hardware:
  use_tensorrt: true
```

---

## FPS Benchmarking

Use `run_headless.py` — it processes all frames and logs `Processed N frames in Xs (Y FPS)` at the end.

```bash
source venv/bin/activate
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# ── Full pipeline: YOLOE + SegFormer-B2 frozen (PyTorch FP16) ──────────────
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4

# ── Semantic-only: disable YOLOE, SegFormer-B2 frozen (PyTorch FP16) ───────
# Edit config.yaml first: models.instance.enabled: false
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4

# ── Semantic-only: SegFormer-B1 (faster, smaller model) ────────────────────
# Edit config.yaml: OPTION B (segformer-b1) + models.instance.enabled: false
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4

# ── Semantic-only: SegFormer-B0 (fastest SegFormer) ────────────────────────
# Edit config.yaml: OPTION A (segformer-b0) + models.instance.enabled: false
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4

# ── Full pipeline with TensorRT (after export_trt.py) ──────────────────────
# Edit config.yaml: hardware.use_tensorrt: true + engine paths set
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4

# ── Limit frames for a quick check ─────────────────────────────────────────
python3 scripts/run_headless.py --source samples/off_road_vid1.mp4 --max-frames 300
```

---

## Scripts Reference

| Script | Purpose | Example |
|---|---|---|
| `run_headless.py` | Headless inference, logs FPS | `python3 scripts/run_headless.py --source samples/video.mp4` |
| `run_player.py` | PyQt5 GUI player | `python3 scripts/run_player.py --source samples/video.mp4` |
| `export_trt.py` | Build TRT `.engine` files | `python3 scripts/export_trt.py --config config/config.yaml` |
| `train_orfd.py` | Fine-tune segmentation model | `python3 scripts/train_orfd.py --model segformer-b2 --freeze-backbone ...` |
| `benchmark_orfd.py` | Accuracy metrics on ORFD val set | `python3 scripts/benchmark_orfd.py --models segformer-b2-frozen` |
| `orfd_semantic_comparison.py` | Side-by-side model comparison strips | `python3 scripts/orfd_semantic_comparison.py` |
| `annotate_images.py` | Annotate a folder of images | `python3 scripts/annotate_images.py --input dir/` |
| `render_samples.py` | Render annotated sample videos | `python3 scripts/render_samples.py` |
| `download_datasets.py` | Download ORFD / GOOSE datasets | `python3 scripts/download_datasets.py` |
| `yoloe_discovery_dump.py` | Dump YOLOE open-vocab detections | `python3 scripts/yoloe_discovery_dump.py` |

---

## Key Config Knobs (`config/config.yaml`)

| Field | What it does |
|---|---|
| `models.instance.enabled` | `false` → skip YOLOE entirely (semantic-only, ~2× faster) |
| `models.semantic.name` | `segformer-b0/b1/b2/b4` or `auriganet` (see OPTION A–F comments) |
| `models.semantic.weights` | Path to fine-tuned `.pth` checkpoint |
| `models.semantic.processor_size` | `256` / `384` / `512` — lower = faster, coarser boundaries |
| `models.semantic.trt_engine_path` | Path to TRT `.engine` file (requires `hardware.use_tensorrt: true`) |
| `models.instance.weights` | `*.engine` for TRT YOLOE, `*.pt` for PyTorch |
| `models.instance.imgsz` | `512` saves ~25% vs default `640` with negligible quality loss |
| `hardware.fp16` | `true` — always on for Jetson |
| `hardware.use_tensorrt` | `true` after running `export_trt.py` |

### Switching semantic model

Edit `config/config.yaml` under `models.semantic:` — uncomment the option you want and comment out the active one:

```yaml
models:
  semantic:
    # OPTION A — fastest
    # name: "segformer-b0"
    # weights: "weights/orfd/frozen_backbone/segformer-b0/best.pth"
    # num_classes: 3

    # OPTION C — best accuracy
    name: "segformer-b2"
    weights: "weights/orfd/frozen_backbone/segformer-b2/best.pth"
    num_classes: 3
```

---

## Troubleshooting

**`Can't initialize NVML` warning**
Benign on Jetson. NVML is the GPU management library used by `nvidia-smi`; it
initialises differently on L4T. PyTorch falls back to its own device queries.
Everything runs correctly — ignore the warning.

**PyQt5 GUI over SSH — no display**
Either connect a monitor or use X11 forwarding:
```bash
# On your laptop
ssh -X simulation-jetson@<jetson-ip>
# Then run run_player.py normally
```
For headless benchmarking, always prefer `run_headless.py`.

**TRT engine fails to load after JetPack upgrade**
Engines are version-locked. After any JetPack / TensorRT upgrade:
```bash
python3 scripts/export_trt.py --config config/config.yaml
# Update config.yaml with new engine paths
```

**HuggingFace model download fails (offline Jetson)**
Pre-download on a connected machine and copy the cache:
```bash
# On a networked machine
python3 -c "from transformers import SegformerForSemanticSegmentation; \
  SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')"
# Copy ~/.cache/huggingface to the Jetson
rsync -avz ~/.cache/huggingface simulation-jetson@<ip>:/home/simulation-jetson/.cache/
```
