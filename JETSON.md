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
ls weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth   # expected

# The CLIP model used by YOLOE must be present in the repo root:
ls weights/mobileclip2_b.ts   # ~240 MB — downloaded during Docker build or copy manually

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
  scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# Save annotated output video
docker compose -f docker-compose.jetson.yml run --rm perception \
  scripts/inference/run_headless.py \
    --source samples/off_road_vid1.mp4 \
    --output /app/samples/annotated/off_road_vid1_annotated.mp4

# Limit to first 200 frames (quick smoke test)
docker compose -f docker-compose.jetson.yml run --rm perception \
  scripts/inference/run_headless.py --source samples/off_road_vid1.mp4 --max-frames 200
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
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# With output video
python3 scripts/inference/run_headless.py \
  --source samples/off_road_vid1.mp4 \
  --output samples/annotated/out.mp4

# GUI player (requires display)
python3 scripts/inference/run_player.py --source samples/off_road_vid1.mp4

# Override config file
python3 scripts/inference/run_headless.py \
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
python3 scripts/tools/export_trt.py --config config/config.yaml

# Export only one model
python3 scripts/tools/export_trt.py --config config/config.yaml --model yoloe
python3 scripts/tools/export_trt.py --config config/config.yaml --model segformer
```

Expected build times on Jetson AGX Orin:
- YOLOE-26L: ~3–5 minutes
- SegFormer-B2 (512px): ~10–15 minutes
- SegFormer-B1 (512px): ~8–12 minutes

After the script prints the engine paths, update `config/config.yaml`:

```yaml
models:
  instance:
    weights: "weights/detection/yoloe-26l-seg.engine"   # printed by export_trt.py

  semantic:
    trt_engine_path: "weights/segmentation/orfd/frozen_backbone/segformer-b2/best-512x512.engine"

hardware:
  use_tensorrt: true
```

---

## Optimization Pipeline (Jetson steps)

The full optimization pipeline lives in `scripts/segmentation/optimization/`. Stages 0–3 run
on the **dev PC** (RTX 5090); Stage 4 runs on the **Jetson** because TRT engines
are not portable between GPU architectures.

### Pre-flight (run once before Stage 4)

```bash
sudo nvpmodel -m 0              # MAXN power mode (max performance)
sudo jetson_clocks               # lock all clocks to maximum
dpkg -l | grep tensorrt          # confirm TensorRT 10.x
ls /usr/local/cuda/lib64/libcusparse_lt.so*  # check cuSPARSELt for 2:4 sparsity
```

### Stage 4: Engine build + authoritative benchmark

Transfer all `.onnx` files from `weights/segmentation/optimization/` on the dev PC to the Jetson,
then run:

```bash
source venv/bin/activate
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

python3 scripts/segmentation/optimization/benchmark_jetson.py \
    --onnx-dir weights/segmentation/optimization/ \
    --val-data datasets/segmentation/ORFD \
    --output reports/segmentation/optimization/benchmark_results.csv

# Optional: 30-minute soak test per variant (adds ~2h total):
python3 scripts/segmentation/optimization/benchmark_jetson.py \
    --onnx-dir weights/segmentation/optimization/ \
    --val-data datasets/segmentation/ORFD \
    --soak
```

This script:
- Builds one TRT `.engine` per `.onnx` via `trtexec` (flags auto-detected from filename).
- Benchmarks latency (p50, p99) and FPS via the real `TensorRTBackend`.
- Re-validates mIoU from engine output (flags if engine drop > 1% vs PyTorch).
- Writes `reports/segmentation/optimization/benchmark_results.csv`.

### Stage 6: Generate report + video comparison (on Jetson or dev PC)

```bash
# Generate Markdown + HTML table from benchmark CSV:
python3 scripts/segmentation/optimization/generate_report.py \
    --csv reports/segmentation/optimization/benchmark_results.csv

# Side-by-side video comparison (engine vs baseline):
python3 scripts/segmentation/optimization/compare_models.py --mode video \
    --model-a pytorch:weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth \
    --model-b engine:weights/segmentation/optimization/qat_int8_256x256.engine \
    --source samples/off_road_vid1.mp4 \
    --output reports/segmentation/optimization/video_compare_baseline_vs_qat.mp4
```

---

## RF-DETR Optimization

`scripts/detection/optimization/` — separate from the segmentation pipeline above, and from
`scripts/detection/training/`/`evaluation/`. Builds and benchmarks a TensorRT engine for the
production `rfdetr-m` checkpoint. Deliberately standalone: `_rfdetr_trt_common.py` and
`benchmark_jetson.py` need only `numpy`/`opencv`/`torch`/`tensorrt` — no full repo checkout, no
`rfdetr` package — so the Jetson side only needs a handful of files, not this whole project.

**Note on this dev kit specifically**: it has no NVMe mounted (only the internal eMMC, ~34G free)
— `/mnt/nvme/avi_ws/...` above does not apply here. This workflow instead uses
`~/perception_optim/` on the eMMC.

### On-device folder layout

```
~/perception_optim/
  env.sh                        # PATH/LD_LIBRARY_PATH — source before every command below
  scripts/                      # benchmark_jetson.py, benchmark_yolo_jetson.py,
                                 # _rfdetr_trt_common.py, _video_bench_common.py
  weights/
    rfdetr-m/                   # rfdetr-m.onnx, rfdetr-m_fp32.engine, rfdetr-m_fp16.engine
    rfdetr-s/                   # rfdetr-s.onnx, rfdetr-s_fp32.engine, rfdetr-s_fp16.engine
    yolo11m_freeze21/           # .pt, .onnx, _fp32.engine, _fp16.engine
  data/
    videos/                     # FPS-benchmark source clips (e.g. the gaza-road footage)
    val_images/ val_labels/     # accuracy sanity-check subset (Detection_Dataset/valid)
  results/
    benchmark_results_{rfdetr-m,rfdetr-s,yolo11m}.csv
```

One subfolder per model under `weights/` is deliberate, not just tidiness — an earlier flat layout
(`weights/*.onnx`, `weights/*.engine` for every model in one directory) led to a real incident: a
`rm -f weights/*.onnx weights/*.engine` meant to clear a bad export wiped out a *different* model's
engines too. Per-model subdirectories make that class of mistake structurally impossible.

### One-time Jetson environment setup

The base JetPack 6 image ships `opencv` + `tensorrt` system-wide but no `torch` and no `pip`:

```bash
# On the Jetson:
wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && python3 /tmp/get-pip.py --user

# NVIDIA's prebuilt torch wheel for this JetPack/L4T/Python combo — check
# https://developer.download.nvidia.com/compute/redist/jp/ for your exact version
# (this device is L4T R36.3.0 = JetPack 6.0, Python 3.10 = cp310):
~/.local/bin/pip install --user "numpy<2,>=1.24" \
  "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp310-cp310-linux_aarch64.whl"
# numpy<2,>=1.24 is required on BOTH sides: numpy 2.x breaks the system opencv build
# (compiled against the numpy 1.x ABI), while this torch build wants a newer 1.x
# C-API than the stock 1.21.5 — 1.26.4 satisfies both. Verify after installing:
#   python3 -c "import cv2, torch; torch.randn(2).cpu().numpy(); print('OK')"

# torch needs cuSPARSELt at runtime, not present on a fresh JetPack 6 image and
# not installable without sudo via apt — fetch it directly, no root needed:
mkdir -p ~/.local/cusparselt && cd ~/.local/cusparselt
wget -q https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.6.2.3-archive.tar.xz
tar -xf libcusparse_lt-linux-aarch64-0.6.2.3-archive.tar.xz

mkdir -p ~/perception_optim/{scripts,weights/rfdetr-m,weights/rfdetr-s,weights/yolo11m_freeze21,data/videos,data/val_images,data/val_labels,results}
cat > ~/perception_optim/env.sh << 'EOF'
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/cusparselt/libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib:$LD_LIBRARY_PATH
EOF
```

Pre-flight (same as the segmentation Stage 4 pre-flight above): `sudo nvpmodel -m 0` (reboots),
`sudo jetson_clocks`, confirm with `nvpmodel -q` → `MAXN`.

### Stage 1: Export ONNX + validate (dev PC, `.venv-rfdetr-train`)

```bash
source .venv-rfdetr-train/bin/activate
python scripts/detection/optimization/export_onnx.py
# -> weights/detection/optimization/rfdetr-m.onnx, numerically validated against
#    the PyTorch reference (mean IoU / confidence diff over real images — see
#    the script's own validation output; fails loudly if the export is broken)

# Other variants — --model-name is inferred from --checkpoint's path if omitted,
# and --shape defaults to the variant's native resolution (rfdetr-s=512, rfdetr-m=576):
python scripts/detection/optimization/export_onnx.py \
    --checkpoint weights/detection/rfdetr-s/detection_dataset_hardneg/conservative_aug/best.pt
```

### Stage 2: Transfer + benchmark (on the Jetson)

```bash
# From the dev PC:
scp scripts/detection/optimization/{benchmark_jetson.py,_rfdetr_trt_common.py,_video_bench_common.py} jetson:~/perception_optim/scripts/
scp weights/detection/optimization/rfdetr-m.onnx jetson:~/perception_optim/weights/rfdetr-m/
scp ~/Music/gaza_road_videos/*.mp4 jetson:~/perception_optim/data/videos/     # or whatever real footage you want FPS on
scp datasets/detection/Detection_Dataset/valid/images/*.jpg jetson:~/perception_optim/data/val_images/
scp datasets/detection/Detection_Dataset/valid/labels/*.txt jetson:~/perception_optim/data/val_labels/

# On the Jetson:
source ~/perception_optim/env.sh
cd ~/perception_optim
python3 scripts/benchmark_jetson.py \
    --onnx weights/rfdetr-m/rfdetr-m.onnx --engine-dir weights/rfdetr-m \
    --model-name rfdetr-m \
    --videos-dir data/videos --val-images data/val_images --val-labels data/val_labels \
    --output results/benchmark_results_rfdetr-m.csv
# For rfdetr-s: --model-name rfdetr-s --shape 512 512 --onnx weights/rfdetr-s/rfdetr-s.onnx
#              --engine-dir weights/rfdetr-s --output results/benchmark_results_rfdetr-s.csv
```

Builds an FP32 and an FP16 `.engine` via `trtexec`, measures real decode+infer FPS/latency over
every video in `--videos-dir` (not synthetic dummy tensors), and a coarse conf=0.4
precision/recall/FP-per-image sanity check against the transferred validation subset — this is a
regression guard, not a publishable mAP number (that machinery is `_ap_utils.py`, which needs the
full repo + ultralytics/rfdetr, deliberately not transplanted onto this minimal venv). A `notes`
column flags an automatic `recall < 0.3` warning — see CLAUDE.md's "RF-DETR / YOLO Jetson TensorRT
Optimization" section for why naive FP16 currently trips this.

**`--harness {naive,optimized,both}`** (default `both`) controls
`_rfdetr_trt_common.RFDETRTensorRTEngine`'s two independent speedups over the plain per-frame
CPU-preprocess + full-stream-sync loop: `gpu_preprocess` (resize/normalize on-device instead of
cv2/numpy) and `cuda_graph` (captures `execute_async_v3` once, replays every frame — falls back to
plain async execution with a logged warning if capture fails on your TRT build). Confirmed on this
device's TRT 8.6.2.3: rfdetr-m FP32 goes from 33.2 FPS (naive) to 51.4 FPS (optimized), matching
`trtexec`'s own 58 FPS benchmark of the identical engine almost exactly — the naive Python loop, not
the engine, was the bottleneck. Verify power/clocks are still locked first (`nvpmodel -q` → `MAXN`;
if not, `jetson_clocks`'s effect can lapse after a reboot or idle period — rerun `sudo jetson_clocks`).

### Stage 3: Report + visual comparison (dev PC)

```bash
# Pull all three models' CSVs back and merge (see generate_report.py's docstring
# for the merge snippet, or just concatenate — same fieldnames across all three):
scp jetson:~/perception_optim/results/benchmark_results_rfdetr-m.csv reports/detection/optimization/
scp jetson:~/perception_optim/results/benchmark_results_rfdetr-s.csv reports/detection/optimization/
scp jetson:~/perception_optim/results/benchmark_results_yolo11m.csv reports/detection/optimization/
python scripts/detection/optimization/generate_report.py \
    --csv reports/detection/optimization/benchmark_results.csv
# -> reports/detection/optimization/RESULTS.md / RESULTS.html

# Side-by-side comparison — N models, any mix of pytorch:/onnx:/engine:/ultralytics:
# specs (standalone, doesn't reuse compare_detection_models.py's dispatch):
python scripts/detection/optimization/compare_models.py --mode video \
    --models pytorch:weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \
             engine:weights/detection/optimization/rfdetr-m_fp16.engine \
    --source ~/Music/gaza_road_videos/tzir-driving.mp4

# 3-way, mixed RF-DETR + YOLO, any precision:
python scripts/detection/optimization/compare_models.py --mode video \
    --models pytorch:weights/detection/rfdetr-s/detection_dataset_hardneg/conservative_aug/best.pt \
             pytorch:weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \
             ultralytics:weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt \
    --source ~/Music/gaza_road_videos/tzir-driving.mp4
```

**Full-length 3-way annotated comparison videos for every clip, run ON THE JETSON against the actual
deployed `.engine` files** (not the dev-PC PyTorch checkpoints — the point is to see what the real
deployment artifacts produce). All three at FP32 — the one precision tier that's healthy across all
three models (FP16 is broken for both RF-DETR variants, see above). `--labels` gives each panel a
short readable title instead of the raw spec string. Saved to `results/3way_comparison/` on-device,
then pulled back to `reports/detection/optimization/3way_comparison/` on the dev PC:

```bash
# On the Jetson:
source ~/perception_optim/env.sh
export PATH=$HOME/.local/bin:$PATH
cd ~/perception_optim
mkdir -p results/3way_comparison
for src in data/videos/*.mp4; do
  name=$(basename "$src" .mp4)
  python3 scripts/compare_models.py --mode video \
    --models engine:weights/rfdetr-s/rfdetr-s_fp32.engine \
             engine:weights/rfdetr-m/rfdetr-m_fp32.engine \
             ultralytics:weights/yolo11m_freeze21/yolo11m_freeze21_fp32.engine \
    --labels "rfdetr-s FP32" "rfdetr-m FP32" "yolo11m FP32" \
    --source "$src" --output "results/3way_comparison/${name}_3way.mp4"
done

# From the dev PC:
scp "jetson:~/perception_optim/results/3way_comparison/*.mp4" reports/detection/optimization/3way_comparison/
```

---

## Best-YOLO Optimization

`scripts/detection/optimization/benchmark_yolo_jetson.py` — same measurement methodology as the
RF-DETR pipeline (`_video_bench_common.py`'s FPS/accuracy code is shared), but exports through
Ultralytics' own native `.export(format="engine")` instead of a hand-rolled TensorRT wrapper —
Ultralytics' `YOLO(path)` already loads `.pt`/`.onnx`/`.engine` uniformly, no NMS-free decode logic
to replicate the way RF-DETR needed. "Best YOLO" per the cached `reports/detection/leaderboard_test.md`
ranking is `weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt` — same 2-class
scheme as the RF-DETR checkpoints, so it's directly comparable (no collapse-map needed).

### One-time additional Jetson setup (on top of the RF-DETR setup above)

Two non-obvious fixes needed for this specific JetPack 6.0 / NVIDIA torch 2.4.0a0 combo:

```bash
source ~/perception_optim/env.sh

# 1. torchvision — no NVIDIA-provided wheel exists for this JetPack version, and a
#    generic PyPI wheel is ABI-incompatible with NVIDIA's custom torch build
#    ("operator torchvision::nms does not exist"). Must build from source against
#    the exact installed torch (~10 min on Orin):
pip install --user cmake ninja
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda   # nvcc
git clone --branch v0.19.0 --depth 1 https://github.com/pytorch/vision.git /tmp/torchvision_src
cd /tmp/torchvision_src
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.7" BUILD_VERSION=0.19.0 python3 setup.py bdist_wheel
pip install --user dist/torchvision-0.19.0-*.whl --no-deps
cd ~/perception_optim && rm -rf /tmp/torchvision_src   # source tree not needed once the wheel is installed
# Verify: python3 -c "from torchvision.ops import nms; import torch; \
#   nms(torch.rand(3,4).cuda(), torch.rand(3).cuda(), 0.5)"

# 2. ultralytics — pin to 8.4.47 (matches the dev PC's main venv), NOT latest.
#    Ultralytics' exporter unconditionally passes dynamo=False to torch.onnx.export
#    for any "torch 2.4.x" — but NVIDIA's Jetson snapshot predates upstream torch
#    actually adding that parameter, so real torch.onnx.export rejects it regardless
#    of ultralytics version. benchmark_yolo_jetson.py patches this itself
#    (_patch_torch_onnx_export_for_old_nvidia_torch) rather than patching ultralytics.
pip install --user "ultralytics==8.4.47" --no-deps
pip install --user "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.82" onnxruntime "numpy<2,>=1.24" \
    psutil polars ultralytics-thop nvidia-ml-py
# Deliberately NOT installing opencv-python (ultralytics' own declared dep) — it
# would clobber the system cv2 the same way described in this repo's requirements.txt
# for the dev PC; verify after every install: python3 -c "import cv2; print(cv2.__version__)"
```

### Run

```bash
# From the dev PC:
scp scripts/detection/optimization/{benchmark_yolo_jetson.py,_video_bench_common.py} jetson:~/perception_optim/scripts/
scp weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt jetson:~/perception_optim/weights/yolo11m_freeze21/yolo11m_freeze21.pt

# On the Jetson:
source ~/perception_optim/env.sh
export PATH=$HOME/.local/bin:$PATH
cd ~/perception_optim
python3 scripts/benchmark_yolo_jetson.py \
    --weights weights/yolo11m_freeze21/yolo11m_freeze21.pt --model-name yolo11m_freeze21 \
    --videos-dir data/videos --val-images data/val_images --val-labels data/val_labels \
    --output results/benchmark_results_yolo11m.csv
```

---

## Per-model annotated videos with a live FPS overlay

`compare_models.py` accepts a single `--models` spec too (no comparison, just one annotated video),
overlaying that model's own rolling-average FPS (last 30 frames, real measured inference latency —
not the aggregate benchmark number) in the top-right corner. Run once per model, saving each into
its own subfolder — this is what generated `reports/detection/optimization/videos_by_model/`:

```bash
# On the Jetson:
source ~/perception_optim/env.sh
export PATH=$HOME/.local/bin:$PATH
cd ~/perception_optim
mkdir -p results/videos_by_model/{rfdetr-s,rfdetr-m,yolo11m}

for model in rfdetr-s rfdetr-m yolo11m; do
  case $model in
    rfdetr-s) spec="engine:weights/rfdetr-s/rfdetr-s_fp32.engine";  label="rfdetr-s FP32" ;;
    rfdetr-m) spec="engine:weights/rfdetr-m/rfdetr-m_fp32.engine";  label="rfdetr-m FP32" ;;
    yolo11m)  spec="ultralytics:weights/yolo11m_freeze21/yolo11m_freeze21_fp32.engine"; label="yolo11m FP32" ;;
  esac
  for src in data/videos/*.mp4; do
    name=$(basename "$src" .mp4)
    python3 scripts/compare_models.py --mode video \
      --models "$spec" --labels "$label" \
      --source "$src" --output "results/videos_by_model/$model/${name}.mp4"
  done
done

# From the dev PC:
scp "jetson:~/perception_optim/results/videos_by_model/rfdetr-s/*.mp4" reports/detection/optimization/videos_by_model/rfdetr-s/
scp "jetson:~/perception_optim/results/videos_by_model/rfdetr-m/*.mp4" reports/detection/optimization/videos_by_model/rfdetr-m/
scp "jetson:~/perception_optim/results/videos_by_model/yolo11m/*.mp4"  reports/detection/optimization/videos_by_model/yolo11m/
```

**Bug fix note**: `_rfdetr_trt_common.py`'s `RFDETRTensorRTEngine`/`RFDETROnnxModel` now auto-detect
the real input resolution from the loaded engine/ONNX graph itself instead of trusting a
caller-supplied default. This matters because different RF-DETR variants use different fixed sizes
(rfdetr-s=512, rfdetr-m=576) — a caller passing the wrong size doesn't error, TensorRT just
misinterprets the buffer according to its own compiled shape, silently producing garbage output
that never crosses the confidence threshold. This exact bug caused rfdetr-s to show **zero
detections in every frame** of the first `compare_models.py`-rendered video batch (which hardcoded
576×576 for every `engine:`/`onnx:` spec) — caught only by visually spot-checking the rendered
videos, not by any error or the numeric benchmark CSVs (those were unaffected — `benchmark_jetson.py`
always passed the correct `--shape` explicitly). If you add a new RF-DETR variant, you no longer
need to track its input size through every caller — the engine/ONNX file is now self-describing.

---

## FPS Benchmarking

Use `run_headless.py` — it processes all frames and logs `Processed N frames in Xs (Y FPS)` at the end.

```bash
source venv/bin/activate
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# ── Full pipeline: YOLOE + SegFormer-B2 frozen (PyTorch FP16) ──────────────
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# ── Semantic-only: disable YOLOE, SegFormer-B2 frozen (PyTorch FP16) ───────
# Edit config.yaml first: models.instance.enabled: false
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# ── Semantic-only: SegFormer-B1 (faster, smaller model) ────────────────────
# Edit config.yaml: OPTION B (segformer-b1) + models.instance.enabled: false
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# ── Semantic-only: SegFormer-B0 (fastest SegFormer) ────────────────────────
# Edit config.yaml: OPTION A (segformer-b0) + models.instance.enabled: false
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# ── Full pipeline with TensorRT (after export_trt.py) ──────────────────────
# Edit config.yaml: hardware.use_tensorrt: true + engine paths set
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4

# ── Limit frames for a quick check ─────────────────────────────────────────
python3 scripts/inference/run_headless.py --source samples/off_road_vid1.mp4 --max-frames 300
```

---

## Scripts Reference

| Script | Purpose | Example |
|---|---|---|
| `run_headless.py` | Headless inference, logs FPS | `python3 scripts/inference/run_headless.py --source samples/video.mp4` |
| `run_player.py` | PyQt5 GUI player | `python3 scripts/inference/run_player.py --source samples/video.mp4` |
| `export_trt.py` | Build TRT `.engine` files (production) | `python3 scripts/tools/export_trt.py --config config/config.yaml` |
| `train_orfd.py` | Fine-tune segmentation model | `python3 scripts/segmentation/training/train_orfd.py --model segformer-b2 --freeze-backbone ...` |
| `compare_semantic_models.py` | Side-by-side model comparison + accuracy metrics (ORFD/zikim/fcdd) | `python3 scripts/segmentation/evaluation/compare_semantic_models.py --dataset orfd --models segformer-b2` |
| `annotate_images.py` | Annotate a folder of images | `python3 scripts/tools/annotate_images.py --input dir/` |
| `render_samples.py` | Render annotated sample videos | `python3 scripts/tools/render_samples.py` |
| `download_datasets.py` | Download ORFD / GOOSE datasets | `python3 scripts/tools/download_datasets.py` |
| `yoloe_discovery_dump.py` | Dump YOLOE open-vocab detections | `python3 scripts/detection/tools/yoloe_discovery_dump.py` |
| **Optimization pipeline** | | |
| `optimization/benchmark_jetson.py` | Stage 4 — TRT engine build + authoritative benchmark | `python3 scripts/segmentation/optimization/benchmark_jetson.py --onnx-dir weights/segmentation/optimization/ --val-data datasets/segmentation/ORFD` |
| `optimization/compare_models.py` | Stage 6a — side-by-side image / video comparison | `python3 scripts/segmentation/optimization/compare_models.py --mode video --model-a pytorch:... --model-b engine:...` |
| `optimization/generate_report.py` | Stage 6b — Markdown + HTML results table | `python3 scripts/segmentation/optimization/generate_report.py --csv reports/segmentation/optimization/benchmark_results.csv` |

---

## Key Config Knobs (`config/config.yaml`)

| Field | What it does |
|---|---|
| `models.instance.enabled` | `false` → skip YOLOE entirely (semantic-only, ~2× faster) |
| `models.semantic.name` | `segformer-b0/b1/b2/b4` (see OPTION A–E comments in config.yaml) |
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
    # weights: "weights/segmentation/orfd/frozen_backbone/segformer-b0/best.pth"
    # num_classes: 3

    # OPTION C — best accuracy
    name: "segformer-b2"
    weights: "weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth"
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
python3 scripts/tools/export_trt.py --config config/config.yaml
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
