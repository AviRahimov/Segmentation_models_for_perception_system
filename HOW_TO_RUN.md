# How to Run — Jetson Inference Reference

Hardware: Jetson AGX Orin 64GB · JetPack 6.x · CUDA 12.2 · TensorRT 10  
Working directory: `/mnt/nvme/avi_ws/Segmentation_models_for_perception_system`

---

## One-time setup (local venv — no Docker needed)

```bash
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Create venv that inherits JetPack system packages (torch, tensorrt, cuda)
python3 -m venv venv --system-site-packages
source venv/bin/activate

pip install -r requirements-jetson.txt
pip install --no-deps -e .

# Lock clocks for consistent FPS measurements
sudo nvpmodel -m 0     # MAXN power mode
sudo jetson_clocks     # lock all clocks to maximum

# Verify GPU is visible
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Headless inference (`run_headless.py`)

No display required. Logs per-frame FPS and saves an annotated MP4 if `--output` is given.

```bash
source venv/bin/activate
cd /mnt/nvme/avi_ws/Segmentation_models_for_perception_system

# Basic run — logs FPS at end:
python3 scripts/inference/run_headless.py --source samples/off_road_vid5.mp4

# Save annotated output video:
python3 scripts/inference/run_headless.py \
    --source samples/off_road_vid5.mp4 \
    --output samples/annotated/out.mp4

# Quick smoke test — first 300 frames only:
python3 scripts/inference/run_headless.py \
    --source samples/off_road_vid5.mp4 \
    --max-frames 300

# Override config file:
python3 scripts/inference/run_headless.py \
    --config config/config.yaml \
    --source samples/off_road_vid5.mp4
```

---

## GUI player (`run_player.py`)

Interactive player with play/pause/seek/fullscreen. Requires a display.

**Option A — Monitor physically connected to Jetson:**
```bash
source venv/bin/activate
python3 scripts/inference/run_player.py --source samples/off_road_vid5.mp4
```

**Option B — X11 forwarding over SSH (run from your laptop):**
```bash
# On your laptop:
ssh -X simulation-jetson@<jetson-ip>

# Then on the Jetson:
source venv/bin/activate
python3 scripts/inference/run_player.py --source samples/off_road_vid5.mp4
```

The player window supports keyboard shortcuts: `Space` = play/pause, `←`/`→` = seek, `F` = fullscreen.

---

## Switching to a TRT engine

After running the optimization pipeline, point the config at the engine file:

```yaml
# config/config.yaml
models:
  semantic:
    trt_engine_path: "weights/segmentation/optimization/qat_int8_256x256.engine"
hardware:
  use_tensorrt: true
```

Then run headless or player as above — no other changes needed.

Available optimized engines (after Stage 4):
```
weights/segmentation/optimization/baseline_fp32_256x256.engine     — FP32, 107 FPS, mIoU 0.8633
weights/segmentation/optimization/qat_int8_256x256.engine          — INT8, 123 FPS, mIoU 0.8464  ← recommended
weights/segmentation/optimization/sparse_qat_int8_256x256_sparse_first.engine  — FP32, 116 FPS, mIoU 0.8549
```

---

## Side-by-side model comparison (`compare_models.py`)

### Image comparison (static PNGs with mIoU)
Produces 4-panel images: Original | Ground Truth | Model A | Model B

```bash
source venv/bin/activate
python3 scripts/segmentation/optimization/compare_models.py --mode images \
    --model-a engine:weights/segmentation/optimization/baseline_fp32_256x256.engine \
    --model-b engine:weights/segmentation/optimization/qat_int8_256x256.engine \
    --test-data datasets/Segmentation_Dataset \
    --n-samples 20
# Output: reports/segmentation/optimization/qualitative/compare_baseline_fp32_256x256_vs_qat_int8_256x256/
```

### Video comparison (side-by-side MP4, traversable overlay only)
Produces a split-screen video: left = Model A (green traversable overlay + FPS), right = Model B.

```bash
source vcp/bin/activate
python3 scripts/segmentation/optimization/compare_models.py --mode video \
    --model-a engine:weights/segmentation/optimization/baseline_fp32_256x256.engine \
    --model-b engine:weights/segmentation/optimization/qat_int8_256x256.engine \
    --source samples/off_road_vid5.mp4
# Output: reports/segmentation/optimization/video_compare_baseline_fp32_256x256_vs_qat_int8_256x256.mp4
```

### Model spec format
```
pytorch:weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth  — PyTorch checkpoint
onnx:weights/segmentation/optimization/qat_int8_256x256.onnx             — ONNX (onnxruntime)
engine:weights/segmentation/optimization/qat_int8_256x256.engine         — TensorRT engine (Jetson only)
```

---

## Building TRT engine files (production pipeline)

Run once on Jetson to build `.engine` files from the repo's default config:

```bash
source venv/bin/activate
python3 scripts/tools/export_trt.py --config config/config.yaml
```

For the optimized variants (output of the QAT/sparse pipeline), use `benchmark_jetson.py` — it builds engines from `.onnx` files automatically:

```bash
python3 scripts/segmentation/optimization/benchmark_jetson.py \
    --onnx-dir weights/segmentation/optimization/ \
    --val-data datasets/Segmentation_Dataset
```

> **Note:** TRT engines are tied to the exact GPU + TensorRT version. Rebuild after any JetPack upgrade.

---

## Minimal standalone engine test (no imports from `src/`)

Use this to quickly verify any `.engine` file works without running the full pipeline:

```python
import torch, tensorrt as trt, numpy as np, cv2
from pathlib import Path
from transformers import SegformerImageProcessor

ENGINE_PATH = "weights/segmentation/optimization/qat_int8_256x256.engine"
RESOLUTION  = 256

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine  = runtime.deserialize_cuda_engine(Path(ENGINE_PATH).read_bytes())
ctx     = engine.create_execution_context()
out_buf = torch.empty(tuple(ctx.get_tensor_shape("logits")),
                       dtype=torch.float32, device="cuda")

proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
proc.size = {"height": RESOLUTION, "width": RESOLUTION}

bgr = cv2.imread("path/to/image.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
pv  = proc(images=[rgb], return_tensors="pt")["pixel_values"].cuda().float()

stream = torch.cuda.current_stream().cuda_stream
ctx.set_tensor_address("pixel_values", pv.data_ptr())
ctx.set_tensor_address("logits", out_buf.data_ptr())
ctx.execute_async_v3(stream)
torch.cuda.current_stream().synchronize()

logits = torch.nn.functional.interpolate(
    out_buf.clone().float(), size=bgr.shape[:2], mode="bilinear", align_corners=False)
mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
print("classes found:", np.unique(mask))  # expect [0, 1, 2]
```

---

## Docker (optional)

Use Docker for a fully reproducible, isolated environment. The venv approach above is simpler for day-to-day use.

```bash
# Build once — ~15 min first time, <1 min cached:
docker build --target headless -t perception:headless -f Dockerfile.jetson .

# Headless run:
docker compose -f docker-compose.jetson.yml run --rm perception \
    scripts/inference/run_headless.py --source samples/off_road_vid5.mp4

# GUI player (X11 forwarding required):
xhost +local:docker
docker compose -f docker-compose.jetson.yml --profile gui up perception-gui
```

---

## Key config knobs (`config/config.yaml`)

| Field | Effect |
|---|---|
| `models.instance.enabled` | `false` → skip YOLOE (semantic-only, ~2× faster) |
| `models.semantic.name` | `segformer-b0/b1/b2/b4` (see OPTION A–E comments) |
| `models.semantic.weights` | Path to fine-tuned `.pth` checkpoint |
| `models.semantic.processor_size` | `256` / `384` / `512` — lower = faster, coarser boundaries |
| `models.semantic.trt_engine_path` | Path to `.engine` file (requires `hardware.use_tensorrt: true`) |
| `hardware.fp16` | `true` — always on for Jetson |
| `hardware.use_tensorrt` | `true` after running `export_trt.py` or the optimization pipeline |

---

## Troubleshooting

**`Can't initialize NVML` warning** — benign on Jetson, ignore.

**GUI player with no display** — use `run_headless.py` or `ssh -X` for X11 forwarding.

**TRT engine fails to load after JetPack upgrade** — engines are version-locked. Rebuild:
```bash
python3 scripts/tools/export_trt.py --config config/config.yaml
```

**HuggingFace download fails (offline Jetson)** — pre-download on a connected machine:
```bash
# On a networked machine:
python3 -c "from transformers import SegformerForSemanticSegmentation; \
  SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')"
rsync -avz ~/.cache/huggingface simulation-jetson@<ip>:/home/simulation-jetson/.cache/
```
