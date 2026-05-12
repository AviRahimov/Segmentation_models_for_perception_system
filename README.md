# Off-Road Robotics Perception System

Real-time perception stack for off-road robots. Combines **open-vocabulary
instance segmentation** (YOLOE-26L), **closed-vocab terrain segmentation**
(SegFormer-B2), **causal temporal smoothing** (logit EMA + IoU-backed instance
association), and a non-blocking **PyQt6 video player**. Everything is driven
by a single `config.yaml`; adding a new class is a single YAML edit.

```
+--------------------------------------------------------------------------+
|  Frame Source  -->  Decoder QThread  -->  bounded queue  -->  Inference  |
|  (video|cam|dir)                                              QThread     |
|                                                                  |        |
|                              +-----------------------------------+        |
|                              v                                            |
|         +-----------------+  +--------------------+  +----------------+   |
|         |  YOLOE  (Inst.) |  |  SegFormer (Sem.)  |  |   SceneCut     |   |
|         |  cached text PE |  |   raw merged       |  |  Bhattacharyya |   |
|         +--------+--------+  |   user-class logits|  +----+-----------+   |
|                  |           +----------+---------+       |               |
|                  v                      v                 |               |
|         +-----------------+    +-----------------+        |               |
|         |  IoU tracker    |    |  LogitsEMA      |  <-----+ reset on cut  |
|         |  tracker        |    |  (causal)       |                        |
|         +--------+--------+    +---------+-------+                        |
|                  |                       |                                |
|                  +---------+-------------+                                |
|                            v                                              |
|                     +--------------+                                      |
|                     |  Renderer    |  per-class display_mode + legend     |
|                     +------+-------+                                      |
|                            v                                              |
|                     +--------------+                                      |
|                     |  PyQt6 UI    |  play / pause / seek / fullscreen    |
|                     +--------------+                                      |
+--------------------------------------------------------------------------+
```

Key design rule: **every box above is a separate module with no cross-
dependency.** Models are loaded by a factory; the pipeline consumes them
through abstract base classes; the renderer and UI know nothing about model
internals; the inference backend (PyTorch / TensorRT) is itself an
abstraction so a new runtime can be plugged in without touching anything
else. See `src/perception/models/backends/tensorrt.py` for the four steps
to enable TensorRT on a deployment target.

---

## Hardware


|      | Development                    | Production                           |
| ---- | ------------------------------ | ------------------------------------ |
| GPU  | NVIDIA RTX 5090 (32 GB)        | Jetson AGX Orin 64GB                 |
| CUDA | 12.x with PyTorch 2.5+         | JetPack 6.x (CUDA 12.2, TensorRT 10) |
| RAM  | >=32 GB                        | 64 GB unified                        |
| OS   | Linux (tested on Ubuntu 24.04) | JetPack Linux                        |


Recommended CPU: 8+ cores. The decoder and inference workers run on
separate threads so a busy CPU does not block the UI.

---

## Installation

```bash
# Python 3.12 is required.
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# requirements.txt sets `--extra-index-url https://download.pytorch.org/whl/cu128`
# inline, so the cu128 (Blackwell-capable) PyTorch wheels are picked
# up automatically:
pip install -r requirements.txt

# OPTIONAL: TensorRT (Jetson: bundled with JetPack; x86: NVIDIA wheel).
# After installing, follow the four integration steps documented at
# src/perception/models/backends/tensorrt.py
```

### Jetson AGX Orin (production target)

The cu128 wheels on `download.pytorch.org` are x86_64-only. On Jetson,
install the NVIDIA-provided aarch64 PyTorch wheel that matches your
JetPack version (see [https://forums.developer.nvidia.com/t/pytorch-for-jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson))
**before** running `pip install -r requirements.txt`, then either edit
the `torch==…` / `torchvision==…` lines out of `requirements.txt` or
install with `--no-deps` for those two lines. The rest of the stack is
arch-agnostic.

### Model weights

- **YOLOE-26L** is downloaded automatically into `./weights/` on first
run (mirrors: GitHub `ultralytics/assets` v8.4.0 → HF Hub
`openvision/yoloe26-l-seg`).
- **SegFormer-B2** is fetched by `transformers` into the standard
Hugging Face cache.

---

## Usage

### GUI player

```bash
python scripts/run_player.py --config config/config.yaml
python scripts/run_player.py --source samples/clip.mp4
python scripts/run_player.py --source-type camera --camera 0
python scripts/run_player.py --source-type image_dir --source datasets/rugd/images
```

Keyboard shortcuts: `Space` play/pause, `Left`/`Right` seek 1 s,
`Shift+Left`/`Shift+Right` seek 10 s, `+` / `-` change speed, `F`
fullscreen, `Esc` exit fullscreen, `Q` quit.

Seeking automatically resets temporal buffers (EMA + tracker) because
temporal context is broken.

**Video player vs ORFD freespace strips:** The Qt player and
`scripts/orfd_semantic_comparison.py` use the same SegFormer (and the same
`config.yaml` weights / class merge rules). The comparison script’s strips
show **binary traversable** predictions (road_ground vs GT path) for IoU;
the player draws the full **argmax** terrain overlay, with each class’s
visibility controlled by `display_mode` and `player.draw_road_ground_semantic_last`
(z-order). There is no second SegFormer—only different visualization.

### Headless inference

```bash
python scripts/run_headless.py --source samples/clip.mp4 --output runs/clip_overlay.mp4
python scripts/run_headless.py --source datasets/rugd/images --max-frames 1000
```

Same pipeline, no Qt - useful for benchmarking and CI.

### Download test datasets

```bash
python scripts/download_datasets.py                         # both
python scripts/download_datasets.py --dataset rugd
python scripts/download_datasets.py --dataset orfd --out /data
```

ORFD is hosted on Google Drive; if the upstream id changes set
`ORFD_GDRIVE_ID=<new-id>` before running. RUGD is fetched directly from
`http://rugd.vision/data/`.

---

## How to add a new class

Append one entry to `config/config.yaml` under `classes:`. **Zero code
changes.**

Open-vocab instance class (just a text prompt):

```yaml
- name: "tent"
  text_prompt: "tent or temporary shelter"
  display_mode: "both"          # both | bbox_only | mask_only | none
  color_rgb: [255, 0, 200]
  is_semantic: false
```

Semantic terrain class (text prompt for the legend + ADE20K channels to
merge):

```yaml
- name: "rocky_terrain"
  text_prompt: "rocky uneven terrain"
  display_mode: "mask_only"
  color_rgb: [180, 100, 70]
  is_semantic: true
  ade20k_indices: [13, 46]      # earth + sand
```

Restart the player; the new class is detected, segmented, smoothed, and
drawn in its assigned colour with the requested display mode.

---

## Configuration reference


| Field                                            | Default                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models.instance.name`                           | `yoloe26l`                                  | Registered instance model. See `INSTANCE_REGISTRY`.                                                                                                                                                                                                                                                                                                                                                                                               |
| `models.instance.weights`                        | `yoloe-26l-seg.pt`                          | Ultralytics YOLOE-seg weights file.                                                                                                                                                                                                                                                                                                                                                                                                               |
| `models.instance.confidence_threshold`           | `0.35`                                      | YOLOE detection threshold (global default; per-class override available - see below).                                                                                                                                                                                                                                                                                                                                                             |
| `models.semantic.name`                           | `segformer-b2`                              | Registered semantic model.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `models.semantic.weights`                        | `nvidia/segformer-b2-finetuned-ade-512-512` | HF Hub id or local path.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `classes[*]`                                     | (required)                                  | Class definitions; see "How to add a class".                                                                                                                                                                                                                                                                                                                                                                                                      |
| `classes[*].confidence_threshold`                | (inherits global)                           | Optional per-class override of `models.instance.confidence_threshold`. Instance classes only. Range `[0, 1]`. Lower for classes the model misses (e.g. small or partially-occluded cars), raise for classes with frequent false positives. The YOLOE wrapper internally calls Ultralytics with the lowest configured threshold across all classes, then per-class filters the output, so a low override on one class does not pollute the others. |
| `temporal.semantic_ema.alpha`                    | `0.35`                                      | EMA weight on the *current* frame's logits.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `temporal.semantic_ema.reset_on_scene_cut`       | `true`                                      | Drop EMA + tracker state on scene cut.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `temporal.semantic_ema.scene_cut_threshold`      | `0.45`                                      | Bhattacharyya distance threshold in [0, 1].                                                                                                                                                                                                                                                                                                                                                                                                       |
| `hardware.device`                                | `cuda`                                      | Torch device.                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `hardware.fp16`                                  | `true`                                      | Run models in half precision on GPU.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `hardware.use_tensorrt`                          | `false`                                     | Use TensorRT backend (see backends/tensorrt.py).                                                                                                                                                                                                                                                                                                                                                                                                  |
| `hardware.text_embed_cache`                      | `true`                                      | Cache YOLOE text embeddings at startup (always recommended).                                                                                                                                                                                                                                                                                                                                                                                      |
| `player.mask_alpha`                              | `0.45`                                      | Mask blend opacity.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `player.show_fps`                                | `true`                                      | Show FPS overlay.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `player.show_class_legend`                       | `true`                                      | Show colour legend.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `player.default_speed`                           | `1.0`                                       | Initial playback speed.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `source.type`                                    | `video`                                     | One of `video`, `camera`, `image_dir`.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `source.path`                                    | `samples/clip.mp4`                          | Video file or image directory.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `source.camera_index`                            | `0`                                         | Webcam index (when `type: camera`).                                                                                                                                                                                                                                                                                                                                                                                                               |
| `source.image_dir_glob`                          | `*.png`                                     | Glob within `source.path` for `image_dir`.                                                                                                                                                                                                                                                                                                                                                                                                        |
| `source.fps_hint`                                | `30.0`                                      | FPS for `image_dir`/`camera` when not reported.                                                                                                                                                                                                                                                                                                                                                                                                   |
| `datasets.download_dir`                          | `./datasets`                                | Default dataset download root.                                                                                                                                                                                                                                                                                                                                                                                                                    |


---

## Architecture (module map)

```
src/perception/
  config/      typed dataclasses + YAML loader
  core/        pure data contracts (Detection, FrameResult, ...)
  io/          FrameSource ABC + video / camera / image-dir
  models/
    backends/  PyTorch (default), TensorRT (documented stub)
    instance/  YOLOE wrapper (cached text embeds)
    semantic/  SegFormer wrapper (raw merged logits via ADE20K LUT)
    factory.py registry-based dispatch
  temporal/    LogitsEMA, scene-cut, IoU instance tracker
  pipeline/    PerceptionPipeline (DI of all of the above)
  render/      overlay primitives + display-mode-aware renderer
  ui/          PyQt6 widgets + decode/inference QThread workers
  datasets/    RUGD + ORFD downloaders
```

SOLID is enforced by the import graph: `core` depends on nothing,
`models` depends only on `core`+`config`, `temporal` only on `core`,
`pipeline` only on the abstract bases of `models`+`temporal`, and `ui`
only on `pipeline`+`render`+`io`. Adding a new model = subclass an ABC,
register in `factory.py`, add a YAML entry. Done.

---

## Testing

```bash
pip install pytest
pytest -q
```

Tests do not require a GPU or model downloads.