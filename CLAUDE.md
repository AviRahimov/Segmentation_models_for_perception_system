# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (Python 3.12 required; cu128 wheels cover both RTX 5090 and Jetson sm_87)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
# supervision/trackers/tidecv transitively pull in plain opencv-python, which
# silently corrupts the pinned opencv-python-headless install (see
# requirements.txt's comment) — always run this after a fresh install:
pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless==4.10.0.84

# Run tests (no GPU or model downloads required)
pytest -q

# Run a single test file
pytest -q tests/test_ema_logits.py

# GUI player (requires display)
python scripts/inference/run_player.py --source samples/clip.mp4

# Headless inference / benchmarking
python scripts/inference/run_headless.py --source samples/clip.mp4 --output runs/out.mp4
python scripts/inference/run_headless.py --source samples/clip.mp4 --max-frames 300

# Training (fine-tune SegFormer on ORFD)
python scripts/segmentation/training/train_orfd.py

# Evaluation
python scripts/segmentation/evaluation/eval_segformer_on_datasets.py
python scripts/segmentation/evaluation/orfd_semantic_comparison.py

# Optimization pipeline (dev PC → Jetson workflow)
python scripts/segmentation/optimization/resolution_sweep.py --checkpoint weights/.../best.pth --data datasets/Segmentation_Dataset
python scripts/segmentation/optimization/export_onnx.py --resolution 256
python scripts/segmentation/optimization/train_qat.py --config config/segmentation/optimization/qat.yaml
python scripts/segmentation/optimization/train_sparse.py --config config/segmentation/optimization/sparse.yaml
# Stage 4 runs on Jetson only:
python scripts/segmentation/optimization/benchmark_jetson.py --onnx-dir weights/optimization/ --val-data datasets/Segmentation_Dataset
python scripts/segmentation/optimization/compare_models.py --mode images \
    --model-a pytorch:weights/.../best.pth --model-b onnx:weights/optimization/qat_int8.onnx \
    --test-data datasets/Segmentation_Dataset

# YOLOE discovery mode dump
python scripts/detection/tools/yoloe_discovery_dump.py --config config/config.yaml \
    --source samples/recording.mp4 --max-frames 200 \
    --jsonl runs/discovery.jsonl --summary-tsv runs/discovery_summary.tsv

# Detection training — Round 1 (YOLO26 and YOLOE-26, scales s/m/l)
python scripts/detection/training/train_round1.py --model yolo26m
python scripts/detection/training/train_round1.py --model yoloe-26m
# Output: weights/detection/{model_name}/round1/best.pt

# Detection training — general (interactive survey: scans datasets/, Enter = defaults)
python scripts/detection/training/train_detector.py
# Classic hyperparameter-sweep CLI (former train_exp.py):
python scripts/detection/training/train_detector.py --model yolo11m --variants freeze10_aug_clean
# Output: weights/detection/{model}/{dataset_slug}/{recipe}/ (interactive)
#         weights/detection/{model}/exp/{variant}/ (CLI sweep)
# Every run is appended to reports/detection/experiments.jsonl (provenance)

# Detection training — RF-DETR (separate venv: see requirements-rfdetr-train.txt
# for why — rfdetr-plus/XL-2XL needs transformers>=5.1, incompatible with the
# main venv's SegFormer pin)
python3.12 -m venv .venv-rfdetr-train && source .venv-rfdetr-train/bin/activate
pip install -r requirements-rfdetr-train.txt
python scripts/detection/training/train_detector_rfdetr.py
# Output: weights/detection/{model}/{dataset_slug}/{coco|ft}/best.pt

# Merged dataset builds — manifest-driven, reproducible (manifests in git)
python scripts/detection/tools/build_dataset.py --manifest config/detection/datasets/merged_2class.yaml
python scripts/detection/tools/build_dataset.py --manifest config/detection/datasets/merged_6class.yaml

# Leaderboard — every checkpoint (any class scheme) ranked on the real val
# benchmark via collapsed AP50 + P/R/FP-per-image at conf 0.40; cached
python scripts/detection/evaluation/leaderboard.py
python scripts/detection/evaluation/leaderboard.py --tta          # + test-time-aug rows
python scripts/detection/evaluation/leaderboard.py --thresholds   # best-F1 per-class conf recommendations
python scripts/detection/evaluation/leaderboard.py --fp-gallery   # annotated false-positive crops

# Detection evaluation
python scripts/detection/evaluation/eval_detection.py \
    --weights weights/detection/yolo26m/round1/best.pt

# Confusion matrix — interactive survey; which class gets confused as which
# other class (leaderboard.py doesn't compute this)
python scripts/detection/evaluation/confusion_matrix.py

# Threshold sweep — conf x NMS-IoU grid search per checkpoint, writes best pair to JSON
python scripts/detection/evaluation/tune_thresholds.py \
    --models pytorch:weights/detection/yolo26m/round1/best.pt \
    --data datasets/Detection_Dataset/data.yaml

# Confidence calibration — fit per-class temperature scaling for one checkpoint
# (interactive checkpoint/benchmark picker if flags omitted); enable via
# postprocess.calibration in config.yaml once fitted
python scripts/detection/evaluation/fit_calibration.py \
    --weights weights/detection/rfdetr-2xl/detection_dataset/coco/best.pt

# Detection model comparison (paper-style — table / images / video)
python scripts/detection/evaluation/compare_detection_models.py --mode table \
    --models pytorch:weights/detection/yolo26s/round1/best.pt \
             pytorch:weights/detection/yolo26m/round1/best.pt \
             pytorch:weights/detection/yoloe-26m/round1/best.pt
python scripts/detection/evaluation/compare_detection_models.py --mode images \
    --models pytorch:weights/detection/yolo26m/round1/best.pt \
             pytorch:weights/detection/yoloe-26m/round1/best.pt \
    --test-data datasets/Detection_Dataset/valid/images --n-samples 20
python scripts/detection/evaluation/compare_detection_models.py --mode video \
    --models pytorch:weights/detection/yolo26m/round1/best.pt \
    --source samples/clip.mp4

# Dataset download
python scripts/tools/download_datasets.py  # both RUGD + ORFD

# Build a 2-class eval set from the synthetic dataset (own class scheme +
# img/ dir Ultralytics can't resolve) for use with compare_detection_models.py
python scripts/detection/tools/prepare_synthesis_eval.py
python scripts/detection/tools/prepare_synthesis_eval.py --n-samples 50  # deterministic subset

# TIDE error-type breakdown (classification/localization/duplicate/background/missed)
python scripts/detection/evaluation/tide_analysis.py
python scripts/detection/evaluation/tide_analysis.py --only weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt

# D-RISE saliency — why does the model fire on one specific detection/box
python scripts/detection/evaluation/drise_explain.py \
    --weights weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \
    --image datasets/Detection_Dataset/test/images/some_frame.jpg

# Full-image FP review — every GT/TP/FP box drawn on the uncropped frame (leaderboard.py's
# --fp-gallery only shows a cropped context window per box)
python scripts/detection/evaluation/fp_full_image_review.py \
    --weights weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt

# Optuna hyperparameter sweep for rfdetr-m (TPE sampler, MedianPruner) — separate venv
source .venv-rfdetr-train/bin/activate
python scripts/detection/training/tune_rfdetr_optuna.py --n-trials 25
python scripts/detection/training/train_rfdetr_optuna_best.py  # retrain with the sweep's winning params
# NOTE: chasing this sweep's mAP50 objective has NOT translated into fewer real FPs on any
# trial tried so far — always re-verify with leaderboard.py --fp-gallery, not just mAP.

# Inpainting-based hard-negative generation — remove labeled Military
# Vehicle/person objects from real training images via ZITS (non-generative
# — can't hallucinate a new object into the hole), for manual review before
# promoting into a NEW dataset copy. Own venv. (LaMa was tried and retired
# in favor of ZITS after a quality comparison; ZITS is now the only backend.)
python3.12 -m venv .venv-inpaint && source .venv-inpaint/bin/activate
pip install iopaint opencv-python-headless numpy
python scripts/detection/tools/generate_inpainted_negatives_iopaint.py --n-images 18  # ZITS
# --> manually delete unwanted candidates from <review_dir>/_preview/, then:
python scripts/detection/tools/init_inpainted_dataset.py           # copy Detection_Dataset_hardneg -> _inpainted
python scripts/detection/tools/promote_inpainted_negatives.py --from-preview \
    --dest datasets/Detection_Dataset_hardneg_inpainted
# promote_inpainted_negatives.py REFUSES to target Detection_Dataset_hardneg directly
# (the dataset the production checkpoint was trained on) without --allow-hardneg.
```

## Architecture

The system is a real-time off-road perception pipeline that runs **SegFormer-B2** (semantic segmentation) and **YOLOE-26L** (open-vocabulary instance detection) in parallel on each frame, applies causal temporal smoothing, and renders the result to either a PyQt5 GUI player or an MP4 file.

```
src/perception/
  config/      typed dataclasses + YAML loader (schema.py + loader.py)
  core/        pure data contracts — Detection, FrameResult, BBox, Color; no deps
  io/          FrameSource ABC + video / camera / image-dir implementations
  models/
    backends/  InferenceBackend ABC — pytorch.py (default) + tensorrt.py (Jetson)
    instance/  YOLOE wrappers (yolo/open.py, yolo/closed.py) + RFDeTR; null.py for disabled
    semantic/  SegFormer wrapper (segformer.py) + _class_catalogues.py for ADE20K LUT
    factory.py registry-based dispatch keyed on YAML model name
  temporal/    LogitsEMA (ema_logits.py), SceneCutDetector (scene_cut.py), IoUTracker
  postprocess/ pure per-frame detection filters — duplicate_filter.py (same-class nested/overlap suppression)
  pipeline/    PerceptionPipeline — DI container; owns nothing, consumes ABCs
  render/      overlay.py primitives + renderer.py (display-mode-aware, z-order)
  ui/          PyQt5 main_window, video_widget, controls; decode + inference QThread workers
  datasets/    RUGD + ORFD downloaders and torch Dataset classes
```

**Import graph (strictly enforced):** `core` → nothing. `models` → `core`+`config`. `temporal` → `core`. `pipeline` → abstract bases of `models`+`temporal`. `ui` → `pipeline`+`render`+`io`. Breaking this layering is a bug.

**Class system is entirely YAML-driven** (`config/config.yaml`). Adding or changing a class is a YAML-only edit — no code changes. Semantic classes merge ADE20K channel logits via a LUT in `_class_catalogues.py`; instance classes use text prompts passed to YOLOE.

**Adding a new model:** subclass the relevant ABC (`semantic/base.py` or `instance/base.py`), register the name in `models/factory.py`, add a YAML entry. That's the entire integration surface.

**TensorRT backend:** `models/backends/tensorrt.py` documents the four steps to enable TRT on a deployment target. Engines are version-locked to the GPU + TensorRT version; rebuild after any JetPack upgrade.

## Key Design Decisions

- **Softmax-then-merge** (not raw-logit sum) when combining ADE20K channels into user classes — preserves probability semantics.
- **Causal EMA only** (`temporal/ema_logits.py`) — never looks ahead; safe for real-time streams. EMA and IoU tracker are reset on scene cuts detected via Bhattacharyya distance on HSV histograms.
- **Frozen dataclasses** for `Detection`, `FrameResult`, etc. — thread-safe pass-by-reference between the decoder thread and inference thread.
- **YOLOE text embeddings cached at warmup** — calling `cache_text_embeddings()` once avoids repeated GPU encode calls per frame.
- **opencv-python-headless** (not `opencv-python`) — prevents Qt plugin conflict with PyQt5.

## Active Models

| Model | Status | Notes |
|---|---|---|
| SegFormer-B2 | Primary semantic | mIoU=0.279 on GOOSE-Ex, ~19 ms/frame on RTX 5090 |
| SegFormer-B4 | Available | Slightly lower mIoU (0.268), ~24 ms |
| YOLOE-26L | Primary instance | Text embeds cached at warmup; discovery mode available |
| DDRNet-39 | Broken | GOOSE-12 channel ordering unconfirmed, IoU≈0.002; do not use |
| PP-LiteSeg | Shelved | Wrapper exists but raises `NotImplementedError` |

## Config Knobs

Key fields in `config/config.yaml` to know about:

| Field | Effect |
|---|---|
| `models.instance.enabled` | `false` → skip YOLOE (~2× faster, semantic-only) |
| `models.instance.profile` | Selects the active class block from `instance_profiles:` (`2class` / `6class` / `yoloe`) — must match the checkpoint's scheme |
| `models.semantic.processor_size` | `256`/`384`/`512` — lower = faster, coarser boundaries |
| `models.semantic.trt_engine_path` | Path to `.engine` (requires `hardware.use_tensorrt: true`) |
| `models.instance.prompt_mode` | `production` (text_prompt per class) or `discovery` (vocab file) |
| `temporal.semantic_ema.alpha` | EMA weight on current frame's logits (default 0.35) |
| `postprocess.duplicate_filter.enabled` | Drop same-class nested/overlapping duplicate boxes before tracking |
| `temporal.instance_tracker.enabled` | `false` → bypass tracking entirely (raw per-frame detections, no smoothing/hold) |
| `temporal.instance_tracker.use_hungarian_matching` | `true` → globally-optimal one-to-one IoU assignment instead of greedy best-first |
| `temporal.instance_tracker.min_hits` | `N>1` → a track must match N consecutive frames before display (suppresses single-frame FP flicker) |
| `models.instance.low_conf_recovery.enabled` | `true` → an already-confirmed track may accept a sub-threshold detection to keep following the real position instead of freezing via hold |
| `player.draw_road_ground_semantic_last` | z-order: render road_ground on top of other semantic classes |

## Detection FP/FN Investigation (rfdetr-m)

TIDE (`tide_analysis.py`) on the production rfdetr-m checkpoint shows its residual
error profile is dominated by **background error** (6/10 FPs — genuine
hallucination on empty background, e.g. an ego-vehicle-mounted rig or a loading
ramp) and **missed detections** (19 FN, the single biggest driver of AP loss).
Two independent fixes were tried and **both rejected** — every tried variant
increased FP or degraded recall relative to the untouched baseline:
- **Optuna hyperparameter tuning** (3 trials retrained+evaluated): mAP50 improved
  on all of them, but FP roughly doubled in every case. Chasing mAP50 as the
  sweep objective does not track real FP/FN — always re-verify with
  `leaderboard.py --fp-gallery`, not the sweep's own metric.
- **Inpainting-based hard negatives** (266 promoted, LaMa/ZITS, `Detection_Dataset_hardneg_inpainted`):
  background error was literally unchanged (6→6) and FN got worse (19→22).

The production checkpoint (`weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt`)
remains the best known rfdetr-m checkpoint. See `reports/detection/phase7_closing_summary.md`
for the full comparison table before proposing another retrain along either of these two axes.

## Jetson / Production Notes

- Development target: RTX 5090 (sm_120) — requires `cu128` PyTorch wheels.
- Production target: Jetson AGX Orin 64GB, JetPack 6.x, CUDA 12.2, TensorRT 10.
- On Jetson: install the NVIDIA-provided aarch64 PyTorch wheel *before* `pip install -r requirements.txt`, then exclude the `torch`/`torchvision` lines to avoid overwriting it.
- Use `requirements-jetson.txt` + `Dockerfile.jetson` / `docker-compose.jetson.yml` for the production environment.
- See `HOW_TO_RUN.md` and `JETSON.md` for the full Jetson-side workflow.
