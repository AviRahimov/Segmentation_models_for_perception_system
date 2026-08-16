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

# Evaluation — compare N models on ORFD/zikim (labeled, mIoU + freespace metrics)
# or fcdd (qualitative only, no verified label mapping yet); omit --dataset/--models
# to pick interactively from what's on disk
python scripts/segmentation/evaluation/compare_semantic_models.py \
    --dataset orfd --models segformer-b2 mask2former-large
python scripts/segmentation/evaluation/compare_semantic_models.py  # interactive
# Params + latency only, any --models, no dataset needed
python scripts/segmentation/evaluation/compare_semantic_models.py \
    --latency-only --models segformer-b2 mask2former-large auriganet
# Qualitative-only N-way comparison on raw, unlabeled video clips (no GT/metrics);
# omit --videos-dir/--models to pick interactively
python scripts/segmentation/evaluation/compare_on_raw_video.py \
    --videos-dir "datasets/segmentation/Off_Road_ShutterStcok_Videos&Frames" \
    --models mask2former-large mask2former-base segformer-b2

# Optimization pipeline (dev PC → Jetson workflow)
python scripts/segmentation/optimization/resolution_sweep.py --checkpoint weights/.../best.pth --data datasets/segmentation/ORFD
python scripts/segmentation/optimization/export_onnx.py --resolution 256
python scripts/segmentation/optimization/train_qat.py --config config/segmentation/optimization/qat.yaml
python scripts/segmentation/optimization/train_sparse.py --config config/segmentation/optimization/sparse.yaml
# Stage 4 runs on Jetson only:
python scripts/segmentation/optimization/benchmark_jetson.py --onnx-dir weights/segmentation/optimization/ --val-data datasets/segmentation/ORFD
python scripts/segmentation/optimization/compare_models.py --mode images \
    --model-a pytorch:weights/.../best.pth --model-b onnx:weights/segmentation/optimization/qat_int8.onnx \
    --test-data datasets/segmentation/ORFD
# Stage 6b: render benchmark_jetson.py's CSV -> colour-coded Markdown/HTML
python scripts/segmentation/optimization/generate_report.py --csv reports/segmentation/optimization/benchmark_results.csv

# RF-DETR TensorRT optimization (separate from the segmentation pipeline above —
# see JETSON.md's "RF-DETR Optimization" / "Best-YOLO Optimization" sections for
# the full SSH-driven workflow, incl. harness perf tuning and the rfdetr-s/YOLO variants)
source .venv-rfdetr-train/bin/activate
python scripts/detection/optimization/export_onnx.py   # dev PC: export + validate ONNX vs PyTorch
# --checkpoint/--model-name/--shape generalize this to rfdetr-s/rfdetr-l too
# Transfer the .onnx + gaza clips + val subset to the Jetson (one weights/{model}/
# subdir per model — see JETSON.md's on-device folder layout), then on-device:
python3 scripts/benchmark_jetson.py \
    --onnx weights/rfdetr-m/rfdetr-m.onnx --engine-dir weights/rfdetr-m --model-name rfdetr-m \
    --videos-dir data/videos --val-images data/val_images --val-labels data/val_labels \
    --output results/benchmark_results_rfdetr-m.csv
# --harness {naive,optimized,both} (default both) compares the plain CPU-preprocess/
# full-sync loop against GPU-preprocess + CUDA-graph replay — see CLAUDE.md's
# "RF-DETR / YOLO Jetson TensorRT Optimization" section for measured numbers
# Best-YOLO checkpoint gets the same real-video FPS/accuracy treatment via Ultralytics'
# own native TensorRT exporter (needs a source-built torchvision — see JETSON.md):
python3 scripts/benchmark_yolo_jetson.py \
    --weights weights/yolo11m_freeze21/yolo11m_freeze21.pt --model-name yolo11m_freeze21 \
    --videos-dir data/videos --val-images data/val_images --val-labels data/val_labels \
    --output results/benchmark_results_yolo11m.csv
# Back on the dev PC, merge the 3 pulled-back CSVs and render:
python scripts/detection/optimization/generate_report.py --csv reports/detection/optimization/benchmark_results.csv
# Full-length 3-way annotated comparison videos (all 11 gaza clips) — see JETSON.md's
# Stage 3 for the loop; saved to reports/detection/optimization/3way_comparison/
# N-way comparison — any mix of pytorch:/onnx:/engine:/ultralytics: specs:
python scripts/detection/optimization/compare_models.py --mode video \
    --models pytorch:weights/detection/rfdetr-s/detection_dataset_hardneg/conservative_aug/best.pt \
             pytorch:weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \
             ultralytics:weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt \
    --source samples/clip.mp4

# Detection training — Round 1 (YOLO11/YOLO26, scales s/m/l)
python scripts/detection/training/train_round1.py --model yolo26m
# Output: weights/detection/{model_name}/round1/best.pt

# Detection training — general (interactive survey: scans datasets/detection/, Enter = defaults)
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
    --data datasets/detection/Detection_Dataset/data.yaml

# Confidence calibration — fit per-class temperature scaling for one checkpoint
# (interactive checkpoint/benchmark picker if flags omitted); enable via
# postprocess.calibration in config.yaml once fitted
python scripts/detection/evaluation/fit_calibration.py \
    --weights weights/detection/rfdetr-2xl/detection_dataset/coco/best.pt

# Detection model comparison (paper-style — table / images / video)
python scripts/detection/evaluation/compare_detection_models.py --mode table \
    --models pytorch:weights/detection/yolo26s/round1/best.pt \
             pytorch:weights/detection/yolo26m/round1/best.pt \
             pytorch:weights/detection/rfdetr-m/round1/best.pt
python scripts/detection/evaluation/compare_detection_models.py --mode images \
    --models pytorch:weights/detection/yolo26m/round1/best.pt \
             pytorch:weights/detection/rfdetr-m/round1/best.pt \
    --test-data datasets/detection/Detection_Dataset/valid/images --n-samples 20
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
    --image datasets/detection/Detection_Dataset/test/images/some_frame.jpg

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
    --dest datasets/detection/Detection_Dataset_hardneg_inpainted
# promote_inpainted_negatives.py REFUSES to target Detection_Dataset_hardneg directly
# (the dataset the production checkpoint was trained on) without --allow-hardneg.
```

## Architecture

The system is a real-time off-road perception pipeline that runs **SegFormer-B2** (semantic segmentation) and **RF-DETR** (closed-vocabulary instance detection) in parallel on each frame, applies causal temporal smoothing, and renders the result to either a PyQt5 GUI player or an MP4 file.

```
src/perception/
  config/      typed dataclasses + YAML loader (schema.py + loader.py)
  core/        pure data contracts — Detection, FrameResult, BBox, Color; no deps
  io/          FrameSource ABC + video / camera / image-dir implementations
  models/
    backends/  InferenceBackend ABC — pytorch.py (default) + tensorrt.py (Jetson)
    instance/  YOLO closed-vocab wrapper (yolo/closed.py) + RFDeTR; null.py for disabled
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

**Class system is entirely YAML-driven** (`config/config.yaml`). Adding or changing a class is a YAML-only edit — no code changes. Semantic classes merge ADE20K channel logits via a LUT in `_class_catalogues.py`; instance classes match a closed-vocab detector's (RF-DETR / YOLO) output via `coco_classes`.

**Adding a new model:** subclass the relevant ABC (`semantic/base.py` or `instance/base.py`), register the name in `models/factory.py`, add a YAML entry. That's the entire integration surface.

**TensorRT backend:** `models/backends/tensorrt.py` documents the four steps to enable TRT on a deployment target. Engines are version-locked to the GPU + TensorRT version; rebuild after any JetPack upgrade.

## Key Design Decisions

- **Softmax-then-merge** (not raw-logit sum) when combining ADE20K channels into user classes — preserves probability semantics.
- **Causal EMA only** (`temporal/ema_logits.py`) — never looks ahead; safe for real-time streams. EMA and IoU tracker are reset on scene cuts detected via Bhattacharyya distance on HSV histograms.
- **Frozen dataclasses** for `Detection`, `FrameResult`, etc. — thread-safe pass-by-reference between the decoder thread and inference thread.
- **opencv-python-headless** (not `opencv-python`) — prevents Qt plugin conflict with PyQt5.

## Active Models

| Model | Status | Notes |
|---|---|---|
| SegFormer-B2 | Primary semantic | mIoU=0.279 on GOOSE-Ex, ~19 ms/frame on RTX 5090; ORFD 3-class mIoU=0.8624 |
| SegFormer-B2 (distilled) | Candidate, not yet promoted | ORFD 3-class mIoU=0.8813 — beats production and its Mask2Former-Large teacher; same architecture/speed as production. See "Segmentation Architecture Comparison + Distillation" below before promoting |
| SegFormer-B4 | Available | Slightly lower mIoU (0.268), ~24 ms |
| RF-DETR-M | Primary instance | Closed-vocab (`coco_classes`); see "RF-DETR / YOLO Jetson TensorRT Optimization" below |
| YOLOE-26L | Removed from `src/` | Was the original primary instance model (open-vocabulary, text-prompt-driven) — researched, trained, and directly compared against closed-vocab YOLO variants before RF-DETR was adopted for production. Superseded because RF-DETR gave better real accuracy/FPS on this project's own footage (see the Jetson optimization section below); YOLOE's wrapper/config/training code was fully deleted from `src/` once no longer used, but this evaluation history is intentionally kept here rather than erased. |
| DDRNet-39 | Removed | Fully deleted from `src/` — was broken (GOOSE-12 channel ordering unconfirmed, IoU≈0.002) |
| PP-LiteSeg | Removed | Fully deleted from `src/` — wrapper had raised `NotImplementedError` |

## Segmentation Architecture Comparison + Distillation (ORFD)

Production SegFormer-B2 (`weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth`,
**0.8624 mIoU**, current 3-class metric) was compared against 6 new architectures fine-tuned on the
same ORFD split with the identical `compute_miou`: AurigaNet (resurrected, 0.8439), UPerNet/ConvNeXt-B
(0.8632), DINOv2-Base/Large + linear head (0.8592 / 0.8534 — bigger hurt here), and Mask2Former
Swin-Base/Large (0.8742 / 0.8780 — bigger helped here). **Important correction**: "0.852", cited
repeatedly in earlier reporting as production's mIoU, was actually `benchmark_orfd.py`'s stale
**2-class** metric (no sky) — always use the 3-class metric (`_orfd_common.compute_miou`,
`NUM_CLASSES=3`) for any new comparison.

**Distillation (Mask2Former-Large → SegFormer-B2) succeeded**: per the user's own visual review of
real footage (Mask2Former-Large handled two new domains — Zikim and ShutterStock clips — better than
its own metrics suggested), it was used as a response-based KD teacher for production's
architecture. Method: student warm-starts from the production checkpoint (not ADE20K cold-start);
loss = hard-label Dice+CE + temperature-scaled KL divergence (T=3) against the teacher's dense
per-pixel probabilities (`masks_classes = softmax(class_queries_logits)[...,:-1]`,
`masks_probs = sigmoid(masks_queries_logits)`, `probs = einsum("bqc,bqhw->bchw", ...)`, normalized).
Result: **0.8813 mIoU** (`weights/segmentation/orfd/distilled_segformer-b2/best.pth`) — beats
production (+0.0189) *and its own teacher* (+0.0033), while keeping SegFormer-B2's architecture and
the Jetson-proven ~150–160 FPS optimized speed unchanged (no re-export/re-optimization needed to
deploy it). Script: `scripts/segmentation/training/train_distill.py`.

**LoRA-adapting Mask2Former-Large's Swin backbone was tried and rejected**: instead of keeping the
backbone fully frozen (the recipe that reached 0.8780), `train_mask2former.py --lora` (targeting only
`attention.self.{query,value}` inside `pixel_level_module.encoder`, everything else left trainable)
peaked at **0.8639 at epoch 1** and never recovered, oscillating lower for the rest of training —
partial backbone adaptation made this model worse, not better.

**A genuine, disclosed complication**: on ORFD's own narrower binary-traversable-only spot check (a
different, stricter metric used elsewhere in this project, not the 3-class mIoU above), production
SegFormer-B2 still leads (0.947) over Mask2Former-Large (0.909), Mask2Former-Base (0.911), and even
the distilled model (0.895) — the distilled student appears to have partly inherited the teacher's
comparatively weaker showing on this specific check along with its strength on the full metric. No
single number is the whole story here; the distilled checkpoint is the strongest candidate on the
primary metric and worth deploying, but this tradeoff should be understood, not hidden, before
promoting it over production.

**The binary-traversable metric above is argmax-based by default, but doesn't have to be** —
`compare_semantic_models.py --traversable-threshold P` (or `orfd_semantic_comparison.freespace_merged_prob_floor`
in `config.yaml`) scores traversable as `P(road_ground) >= P` instead. This knob only affects that
one eval metric, not the live deployed pipeline (`PerceptionPipeline`/`Renderer` always argmax
regardless of this setting). A quick sweep on production SegFormer-B2 (200 ORFD training-split
samples, 2 seeds) found the config's current default (0.25, mean binary-trav IoU 0.852/0.852 across
seeds) is actually *worse* than plain argmax (0.895/0.890), and a floor around **0.75-0.78** does
meaningfully better than either (0.914/0.901) — a real, reproducible +6-9 point gap on this metric,
not yet applied anywhere (config.yaml's default is unchanged pending a decision on whether to adopt
it, and this hasn't been checked against zikim/fcdd or other checkpoints).

## Config Knobs

Key fields in `config/config.yaml` to know about:

| Field | Effect |
|---|---|
| `models.instance.enabled` | `false` → skip the instance model (~2× faster, semantic-only) |
| `models.instance.profile` | Selects the active class block from `instance_profiles:` (`2class` / `6class` / `rfdetr_2class` / `rfdetr_6class`) — must match the checkpoint's scheme |
| `models.semantic.processor_size` | `256`/`384`/`512` — lower = faster, coarser boundaries |
| `models.semantic.trt_engine_path` | Path to `.engine` (requires `hardware.use_tensorrt: true`) |
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

## RF-DETR Confidence Threshold Selection (rfdetr-m production checkpoint)

The production `config.yaml` values (`rfdetr_2class` profile) were originally both `0.50`, set from a
single best-F1 read on the 55-image test split — and the mil-vehicle line's own comment cited a
*different* model/dataset (`rfdetr-2xl/detection_dataset/coco`), not the actual deployed
`rfdetr-m/detection_dataset_hardneg/conservative_aug` checkpoint. A deeper, real-numbers sweep was run
instead of trusting that value: RF-DETR has **no NMS-IoU knob at all** (confirmed by reading
`src/perception/models/instance/rfdetr/model.py`'s `predict()` call — only `threshold` is passed — and
grepping the installed `rfdetr` package source for `nms`, zero matches; it's a DETR-family set-prediction
model, genuinely NMS-free), so `tune_thresholds.py`'s YOLO-only conf×IoU sweep doesn't apply here.
`leaderboard.py --thresholds` already computes a full 19-point conf-sweep curve (0.05→0.95 step 0.05, via
`_ap_utils.threshold_sweep`) but was discarding everything except the single best-F1 row — extended to
also write the full curve to `reports/detection/threshold_recommendations.md`, and a `--deploy-conf` flag
was added (both `leaderboard.py` and `fp_full_image_review.py`) so `--fp-gallery`/full-image FP review can
be re-run at any candidate threshold, not just the fixed 0.40 both tools previously hardcoded.

**A real bug was found and fixed while cross-checking qualitatively**: `fp_full_image_review.py`'s
`--conf` flag only set the prediction-collection floor passed to the model — the actual TP/FP
classification (`_classify_boxes`) hardcoded `min_score=_DEPLOY_CONF` (0.40) regardless of `--conf`,
so passing `--conf 0.55` silently still classified FPs at 0.40 unless `--conf` happened to exceed 0.40
(in which case the collection floor became the *effective* threshold by coincidence, masking the bug).
Fixed by threading a proper `--deploy-conf` flag into `_classify_boxes` itself.

The sweep was run on **both** `Detection_Dataset/valid` (34 images) and `/test` (55 images)
independently — trusting either split alone risks fitting sample noise, the same lesson as this
project's other single-metric pitfalls:

| Class | Conf | Split | Precision | Recall | F1 | FP/img |
|---|---|---|---|---|---|---|
| Military Vehicle | 0.50 (old) | valid / test | 0.966/0.814/0.884/0.059 | — | — | 0.900/0.955/0.926/0.127 |
| Military Vehicle | **0.60 (new)** | valid / test | 1.000/0.786/0.880/0.000 | — | — | 0.938/0.924/0.931/0.073 |
| person | 0.50 (old) | valid / test | 0.897/0.778/0.833/0.118 | — | — | 0.929/0.696/0.796/0.055 |
| person | **0.55 (new)** | valid / test | 0.946/0.778/0.854/0.059 | — | — | 0.971/0.607/0.747/0.018 |

(Each cell is P/R/F1/FP-per-image for that split.) Military Vehicle's new value is an unambiguous win —
F1 flat-to-better on both splits while FP/img drops substantially, not a recall-for-FP tradeoff. Person is
a genuine, disclosed tradeoff: FP/img drops meaningfully on both splits and F1 *improves* on valid, but
test-split recall costs more (0.696→0.607) — the two splits disagreed on person's own best-F1 point
(0.55 on valid vs. 0.30 on test), a real small-sample disagreement, not hidden.

TIDE (`tide_analysis.py --conf-thr`) at 0.50/0.60/0.70 (uniform across both classes, since TIDE doesn't
support per-class thresholds) confirmed the expected Bkg-vs-Miss trade directionally: Bkg hallucinations
6→5→4, but Miss (FN) rose 21→28→33 — a real cost from raising a *shared* threshold too far, which is
exactly why the final choice uses **per-class** thresholds (config.yaml already supports independent
values per class) rather than one shared value. Qualitative cross-check
(`fp_full_image_review.py --deploy-conf`) at the chosen values found the remaining FPs were the same
already-known error modes — a genuine unlabeled second vehicle in dust/haze, a near-duplicate person box
on an already-correctly-detected person, and a loading-ramp/rig hallucination matching the background-error
pattern already documented above — not new or surprising failure modes.

**Applied to production**: `config/config.yaml`'s `rfdetr_2class` profile now uses
`confidence_threshold: 0.60` (Military Vehicle) and `0.55` (person), replacing both `0.50` values, with
inline comments citing this methodology and the correct checkpoint.

## RF-DETR / YOLO Jetson TensorRT Optimization

`scripts/detection/optimization/` (export_onnx.py, `_rfdetr_trt_common.py`, `_video_bench_common.py`,
benchmark_jetson.py, benchmark_yolo_jetson.py, compare_models.py, generate_report.py) builds and
benchmarks TensorRT engines for rfdetr-m, rfdetr-s, and the best YOLO checkpoint, separate from the
training/evaluation scripts — see JETSON.md's "RF-DETR Optimization" / "Best-YOLO Optimization"
sections for the full dev-PC → Jetson workflow. Measured on a real AGX Orin (TensorRT 8.6.2.3, MAXN,
real decode+infer over 11 gaza-road clips — see `reports/detection/optimization/RESULTS.md` for the
full table):

| Model | Precision | Harness | FPS | Recall @0.4 | Notes |
|---|---|---|---|---|---|
| rfdetr-m | FP32 | naive (cv2 preprocess, full sync, no CUDA graph) | 33.2 | 0.79 | Healthy baseline |
| rfdetr-m | FP32 | **optimized** (GPU preprocess + CUDA graph) | **51.4** | 0.72 | Matches RidgeRun's own 52 FPS reference almost exactly |
| rfdetr-m | FP16 | either | 52.7 / 115.1 | **0.00** | **Broken** — logits systematically suppressed (max ≈ -3.0 vs FP32's ≈ +2.7), not NaN, just severe precision loss through the transformer. Confirmed independent of harness. |
| rfdetr-s | FP32 | naive | 42.1 | 0.80 | Better accuracy than rfdetr-m at higher FPS |
| rfdetr-s | FP32 | **optimized** | **68.9** | 0.77 | Matches RidgeRun's rfdetr-s reference (69 FPS) almost exactly |
| rfdetr-s | FP16 | either | 57.1 / 137.7 | **0.00** | Same FP16 breakage as rfdetr-m |
| yolo11m (best, see below) | FP32 | Ultralytics native | 49.6 | 0.57 | Notably lower accuracy than either RF-DETR variant |
| yolo11m (best, see below) | FP16 | Ultralytics native | **71.5** | 0.57 | **FP16 works fine for YOLO** — recall identical to FP32, unlike RF-DETR's total collapse; TensorRT did warn about 99 subnormal-FP16 weights but it didn't show up in real accuracy |

**Harness optimization** (`_rfdetr_trt_common.RFDETRTensorRTEngine`'s `gpu_preprocess=True,
cuda_graph=True`) closes most of the gap between our own Python inference loop and `trtexec`'s own
optimized benchmark of the identical engine (which gets 58 FPS on rfdetr-m FP32 — confirming the
*engine* was never the bottleneck, the naive harness was): GPU-side resize/normalize (CUDA bilinear
interpolate needs a float cast first — doesn't accept uint8) plus CUDA-graph capture around
`execute_async_v3` (capture succeeds cleanly on this TRT 8.6.2.3 build; wrapped in try/except since
NVIDIA/TensorRT#2603 reports capture failing on TRT 8.5.2.2). Real, measured cost: switching from cv2
resize to GPU resize shifts rfdetr-m's recall 0.79→0.72 — a genuine accuracy delta from the different
bilinear implementation, not free, disclosed rather than hidden.

**Naive FP16 export is rejected as-is** for both rfdetr-m and rfdetr-s — confirms the exact risk
flagged in RidgeRun's own `deepstream-rfdetr` README ("detection quality degraded considerably" under
FP16) and in `infracv/rf-detr-cpp`'s issue tracker (post-attention LayerNorm overflow). The large FPS
gain is not usable at the cost of zero real detections. A follow-up worth trying before giving up on
FP16 entirely: per-layer precision constraints (keep the attention/LayerNorm layers in FP32 via
`trtexec --layerPrecisions`/`--precisionConstraints` while leaving the rest FP16) rather than a
wholesale FP32 fallback — not yet attempted.

**rfdetr-s is a strong alternative to rfdetr-m**: `weights/detection/rfdetr-s/detection_dataset_hardneg/conservative_aug/best.pt`
already existed from a prior training run (same recipe used for rfdetr-m's production checkpoint, no
retraining needed) and beats rfdetr-m on both axes — higher FPS (68.9 vs 51.4 optimized FP32) and
comparable-or-better accuracy (precision 0.926 vs 0.902, recall 0.77 vs 0.72).

**Best YOLO checkpoint** per the cached `reports/detection/leaderboard_test.md` ranking is
`weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt` (mAP50=0.8677, same 2-class
scheme as the RF-DETR checkpoints — directly comparable, no collapse-map needed). Exporting it to
TensorRT on this JetPack 6.0/torch 2.4.0a0 combo needed two non-obvious fixes, both documented in
JETSON.md: (1) `torchvision` has no NVIDIA-provided wheel for this JetPack version — must be built
from source against the exact NVIDIA torch build (a generic PyPI wheel fails with
`operator torchvision::nms does not exist`, an ABI mismatch); (2) Ultralytics' exporter unconditionally
passes `dynamo=False` to `torch.onnx.export` for any "torch 2.4.x", but NVIDIA's Jetson snapshot
predates upstream torch actually adding that parameter — `benchmark_yolo_jetson.py` patches
`torch.onnx.export` to drop unsupported kwargs rather than patching Ultralytics itself.

**But real accuracy on our own footage tells a different story than the leaderboard rank**: despite
being the top-ranked YOLO checkpoint by collapsed mAP50, this model's real precision/recall at conf=0.4
(0.78/0.57) is meaningfully behind *both* RF-DETR variants (0.90+/0.72+) on the same validation images —
another instance of this repo's recurring lesson that a single aggregate metric doesn't reliably predict
real deployment behavior (see the Optuna note above). **rfdetr-s remains the strongest overall
candidate**: best accuracy, second-best FPS (67.9, only behind YOLO's FP16 73.1), and no FP16 precision
cliff to work around.

**Bug found and fixed: `compare_models.py` silently fed rfdetr-s the wrong input resolution.**
Different RF-DETR variants use different fixed engine input sizes (rfdetr-s=512×512,
rfdetr-m=576×576), but `_rfdetr_trt_common.load_model()` hardcoded a single default (576×576) for
every `engine:`/`onnx:` spec regardless of which model it actually was. Feeding a 576×576-sized
buffer into an engine statically compiled for 512×512 doesn't error — TensorRT just reads the
memory according to its own compiled shape, silently misinterpreting the data — so rfdetr-s
produced garbage logits that never crossed the confidence threshold: **0 detections in every frame
of every video**, discovered only by visually spot-checking the rendered comparison videos (the
numeric benchmark CSVs were unaffected, since `benchmark_jetson.py` always passed the correct
`--shape` explicitly). Fixed by auto-detecting the real input size from the loaded engine/ONNX
graph itself (`RFDETRTensorRTEngine`/`RFDETROnnxModel` now read it from
`engine.get_tensor_shape()`/`session.get_inputs()[0].shape`), so a caller-supplied size is now just
an optional override that gets corrected (with a logged warning) rather than silently trusted.
**Lesson: always visually spot-check rendered detection output, not just aggregate metrics** — this
exact class of bug (silent wrong-shape input) produces no exception anywhere in the stack.

## Combined Detection+Segmentation Jetson Benchmark

`scripts/tools/jetson_combined_survey.py` — standalone (no `src/perception`, see JETSON.md's
"Combined detection + segmentation survey"), runs one detection engine + one segmentation engine
together per frame on a real video and reports real combined FPS on the AGX Orin. Neither
`~/perception_optim/` nor `~/perception_optim/segformer_repo/` can run the full
`PerceptionPipeline` (no `models`/`pipeline`/`render` package installed on-device), so this reuses
each family's own proven TensorRT-spec loader (`_rfdetr_trt_common.load_model()` /
`_segformer_trt_common.SegformerTensorRTEngine`) directly instead.

The distilled SegFormer-B2 checkpoint (see "Segmentation Architecture Comparison + Distillation"
above) was exported to ONNX/TensorRT for the first time this session, using the exact same
`export_onnx.py` → `benchmark_jetson.py` pipeline already proven for production SegFormer-B2 (no new
export code needed — same architecture, different weights). Real engine mIoU: **0.8720** (FP32 and
FP16 identical — SegFormer has no FP16 precision cliff, unlike RF-DETR), FPS optimized (GPU-preprocess
+ CUDA graph): **156.3 FP16 / 158.1 FP32** — matching production SegFormer-B2's speed almost exactly,
as expected (same architecture/resolution).

Real combined (sequential, both models every frame, `tzir-driving.mp4`, det-conf=0.35, MAXN power
mode confirmed via `nvpmodel -q`, `sudo jetson_clocks` applied and the full sweep re-run afterward —
numbers below are unchanged from before the explicit clock lock, within ±0.3 FPS noise, confirming
the device was already at its real ceiling and clock throttling was never the bottleneck):

| Detection | Segmentation | Combined FPS | Detection-only | Segmentation-only |
|---|---|---:|---:|---:|
| rfdetr-s (FP32) | distilled (FP16) | 51.0 | 70.3 | 185.1 |
| rfdetr-s (FP32) | production (FP16) | 51.1 | 70.5 | 184.9 |
| rfdetr-m (FP32) | distilled (FP16) | 40.8 | 52.3 | 186.0 |
| rfdetr-m (FP32) | production (FP16) | 40.8 | 52.2 | 186.0 |
| yolo11m_freeze21 (FP16) | distilled (FP16) | 57.6 | 82.8 | 189.4 |
| yolo11m_freeze21 (FP16) | production (FP16) | 58.2 | 83.7 | 191.7 |

Detection-only numbers here are consistent with the isolated detection-only phase above (rfdetr-s
~69 FPS, rfdetr-m ~51 FPS, yolo11m FP16 ~71.5 FPS there vs ~70-84 FPS here — real hardware run-to-run
variance, not a methodology change). **Which segmentation checkpoint is paired barely matters for
combined FPS** (both are the same architecture/resolution) — **detection model choice dominates**.
All six combinations stay comfortably above real-time (30 FPS). rfdetr-m/rfdetr-s FP16 are
deliberately not offered in this script's registry — both have the confirmed real precision collapse
documented above (recall→0), not a speed tradeoff worth exposing as a default choice.

**GPU-stream concurrency was tried and didn't help.** Both engines currently run fully sequentially
(each `.infer()` call blocks on a `cudaEvent` sync before returning) on the default CUDA stream.
`scripts/tools/_stream_overlap_probe.py` (one-off, not part of the survey script) tested putting each
engine's CUDA-graph replay + GPU preprocessing on its own dedicated stream and syncing both only at
the end, so the GPU scheduler could interleave their kernels if there was spare SM capacity — measured
**52.5 FPS vs 54.1 FPS sequential (rfdetr-s + distilled, same clip) — 3% *slower*, not faster.**
Orin's iGPU has no real concurrency headroom left once one model (rfdetr-s alone already uses most of
it) is running; the extra stream-management overhead is pure loss. The sequential design is already
close to this hardware's ceiling for this pairing — the effective lever for combined FPS is detection
model choice (see table above), not execution scheduling.

**EMA smoothing added to the standalone survey script** (`jetson_combined_survey.py --ema
--ema-alpha 0.35`, off by default). Production's `PerceptionPipeline` already has causal EMA smoothing
(`src/perception/temporal/ema_logits.py`) wired in and active on every frame regardless of backend — but
that pipeline has never actually run on the Jetson device (the on-device checkout deliberately lacks
`models`/`pipeline`/`render`, only `datasets/`). `LogitsEMA` was duplicated (not imported — importing
`perception.temporal` on-device fails at package-import time, since even its own `__init__.py` eagerly
pulls in `factory.py`, which reaches into `models`/`config`) directly into `_segformer_trt_common.py`,
plus a new `SegformerTensorRTEngine.infer_smoothed()` method that blends raw engine logits through the
duplicate before upsample+argmax (`.infer()` itself is unchanged). Verified frame-1 bit-identical between
`.infer()`/`.infer_smoothed()` (EMA is pass-through on the first frame) with zero diff on a real engine.
Measured cost: essentially free — segmentation-only FPS 172.7→167.9, combined 49.8→49.4 (rfdetr-s +
gaza_joint_hardneg_tversky_b2_fp16, `tzir-driving.mp4`). Frame-to-frame overlay pixel diff over 5
consecutive frames dropped from a mean of 7.02 (off) to 6.69 (on), and the qualitative render showed
visibly calmer segmentation-boundary edges — a real, if modest, reduction (most of the remaining
frame-to-frame diff is genuine camera motion, not classification jitter).

**A significant, disclosed correction to this section's own earlier "clock throttling was never the
bottleneck" claim above**: `jetson_clocks` does **not persist across a reboot or a new SSH session** —
it must be re-applied every session, and skipping it produces a dramatic, real FPS gap, not noise. This
session's Jetson connection started with the CPU governor at `schedutil`/1.42GHz (not locked), and the
*exact same* rfdetr-s + gaza_joint_hardneg_tversky_b2_fp16 pairing measured **29.2 FPS combined / 101.6
FPS segmentation-only** under those conditions — closely matching the previously-reported "combined FPS
dropped from 40.8 to 29" anomaly investigated earlier this session, which had been root-caused (at the
time) to CPU/GPU contention from interleaved decode+infer+render+write. After `sudo jetson_clocks` was
applied (confirmed via `cat scaling_cur_freq` → locked at 2.2016GHz, not just `nvpmodel -q`'s power-mode
check, which only confirms the mode allows max clocks, not that they're actually locked there), the
*identical* pairing measured **49.3-49.9 FPS combined / 167-174 FPS segmentation-only** — a ~1.7x jump
from clock state alone. This strongly suggests the earlier interleaving root-cause theory was at least
partially confounded by the same non-locked-clocks condition rather than being a purely architectural
bottleneck (see the decode/write-threading result immediately below, which found ~0% additional gain once
clocks were properly locked). **Always verify `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`
directly each session** — `nvpmodel -q` alone is not sufficient confirmation.

**GStreamer/NVDEC hardware decode is available on this device but not attempted this round.**
`cv2.getBuildInformation()` on-device shows `GStreamer: YES (1.20.3)`, and `gst-inspect-1.0` confirms both
`nvv4l2decoder` (hardware H.264/H.265 decode) and `nvvidconv` are installed — contrary to this plan's own
prior assumption that a default JetPack 6 OpenCV build likely lacks this. A full `cv2.VideoCapture`
GStreamer-pipeline rewrite is a larger, higher-risk change than what was attempted this round, though, and
with clocks properly locked the interleaving bottleneck this would have targeted turned out not to be the
dominant factor after all (see above) — flagged as a real, available lever for a future round, not pursued
here.

**Opt-in decode/write threading was tried and measured ~0% gain once clocks were locked.**
`jetson_combined_survey.py --pipeline-io` (background `threading.Thread` + bounded `queue.Queue` for
`cv2.VideoCapture.read()`/`cv2.VideoWriter.write()`, all CUDA calls kept on the main thread — a genuinely
different mechanism from the rejected GPU-stream-concurrency approach above, which overlapped CUDA kernels
rather than CPU-bound I/O) and a `--no-write` diagnostic flag were added and validated: output is
pixel-identical to the unthreaded baseline (900/900 frames, max diff 0, confirming no dropped/reordered
frames), but FPS was flat across every combination tested with locked clocks — baseline 49.3-49.9,
`--no-write` 50.0, `--pipeline-io` 49.6, both together 49.8 (rfdetr-s + gaza_joint_hardneg_tversky_b2_fp16,
`tzir-driving.mp4`, both 300-frame and full 900-frame runs). Kept in the script (correct, harmless, opt-in)
but **not enabled in the final validation run below** — it earned zero measured benefit once the real
bottleneck (unlocked clocks) was addressed.

**Final combined validation** (EMA on, alpha=0.35; rfdetr-s at the new per-class-averaged single
`--det-conf 0.55` — this standalone script has no per-class threshold support, unlike production
`config.yaml`, a disclosed simplification; `gaza_joint_hardneg_tversky_b2_fp16`; clocks locked; 300 frames
per clip; all 11 real Gaza/desert clips under `~/perception_optim/data/videos/`):

| Video | Combined FPS | Detection-only | Segmentation-only |
|---|---:|---:|---:|
| tzir-driving.mp4 | 43.2 | 61.4 | 145.8 |
| colisim.mp4 | 49.6 | 70.7 | 166.9 |
| open-field-without-mg.mp4 | 49.5 | 70.5 | 166.2 |
| armoured-bulldozers-sand-humvee | 49.6 | 70.3 | 168.8 |
| gaza-israel-jeep-combing-road | 49.4 | 70.0 | 167.4 |
| gaza-palestine-jeeps-tanks-western-edge | 48.9 | 69.6 | 164.4 |
| gaza-palestine-jeep-combing-western | 43.3 | 62.1 | 143.2 |
| military-armed-vehicle-ceasefire | 44.4 | 63.2 | 149.1 |
| military-vehicle-streets-bombardment-ceasefire | 42.2 | 59.8 | 143.0 |
| pov-military-ground-vehicle-destroyed-streets | 43.5 | 61.7 | 147.0 |
| tank-with-smoke.mp4 | 44.8 | 63.4 | 152.8 |

All 11 clips stay comfortably above real-time (30 FPS), ranging 42.2-49.6 FPS combined — the spread
across clips (vs. the single-clip numbers above) reflects differing native video resolutions affecting
CPU-side render/overlay cost, confirmed not thermal throttling (temps 42-49°C, CPU still locked at
2.2016GHz throughout). Rendered videos spot-checked visually: segmentation boundaries calm and consistent
with EMA on, the tuned per-class thresholds show no new/surprising false positives beyond the
already-documented error modes, and Stage 6's rock/rubble-vs-traversable fix holds up on real Gaza
footage (`colisim.mp4`'s rubble streets correctly excluded from the traversable-path prediction).

## Jetson / Production Notes

- Development target: RTX 5090 (sm_120) — requires `cu128` PyTorch wheels.
- Production target: Jetson AGX Orin 64GB, JetPack 6.x, CUDA 12.2, TensorRT 10.
- On Jetson: install the NVIDIA-provided aarch64 PyTorch wheel *before* `pip install -r requirements.txt`, then exclude the `torch`/`torchvision` lines to avoid overwriting it.
- Use `requirements-jetson.txt` + `Dockerfile.jetson` / `docker-compose.jetson.yml` for the production environment.
- See `HOW_TO_RUN.md` and `JETSON.md` for the full Jetson-side workflow.
