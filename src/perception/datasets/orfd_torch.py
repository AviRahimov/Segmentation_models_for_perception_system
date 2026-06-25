"""PyTorch Dataset for the ORFD binary freespace dataset.

Label encoding (``*_fillcolor.png`` grayscale):
    255 → 1  (traversable)
    0   → 0  (non-traversable)
    128 → 2  (sky — explicit third class so the model learns sky features)

Pairing strategy: both ``image_data/`` and ``gt_image/`` files are named
by Unix timestamps in milliseconds.  GT filenames append ``_fillcolor``
(e.g. ``1623697255837_fillcolor.png``).  We extract the numeric timestamp
from each GT file and search for a matching image file with the same
timestamp.  Frames that have a GT but no image (or vice-versa) are silently
dropped; a warning is emitted if more than 10 % of files cannot be paired.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Label remap: source gray value → model class index (255 = ignore).
_LABEL_REMAP: dict[int, int] = {
    255: 1,  # traversable → class 1
    0:   0,  # non-traversable → class 0
    128: 2,  # sky → class 2 (explicit supervision; prevents sky-as-traversable artifact)
}

# ImageNet normalisation stats (same as the original SegFormer training).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model input size used during training.
TRAIN_SIZE = 512


def _remap_label(gt_u8: np.ndarray) -> np.ndarray:
    """Convert ORFD fillcolor values to {0, 1, 255}."""
    out = np.full_like(gt_u8, 255, dtype=np.uint8)
    for src, dst in _LABEL_REMAP.items():
        out[gt_u8 == src] = dst
    return out


def _pair_files(split_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted list of (image_path, label_path) pairs.

    Matching is by exact timestamp extracted from the GT filename.
    """
    img_dir = split_dir / "image_data"
    gt_dir  = split_dir / "gt_image"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"ORFD image_data directory not found: {img_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"ORFD gt_image directory not found: {gt_dir}")

    # Build a fast lookup: timestamp (str) → image path.
    img_by_ts: dict[str, Path] = {}
    for p in img_dir.glob("*.png"):
        img_by_ts[p.stem] = p  # stem == timestamp string

    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for gt_path in sorted(gt_dir.glob("*_fillcolor.png")):
        # GT stem: "<timestamp>_fillcolor"
        ts_str = gt_path.stem.removesuffix("_fillcolor")
        img_path = img_by_ts.get(ts_str)
        if img_path is None:
            skipped += 1
            continue
        pairs.append((img_path, gt_path))

    total_gt = len(list(gt_dir.glob("*_fillcolor.png")))
    if total_gt > 0 and skipped / total_gt > 0.10:
        logger.warning(
            "ORFD %s: %d / %d GT labels had no matching image (>10%% skip rate). "
            "Check dataset integrity.",
            split_dir.name, skipped, total_gt,
        )

    pairs.sort(key=lambda t: t[0].stem)
    logger.info("ORFD %s: %d paired samples (%d GT skipped).",
                split_dir.name, len(pairs), skipped)
    return pairs


class ORFDDataset(Dataset):
    """ORFD binary freespace dataset.

    Args:
        root:        Path to the ORFD root (contains training/, validation/,
                     testing/ subdirectories).
        split:       One of ``"training"``, ``"validation"``, ``"testing"``.
        augment:     If ``True``, apply training-time augmentations (random
                     flip, crop, scale, colour jitter). Set ``False`` for
                     validation / testing.
        input_size:  Square crop / resize target (default 512).
    """

    CLASSES = ("non_traversable", "traversable", "sky")
    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        augment: bool = True,
        input_size: int = TRAIN_SIZE,
    ) -> None:
        super().__init__()
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"ORFD root not found: {root}")
        valid_splits = {"training", "validation", "testing"}
        if split not in valid_splits:
            raise ValueError(f"split must be one of {valid_splits}, got {split!r}")

        self._split_dir = root / split
        self._augment = augment
        self._size = input_size
        self._pairs = _pair_files(self._split_dir)
        if not self._pairs:
            raise RuntimeError(
                f"No paired samples found in {self._split_dir}. "
                "Verify the ORFD dataset download."
            )

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, gt_path = self._pairs[idx]

        # --- Load ---
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise OSError(f"Cannot load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            raise OSError(f"Cannot load GT: {gt_path}")

        label = _remap_label(gt_raw)  # uint8 {0, 1, 255}

        # --- Augment or resize ---
        if self._augment:
            rgb, label = _augment_train(rgb, label, self._size)
        else:
            rgb, label = _resize_val(rgb, label, self._size)

        # --- To tensor ---
        img_t = _to_normalized_tensor(rgb)           # (3, H, W) float32
        lbl_t = torch.from_numpy(label.astype(np.int64))  # (H, W) int64

        return img_t, lbl_t

    # ------------------------------------------------------------------ #

    @property
    def pairs(self) -> list[tuple[Path, Path]]:
        """Read-only list of (image_path, label_path) pairs."""
        return list(self._pairs)


# --------------------------------------------------------------------------- #
# Preprocessing helpers                                                        #
# --------------------------------------------------------------------------- #


def _to_normalized_tensor(rgb: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB → CHW float32 tensor, ImageNet normalised."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def _augment_train(
    rgb: np.ndarray,
    label: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Training augmentation pipeline.

    Operations (all consistent between image and label unless noted):
        1. Random scale in [0.5, 2.0]
        2. Random horizontal flip (p=0.5)
        3. Random rotation ±15° (p=0.5)           [image+label]
        4. Random 512×512 crop (or pad if smaller)
        5. Gaussian blur (p=0.3)                   [image only]
        6. Colour + hue jitter                     [image only]
        7. Coarse dropout (p=0.4)                  [image+label → ignore]
    """
    h, w = rgb.shape[:2]

    # 1. Random scale.
    scale = np.random.uniform(0.5, 2.0)
    new_h = int(h * scale)
    new_w = int(w * scale)
    rgb   = cv2.resize(rgb,   (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 2. Random horizontal flip.
    if np.random.random() > 0.5:
        rgb   = np.fliplr(rgb).copy()
        label = np.fliplr(label).copy()

    # 3. Random rotation ±15°.
    rgb, label = _random_rotate(rgb, label)

    # 4. Pad if smaller than crop size, then random crop.
    h, w = rgb.shape[:2]
    pad_h = max(size - h, 0)
    pad_w = max(size - w, 0)
    if pad_h > 0 or pad_w > 0:
        rgb   = np.pad(rgb,   ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        label = np.pad(label, ((0, pad_h), (0, pad_w)),          mode="constant", constant_values=255)

    h, w = rgb.shape[:2]
    y0 = np.random.randint(0, h - size + 1)
    x0 = np.random.randint(0, w - size + 1)
    rgb   = rgb  [y0:y0 + size, x0:x0 + size]
    label = label[y0:y0 + size, x0:x0 + size]

    # 5. Gaussian blur (image only).
    rgb = _gaussian_blur(rgb)

    # 6. Colour + hue jitter (image only).
    rgb = _color_jitter(rgb)

    # 7. Coarse dropout.
    rgb, label = _coarse_dropout(rgb, label)

    return rgb, label


def _random_rotate(
    rgb: np.ndarray,
    label: np.ndarray,
    max_angle: float = 15.0,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image and label by a small random angle."""
    if np.random.random() > p:
        return rgb, label
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rgb   = cv2.warpAffine(rgb,   M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    label = cv2.warpAffine(label, M, (w, h),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rgb, label


def _gaussian_blur(rgb: np.ndarray, p: float = 0.3) -> np.ndarray:
    """Randomly apply Gaussian blur to simulate motion blur / defocus."""
    if np.random.random() > p:
        return rgb
    kernel_size = int(np.random.choice([3, 5, 7]))
    sigma = np.random.uniform(0.5, 2.0)
    return cv2.GaussianBlur(rgb, (kernel_size, kernel_size), sigma)


def _color_jitter(
    rgb: np.ndarray,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1,
) -> np.ndarray:
    """Independent random brightness, contrast, saturation, and hue jitter."""
    b = 1.0 + np.random.uniform(-brightness, brightness)
    c = 1.0 + np.random.uniform(-contrast, contrast)
    s = 1.0 + np.random.uniform(-saturation, saturation)
    h_shift = np.random.uniform(-hue * 180, hue * 180)  # OpenCV H is 0–180

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180          # hue shift
    hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)        # saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * b, 0, 255)        # brightness
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    mean = rgb.mean()
    rgb = np.clip((rgb.astype(np.float32) - mean) * c + mean, 0, 255).astype(np.uint8)
    return rgb


def _coarse_dropout(
    rgb: np.ndarray,
    label: np.ndarray,
    max_holes: int = 4,
    max_size: int = 64,
    p: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly erase rectangular patches to simulate occlusion."""
    if np.random.random() > p:
        return rgb, label
    h, w = rgb.shape[:2]
    rgb   = rgb.copy()
    label = label.copy()
    n_holes = np.random.randint(1, max_holes + 1)
    for _ in range(n_holes):
        hole_h = np.random.randint(16, max_size + 1)
        hole_w = np.random.randint(16, max_size + 1)
        y1 = np.random.randint(0, max(1, h - hole_h))
        x1 = np.random.randint(0, max(1, w - hole_w))
        y2, x2 = y1 + hole_h, x1 + hole_w
        rgb[y1:y2, x1:x2] = np.random.randint(0, 256, (1, 1, 3), dtype=np.uint8)
        label[y1:y2, x1:x2] = 255  # ignore erased regions
    return rgb, label


def _resize_val(
    rgb: np.ndarray,
    label: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validation / test: resize shorter side to ``size``, then centre-crop to size×size."""
    h, w = rgb.shape[:2]
    if h <= w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    rgb   = cv2.resize(rgb,   (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # Centre crop so every sample has identical spatial dimensions for batching.
    top  = (new_h - size) // 2
    left = (new_w - size) // 2
    rgb   = rgb  [top:top + size, left:left + size]
    label = label[top:top + size, left:left + size]
    return rgb, label
