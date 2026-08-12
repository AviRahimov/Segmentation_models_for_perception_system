"""PyTorch Dataset for the Gaza-domain fine-grained-labeled image set.

Produced by the SAM3 auto-labeling pipeline
(scripts/segmentation/tools/run_sam3_labeling.py -> rasterize_sam3_labels.py
-> promote_gaza_labels.py) -- targets the exact domain gap the earlier
research identified as completely missing from ORFD (rubble/urban-debris
Gaza terrain), not a replacement for ORFD.

Layout (flat, unlike ORFD's image_data/gt_image/_fillcolor convention --
this is a single promoted image+label set, no calib/dense_depth):
    <root>/images/<stem>.<ext>
    <root>/labels/<stem>.png   (fine-grained class index, or 255=ignore)

Label encoding: the fine-grained SAM3_FINEGRAINED_NAMES class index (see
src/perception/models/semantic/_class_catalogues.py) is collapsed to
ORFDDataset's 3 coarse classes at load time -- a plain per-pixel remap
(hard labels), NOT the soft/batched LUT-merge used for the fine-grained-head
training path (scripts/segmentation/training/_fine_to_coarse.py) -- this
dataset feeds the existing, unchanged 3-class SegFormer training pipeline
(Stage 4a: "SAM3 as a better 3-class labeler", no new architecture).

The index-keyed LUT below is derived from _class_catalogues.py's canonical
name-keyed SAM3_FINEGRAINED_NAMES/FINE_TO_COARSE/COARSE_CLASSES (the single
source of truth -- adding a class only means editing that one file), loaded
directly by file path rather than via `from perception.models...` to avoid
executing perception/models/__init__.py's heavy factory/YOLO import chain
for what should be a few small dicts (same rationale as
scripts/segmentation/_class_catalogues_loader.py, which can't be imported
here since src/ doesn't depend on scripts/).
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .orfd_torch import ORFDDataset, TRAIN_SIZE, _augment_train, _resize_val, _to_normalized_tensor

logger = logging.getLogger(__name__)

_IGNORE_INDEX = 255

_catalogues_path = Path(__file__).resolve().parents[1] / "models" / "semantic" / "_class_catalogues.py"
_spec = importlib.util.spec_from_file_location("_class_catalogues", _catalogues_path)
_class_catalogues = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_class_catalogues)

_COARSE_LUT = np.full(256, _IGNORE_INDEX, dtype=np.uint8)
for _fine_idx, _fine_name in enumerate(_class_catalogues.SAM3_FINEGRAINED_NAMES):
    _coarse_name = _class_catalogues.FINE_TO_COARSE[_fine_name]
    _COARSE_LUT[_fine_idx] = _class_catalogues.COARSE_CLASSES.index(_coarse_name)


def _remap_to_coarse(fine_label: np.ndarray) -> np.ndarray:
    """Fine-grained class index or 255=ignore -> ORFD's 3-class scheme."""
    return _COARSE_LUT[fine_label]


class GazaDomainDataset(Dataset):
    """Gaza-domain image set, labels collapsed to ORFD's 3 coarse classes.

    Args mirror ORFDDataset for drop-in use alongside it (e.g. via
    ConcatDataset for joint training) -- same CLASSES/IGNORE_INDEX and
    (img_t, lbl_t) return contract.
    """

    CLASSES = ORFDDataset.CLASSES
    IGNORE_INDEX = _IGNORE_INDEX

    def __init__(self, root: str | Path, augment: bool = True, input_size: int = TRAIN_SIZE,
                 stems: set[str] | None = None) -> None:
        """``stems``, if given, restricts the dataset to only those image
        stems -- used to serve clip-grouped train/val splits (see
        scripts/segmentation/tools/split_gaza_domain.py) from this same
        class without duplicating the pairing/weighting logic below."""
        super().__init__()
        root = Path(root)
        images_dir, labels_dir = root / "images", root / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Expected {root}/images/ and {root}/labels/ -- run promote_gaza_labels.py first")

        self._augment = augment
        self._size = input_size
        self._pairs: list[tuple[Path, Path]] = []
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if stems is not None and img_path.stem not in stems:
                continue
            label_path = labels_dir / f"{img_path.stem}.png"
            if label_path.is_file():
                self._pairs.append((img_path, label_path))
        if not self._pairs:
            raise RuntimeError(f"No image/label pairs found under {root} -- run promote_gaza_labels.py first")
        logger.info("GazaDomainDataset: %d image/label pairs from %s", len(self._pairs), root)

        # Rare-class (within this dataset) up-weighting. Based on the real
        # per-class image counts from the full 241-image labeling run (not a
        # guess, and corrected once already after the first run showed
        # vehicle/rubble were actually common, not rare): tent appears in
        # only 4/241 images (1.7%), animal in 36/241 (15%), tarp in 67/241
        # (28%) -- genuinely rare. rock turned out common (189/241 = 78%,
        # the same mistake vehicle/rubble made the first time) and vehicle/
        # rubble remain common, so all three are deliberately left at normal
        # weight. Computed once here (fine-grained values, pre-collapse)
        # since __getitem__ only ever sees the already-collapsed 3-class
        # label. Re-verify this set against real counts if the image set
        # changes significantly.
        _rare_names = {"animal", "tent", "tarp"} & set(_class_catalogues.SAM3_FINEGRAINED_NAMES)
        _RARE_FINE_INDICES = {_class_catalogues.SAM3_FINEGRAINED_NAMES.index(n) for n in _rare_names}
        self.sample_weights: list[float] = []
        for _, label_path in self._pairs:
            fine = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            has_rare = bool(_RARE_FINE_INDICES & set(np.unique(fine).tolist()))
            self.sample_weights.append(3.0 if has_rare else 1.0)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self._pairs[idx]

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise OSError(f"Cannot load image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        fine_label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if fine_label is None:
            raise OSError(f"Cannot load label: {label_path}")
        label = _remap_to_coarse(fine_label)

        if self._augment:
            rgb, label = _augment_train(rgb, label, self._size)
        else:
            rgb, label = _resize_val(rgb, label, self._size)

        img_t = _to_normalized_tensor(rgb)
        lbl_t = torch.from_numpy(label.astype(np.int64))
        return img_t, lbl_t

    @property
    def pairs(self) -> list[tuple[Path, Path]]:
        return list(self._pairs)
