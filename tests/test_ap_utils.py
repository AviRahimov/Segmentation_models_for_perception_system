"""Unit tests for the collapsed-AP evaluator (scripts/detection/evaluation/_ap_utils.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "detection" / "evaluation"))

from _ap_utils import GT, Pred, ap_per_class, collapse_preds, iou_xyxy  # noqa: E402


def _box(x=0, y=0, s=100):
    return (x, y, x + s, y + s)


def test_iou_identical_is_one():
    assert iou_xyxy(_box(), _box()) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou_xyxy(_box(0, 0), _box(500, 500)) == 0.0


def test_perfect_predictions_ap_one():
    gts = [GT("a", "vehicle", _box()), GT("b", "vehicle", _box(50, 50))]
    preds = [Pred("a", "vehicle", 0.9, _box()),
             Pred("b", "vehicle", 0.8, _box(50, 50))]
    ap = ap_per_class(preds, gts, ["vehicle"])
    assert ap["vehicle"] == pytest.approx(1.0)


def test_no_predictions_ap_zero():
    gts = [GT("a", "vehicle", _box())]
    ap = ap_per_class([], gts, ["vehicle"])
    assert ap["vehicle"] == pytest.approx(0.0)


def test_no_gt_is_nan():
    preds = [Pred("a", "person", 0.9, _box())]
    ap = ap_per_class(preds, [], ["person"])
    assert ap["person"] != ap["person"]  # NaN


def test_false_positive_halves_precision_not_recall():
    # 1 GT, 2 preds: correct one at high score, FP at lower score.
    # PR curve: (r=1, p=1) then (r=1, p=0.5) → AP = 1.0
    gts = [GT("a", "vehicle", _box())]
    preds = [Pred("a", "vehicle", 0.9, _box()),
             Pred("a", "vehicle", 0.5, _box(400, 400))]
    ap = ap_per_class(preds, gts, ["vehicle"])
    assert ap["vehicle"] == pytest.approx(1.0)


def test_fp_ranked_above_tp_lowers_ap():
    # FP at 0.9, TP at 0.5: precision at recall 1.0 is 0.5 → AP = 0.5
    gts = [GT("a", "vehicle", _box())]
    preds = [Pred("a", "vehicle", 0.9, _box(400, 400)),
             Pred("a", "vehicle", 0.5, _box())]
    ap = ap_per_class(preds, gts, ["vehicle"])
    assert ap["vehicle"] == pytest.approx(0.5)


def test_one_gt_matched_once():
    # Two identical predictions on one GT: second is a duplicate FP.
    gts = [GT("a", "vehicle", _box())]
    preds = [Pred("a", "vehicle", 0.9, _box()),
             Pred("a", "vehicle", 0.8, _box())]
    ap = ap_per_class(preds, gts, ["vehicle"])
    assert ap["vehicle"] == pytest.approx(1.0)  # TP first, dup FP after full recall


def test_missed_gt_caps_recall():
    # 2 GTs, only 1 detected → recall caps at 0.5 → AP = 0.5
    gts = [GT("a", "vehicle", _box()), GT("a", "vehicle", _box(300, 300))]
    preds = [Pred("a", "vehicle", 0.9, _box())]
    ap = ap_per_class(preds, gts, ["vehicle"])
    assert ap["vehicle"] == pytest.approx(0.5)


def test_collapse_mapping_and_drop():
    preds = [
        Pred("a", "tank", 0.9, _box()),
        Pred("a", "soldier", 0.8, _box(200, 200)),
        Pred("a", "civilian_vehicle", 0.7, _box(400, 400)),  # dropped
    ]
    collapse = {"tank": "Military Vehicle", "soldier": "person"}
    out = collapse_preds(preds, collapse)
    assert [p.class_name for p in out] == ["Military Vehicle", "person"]


def test_collapsed_end_to_end_ap():
    # 6-class preds scored against 2-class GT via collapse.
    gts = [GT("a", "Military Vehicle", _box()),
           GT("a", "person", _box(200, 200))]
    preds = [Pred("a", "tank", 0.9, _box()),
             Pred("a", "soldier", 0.85, _box(200, 200))]
    collapse = {"tank": "Military Vehicle", "truck": "Military Vehicle",
                "armored_vehicle": "Military Vehicle",
                "soldier": "person", "civilian": "person"}
    out = ap_per_class(collapse_preds(preds, collapse), gts,
                       ["Military Vehicle", "person"])
    assert out["Military Vehicle"] == pytest.approx(1.0)
    assert out["person"] == pytest.approx(1.0)


def test_class_confusion_is_fp():
    # Predicting person on a vehicle GT: FP for person (no person GT → NaN),
    # and vehicle has no predictions → AP 0.
    gts = [GT("a", "Military Vehicle", _box())]
    preds = [Pred("a", "person", 0.9, _box())]
    out = ap_per_class(preds, gts, ["Military Vehicle", "person"])
    assert out["Military Vehicle"] == pytest.approx(0.0)
    assert out["person"] != out["person"]  # NaN — no person GT


def test_load_yolo_gts_polygon_labels(tmp_path):
    """Roboflow polygon labels (class + xy pairs) must become tight bboxes."""
    import numpy as np
    import cv2
    img_dir = tmp_path / "images"; img_dir.mkdir()
    lbl_dir = tmp_path / "labels"; lbl_dir.mkdir()
    cv2.imwrite(str(img_dir / "a.png"), np.zeros((100, 200, 3), dtype=np.uint8))
    # triangle polygon: (0.1,0.2) (0.5,0.2) (0.3,0.8) on a 200x100 image
    (lbl_dir / "a.txt").write_text("0 0.1 0.2 0.5 0.2 0.3 0.8\n")
    from _ap_utils import load_yolo_gts
    pairs = load_yolo_gts(img_dir, lbl_dir, ["vehicle"])
    (_, gts), = pairs
    assert gts[0].box == pytest.approx((20.0, 20.0, 100.0, 80.0))


# --------------------------------------------------------------------------- #
# Operating-point / sweep / size-bucket metrics                                #
# --------------------------------------------------------------------------- #

from _ap_utils import (  # noqa: E402
    false_positives,
    operating_point,
    size_bucketed_recall,
    threshold_sweep,
)


def test_operating_point_counts():
    # 2 GT vehicles; preds: 1 TP @0.9, 1 FP @0.6, 1 below-threshold TP @0.2 (ignored)
    gts = [GT("a", "vehicle", _box()), GT("b", "vehicle", _box())]
    preds = [Pred("a", "vehicle", 0.9, _box()),
             Pred("a", "vehicle", 0.6, _box(400, 400)),
             Pred("b", "vehicle", 0.2, _box())]
    op = operating_point(preds, gts, ["vehicle"], n_images=2, conf_thr=0.4)["vehicle"]
    assert (op["tp"], op["fp"], op["fn"]) == (1, 1, 1)
    assert op["precision"] == pytest.approx(0.5)
    assert op["recall"] == pytest.approx(0.5)
    assert op["fp_per_image"] == pytest.approx(0.5)  # 1 FP / 2 images


def test_operating_point_no_preds_above_threshold():
    gts = [GT("a", "vehicle", _box())]
    preds = [Pred("a", "vehicle", 0.1, _box())]
    op = operating_point(preds, gts, ["vehicle"], n_images=1, conf_thr=0.4)["vehicle"]
    assert op["recall"] == pytest.approx(0.0)
    assert op["precision"] != op["precision"]  # NaN — no predictions kept


def test_threshold_sweep_finds_obvious_best():
    # TPs at 0.8/0.7; FPs at 0.3/0.2 → any threshold in (0.3, 0.7] gives P=R=F1=1
    gts = [GT("a", "vehicle", _box()), GT("a", "vehicle", _box(300, 300))]
    preds = [Pred("a", "vehicle", 0.8, _box()),
             Pred("a", "vehicle", 0.7, _box(300, 300)),
             Pred("a", "vehicle", 0.3, _box(600, 0)),
             Pred("a", "vehicle", 0.2, _box(0, 600))]
    sweep = threshold_sweep(preds, gts, "vehicle", n_images=1)
    assert sweep["best"]["f1"] == pytest.approx(1.0)
    assert 0.3 < sweep["best"]["conf"] <= 0.7


def test_size_buckets_data_driven_boundaries():
    # 9 GTs with sizes 10..90 px → terciles ≈ 36 / 63; only large ones detected.
    gts, preds = [], []
    for i, s in enumerate([10, 20, 30, 40, 50, 60, 70, 80, 90]):
        b = (0, 0, s, s)
        gts.append(GT(f"img{i}", "person", b))
        if s >= 70:
            preds.append(Pred(f"img{i}", "person", 0.9, b))
    res = size_bucketed_recall(preds, gts, ["person"], conf_thr=0.4)["person"]
    b1, b2 = res["boundaries_px"]
    assert 30 <= b1 <= 40 and 60 <= b2 <= 70
    assert res["recall"]["small"] == pytest.approx(0.0)
    assert res["recall"]["large"] == pytest.approx(1.0)
    assert res["counts"]["small"] + res["counts"]["medium"] + res["counts"]["large"] == 9


def test_false_positives_extraction():
    gts = [GT("a", "vehicle", _box())]
    preds = [Pred("a", "vehicle", 0.9, _box()),           # TP
             Pred("a", "vehicle", 0.7, _box(400, 400)),   # FP
             Pred("a", "vehicle", 0.2, _box(600, 600))]   # below threshold
    fps = false_positives(preds, gts, ["vehicle"], conf_thr=0.4)
    assert len(fps) == 1 and fps[0].score == pytest.approx(0.7)
