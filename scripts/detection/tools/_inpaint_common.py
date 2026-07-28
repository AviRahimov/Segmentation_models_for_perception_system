"""Shared sampling/masking/preview logic for inpainting-based hard-negative
generation, used by generate_inpainted_negatives_iopaint.py (ZITS).

Kept as its own module (rather than folded into that one script) so any
future backend can reuse the exact same sampling/masking -- an earlier LaMa
backend (generate_inpainted_negatives.py) was compared against ZITS this way
before being retired once ZITS won on quality; a new backend added later
would plug in the same way.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = _ROOT / "datasets/detection/Detection_Dataset/train"
CLASS_NAMES = ["Military Vehicle", "person"]

# build_mask()'s size-adaptive dilation: margin = clip(diag * DILATE_FRAC, DILATE_MIN_PX, DILATE_MAX_PX).
# A fixed-pixel margin (the original approach) is a tiny, near-invisible fraction
# of a large vehicle's own size but a huge fraction of a small distant person's --
# scaling to each instance's own bounding-box diagonal keeps the margin
# proportionally sensible at both ends.
DILATE_FRAC = 0.035
DILATE_MIN_PX = 6
DILATE_MAX_PX = 40
CLOSE_PX = 15  # morphological-close radius applied to the unioned mask
PROTECT_MARGIN_PX = 7  # safety dilation on protected (kept-class) regions before subtracting
# If subtracting the protect region would wipe out more than this fraction of
# a single removed instance's own mask, skip protection for that instance
# entirely rather than let it collapse to (near-)nothing.
PROTECT_MIN_SURVIVING_FRACTION = 0.3
# filter_surviving_kept_lines(): a kept object whose own raw polygon area ends
# up covered by more than this fraction of the final removal mask is treated
# as having been visually erased (protection was abandoned for it -- see
# PROTECT_MIN_SURVIVING_FRACTION above) and its GT line is dropped rather than
# kept -- otherwise the promoted label would claim an object is present at a
# location the inpainting model actually painted over. Empirically, fully
# protected kept objects land near 0% overlap and abandoned ones land near
# 100%, so 50% cleanly separates the two cases (confirmed by auditing all
# vehicle_only/person_only candidates from a real review batch: 10/131 had a
# kept-object overlap this high, none had a borderline value near 50%).
KEPT_SURVIVAL_OVERLAP_THRESHOLD = 0.5


def sample_images(src_dir: Path, n_images: int, seed: int) -> list[Path]:
    img_dir, lbl_dir = src_dir / "images", src_dir / "labels"
    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    candidates = [p for p in all_images if (lbl_dir / (p.stem + ".txt")).exists()
                 and (lbl_dir / (p.stem + ".txt")).stat().st_size > 0]
    random.seed(seed)
    return random.sample(candidates, min(n_images, len(candidates)))


def parse_label_lines(label_path: Path, w: int, h: int) -> list[tuple[int, str, np.ndarray]]:
    """Returns (class_id, raw_line, pixel-space polygon Nx2) per label line."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        else:
            xs = [c * w for c in coords[0::2]]
            ys = [c * h for c in coords[1::2]]
            poly = np.array(list(zip(xs, ys)), dtype=np.int32)
        out.append((cid, line, poly))
    return out


def build_mask(polys: list[np.ndarray], h: int, w: int,
              protect_polys: list[np.ndarray] | None = None) -> np.ndarray:
    """Per-instance convex hull + size-adaptive elliptical dilation, then a
    morphological close over the union, then subtract any protected regions.

    Why the hull: an object's polygon is traced from its VISIBLE silhouette,
    so when it's partially occluded (e.g. a soldier sitting on/in front of a
    vehicle) the polygon has a concave notch carved out of it right at the
    occlusion seam. That notch becomes a jagged, hard-to-blend boundary for
    the inpainting model. The hull smooths over it -- confirmed by direct
    review of the pilot batch that this notching, not just model quality, was
    a real source of bad vehicle removals.

    Why per-instance (not one hull over every object unioned together):
    hulling the whole merged mask at once would bridge the empty gap between
    two separate, unrelated objects (e.g. two Humvees several meters apart)
    into one giant masked blob. Hulling each polygon on its own avoids that.

    Why the closing step: fills any small residual gaps left where two
    instances' dilated masks nearly but don't quite touch.

    Why protect_polys: closing a vehicle's occlusion notch necessarily also
    covers whatever WAS occluding it -- typically a person sitting on/in front
    of it. For a variant like "vehicle_only_removed" that keeps that person's
    GT label in the output, silently erasing their pixels too would corrupt
    the label (it would point at inpainted background instead of a person) --
    confirmed empirically: without this subtraction step, a real LaMa run
    smudged out two soldiers' heads/torsos standing at a tank's hull. Any
    polygon passed here is carved back out of each removed instance's own
    mask (with a small safety dilation of its own, so we don't leave a thin
    ring of "remove" mask hugging the kept object's exact silhouette).

    Why the protect region itself is built from the RAW polygon, NOT its own
    hull: hulling the protected (kept) object closes ITS OWN occlusion notch
    too -- but that notch is, by definition, exactly the region occupied by
    the OTHER object (the one currently being removed). Protecting the kept
    object's hull therefore reaches directly into the area we're trying to
    clear, cutting real "remove this" pixels that just happen to fall inside
    the kept object's 2D bounding hull. Confirmed empirically on a real image
    (person standing against a large tank, person_only_removed): protecting
    via the vehicle's hull left only 64% of the person's own mask surviving
    (a visibly incomplete, partial-body mask); switching to the vehicle's raw
    polygon for protection raised that to 81%, visually covering the whole
    body. The raw polygon doesn't have this problem since it stops exactly
    at the true occlusion boundary rather than bridging across it.

    Why per-instance with a survival-fraction fallback (not a flat subtraction
    over the whole unioned mask): a person sitting ON/IN a vehicle (common in
    this dataset -- e.g. two soldiers riding on a Humvee's truck bed) has a
    removal region that sits almost entirely INSIDE the vehicle's own visible
    silhouette, since removing them exposes more vehicle, not background.
    Protecting the vehicle there would subtract nearly the whole person mask,
    producing an empty mask and silently no-op'ing the removal -- confirmed
    empirically (person_only_removed on such an image produced a 0-pixel
    mask). Checking each instance's own surviving fraction after subtraction
    and falling back to its unprotected mask when protection would gut it
    keeps the notch-fix intact for the common case (a person merely occluding
    part of a vehicle from the front) without silently cancelling removal in
    the sitting-on-it case.
    """
    protect_mask = None
    if protect_polys:
        protect_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in protect_polys:
            cv2.fillPoly(protect_mask, [poly], 255)  # raw polygon -- see docstring, NOT hulled
        protect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (PROTECT_MARGIN_PX * 2 + 1, PROTECT_MARGIN_PX * 2 + 1))
        protect_mask = cv2.dilate(protect_mask, protect_kernel)

    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        hull = cv2.convexHull(poly)
        _, _, bw, bh = cv2.boundingRect(hull)
        diag = float(np.hypot(bw, bh))
        margin_px = int(np.clip(diag * DILATE_FRAC, DILATE_MIN_PX, DILATE_MAX_PX))
        inst_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inst_mask, [hull], 255)
        if margin_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1, margin_px * 2 + 1))
            inst_mask = cv2.dilate(inst_mask, kernel, iterations=1)

        if protect_mask is not None:
            original_area = int((inst_mask > 0).sum())
            protected_inst = inst_mask.copy()
            protected_inst[protect_mask > 0] = 0
            surviving_fraction = (protected_inst > 0).sum() / original_area if original_area else 1.0
            if surviving_fraction >= PROTECT_MIN_SURVIVING_FRACTION:
                inst_mask = protected_inst
            # else: protection would gut this instance -- keep it unprotected.

        mask = cv2.bitwise_or(mask, inst_mask)
    if polys:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX * 2 + 1, CLOSE_PX * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def build_variants(
    lines: list[tuple[int, str, np.ndarray]],
) -> list[tuple[str, list[np.ndarray], list[tuple[str, np.ndarray]], list[np.ndarray]]]:
    """Up to 3 (variant_name, polys_to_remove, kept_lines, protect_polys)
    tuples, skipping any variant that would be a no-op or a duplicate of
    another.

    kept_lines is (raw_label_line, raw_polygon) for the OTHER class's
    instances this variant claims to keep -- returned as data, NOT a
    pre-joined string, because whether each one actually survives can only be
    known after build_mask() runs (see filter_surviving_kept_lines()).
    protect_polys is just the polygons from kept_lines, passed to
    build_mask()'s protect_polys= so a removal-mask notch-fix doesn't blindly
    erase an object this variant is supposed to keep intact.
    """
    vehicle_lines = [(c, r, p) for c, r, p in lines if CLASS_NAMES[c] == "Military Vehicle"]
    person_lines = [(c, r, p) for c, r, p in lines if CLASS_NAMES[c] == "person"]
    has_vehicle, has_person = bool(vehicle_lines), bool(person_lines)

    variants: list[tuple[str, list[np.ndarray], list[tuple[str, np.ndarray]], list[np.ndarray]]] = []
    if has_vehicle and has_person:
        variants.append(("both_removed", [p for _, _, p in lines], [], []))
        person_kept = [(raw, p) for _, raw, p in person_lines]
        variants.append(("vehicle_only_removed", [p for _, _, p in vehicle_lines],
                         person_kept, [p for _, p in person_kept]))
        vehicle_kept = [(raw, p) for _, raw, p in vehicle_lines]
        variants.append(("person_only_removed", [p for _, _, p in person_lines],
                         vehicle_kept, [p for _, p in vehicle_kept]))
    elif has_vehicle or has_person:
        # Only one class present -> "vehicle_only"/"person_only" would be
        # identical to "both_removed" (same mask, same empty label) -- skip
        # the redundant duplicate, generate once under both_removed.
        variants.append(("both_removed", [p for _, _, p in lines], [], []))
    return variants


def filter_surviving_kept_lines(
    mask: np.ndarray,
    kept_lines: list[tuple[str, np.ndarray]],
    overlap_threshold: float = KEPT_SURVIVAL_OVERLAP_THRESHOLD,
) -> str:
    """Join kept_lines' raw text, dropping any whose own polygon area is
    mostly covered by the final removal mask (see KEPT_SURVIVAL_OVERLAP_THRESHOLD).
    Must be called AFTER build_mask() -- this is exactly why kept_lines isn't
    collapsed into a string inside build_variants() itself."""
    if not kept_lines:
        return ""
    h, w = mask.shape[:2]
    surviving_raw = []
    for raw, poly in kept_lines:
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 255)
        area = int((poly_mask > 0).sum())
        if area == 0:
            continue
        overlap = ((poly_mask > 0) & (mask > 0)).sum() / area
        if overlap <= overlap_threshold:
            surviving_raw.append(raw)
    return "\n".join(surviving_raw)


def preview_panel(orig_bgr: np.ndarray, mask: np.ndarray, result_bgr: np.ndarray, title: str) -> np.ndarray:
    overlay = orig_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.5
    overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)

    def _label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    return np.hstack([_label(orig_bgr, "original"), _label(overlay, "mask"), _label(result_bgr, title)])
