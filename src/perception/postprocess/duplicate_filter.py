"""Suppress duplicate same-class detections (nested or heavily overlapping).

Plain NMS misses the nested case: a small box fully inside a large one can
have *low* IoU (inter / union is dominated by the large area), so both
survive. Physically a vehicle cannot be inside another vehicle, nor a person
inside another person, so same-class containment means one of the two boxes
is wrong. IoM (intersection over the *minimum* of the two areas) is 1.0 for
full containment regardless of the size ratio, which is why the duplicate
test here is ``IoU >= iou_threshold or IoM >= containment_threshold``.

Survivor choice per duplicate cluster: highest score wins, except when a
smaller (tighter) box scores within ``score_margin`` of the best — then the
smaller box is kept, because the spurious box is usually the loose outer one.

Different classes never suppress each other: a person inside a vehicle is a
legitimate scene.
"""
from __future__ import annotations

from ..core.types import Detection


def _box_area(b: tuple[int, int, int, int]) -> float:
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _iou_iom(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[float, float]:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
    if inter <= 0.0:
        return 0.0, 0.0
    area_a, area_b = _box_area(a), _box_area(b)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    iou = inter / union if union > 0 else 0.0
    iom = inter / smaller if smaller > 0 else 0.0
    return iou, iom


def filter_duplicates(
    detections: list[Detection],
    *,
    iou_threshold: float = 0.55,
    containment_threshold: float = 0.85,
    score_margin: float = 0.05,
) -> list[Detection]:
    """Return detections with same-class nested/overlapping duplicates removed.

    Order of the surviving detections follows the input order.
    """
    n = len(detections)
    if n < 2:
        return detections

    # Union-find over duplicate pairs (same class only).
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if detections[i].class_name != detections[j].class_name:
                continue
            iou, iom = _iou_iom(detections[i].bbox_xyxy, detections[j].bbox_xyxy)
            if iou >= iou_threshold or iom >= containment_threshold:
                union(i, j)

    # Pick one survivor per cluster.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    keep: set[int] = set()
    for members in clusters.values():
        if len(members) == 1:
            keep.add(members[0])
            continue
        best_score = max(detections[i].score for i in members)
        # Candidates within the margin of the best score → tightest box wins.
        candidates = [i for i in members
                      if detections[i].score >= best_score - score_margin]
        survivor = min(candidates, key=lambda i: _box_area(detections[i].bbox_xyxy))
        keep.add(survivor)

    return [d for i, d in enumerate(detections) if i in keep]
