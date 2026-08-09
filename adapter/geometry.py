"""Box geometry helpers: IoU, size, aspect ratio, centroid distance."""
from __future__ import annotations

import math

Box = tuple[float, float, float, float]


def area(box: Box) -> float:
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0

def overlap_ratio(a: Box, b: Box) -> float:
    """Intersection over the *smaller* box's area (more robust to duplicate
    boxes of slightly different scale than plain IoU)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    smaller = min(area(a), area(b))
    return inter / smaller if smaller > 0 else 0.0


def centroid(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def centroid_distance(a: Box, b: Box) -> float:
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return math.hypot(ax - bx, ay - by)


def width_height(box: Box) -> tuple[float, float]:
    return (box[2] - box[0], box[3] - box[1])


def aspect_ratio(box: Box) -> float:
    """>= 1.0, long side over short side."""
    w, h = width_height(box)
    w, h = max(w, 1e-6), max(h, 1e-6)
    return max(w, h) / min(w, h)


def diag(box: Box) -> float:
    w, h = width_height(box)
    return math.hypot(w, h)
