from __future__ import annotations

from typing import Tuple

BBox = Tuple[float, float, float, float]


def clamp_bbox_xyxy(box, width, height):
    """Clamp an XYXY box to image boundaries."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return x1, y1, x2, y2


def normalize_bbox_xyxy(box):
    """Ensure x1 <= x2 and y1 <= y2."""
    x1, y1, x2, y2 = box
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return left, top, right, bottom


def bbox_center(box):
    x1, y1, x2, y2 = normalize_bbox_xyxy(box)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_bottom_center(box):
    x1, _y1, x2, y2 = normalize_bbox_xyxy(box)
    return (x1 + x2) / 2.0, y2


def point_inside_box(
    point_xy,
    box_xyxy,
    inclusive = True,
):
    cx, cy = point_xy
    x1, y1, x2, y2 = normalize_bbox_xyxy(box_xyxy)
    if inclusive:
        return x1 <= cx <= x2 and y1 <= cy <= y2
    return x1 < cx < x2 and y1 < cy < y2


def expand_bbox_xyxy(box, width, height, margin_ratio = 0.2):
    """Expand a box by margin_ratio on both axes, then clamp."""
    x1, y1, x2, y2 = normalize_bbox_xyxy(box)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * margin_ratio
    my = bh * margin_ratio
    expanded = (x1 - mx, y1 - my, x2 + mx, y2 + my)
    return clamp_bbox_xyxy(expanded, width=width, height=height)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = normalize_bbox_xyxy(a)
    bx1, by1, bx2, by2 = normalize_bbox_xyxy(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def euclidean_distance_xy(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5
