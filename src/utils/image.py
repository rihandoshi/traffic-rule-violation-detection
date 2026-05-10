from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import BBox, clamp_bbox_xyxy, normalize_bbox_xyxy


def load_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return image


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def crop_xyxy(image: np.ndarray, box: BBox) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox_xyxy(normalize_bbox_xyxy(box), width=w, height=h)
    xi1, yi1, xi2, yi2 = map(int, (x1, y1, x2, y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return np.zeros((0, 0, 3), dtype=image.dtype)
    return image[yi1:yi2, xi1:xi2].copy()


def draw_boxes(
    image_bgr: np.ndarray,
    boxes: Sequence[BBox],
    labels: Optional[Iterable[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = image_bgr.copy()
    labels_list = list(labels) if labels is not None else []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, normalize_bbox_xyxy(box))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if i < len(labels_list):
            text = labels_list[i]
            ty = max(20, y1 - 8)
            cv2.putText(
                out,
                text,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return out
