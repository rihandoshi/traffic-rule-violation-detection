from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.utils.geometry import (
    BBox,
    bbox_bottom_center,
    bbox_center,
    euclidean_distance_xy,
    expand_bbox_xyxy,
    iou_xyxy,
    normalize_bbox_xyxy,
    point_inside_box,
)


def bottom_center_inside_box(
    person_box_xyxy: BBox,
    bike_box_xyxy: BBox,
    inclusive: bool = True,
) -> bool:
    """Check whether rider bottom-center lies inside bike box."""
    return point_inside_box(bbox_bottom_center(person_box_xyxy), bike_box_xyxy, inclusive=inclusive)


def associate_riders_to_bikes(
    bike_boxes: List[Tuple[BBox, float]],
    rider_boxes: List[Tuple[BBox, float]],
    image_width: int,
    image_height: int,
    bike_expand_ratio: float = 0.2,
    min_iou_for_candidate: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Assign each rider to at most one bike.

    Strategy:
    1) Expand bike box
    2) Candidate if rider bottom-center is in expanded bike box OR IoU > threshold
    3) If multiple candidates, choose highest IoU, then nearest bike center
    """
    normalized_bikes: List[Tuple[BBox, float]] = [
        (normalize_bbox_xyxy(bbox), float(score)) for bbox, score in bike_boxes
    ]
    normalized_riders: List[Tuple[BBox, float]] = [
        (normalize_bbox_xyxy(bbox), float(score)) for bbox, score in rider_boxes
    ]

    groups: List[Dict[str, Any]] = [
        {
            "bike_bbox": [float(v) for v in bike_bbox],
            "bike_score": bike_score,
            "riders": [],
        }
        for bike_bbox, bike_score in normalized_bikes
    ]

    for rider_idx, (rider_bbox, rider_score) in enumerate(normalized_riders):
        candidates: List[Tuple[int, float, float]] = []  # (bike_idx, iou, distance)
        rider_center = bbox_center(rider_bbox)

        for bike_idx, (bike_bbox, _bike_score) in enumerate(normalized_bikes):
            expanded_bike = expand_bbox_xyxy(
                bike_bbox,
                width=image_width,
                height=image_height,
                margin_ratio=bike_expand_ratio,
            )
            rider_iou = iou_xyxy(rider_bbox, bike_bbox)
            in_contact_region = bottom_center_inside_box(rider_bbox, expanded_bike)

            if not in_contact_region and rider_iou < min_iou_for_candidate:
                continue

            bike_center = bbox_center(bike_bbox)
            distance = euclidean_distance_xy(rider_center, bike_center)
            candidates.append((bike_idx, rider_iou, distance))

        if not candidates:
            continue

        # max IoU first, then min distance
        best_bike_idx = sorted(candidates, key=lambda x: (-x[1], x[2]))[0][0]
        groups[best_bike_idx]["riders"].append(
            {
                "bbox": [float(v) for v in rider_bbox],
                "score": rider_score,
                "rider_index": rider_idx,
            }
        )

    return groups
