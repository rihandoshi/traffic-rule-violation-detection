from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

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


def bottom_center_inside_box(person_box_xyxy, bike_box_xyxy, inclusive = True):
    # Here we should check is rider ka bottom center lies inside the bike box
    return point_inside_box(bbox_bottom_center(person_box_xyxy), bike_box_xyxy, inclusive=inclusive)


def get_median_depth(depth_map, bbox):
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = normalize_bbox_xyxy(bbox)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float(np.median(crop))


def associate_riders_to_bikes(bike_boxes, rider_boxes, image_width, image_height, bike_expand_ratio = 0.2, min_iou_for_candidate = 0.01, depth_map = None, max_depth_diff = 80.0):
    # Strategy: We assign one rider to one bike. So we expand bike box by a little, then if rider ka bottom 
    # center is in the expanded box or if the iou > 0.01 in this case then we those that. 
    # If depth_map is there then we also ensure the absolute difference in depth is < max depth difference.
    # If more than one then highest iou or nearst bike center.
    normalized_bikes = [
        (normalize_bbox_xyxy(bbox), float(score)) for bbox, score in bike_boxes
    ]
    normalized_riders= [
        (normalize_bbox_xyxy(bbox), float(score)) for bbox, score in rider_boxes
    ]

    groups = [
        {
            "bike_bbox": [float(v) for v in bike_bbox],
            "bike_score": bike_score,
            "riders": [],
        }
        for bike_bbox, bike_score in normalized_bikes
    ]

    for rider_idx, (rider_bbox, rider_score) in enumerate(normalized_riders):
        candidates = [] 
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

            # 3D Depth Check
            if depth_map is not None:
                rider_depth = get_median_depth(depth_map, rider_bbox)
                bike_depth = get_median_depth(depth_map, bike_bbox)
                if abs(rider_depth - bike_depth) > max_depth_diff:
                    continue

            bike_center = bbox_center(bike_bbox)
            distance = euclidean_distance_xy(rider_center, bike_center)
            candidates.append((bike_idx, rider_iou, distance))

        if not candidates:
            continue

        # max iou first to assign , then we check min distance
        best_bike_idx = sorted(candidates, key=lambda x: (-x[1], x[2]))[0][0]
        groups[best_bike_idx]["riders"].append(
            {
                "bbox": [float(v) for v in rider_bbox],
                "score": rider_score,
                "rider_index": rider_idx,
            }
        )

    return groups
