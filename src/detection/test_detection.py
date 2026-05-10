from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "test_images").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.association.assign import associate_riders_to_bikes  # noqa: E402

PERSON = 0
BICYCLE = 1
MOTORCYCLE = 3
TARGET_CLASSES = [PERSON, BICYCLE, MOTORCYCLE]

COLORS = {
    PERSON: (0, 255, 0),        # green
    BICYCLE: (0, 165, 255),     # orange
    MOTORCYCLE: (255, 0, 255),  # magenta
}


def load_model() -> YOLO:
    model_dir = PROJECT_ROOT / "models"
    weights = model_dir / "yolov8n.pt"
    if not weights.is_file():
        weights = PROJECT_ROOT / "src" / "detection" / "yolov8n.pt"
    if not weights.is_file():
        weights = Path("yolov8n.pt")
    print("Using weights:", weights.resolve())
    return YOLO(str(weights))


def draw_detection_boxes(
    image_bgr: np.ndarray,
    results,
    names: dict,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    out = image_bgr.copy()
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out

    for i in range(len(r.boxes)):
        conf = float(r.boxes.conf[i])
        if conf < conf_threshold:
            continue
        cls_id = int(r.boxes.cls[i])
        if cls_id not in COLORS:
            continue

        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i].tolist())
        label = names.get(cls_id, str(cls_id))
        caption = f"{label} {conf:.2f}"
        color = COLORS[cls_id]

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            caption,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def parse_boxes(results, conf_threshold: float = 0.25) -> Tuple[List[Tuple[Tuple[float, float, float, float], float]], List[Tuple[Tuple[float, float, float, float], float]]]:
    bike_boxes: List[Tuple[Tuple[float, float, float, float], float]] = []
    rider_boxes: List[Tuple[Tuple[float, float, float, float], float]] = []

    r = results[0]
    if r.boxes is None:
        return bike_boxes, rider_boxes

    for i in range(len(r.boxes)):
        conf = float(r.boxes.conf[i])
        if conf < conf_threshold:
            continue
        cls_id = int(r.boxes.cls[i])
        bbox = tuple(map(float, r.boxes.xyxy[i].tolist()))

        if cls_id in (BICYCLE, MOTORCYCLE):
            bike_boxes.append((bbox, conf))
        elif cls_id == PERSON:
            rider_boxes.append((bbox, conf))
    return bike_boxes, rider_boxes


def draw_association_overlay(image_bgr: np.ndarray, groups: list) -> np.ndarray:
    out = image_bgr.copy()
    for idx, group in enumerate(groups):
        bx1, by1, bx2, by2 = map(int, group["bike_bbox"])
        riders = group["riders"]
        bike_color = (255, 255, 0)  # cyan
        cv2.rectangle(out, (bx1, by1), (bx2, by2), bike_color, 2)
        cv2.putText(
            out,
            f"bike_{idx} riders={len(riders)}",
            (bx1, max(by1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            bike_color,
            2,
            cv2.LINE_AA,
        )

        for rider in riders:
            x1, y1, x2, y2 = map(int, rider["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                out,
                f"r{rider['rider_index']}",
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def main() -> None:
    model = load_model()

    image_name = "6.jpg"
    image_path = PROJECT_ROOT / "test_images" / image_name
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    conf = 0.25
    results = model.predict(source=image_bgr, classes=TARGET_CLASSES, conf=conf, verbose=False)

    det_vis = draw_detection_boxes(image_bgr, results, model.names, conf_threshold=conf)
    bike_boxes, rider_boxes = parse_boxes(results, conf_threshold=conf)

    h, w = image_bgr.shape[:2]
    groups = associate_riders_to_bikes(
        bike_boxes=bike_boxes,
        rider_boxes=rider_boxes,
        image_width=w,
        image_height=h,
        bike_expand_ratio=0.2,
        min_iou_for_candidate=0.01,
    )

    print(f"Detected bikes/two-wheelers: {len(bike_boxes)}")
    print(f"Detected riders/persons: {len(rider_boxes)}")
    print("Association summary:")
    for i, group in enumerate(groups):
        print(f"  bike_{i}: riders={len(group['riders'])}")

    assoc_vis = draw_association_overlay(det_vis, groups)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "association_debug.jpg"
    cv2.imwrite(str(out_path), assoc_vis)
    print(f"Saved debug visualization: {out_path}")


if __name__ == "__main__":
    main()