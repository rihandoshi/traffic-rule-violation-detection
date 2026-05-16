from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(r"D:/Projects/CV Project")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.association.assign import associate_riders_to_bikes
from src.detection.helmets import HelmetDetector
from src.depth.estimator import DepthEstimator
from src.utils.image import crop_xyxy, enhance_image_for_detection

PERSON = 0
BICYCLE = 1
MOTORCYCLE = 3
TARGET_CLASSES = [PERSON, BICYCLE, MOTORCYCLE]

COLORS = {
    PERSON: (0, 255, 0),  
    BICYCLE: (0, 165, 255),   
    MOTORCYCLE: (255, 0, 255),  
}


def load_model():
    weights = PROJECT_ROOT / "models" / "yolov8s.pt"
    if not weights.is_file():
        raise FileNotFoundError(
            f"Expected weights at '{weights}'. Keep a single copy in models/."
        )
    print("Using weights:", weights.resolve())
    return YOLO(str(weights))


def draw_detection_boxes(
    image_bgr: np.ndarray,
    results,
    names: dict,
    conf_threshold: float = 0.25,
):
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


def parse_boxes(results, conf_threshold: float = 0.25):
    bike_boxes = []
    rider_boxes = []

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


def draw_association_overlay(image_bgr: np.ndarray, groups: list, original_bgr: np.ndarray = None, helmet_detector=None):
    out = image_bgr.copy()
    for idx, group in enumerate(groups):
        bx1, by1, bx2, by2 = map(int, group["bike_bbox"])
        riders = group["riders"]
        bike_color = (255, 255, 0) # The color will be blue
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
            
            has_helmet_text = ""
            box_color = (0, 255, 255) # Yellow default
            
            if original_bgr is not None and helmet_detector is not None:
                crop = crop_xyxy(original_bgr, rider["bbox"])
                has_helmet = helmet_detector.predict(crop)
                if has_helmet:
                    has_helmet_text = " (Helmet)"
                    box_color = (0, 255, 0) # Green
                else:
                    has_helmet_text = " (No Helmet)"
                    box_color = (0, 0, 255) # Red

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                out,
                f"r{rider['rider_index']}{has_helmet_text}",
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA,
            )
    return out


def main():
    model = load_model()

    image_name = "2.jpg"
    image_path = PROJECT_ROOT / "test_images" / image_name
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = image_bgr.shape[:2]
    imgsz = max(640, min(1280, max(h, w)))
    imgsz = int((imgsz + 31) // 32 * 32)
    
    enhanced_bgr = enhance_image_for_detection(image_bgr)

    # Lower conf + higher imgsz can improve small bike recall. 
    # TODO fine tune this later pls dont forget
    conf = 0.20
    results = model.predict(
        source=enhanced_bgr,
        classes=TARGET_CLASSES,
        conf=conf,
        imgsz=imgsz,
        half=(model.device.type != "cpu"),
        verbose=False,
    )

    det_vis = draw_detection_boxes(image_bgr, results, model.names, conf_threshold=conf)
    bike_boxes, rider_boxes = parse_boxes(results, conf_threshold=conf)

    # Using depth anything model to better differentiate for things like walking pedestrians from a moving bike 
    depth_estimator = DepthEstimator(cache_dir=str(PROJECT_ROOT / "models"))
    depth_map = depth_estimator.predict(enhanced_bgr)

    h, w = image_bgr.shape[:2]
    groups = associate_riders_to_bikes(
        bike_boxes=bike_boxes,
        rider_boxes=rider_boxes,
        image_width=w,
        image_height=h,
        bike_expand_ratio=0.2,
        min_iou_for_candidate=0.01,
        depth_map=depth_map,
        max_depth_diff=80.0,
    )

    print(f"Detected bikes: {len(bike_boxes)}")
    print(f"Detected riders or ppl maybe: {len(rider_boxes)}")
    print(f"Run config: conf={conf}, imgsz={imgsz}")
    print("Association summary:")
    for i, group in enumerate(groups):
        print(f"  bike_{i}: riders={len(group['riders'])}")

    helmet_weights = PROJECT_ROOT / "models" / "helmet_yolov8s.pt"
    if not helmet_weights.is_file():
        helmet_weights = PROJECT_ROOT / "models" / "yolov8s.pt" 
    helmet_detector = HelmetDetector(weights_path=helmet_weights)

    assoc_vis = draw_association_overlay(det_vis, groups, image_bgr, helmet_detector)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = output_dir / "association_debug.jpg"
    cv2.imwrite(str(out_path), assoc_vis)
    print(f"Saved debug visualization: {out_path}")
    
    if depth_map is not None:
        depth_vis = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_out_path = output_dir / "depth_debug.jpg"
        cv2.imwrite(str(depth_out_path), depth_vis)
        print(f"Saved depth visualization: {depth_out_path}")


if __name__ == "__main__":
    main()