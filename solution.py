from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.association.assign import associate_riders_to_bikes
from src.detection.bikes import BikeDetector
from src.detection.riders import RiderDetector
from src.detection.helmets import HelmetDetector
from src.depth.estimator import DepthEstimator
from src.utils.image import load_image_bgr, crop_xyxy, enhance_image_for_detection
import numpy as np


class TrafficViolationDetector:
    def __init__(self, model_dir: str = "./models"):
        """
        Initialize and load all models here only.
        model_dir: path to directory containing model weights.
        """
        self.model_dir = Path(model_dir)
        weights_path = self.model_dir / "yolov8s.pt"
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Expected weights at '{weights_path}'. Put yolov8s.pt in models/."
            )
            
        helmet_weights_path = self.model_dir / "helmet_yolov8n.pt"
        if not helmet_weights_path.is_file():
            print(f"WARNING: Helmet weights missing at '{helmet_weights_path}'. Using base model for fallback testing.")
            helmet_weights_path = weights_path

        # This is detection of bikes / riders, assosiation and helmet detection
        self.bike_detector = BikeDetector(weights_path=weights_path, conf_threshold=0.2, imgsz=960)
        self.rider_detector = RiderDetector(weights_path=weights_path, conf_threshold=0.2, imgsz=960)
        self.helmet_detector = HelmetDetector(weights_path=helmet_weights_path, conf_threshold=0.25)
        self.depth_estimator = DepthEstimator(cache_dir=str(self.model_dir))

    def predict(self, image_path: str):
        """
        Input:
        image_path: Path to input image
        Output:
        {
        "violations": [
            {
            "num_riders": int,
            "helmet_violations": int,
            "license_plate": "string"
            }
        ]
        }
        """
        try:
            image_bgr = load_image_bgr(image_path)
            if image_bgr is None or image_bgr.size == 0:
                print(f"Warning: Failed to read image {image_path}")
                return {"violations": []}

            h, w = image_bgr.shape[:2]

            # Dynamic Inference Size: Cap at 1280, floor at 640, round to nearest 32
            imgsz = max(640, min(1280, max(h, w)))
            imgsz = int((imgsz + 31) // 32 * 32)

            # Pre-process image for better detection in low light
            enhanced_bgr = enhance_image_for_detection(image_bgr)

            # Get depth map (returns None if not available)
            depth_map = self.depth_estimator.predict(enhanced_bgr)

            bike_preds = self.bike_detector.predict(enhanced_bgr, imgsz=imgsz)
            rider_preds = self.rider_detector.predict(enhanced_bgr, imgsz=imgsz)

            bike_boxes = [(bbox, score) for bbox, score, _cls_name in bike_preds]
            groups = associate_riders_to_bikes(
                bike_boxes=bike_boxes,
                rider_boxes=rider_preds,
                image_width=w,
                image_height=h,
                bike_expand_ratio=0.2,
                min_iou_for_candidate=0.01,
                depth_map=depth_map,
                max_depth_diff=80.0,
            )

            # Final rule violations check
            violations: List[Dict[str, int | str]] = []
            for g in groups:
                num_riders = len(g["riders"])
                helmet_violations = 0
                
                for rider in g["riders"]:
                    crop = crop_xyxy(image_bgr, rider["bbox"])  # Use original crop for true colors
                    has_helmet = self.helmet_detector.predict(crop)
                    if not has_helmet:
                        helmet_violations += 1

                # A violation occurs if more than 2 riders, OR if someone is missing a helmet
                if num_riders > 2 or helmet_violations > 0:
                    violations.append(
                        {
                            "num_riders": num_riders,
                            "helmet_violations": helmet_violations,
                            "license_plate": "",
                        }
                    )

            return {"violations": violations}
        
        except Exception as e:
            print(f"Crash prevented! Error processing image {image_path}: {e}")
            return {"violations": []}
