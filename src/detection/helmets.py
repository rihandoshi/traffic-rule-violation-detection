from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from ultralytics import YOLO


class HelmetDetector:
    # This is my attempt at a helmet detector.

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        imgsz: int = 224, 
    ):
        """
        Args:
            weights_path: path to the trained model.
            device: 
            conf_threshold: minimum confidence to accept helmet as a yes
            imgsz: Input image size for the YOLO model.
        """
        self.model = YOLO(str(weights_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

       
        self.is_classification = hasattr(self.model.model, 'names') and 'helmet' in str(self.model.model.names).lower()
        

    def predict(self, rider_crop_bgr):
        
        if rider_crop_bgr.size == 0:
            return False

        h, w = rider_crop_bgr.shape[:2]
        crop_size = max(h, w)
        
        # We lower confidence requirement for tiny riders koz blurry things can make it hard to detect helmets so false negatives can happen
        dynamic_conf = self.conf_threshold
        if crop_size < 80:
            dynamic_conf = max(0.10, self.conf_threshold - 0.10)

        results = self.model.predict(
            source=rider_crop_bgr,
            conf=dynamic_conf,
            imgsz=self.imgsz,
            device=self.device,
            half=(self.device != "cpu"),
            verbose=False,
        )

        result = results[0]

        if hasattr(result, "probs") and result.probs is not None:
            top_class_idx = int(result.probs.top1)
            class_name = self.model.names[top_class_idx].lower()
            
            if "no" in class_name or "without" in class_name:
                return False
            if "helmet" in class_name:
                return True
            
            return top_class_idx == 0 

        elif hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            if len(boxes) == 0:
                return False
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                class_name = self.model.names.get(cls_id, str(cls_id)).lower()
                
                if "no" in class_name or "head" in class_name or "without" in class_name:
                    return False
                
                if "helmet" in class_name:
                    return True
                if cls_id == 0: 
                    return True
                    
        return False
