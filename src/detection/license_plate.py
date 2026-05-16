from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

BBox = Tuple[float, float, float, float]


class LPDetector:
    # This class is used to detect license plates

    LP_CLASS_IDS = (0,)  # Example class IDs for license plates

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cpu",
        conf_threshold: float = 0.2,
        imgsz: int = 960, #1280
    ):
        self.model = YOLO(str(weights_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def predict(self, image_bgr: np.ndarray):
        results = self.model.predict(
            source=image_bgr,
            classes=list(self.LP_CLASS_IDS),
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: List[Tuple[BBox, float, str]] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            bbox = tuple(map(float, boxes.xyxy[i].tolist()))
            class_name = self.model.names.get(cls_id, str(cls_id))
            out.append((bbox, conf, class_name))
        return out
