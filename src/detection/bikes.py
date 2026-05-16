from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

BBox = Tuple[float, float, float, float]


class BikeDetector:
    # This class is used to detect 2 wheelers

    TWO_WHEELER_CLASS_IDS = (1, 3)

    def __init__(
        self,
        weights_path,
        device= "cpu",
        conf_threshold= 0.2,
        imgsz= 960,
    ):
        self.model = YOLO(str(weights_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def predict(self, image_bgr, imgsz=None):
        target_imgsz = imgsz if imgsz is not None else self.imgsz
        results = self.model.predict(
            source=image_bgr,
            classes=list(self.TWO_WHEELER_CLASS_IDS),
            conf=self.conf_threshold,
            imgsz=target_imgsz,
            device=self.device,
            half=(self.device != "cpu"),
            verbose=False,
        )
        out = []
        boxes = results[0].boxes
        if boxes is None:
            return out
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            bbox = tuple(map(float, boxes.xyxy[i].tolist()))
            class_name = self.model.names.get(cls_id, str(cls_id))
            out.append((bbox, conf, class_name))
        return out
