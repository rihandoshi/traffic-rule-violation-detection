from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class DepthEstimator:
    def __init__(self, device: str = "cpu", cache_dir: str = "./models"):
        if pipeline is None:
            # We disable
            self.pipe = None
            return
            
        import os
        os.environ["HF_HUB_CACHE"] = cache_dir
        dev_idx = 0 if device == "cuda" else -1
        try:
            self.pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=dev_idx,
                model_kwargs={"cache_dir": cache_dir}
            )
        except Exception as e:
            self.pipe = None

    def predict(self, image_bgr: np.ndarray) -> np.ndarray | None:
        # Depth model here

        if self.pipe is None:
            return None
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        
        try:
            result = self.pipe(pil_img)
            depth_map = np.array(result["depth"])
            return depth_map
        except Exception as e:
            return None
