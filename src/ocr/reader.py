import cv2
from paddleocr import PaddleOCR
from pathlib import Path
import traceback

from src.ocr.preprocessing import (
    get_grayscale,
    scale_image,
    remove_noise,
    thresholding,
    closing,
    deskew,
)


class OCRReader:
    def __init__(self):
        model_dir=Path("./models"),
        self.ocr = PaddleOCR(
                use_textline_orientation=False,
                lang='en',
                det_model_dir=str(model_dir / "PP-OCRv5_server_det"),
                rec_model_dir=str(model_dir / "PP-OCRv5_mobile_rec"),
        )

    def extract_text(self, plate_crop):
        try:
            #gray = get_grayscale(plate_crop)

            scaled = scale_image(plate_crop)

            denoised = remove_noise(scaled)

            #thresh = thresholding(denoised)

            #morphed = closing(thresh)

            processed = denoised
            # if(len(processed.shape) == 2):
            #     processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            results = self.ocr.predict(processed)

            if not results:
                return ""

            texts = []

            for res in results:
                if "rec_texts" in res:
                    texts.extend(res["rec_texts"])

            if not texts:
                return ""

            final_text = " ".join(texts)

            final_text = (
                final_text
                .replace(" ", "")
                .replace("-", "")
                .upper()
            )

            return final_text

        except Exception as e:
            print(f"OCR failed: {e}")
            traceback.print_exc()
            return ""