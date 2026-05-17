import easyocr
import cv2

from src.ocr.preprocessing import (
    scale_image,
    remove_noise,
)


class OCRReader:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False, model_storage_directory="./models/easyocr")

    def extract_text(self, plate_crop):
        try:
            scaled = scale_image(plate_crop)

            denoised = remove_noise(scaled)

            results = self.reader.readtext(denoised)

            if not results:
                return ""

            texts = [r[1] for r in results]

            final_text = "".join(texts)

            final_text = (
                final_text
                .replace(" ", "")
                .replace("-", "")
                .upper()
            )

            return final_text

        except Exception as e:
            print(f"OCR failed: {e}")
            return ""