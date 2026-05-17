from paddleocr import PaddleOCR

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
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en'
        )

    def extract_text(self, plate_crop):
        try:
            gray = get_grayscale(plate_crop)

            scaled = scale_image(gray)

            denoised = remove_noise(scaled)

            thresh = thresholding(denoised)

            morphed = closing(thresh)

            processed = deskew(morphed)
            
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
            return ""