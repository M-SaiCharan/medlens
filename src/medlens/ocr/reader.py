"""Image -> text for medicine strips, boxes and prescriptions.

Uses RapidOCR (PaddleOCR detection + recognition models on ONNX Runtime): no
torch dependency, fast on CPU, so it runs on a free hosting tier.
"""
import io

import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1600  # downscale huge phone photos; text stays readable, latency drops


class OCRReader:
    def __init__(self, min_confidence: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()
        self.min_confidence = min_confidence

    @staticmethod
    def _load(image: bytes) -> np.ndarray:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        return np.asarray(img)

    def read_lines(self, image: bytes) -> list[tuple[str, float]]:
        """Detected text lines, top-to-bottom, with recognition confidence."""
        result, _ = self._engine(self._load(image))
        if not result:
            return []
        # result rows: [box (4 points), text, confidence]; sort by box top, then left.
        result.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        return [(text, float(conf)) for _, text, conf in result if float(conf) >= self.min_confidence]

    def read(self, image: bytes) -> str:
        return "\n".join(text for text, _ in self.read_lines(image))
