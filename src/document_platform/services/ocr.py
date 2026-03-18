from __future__ import annotations

import io

from src.document_platform.config import AppConfig
from src.document_platform.models import OCRResult


class OCRService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def extract_from_pdf(self, file_bytes: bytes) -> OCRResult:
        try:
            import fitz  # type: ignore
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return OCRResult(used=False, text="", engine="tesseract", metadata={"error": str(exc)})

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        pages: list[str] = []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page in pdf:
                    pixmap = page.get_pixmap(dpi=220)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    pages.append(pytesseract.image_to_string(image, lang="eng+fra"))
        except Exception as exc:
            return OCRResult(used=False, text="", engine="tesseract", metadata={"error": str(exc)})

        text = "\n".join(pages).strip()
        return OCRResult(
            used=bool(text),
            text=text,
            engine="tesseract",
            metadata={"pages_ocrd": len(pages)},
        )

    def extract_from_image(self, file_bytes: bytes) -> OCRResult:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return OCRResult(used=False, text="", engine="tesseract", metadata={"error": str(exc)})

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        try:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang="eng+fra")
        except Exception as exc:
            return OCRResult(used=False, text="", engine="tesseract", metadata={"error": str(exc)})

        return OCRResult(
            used=bool(text.strip()),
            text=text.strip(),
            engine="tesseract",
            metadata={"pages_ocrd": 1},
        )
