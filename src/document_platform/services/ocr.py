from __future__ import annotations

import io
from typing import Callable

from src.document_platform.config import AppConfig
from src.document_platform.models import OCRResult


class OCRService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def extract_from_pdf(self, file_bytes: bytes, engine: str | None = None) -> OCRResult:
        selected_engine = (engine or self.config.ocr_engine or "auto").lower()
        return self._run_with_engine(
            selected_engine,
            lambda preferred: self._extract_pdf_with_engine(file_bytes, preferred),
        )

    def extract_from_image(self, file_bytes: bytes, engine: str | None = None) -> OCRResult:
        selected_engine = (engine or self.config.ocr_engine or "auto").lower()
        return self._run_with_engine(
            selected_engine,
            lambda preferred: self._extract_image_with_engine(file_bytes, preferred),
        )

    def _run_with_engine(self, engine: str, extractor: Callable[[str], OCRResult]) -> OCRResult:
        if engine == "auto":
            for candidate in ["paddleocr", "tesseract"]:
                result = extractor(candidate)
                if result.used:
                    return result
            return result

        return extractor(engine)

    def _extract_pdf_with_engine(self, file_bytes: bytes, engine: str) -> OCRResult:
        if engine == "paddleocr":
            return self._extract_pdf_with_paddle(file_bytes)
        return self._extract_pdf_with_tesseract(file_bytes)

    def _extract_image_with_engine(self, file_bytes: bytes, engine: str) -> OCRResult:
        if engine == "paddleocr":
            return self._extract_image_with_paddle(file_bytes)
        return self._extract_image_with_tesseract(file_bytes)

    def _extract_pdf_with_tesseract(self, file_bytes: bytes) -> OCRResult:
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

    def _extract_image_with_tesseract(self, file_bytes: bytes) -> OCRResult:
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

    def _extract_pdf_with_paddle(self, file_bytes: bytes) -> OCRResult:
        try:
            import fitz  # type: ignore
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return OCRResult(used=False, text="", engine="paddleocr", metadata={"error": str(exc)})

        paddle = self._build_paddle_engine()
        if isinstance(paddle, OCRResult):
            return paddle

        pages: list[str] = []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page in pdf:
                    pixmap = page.get_pixmap(dpi=250)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                    text = self._paddle_image_to_text(np.array(image), paddle)
                    pages.append(text)
        except Exception as exc:
            return OCRResult(used=False, text="", engine="paddleocr", metadata={"error": str(exc)})

        text = "\n".join(part for part in pages if part).strip()
        return OCRResult(
            used=bool(text),
            text=text,
            engine="paddleocr",
            metadata={"pages_ocrd": len(pages)},
        )

    def _extract_image_with_paddle(self, file_bytes: bytes) -> OCRResult:
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:
            return OCRResult(used=False, text="", engine="paddleocr", metadata={"error": str(exc)})

        paddle = self._build_paddle_engine()
        if isinstance(paddle, OCRResult):
            return paddle

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            text = self._paddle_image_to_text(np.array(image), paddle)
        except Exception as exc:
            return OCRResult(used=False, text="", engine="paddleocr", metadata={"error": str(exc)})

        return OCRResult(
            used=bool(text.strip()),
            text=text.strip(),
            engine="paddleocr",
            metadata={"pages_ocrd": 1},
        )

    def _build_paddle_engine(self):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:
            return OCRResult(used=False, text="", engine="paddleocr", metadata={"error": str(exc)})

        try:
            return PaddleOCR(
                use_textline_orientation=True,
                lang="fr",
            )
        except Exception:
            try:
                return PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                )
            except Exception as exc:
                return OCRResult(used=False, text="", engine="paddleocr", metadata={"error": str(exc)})

    def _paddle_image_to_text(self, image_array, paddle) -> str:
        try:
            result = paddle.ocr(image_array, cls=True)
        except TypeError:
            result = paddle.ocr(image_array)
        lines: list[str] = []
        for page in result or []:
            for item in page or []:
                if isinstance(item, dict):
                    text = str(item.get("rec_text", "")).strip()
                    if text:
                        lines.append(text)
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    candidate = item[1]
                    if isinstance(candidate, (list, tuple)) and candidate:
                        text = str(candidate[0]).strip()
                    else:
                        text = str(candidate).strip()
                    if text:
                        lines.append(text)
        return "\n".join(lines).strip()
