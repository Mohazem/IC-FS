from __future__ import annotations

import io

from src.document_platform.models import DocumentInput, ExtractionResult


class TextExtractionService:
    def extract(self, document: DocumentInput) -> ExtractionResult:
        if document.manual_text.strip():
            return ExtractionResult(
                text=document.manual_text.strip(),
                source_type="manual_text",
                page_count=1,
            )

        file_name = document.file_name.lower()
        if file_name.endswith(".pdf"):
            return self._extract_pdf(document.file_bytes or b"")
        if file_name.endswith((".csv", ".xlsx", ".xls")):
            return self._extract_spreadsheet(document.file_name, document.file_bytes or b"")
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            return ExtractionResult(
                text="",
                source_type="image",
                page_count=1,
                metadata={"needs_ocr": True},
            )

        text = (document.file_bytes or b"").decode("utf-8", errors="ignore")
        return ExtractionResult(
            text=text.strip(),
            source_type="text_file",
            page_count=1,
        )

    def _extract_pdf(self, file_bytes: bytes) -> ExtractionResult:
        text_parts: list[str] = []
        page_count = 0
        metadata: dict[str, object] = {"strategy": []}

        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            metadata["strategy"].append("pdfplumber")
        except Exception as exc:
            metadata["pdfplumber_error"] = str(exc)

        if "".join(text_parts).strip():
            return ExtractionResult(
                text="\n".join(text_parts).strip(),
                source_type="pdf",
                page_count=page_count or 1,
                metadata=metadata,
            )

        try:
            import fitz  # type: ignore

            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                page_count = pdf.page_count
                text_parts = [page.get_text("text") for page in pdf]
            metadata["strategy"].append("pymupdf")
        except Exception as exc:
            metadata["pymupdf_error"] = str(exc)

        return ExtractionResult(
            text="\n".join(text_parts).strip(),
            source_type="pdf",
            page_count=page_count or 1,
            used_pdf_ocr_fallback=not bool("".join(text_parts).strip()),
            metadata=metadata,
        )

    def _extract_spreadsheet(self, file_name: str, file_bytes: bytes) -> ExtractionResult:
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            return ExtractionResult(
                text="",
                source_type="spreadsheet",
                page_count=1,
                metadata={"error": str(exc)},
            )

        try:
            if file_name.lower().endswith(".csv"):
                dataframe = pd.read_csv(io.BytesIO(file_bytes))
            else:
                dataframe = pd.read_excel(io.BytesIO(file_bytes))
            text = dataframe.astype(str).fillna("").to_csv(index=False)
            return ExtractionResult(
                text=text.strip(),
                source_type="spreadsheet",
                page_count=1,
                metadata={"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
            )
        except Exception as exc:
            return ExtractionResult(
                text="",
                source_type="spreadsheet",
                page_count=1,
                metadata={"error": str(exc)},
            )
