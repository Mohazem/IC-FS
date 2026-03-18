from __future__ import annotations

from src.document_platform.config import AppConfig
from src.document_platform.models import DocumentInput, OCRResult, ProcessingResult
from src.document_platform.services.business_rules import BusinessRulesService
from src.document_platform.services.extraction import TextExtractionService
from src.document_platform.services.financial_parser import FinancialStatementParser
from src.document_platform.services.indexing import IndexingService
from src.document_platform.services.ocr import OCRService
from src.document_platform.services.storage import StorageService
from src.document_platform.services.structured_extraction import StructuredExtractionService


class DocumentPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.extraction = TextExtractionService()
        self.ocr = OCRService(config)
        self.structured_extraction = StructuredExtractionService(config)
        self.business_rules = BusinessRulesService()
        self.financial_parser = FinancialStatementParser()
        self.indexing = IndexingService(config)
        self.storage = StorageService(config)

    def run(
        self,
        file_name: str,
        file_bytes: bytes | None = None,
        manual_text: str = "",
        force_ocr: bool = False,
        extraction_mode: str = "local",
        store_in_qdrant: bool = True,
    ) -> dict:
        document = DocumentInput(file_name=file_name, file_bytes=file_bytes, manual_text=manual_text)
        extraction = self.extraction.extract(document)

        ocr_result = OCRResult(used=False, text="", engine="tesseract", metadata={})
        lower_name = file_name.lower()
        is_pdf = lower_name.endswith(".pdf")
        is_image = lower_name.endswith((".png", ".jpg", ".jpeg"))
        should_run_ocr = bool(file_bytes and (force_ocr or not extraction.text.strip()) and (is_pdf or is_image))
        if should_run_ocr and is_pdf:
            ocr_result = self.ocr.extract_from_pdf(file_bytes or b"")
        elif should_run_ocr and is_image:
            ocr_result = self.ocr.extract_from_image(file_bytes or b"")

        final_text = ocr_result.text if ocr_result.used else extraction.text
        if extraction_mode == "huggingface":
            structured = self.structured_extraction.extract(final_text)
        else:
            structured = self.structured_extraction.fallback(final_text, reason="local_python_mode")

        financial_statements = self.financial_parser.extract_financial_statements(
            file_name=file_name,
            file_bytes=file_bytes,
        )
        comparative_key_metrics = self.financial_parser.extract_period_metrics(
            file_name=file_name,
            file_bytes=file_bytes,
            default_year=self._guess_year(structured),
        )
        if financial_statements or comparative_key_metrics:
            structured.setdefault("financial_data", {})
        if financial_statements:
            structured["financial_data"]["financial_statements"] = financial_statements
        if comparative_key_metrics:
            structured["financial_data"]["key_metrics"] = comparative_key_metrics

        rules = self.business_rules.evaluate(structured, final_text)
        result = ProcessingResult.create(
            file_name=file_name,
            text=final_text,
            extraction=extraction,
            ocr=ocr_result,
            structured_extraction=structured,
            business_rules=rules,
            indexing={"enabled": store_in_qdrant, "indexed_chunks": 0, "collection": self.config.qdrant_collection, "status": "pending"},
            storage={"saved": False},
        )

        indexing_result = self.indexing.index(
            text=final_text,
            metadata={"file_name": file_name, "run_id": result.run_id, "document_type": structured.get("document_type", "unknown")},
            enabled=store_in_qdrant,
        )
        result.indexing = indexing_result

        result.storage = {
            "json_path": str(self.storage.get_json_path(result.run_id)),
            "sqlite_path": str(self.config.sqlite_path),
            "saved": False,
        }
        storage_result = self.storage.save(result.to_dict())
        result.storage = storage_result

        return result.to_dict()

    def _guess_year(self, structured: dict) -> str | None:
        reporting_period = structured.get("financial_data", {}).get("reporting_period", "")
        for token in str(reporting_period).split():
            if token.isdigit() and len(token) == 4:
                return token
        return None
