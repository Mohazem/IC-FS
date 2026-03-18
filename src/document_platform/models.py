from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DocumentInput:
    file_name: str
    file_bytes: bytes | None = None
    manual_text: str = ""


@dataclass(slots=True)
class ExtractionResult:
    text: str
    source_type: str
    page_count: int
    used_pdf_ocr_fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRResult:
    used: bool
    text: str
    engine: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingResult:
    run_id: str
    created_at: str
    file_name: str
    text: str
    extraction: dict[str, Any]
    ocr: dict[str, Any]
    structured_extraction: dict[str, Any]
    business_rules: dict[str, Any]
    indexing: dict[str, Any]
    storage: dict[str, Any]

    @classmethod
    def create(
        cls,
        file_name: str,
        text: str,
        extraction: Any,
        ocr: Any,
        structured_extraction: dict[str, Any],
        business_rules: dict[str, Any],
        indexing: dict[str, Any],
        storage: dict[str, Any],
    ) -> "ProcessingResult":
        return cls(
            run_id=str(uuid4()),
            created_at=utc_now_iso(),
            file_name=file_name,
            text=text,
            extraction=asdict(extraction),
            ocr=asdict(ocr),
            structured_extraction=structured_extraction,
            business_rules=business_rules,
            indexing=indexing,
            storage=storage,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
