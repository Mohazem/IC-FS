from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    data_dir: Path
    sqlite_path: Path
    hf_token: str | None
    hf_base_url: str
    hf_model: str
    hf_embed_model: str
    qdrant_url: str
    qdrant_collection: str
    tesseract_cmd: str | None
    ocr_engine: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = Path(os.getenv("APP_DATA_DIR", "data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "json").mkdir(parents=True, exist_ok=True)

        return cls(
            data_dir=data_dir,
            sqlite_path=data_dir / "results.db",
            hf_token=os.getenv("HF_TOKEN"),
            hf_base_url=os.getenv("HF_BASE_URL", "https://router.huggingface.co"),
            hf_model=os.getenv("HF_MODEL", "katanemo/Arch-Router-1.5B:hf-inference"),
            hf_embed_model=os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "documents"),
            tesseract_cmd=os.getenv("TESSERACT_CMD"),
            ocr_engine=os.getenv("OCR_ENGINE", "auto"),
        )
