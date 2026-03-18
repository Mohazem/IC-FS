from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.document_platform.config import AppConfig


class StorageService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.config.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save(self, payload: dict) -> dict:
        run_id = payload["run_id"]
        json_path = self.get_json_path(run_id)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        with sqlite3.connect(self.config.sqlite_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO processing_runs (run_id, created_at, file_name, result_json) VALUES (?, ?, ?, ?)",
                (run_id, payload["created_at"], payload["file_name"], json.dumps(payload, ensure_ascii=False)),
            )
            connection.commit()

        return {
            "json_path": str(json_path),
            "sqlite_path": str(self.config.sqlite_path),
            "saved": True,
        }

    def get_json_path(self, run_id: str) -> Path:
        return Path(self.config.data_dir / "json" / f"{run_id}.json")
