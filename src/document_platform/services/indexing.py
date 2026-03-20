from __future__ import annotations

import hashlib
import re
from uuid import uuid4

import requests

from src.document_platform.config import AppConfig


class IndexingService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def index(self, text: str, metadata: dict, enabled: bool = True) -> dict:
        chunks = self.chunk_text(text)
        if not chunks:
            return {"enabled": enabled, "indexed_chunks": 0, "collection": self.config.qdrant_collection, "status": "skipped_empty"}

        embeddings: list[list[float]] = []
        fallback_used = False
        embedding_errors: list[str] = []
        for chunk in chunks:
            vector, used_fallback, error = self.embed_text(chunk)
            embeddings.append(vector)
            fallback_used = fallback_used or used_fallback
            if error:
                embedding_errors.append(error)
        result = {
            "enabled": enabled,
            "indexed_chunks": len(chunks),
            "collection": self.config.qdrant_collection,
            "status": "prepared",
            "embedding_fallback_used": fallback_used,
        }
        if embedding_errors:
            result["embedding_errors"] = embedding_errors[:3]

        if not enabled:
            result["status"] = "disabled"
            return result

        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http.models import Distance, PointStruct, VectorParams  # type: ignore

            client = QdrantClient(url=self.config.qdrant_url)
            vector_size = len(embeddings[0])
            collection_name = self._resolved_collection_name(vector_size)
            existing_collections = {collection.name for collection in client.get_collections().collections}
            if collection_name not in existing_collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            points = [
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={"text": chunk, **metadata},
                )
                for chunk, vector in zip(chunks, embeddings, strict=True)
            ]
            client.upsert(collection_name=collection_name, points=points)
            result["status"] = "indexed"
            result["collection"] = collection_name
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)

        return result

    def chunk_text(self, text: str, chunk_size: int = 800) -> list[str]:
        cleaned = " ".join(text.split())
        return [cleaned[i : i + chunk_size] for i in range(0, len(cleaned), chunk_size) if cleaned[i : i + chunk_size]]

    def embed_text(self, text: str) -> tuple[list[float], bool, str | None]:
        if not self.config.hf_token:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            return [byte / 255 for byte in digest], True, "huggingface_not_configured"

        try:
            response = requests.post(
                f"{self.config.hf_base_url}/hf-inference/models/{self.config.hf_embed_model}",
                headers={
                    "Authorization": f"Bearer {self.config.hf_token}",
                    "Content-Type": "application/json",
                },
                json={"inputs": text},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
                return [float(value) for value in payload], False, None
            if isinstance(payload, list) and payload and isinstance(payload[0], list):
                return [float(value) for value in payload[0]], False, None
        except Exception as exc:
            error = str(exc)
        else:
            error = "invalid_embedding_response"

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest], True, error

    def search(self, query: str, run_id: str, limit: int = 5) -> dict:
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "matches": []}

        vector, used_fallback, error = self.embed_text(query)
        collection_name = self._resolved_collection_name(len(vector))
        try:
            client = QdrantClient(url=self.config.qdrant_url)
            matches = client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="run_id",
                            match=MatchValue(value=run_id),
                        )
                    ]
                ),
            )
            return {
                "status": "ok",
                "collection": collection_name,
                "embedding_fallback_used": used_fallback,
                "embedding_error": error,
                "matches": [
                    {
                        "score": match.score,
                        "text": (match.payload or {}).get("text", ""),
                        "payload": dict(match.payload or {}),
                    }
                    for match in matches
                ],
            }
        except Exception as exc:
            return {
                "status": "failed",
                "collection": collection_name,
                "embedding_fallback_used": used_fallback,
                "embedding_error": error,
                "error": str(exc),
                "matches": [],
            }

    def _resolved_collection_name(self, vector_size: int) -> str:
        base_name = self.config.qdrant_collection.strip() or "documents"
        model_token = self._sanitize_collection_token(self.config.hf_embed_model or "local")
        return f"{base_name}_{model_token}_{vector_size}"

    def _sanitize_collection_token(self, value: str) -> str:
        compact = value.split("/")[-1].lower()
        compact = re.sub(r"[^a-z0-9]+", "_", compact).strip("_")
        return compact[:40] or "default"
