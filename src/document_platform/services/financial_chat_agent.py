from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from src.document_platform.config import AppConfig
from src.document_platform.services.rag_chat import FinancialRAGService


class FinancialChatAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.rag = FinancialRAGService(config)

    def answer(self, question: str, result: dict[str, Any]) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"answer": "Pose une question sur l'etat financier pour lancer l'analyse.", "contexts": [], "mode": "empty"}

        section_answer = self._answer_from_section_listing(question, result)
        if section_answer:
            return {
                "answer": section_answer["answer"],
                "contexts": section_answer["contexts"],
                "mode": "agent_section_listing",
                "tools_used": ["find_section_items", "search_local_text"],
            }

        metric_answer = self._answer_from_metric_listing(question, result)
        if metric_answer:
            return {
                "answer": metric_answer["answer"],
                "contexts": metric_answer["contexts"],
                "mode": "agent_metric_listing",
                "tools_used": ["get_key_metrics", "get_statement_lines"],
            }

        fallback = self.rag.answer(question, result)
        fallback["mode"] = f"agent->{fallback.get('mode', 'rag')}"
        fallback["tools_used"] = ["fallback_rag"]
        return fallback

    def _answer_from_section_listing(self, question: str, result: dict[str, Any]) -> dict[str, Any] | None:
        if not self._is_listing_question(question):
            return None

        target = self._extract_section_target(question)
        if not target:
            return None

        structured_match = self._find_structured_section_items(target, result)
        if structured_match:
            label, items, contexts = structured_match
            rendered = "\n".join(f"- {item}" for item in items)
            return {
                "answer": f"{label} :\n{rendered}",
                "contexts": contexts,
            }

        text_match = self._find_text_section_items(target, result.get("text", ""))
        if text_match:
            label, items = text_match
            rendered = "\n".join(f"- {item}" for item in items)
            contexts = [{"text": f"{label} | {item}", "source": "document_text", "doc_type": "section_item"} for item in items[:5]]
            return {
                "answer": f"{label} :\n{rendered}",
                "contexts": contexts,
            }

        return None

    def _answer_from_metric_listing(self, question: str, result: dict[str, Any]) -> dict[str, Any] | None:
        normalized = self._normalize(question)
        if "liste" not in normalized and "quels sont" not in normalized and "quelles sont" not in normalized:
            return None

        financial_data = result.get("structured_extraction", {}).get("financial_data", {})
        key_metrics = financial_data.get("key_metrics", {})
        metric_contexts = []
        matches = []
        for metric_name, values in key_metrics.items():
            if not isinstance(values, dict):
                continue
            if metric_name in normalized:
                rendered = ", ".join(f"{year}: {value}" for year, value in values.items() if value is not None)
                if rendered:
                    label = metric_name.replace("_", " ").title()
                    matches.append(f"{label} -> {rendered}")
                    metric_contexts.append({"text": f"{label} | {rendered}", "source": "structured", "doc_type": "key_metric"})

        if not matches:
            return None

        return {
            "answer": "\n".join(f"- {item}" for item in matches),
            "contexts": metric_contexts,
        }

    def _find_structured_section_items(self, target: str, result: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]] | None:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        best: tuple[float, str, list[str], list[dict[str, Any]]] | None = None

        for statement_key, statement in statements.items():
            line_items = statement.get("line_items", [])
            for index, item in enumerate(line_items):
                label = str(item.get("label", "")).strip()
                if not label:
                    continue
                score = self._similarity(self._normalize(label), target)
                if score < 0.58:
                    continue

                collected: list[str] = []
                contexts: list[dict[str, Any]] = []
                for probe in line_items[index + 1 : index + 9]:
                    next_label = str(probe.get("label", "")).strip()
                    if not next_label:
                        continue
                    normalized_next = self._normalize(next_label)
                    if self._is_section_stop(normalized_next):
                        break
                    values = probe.get("values", {})
                    rendered_values = ", ".join(f"{year}: {value}" for year, value in values.items() if value is not None)
                    rendered_line = f"{next_label} | {rendered_values}" if rendered_values else next_label
                    collected.append(rendered_line)
                    contexts.append({"text": f"{statement.get('title', statement_key)} | {rendered_line}", "source": statement_key, "doc_type": "statement_line"})

                if collected and (best is None or score > best[0]):
                    best = (score, label, collected, contexts)

        if best:
            return best[1], best[2], best[3]
        return None

    def _find_text_section_items(self, target: str, text: str) -> tuple[str, list[str]] | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        best_index = -1
        best_label = ""
        best_score = 0.0

        for index, line in enumerate(lines):
            normalized_line = self._normalize(line)
            score = self._similarity(normalized_line, target)
            if score > best_score:
                best_score = score
                best_index = index
                best_label = line

        if best_score < 0.52 or best_index < 0:
            return None

        items: list[str] = []
        for line in lines[best_index + 1 : best_index + 10]:
            normalized_line = self._normalize(line)
            if self._is_section_stop(normalized_line):
                break
            if re.fullmatch(r"(19|20)\d{2}", line):
                continue
            items.append(line)

        if not items:
            return None

        return best_label, self._consolidate_text_section_items(items)

    def _extract_section_target(self, question: str) -> str:
        normalized = self._normalize(question)
        normalized = re.sub(r"^(liste|listes|quels sont|quelles sont|detaille|detaillez|montre|montrez)\s+", "", normalized).strip()
        normalized = re.sub(r"^(les|des|du|de la|de l|la|le)\s+", "", normalized).strip()
        return normalized

    def _is_listing_question(self, question: str) -> bool:
        normalized = self._normalize(question)
        markers = [
            "liste",
            "quels sont",
            "quelles sont",
            "detaille",
            "detaillez",
            "compose",
            "composent",
            "montre",
            "montrez",
        ]
        return any(marker in normalized for marker in markers)

    def _is_section_stop(self, normalized_line: str) -> bool:
        stop_tokens = [
            "total",
            "passif",
            "passifs",
            "depenses",
            "charges",
            "recette",
            "recettes",
            "revenus",
            "actif net",
            "surplus",
            "deficit",
            "resultat",
            "etat des",
            "bilan",
            "actif ",
            "actifs ",
        ]
        if any(normalized_line.startswith(token) for token in stop_tokens):
            return True
        return normalized_line.isupper()

    def _normalize(self, value: str) -> str:
        value = value.lower()
        value = value.replace("à", "a").replace("â", "a").replace("ä", "a")
        value = value.replace("é", "e").replace("è", "e").replace("ê", "e").replace("ë", "e")
        value = value.replace("î", "i").replace("ï", "i")
        value = value.replace("ô", "o").replace("ö", "o")
        value = value.replace("ù", "u").replace("û", "u").replace("ü", "u")
        value = value.replace("ç", "c")
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        token_overlap = len(set(a.split()).intersection(set(b.split()))) / max(len(set(b.split())), 1)
        fuzzy = SequenceMatcher(None, a, b).ratio()
        return max(fuzzy, token_overlap)

    def _consolidate_text_section_items(self, items: list[str]) -> list[str]:
        consolidated: list[str] = []
        index = 0

        while index < len(items):
            current = items[index].strip()
            next_item = items[index + 1].strip() if index + 1 < len(items) else ""

            if self._is_amount_only(current):
                if consolidated:
                    consolidated[-1] = f"{consolidated[-1]} {current}".strip()
                index += 1
                continue

            if next_item and self._is_amount_only(next_item):
                consolidated.append(f"{current} {next_item}".strip())
                index += 2
                continue

            consolidated.append(current)
            index += 1

        return [item for item in consolidated if item]

    def _is_amount_only(self, value: str) -> bool:
        cleaned = value.replace("$", "").replace(",", ".").strip()
        return bool(re.fullmatch(r"\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})?|\d{4,}(?:[.,]\d{2})?", cleaned))
