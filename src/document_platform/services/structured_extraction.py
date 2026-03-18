from __future__ import annotations

import ast
import json
import re
from collections import Counter
from unicodedata import normalize

import requests

from src.document_platform.config import AppConfig


class StructuredExtractionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def extract(self, text: str) -> dict:
        snippet = self._prepare_snippet(text)
        if not snippet:
            return self._fallback("", reason="empty_text")

        prompt = (
            "Extract a JSON object with keys: title, document_type, summary, entities, dates, amounts, "
            "language, confidence, financial_data. Keep entities, dates, and amounts short. "
            "Return JSON only.\n\n"
            f"Document:\n{snippet}"
        )

        if not self.config.hf_token:
            return self._fallback(text, reason="huggingface_not_configured")

        try:
            response = requests.post(
                f"{self.config.hf_base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.hf_model,
                    "messages": [
                        {"role": "system", "content": "Return valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 900,
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            raw_response = payload["choices"][0]["message"]["content"]
            parsed = self._parse_llm_json(raw_response)
            parsed = self._normalize_llm_output(parsed)
            parsed["_provider"] = "huggingface"
            parsed["_model"] = self.config.hf_model
            return parsed
        except Exception as exc:
            return self._fallback(text, reason=self._classify_reason(exc))

    def fallback(self, text: str, reason: str = "llm_disabled") -> dict:
        return self._fallback(text, reason=reason)

    def _prepare_snippet(self, text: str) -> str:
        cleaned = " ".join(text.split())
        return cleaned[:2500].strip()

    def _fallback(self, text: str, reason: str) -> dict:
        cleaned_text = self._clean_text(text)
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        title = self._extract_title(lines)
        summary = self._build_summary(lines)
        dates = self._extract_dates(cleaned_text)
        amounts = self._extract_amounts(cleaned_text)
        entities = self._extract_entities(cleaned_text)
        language = self._detect_language(cleaned_text)
        document_type = self._detect_document_type(cleaned_text, title)
        confidence = self._estimate_confidence(title, summary, document_type, dates, amounts, entities)
        financial_data = self._extract_financial_data(cleaned_text, title, dates, amounts)

        return {
            "title": title,
            "document_type": document_type,
            "summary": summary,
            "entities": entities,
            "dates": dates,
            "amounts": amounts,
            "language": language,
            "confidence": confidence,
            "financial_data": financial_data,
            "_provider": "fallback",
            "_reason": reason,
        }

    def _clean_text(self, text: str) -> str:
        normalized = text.replace("\x0c", "\n")
        normalized = re.sub(r"[|_]{2,}", " ", normalized)
        normalized = re.sub(r"[^\S\r\n]+", " ", normalized)
        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
        normalized = re.sub(r"([A-Za-z])\s{2,}([A-Za-z])", r"\1 \2", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _extract_title(self, lines: list[str]) -> str:
        for line in lines[:8]:
            if len(line) < 5:
                continue
            if sum(char.isalpha() for char in line) < 4:
                continue
            return line[:120]
        return "Untitled document"

    def _build_summary(self, lines: list[str]) -> str:
        meaningful: list[str] = []
        for line in lines:
            if len(line) < 20:
                continue
            if line.lower() in meaningful:
                continue
            meaningful.append(line)
            if len(meaningful) == 3:
                break
        return " ".join(meaningful)[:600]

    def _extract_dates(self, text: str) -> list[str]:
        patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
        ]
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        return self._unique_preserve_order(matches, limit=8)

    def _extract_amounts(self, text: str) -> list[str]:
        pattern = r"(?:[$€£]?\s?\d{1,3}(?:[ .,\u00A0]\d{3})+(?:[.,]\d{2})?|\b\d{4,}(?:[.,]\d{2})?\b)(?:\s?(?:USD|EUR|CAD|MAD|GBP|dh|DHS))?"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        cleaned = [match.strip() for match in matches if any(char.isdigit() for char in match)]
        return self._unique_preserve_order(cleaned, limit=8)

    def _extract_entities(self, text: str) -> list[str]:
        candidates = re.findall(r"\b[A-Z][A-Za-zÀ-ÿ&'.-]+(?:\s+[A-Z][A-Za-zÀ-ÿ&'.-]+){0,4}\b", text)
        blacklist = {
            "Exemple",
            "Etats Financiers",
            "États Financiers",
            "Document Processing Platform",
        }
        filtered = [candidate.strip() for candidate in candidates if candidate.strip() not in blacklist and len(candidate.strip()) > 3]
        ranked = Counter(filtered)
        return [item for item, _count in ranked.most_common(8)]

    def _detect_language(self, text: str) -> str:
        lower = self._normalize_for_match(text)
        french_markers = ["etats financiers", "fondation", "exemple", "ontario", "annee", "bilan", "charges", "produits"]
        english_markers = ["financial statements", "foundation", "statement", "revenue", "expenses", "report"]

        french_score = sum(marker in lower for marker in french_markers)
        english_score = sum(marker in lower for marker in english_markers)

        if french_score > english_score:
            return "fr"
        if english_score > french_score:
            return "en"
        return "unknown"

    def _detect_document_type(self, text: str, title: str) -> str:
        combined = self._normalize_for_match(f"{title}\n{text}")
        if any(keyword in combined for keyword in ["etat financier", "financial statement", "statement of operations", "statement of financial position"]):
            return "financial_statement"
        if any(keyword in combined for keyword in ["balance sheet", "bilan", "assets", "liabilities", "revenus", "charges"]):
            return "financial_statement"
        if any(keyword in combined for keyword in ["invoice", "facture"]):
            return "invoice"
        if any(keyword in combined for keyword in ["report", "rapport annuel", "annual report"]):
            return "report"
        if any(keyword in combined for keyword in ["contract", "contrat", "agreement"]):
            return "contract"
        return "unknown"

    def _estimate_confidence(
        self,
        title: str,
        summary: str,
        document_type: str,
        dates: list[str],
        amounts: list[str],
        entities: list[str],
    ) -> float:
        score = 0.25
        if title != "Untitled document":
            score += 0.15
        if len(summary) > 40:
            score += 0.15
        if document_type != "unknown":
            score += 0.2
        if dates:
            score += 0.1
        if amounts:
            score += 0.1
        if entities:
            score += 0.1
        return min(round(score, 2), 0.85)

    def _extract_financial_data(self, text: str, title: str, dates: list[str], amounts: list[str]) -> dict:
        statement_body = self._select_statement_body(text)
        lower = self._normalize_for_match(statement_body)
        statement_sections = []
        section_map = {
            "balance_sheet": ["balance sheet", "bilan", "statement of financial position"],
            "income_statement": ["statement of operations", "income statement", "etat des resultats"],
            "cash_flow": ["cash flow", "flux de tresorerie"],
        }
        for section, keywords in section_map.items():
            if any(keyword in lower for keyword in keywords):
                statement_sections.append(section)

        currency = "unknown"
        if any(token in statement_body for token in ["CAD", "CA$", "Ontario", "Quebec", "Québec"]):
            currency = "CAD"
        elif "USD" in statement_body:
            currency = "USD"
        elif any(token in statement_body for token in ["EUR", "€"]):
            currency = "EUR"
        elif "MAD" in statement_body:
            currency = "MAD"
        elif "$" in statement_body:
            currency = "unknown_dollar"

        key_metrics = self._extract_key_metrics(statement_body)
        comparative = self._extract_comparative_metrics(statement_body, dates)
        if comparative["current_period"].get("assets"):
            key_metrics["assets"] = comparative["current_period"]["assets"]
        if comparative["current_period"].get("liabilities"):
            key_metrics["liabilities"] = comparative["current_period"]["liabilities"]
        if comparative["current_period"].get("equity"):
            key_metrics["equity"] = comparative["current_period"]["equity"]

        return {
            "organization_name": self._guess_organization_name(title, statement_body),
            "reporting_period": dates[0] if dates else "unknown",
            "currency": currency,
            "statement_sections_detected": statement_sections,
            "key_metrics": key_metrics,
            "comparative_metrics": comparative,
            "risk_indicators": self._build_risk_indicators(key_metrics),
            "tender_alignment": {
                "use_case": "financial_statement_extraction",
                "supported_inputs": ["pdf", "image", "excel", "csv", "text"],
                "export_targets": ["json", "sqlite", "qdrant"],
                "automation_ready": True,
            },
        }

    def _extract_key_metrics(self, text: str) -> dict:
        metrics = {
            "revenue": self._find_metric_amount(
                text,
                primary_patterns=["total des revenus", "total revenues", "recettes", "revenus"],
                fallback_labels=["revenue", "revenu", "revenus", "sales"],
            ),
            "expenses": self._find_metric_amount(
                text,
                primary_patterns=["total des depenses", "total expenses", "depenses", "charges"],
                fallback_labels=["expense", "expenses", "charge", "charges", "depense", "depenses"],
            ),
            "net_income": self._find_metric_amount(
                text,
                primary_patterns=["surplus", "excedent", "resultat net", "net income"],
                fallback_labels=["net income", "resultat net", "profit", "surplus", "excedent"],
            ),
            "assets": self._find_metric_amount(
                text,
                primary_patterns=["total de l actif", "total des actifs", "total assets"],
                fallback_labels=["assets", "actif", "actifs"],
            ),
            "liabilities": self._find_metric_amount(
                text,
                primary_patterns=["total du passif", "total liabilities", "total des passifs"],
                fallback_labels=["liabilities", "passif", "dettes"],
            ),
            "equity": self._find_metric_amount(
                text,
                primary_patterns=["actif net a la fin", "total de l actif net", "equity"],
                fallback_labels=["equity", "capitaux propres", "actif net"],
            ),
        }
        return metrics

    def _find_metric_amount(self, text: str, primary_patterns: list[str], fallback_labels: list[str]) -> str | None:
        lines = text.splitlines()
        normalized_primary = [self._normalize_for_match(label) for label in primary_patterns]
        normalized_labels = [self._normalize_for_match(label) for label in fallback_labels]
        candidates: list[str] = []

        for index, line in enumerate(lines):
            lower = self._normalize_for_match(line)
            if any(pattern in lower for pattern in normalized_primary):
                search_window = " ".join(lines[index : min(index + 4, len(lines))])
                candidates.extend(self._extract_amounts(search_window))

        primary_pick = self._pick_best_amount(candidates)
        if primary_pick:
            return primary_pick

        for index, line in enumerate(lines):
            lower = self._normalize_for_match(line)
            search_window = " ".join(lines[index : min(index + 3, len(lines))])
            if not any(label in lower for label in normalized_labels):
                if not any(label in self._normalize_for_match(search_window) for label in normalized_labels):
                    continue
            candidates.extend(self._extract_amounts(search_window))

        return self._pick_best_amount(candidates)

    def _pick_best_amount(self, amounts: list[str]) -> str | None:
        scored: list[tuple[float, str]] = []
        for amount in amounts:
            parsed = self._parse_amount(amount)
            if parsed is None:
                continue
            if 1900 <= parsed <= 2100 and float(parsed).is_integer():
                continue
            scored.append((abs(parsed), amount))
        if not scored:
            return amounts[0] if amounts else None
        scored.sort(reverse=True)
        return scored[0][1]

    def _guess_organization_name(self, title: str, text: str) -> str:
        entities = self._extract_entities(f"{title}\n{text}")
        return entities[0] if entities else title

    def _build_risk_indicators(self, key_metrics: dict) -> dict:
        revenue = self._parse_amount(key_metrics.get("revenue"))
        expenses = self._parse_amount(key_metrics.get("expenses"))
        assets = self._parse_amount(key_metrics.get("assets"))
        liabilities = self._parse_amount(key_metrics.get("liabilities"))

        operating_margin = None
        debt_ratio = None
        if revenue and expenses is not None and revenue != 0:
            operating_margin = round((revenue - expenses) / revenue, 4)
        if assets and liabilities is not None and assets != 0:
            debt_ratio = round(liabilities / assets, 4)

        return {
            "operating_margin": operating_margin,
            "debt_ratio": debt_ratio,
            "coverage_ready": any(value is not None for value in [operating_margin, debt_ratio]),
        }

    def _extract_comparative_metrics(self, text: str, dates: list[str]) -> dict:
        lines = text.splitlines()
        detected_years = self._extract_years(text, dates)
        if not detected_years:
            detected_years = ["current_period"]
        current_year = detected_years[0]
        prior_year = detected_years[1] if len(detected_years) > 1 else None
        years = [current_year] + ([prior_year] if prior_year else [])
        result = {
            "years": years,
            "current_period": {"assets": None, "liabilities": None, "equity": None},
            "prior_period": {"assets": None, "liabilities": None, "equity": None} if prior_year else {},
        }

        metric_map = {
            "assets": ["total de l actif", "total assets"],
            "liabilities": ["total du passif", "total liabilities"],
            "equity": ["actif net a la fin", "total de l actif net", "equity"],
        }

        for metric, labels in metric_map.items():
            for index, line in enumerate(lines):
                normalized = self._normalize_for_match(line)
                if not any(label in normalized for label in labels):
                    continue
                for probe in range(index, min(index + 12, len(lines))):
                    amounts = self._extract_amounts(lines[probe])
                    if len(amounts) >= 2:
                        result["current_period"][metric] = amounts[0]
                        if prior_year:
                            result["prior_period"][metric] = amounts[1]
                        break
                if result["current_period"][metric]:
                    break

        current_assets = self._parse_amount(result["current_period"]["assets"])
        current_liabilities = self._parse_amount(result["current_period"]["liabilities"])
        if result["current_period"]["equity"] is None and current_assets is not None and current_liabilities is not None:
            result["current_period"]["equity"] = self._format_amount(current_assets - current_liabilities)

        if prior_year:
            prior_assets = self._parse_amount(result["prior_period"]["assets"])
            prior_liabilities = self._parse_amount(result["prior_period"]["liabilities"])
            if result["prior_period"]["equity"] is None and prior_assets is not None and prior_liabilities is not None:
                result["prior_period"]["equity"] = self._format_amount(prior_assets - prior_liabilities)

        return result

    def _extract_years(self, text: str, dates: list[str]) -> list[str]:
        year_matches = re.findall(r"\b(?:19|20)\d{2}\b", " ".join(dates) + "\n" + text)
        plausible = []
        for year in year_matches:
            numeric_year = int(year)
            if 1990 <= numeric_year <= 2035:
                plausible.append(year)
        normalized = sorted(set(plausible), reverse=True)
        return normalized[:2]

    def _parse_amount(self, raw_value: str | None) -> float | None:
        if not raw_value:
            return None
        cleaned = raw_value.upper().replace("USD", "").replace("EUR", "").replace("CAD", "").replace("MAD", "")
        cleaned = cleaned.replace("$", "").replace("EUR", "").replace("GBP", "").replace("DHS", "").replace("DH", "")
        cleaned = cleaned.replace(" ", "")
        if cleaned.count(",") > 1 and "." not in cleaned:
            cleaned = cleaned.replace(",", "")
        elif cleaned.count(".") > 1 and "," not in cleaned:
            cleaned = cleaned.replace(".", "")
        elif "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _format_amount(self, value: float) -> str:
        return f"{int(round(value)):,}".replace(",", " ")

    def _normalize_for_match(self, value: str) -> str:
        normalized = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        return normalized.lower()

    def _guess_current_year(self, dates: list[str], text: str) -> str:
        joined = " ".join(dates)
        match = re.search(r"(20\d{2})", joined)
        if match:
            return match.group(1)
        match = re.search(r"(20\d{2})", text)
        if match:
            return match.group(1)
        return "current_period"

    def _select_statement_body(self, text: str) -> str:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            normalized = self._normalize_for_match(line)
            if "exemple 1" in normalized or "entreprise abc" in normalized:
                return "\n".join(lines[index:])
        for index, line in enumerate(lines):
            normalized = self._normalize_for_match(line)
            if index > 20 and ("actif" in normalized or "balance sheet" in normalized or "bilan" in normalized):
                return "\n".join(lines[index:])
        return text

    def _classify_reason(self, exc: Exception) -> str:
        message = str(exc).lower()
        if "timed out" in message:
            return "huggingface_timeout_fallback_used"
        if "connection" in message or "failed to establish" in message:
            return "huggingface_unreachable_fallback_used"
        return "huggingface_error_fallback_used"

    def _parse_llm_json(self, raw_response: str) -> dict:
        candidate = raw_response.strip()
        if "```" in candidate:
            candidate = candidate.replace("```json", "```").replace("```JSON", "```")
            parts = [part.strip() for part in candidate.split("```") if part.strip()]
            candidate = parts[0]

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            candidate = candidate[start : end + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("invalid_llm_json")

    def _normalize_llm_output(self, parsed: dict) -> dict:
        normalized = dict(parsed)

        for key in ["entities", "dates", "amounts"]:
            value = normalized.get(key, [])
            if isinstance(value, dict):
                normalized[key] = [str(item) for item in value.values()]
            elif isinstance(value, str):
                normalized[key] = [value]
            elif not isinstance(value, list):
                normalized[key] = []

        confidence = normalized.get("confidence")
        if isinstance(confidence, str):
            mapping = {
                "high": 0.85,
                "medium": 0.65,
                "low": 0.35,
            }
            cleaned_confidence = confidence.strip().lower()
            if cleaned_confidence in mapping:
                normalized["confidence"] = mapping[cleaned_confidence]
            else:
                try:
                    normalized["confidence"] = float(cleaned_confidence)
                except ValueError:
                    normalized["confidence"] = 0.5
        elif not isinstance(confidence, (int, float)):
            normalized["confidence"] = 0.5

        language = normalized.get("language")
        if isinstance(language, str):
            lowered = language.strip().lower()
            if lowered.startswith("fr"):
                normalized["language"] = "fr"
            elif lowered.startswith("en"):
                normalized["language"] = "en"

        financial_data = normalized.get("financial_data")
        if not isinstance(financial_data, dict):
            normalized["financial_data"] = {}

        return normalized

    def _unique_preserve_order(self, values: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = value.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(normalized)
            if len(result) >= limit:
                break
        return result
