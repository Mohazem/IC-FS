from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import requests

from src.document_platform.config import AppConfig
from src.document_platform.services.indexing import IndexingService


class FinancialRAGService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.indexing = IndexingService(config)
        self.metric_labels = {
            "revenue": "Revenus",
            "expenses": "Charges",
            "net_income": "Resultat net",
            "assets": "Actifs",
            "liabilities": "Passifs",
            "equity": "Actif net",
        }
        self.metric_aliases = {
            "revenue": ["revenue", "revenu", "revenus", "recettes", "sales"],
            "expenses": ["expenses", "expense", "depense", "depenses", "charges", "couts"],
            "net_income": ["net income", "resultat net", "surplus", "deficit", "excedent"],
            "assets": ["assets", "actif", "actifs"],
            "liabilities": ["liabilities", "passif", "passifs", "dettes"],
            "equity": ["equity", "actif net", "capitaux propres", "situation financiere nette"],
        }

    def answer(self, question: str, result: dict[str, Any]) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {"answer": "Pose une question sur l'etat financier pour lancer la recherche.", "contexts": [], "mode": "empty"}

        contexts = self._retrieve_context(question, result)
        inferred_answer = self._answer_from_financial_reasoning(question, result)
        if inferred_answer:
            return {
                "answer": inferred_answer,
                "contexts": self._prioritize_reasoning_contexts(contexts)[:5],
                "mode": "financial_reasoning",
            }

        line_item_answer = self._answer_from_line_item(question, result)
        if line_item_answer:
            return {
                "answer": line_item_answer,
                "contexts": self._prioritize_line_item_contexts(contexts)[:5],
                "mode": "direct_line_item",
            }

        direct_answer, metric_name = self._answer_from_metrics(question, result)
        if direct_answer:
            return {
                "answer": direct_answer,
                "contexts": self._prioritize_metric_contexts(contexts, metric_name)[:5],
                "mode": "direct_metric",
            }

        if self.config.hf_token:
            llm_answer = self._answer_with_hf(question, contexts)
            if llm_answer:
                return {
                    "answer": llm_answer,
                    "contexts": contexts[:5],
                    "mode": "rag_huggingface",
                }

        return {
            "answer": self._answer_locally(question, contexts),
            "contexts": contexts[:5],
            "mode": "rag_local",
        }

    def _retrieve_context(self, question: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        local_contexts = self._local_contexts(question, result)
        indexed_contexts = self._indexed_contexts(question, result)
        merged = local_contexts + indexed_contexts
        merged.sort(key=lambda item: item.get("score", 0), reverse=True)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in merged:
            text = item.get("text", "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique.append(item)
            if len(unique) >= 8:
                break
        return unique

    def _local_contexts(self, question: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = self._build_candidate_documents(result)
        scored: list[dict[str, Any]] = []
        normalized_question = self._normalize(question)
        question_tokens = set(normalized_question.split())

        for candidate in candidates:
            text = candidate["text"]
            normalized_text = self._normalize(text)
            token_overlap = len(question_tokens.intersection(set(normalized_text.split())))
            fuzzy = SequenceMatcher(None, normalized_question, normalized_text[: min(len(normalized_text), 180)]).ratio()
            score = token_overlap * 0.35 + fuzzy
            if candidate.get("doc_type") == "statement_line" and any(year in text for year in re.findall(r"(?:19|20)\d{2}", question)):
                score += 0.2
            if score > 0.1:
                scored.append({**candidate, "score": round(score, 4)})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:6]

    def _indexed_contexts(self, question: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        indexing = result.get("indexing", {})
        if indexing.get("status") != "indexed":
            return []

        search_result = self.indexing.search(question, run_id=result["run_id"], limit=4)
        if search_result.get("status") != "ok":
            return []

        contexts = []
        for match in search_result.get("matches", []):
            text = str(match.get("text", "")).strip()
            if not text:
                continue
            contexts.append(
                {
                    "text": text,
                    "score": round(float(match.get("score", 0.0)), 4),
                    "doc_type": "indexed_text",
                    "source": "qdrant",
                }
            )
        return contexts

    def _build_candidate_documents(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        financial_data = result.get("structured_extraction", {}).get("financial_data", {})
        key_metrics = financial_data.get("key_metrics", {})
        for metric_name, values in key_metrics.items():
            if not isinstance(values, dict):
                continue
            rendered = ", ".join(f"{year}: {value}" for year, value in values.items() if value is not None)
            if rendered:
                candidates.append(
                    {
                        "text": f"Key metric {metric_name}: {rendered}",
                        "doc_type": "key_metric",
                        "source": "structured",
                    }
                )

        risk_indicators = financial_data.get("risk_indicators", {})
        if isinstance(risk_indicators, dict) and risk_indicators:
            rendered = ", ".join(f"{name}: {value}" for name, value in risk_indicators.items())
            candidates.append(
                {
                    "text": f"Risk indicators: {rendered}",
                    "doc_type": "risk_indicator",
                    "source": "structured",
                }
            )

        statements = financial_data.get("financial_statements", {}).get("statements", {})
        for statement_key, statement in statements.items():
            title = statement.get("title", statement_key)
            for item in statement.get("line_items", []):
                label = str(item.get("label", "")).strip()
                values = item.get("values", {})
                rendered = ", ".join(f"{year}: {value}" for year, value in values.items() if value is not None)
                if label and rendered:
                    candidates.append(
                        {
                            "text": f"{title} | {label} | {rendered}",
                            "doc_type": "statement_line",
                            "source": statement_key,
                        }
                    )

        for chunk in self.indexing.chunk_text(result.get("text", ""), chunk_size=500):
            candidates.append(
                {
                    "text": chunk,
                    "doc_type": "raw_text",
                    "source": "document_text",
                }
            )

        return candidates

    def _answer_from_financial_reasoning(self, question: str, result: dict[str, Any]) -> str | None:
        normalized_question = self._normalize(question)
        years = re.findall(r"(?:19|20)\d{2}", question)
        financial_data = result.get("structured_extraction", {}).get("financial_data", {})
        key_metrics = financial_data.get("key_metrics", {})

        profitability_markers = [
            "rentable",
            "rentabilite",
            "profit",
            "profitable",
            "benefice",
            "bénéfice",
            "deficit",
            "déficit",
            "perte",
            "surplus",
            "marge",
        ]
        if any(marker in normalized_question for marker in profitability_markers):
            return self._profitability_answer(result, key_metrics, years)

        solvency_markers = ["solvable", "solvabilite", "solvabilité", "endettee", "endettée", "dette", "passif", "liabilities", "structure financiere", "structure financière", "sain", "saine"]
        if any(marker in normalized_question for marker in solvency_markers):
            return self._solvency_answer(result, key_metrics, years)

        return None

    def _profitability_answer(self, result: dict[str, Any], key_metrics: dict[str, Any], years: list[str]) -> str | None:
        target_years = years or self._metric_years(key_metrics)
        findings: list[str] = []
        public_context = self._is_public_or_nonprofit_statement(result)
        financing_before = self._statement_values(result, ["cout de fonctionnement net avant le financement du gouvernement"])
        financing_after = self._statement_values(result, ["cout de fonctionnement net apres le financement du gouvernement", "cout de fonctionnement net après le financement du gouvernement"])
        government_funding = self._statement_values(result, ["encaisse nette fournie par le gouvernement", "financement du gouvernement"])

        for year in target_years:
            revenue = self._amount_for_year(key_metrics, "revenue", year)
            expenses = self._amount_for_year(key_metrics, "expenses", year)
            net_income = self._amount_for_year(key_metrics, "net_income", year)

            if net_income is None and revenue is not None and expenses is not None:
                net_income = revenue - expenses

            if public_context:
                before_funding = self._parse_amount(financing_before.get(year))
                after_funding = self._parse_amount(financing_after.get(year))
                funding = self._parse_amount(government_funding.get(year))
                if revenue is not None and expenses is not None:
                    message = f"En {year}, ce document ressemble davantage a un organisme public ou non lucratif qu'a une societe commerciale."
                    message += f" Les revenus nets sont de {self._format_amount(revenue)} pour des charges de {self._format_amount(expenses)}"
                    if before_funding is not None:
                        message += f", soit un cout net de fonctionnement de {self._format_amount(before_funding)} avant financement gouvernemental"
                    if funding is not None:
                        message += f". Le financement du gouvernement represente {self._format_amount(funding)}"
                    if after_funding is not None:
                        message += f" et le cout net apres financement est de {self._format_amount(after_funding)}"
                    findings.append(message + ".")
                    continue

            if net_income is None:
                continue

            if net_income > 0:
                message = f"En {year}, la societe semble rentable avec un resultat net positif de {self._format_amount(net_income)}"
                if revenue not in (None, 0):
                    margin = (net_income / revenue) * 100
                    message += f" et une marge approximative de {margin:.1f}%"
                findings.append(message + ".")
            elif net_income < 0:
                message = f"En {year}, la societe ne semble pas rentable avec un resultat net negatif de {self._format_amount(abs(net_income))}"
                if revenue not in (None, 0):
                    margin = (net_income / revenue) * 100
                    message += f" et une marge approximative de {margin:.1f}%"
                findings.append(message + ".")
            else:
                findings.append(f"En {year}, la societe est a l'equilibre avec un resultat net nul.")

        if findings:
            return " ".join(findings)
        return None

    def _solvency_answer(self, result: dict[str, Any], key_metrics: dict[str, Any], years: list[str]) -> str | None:
        target_years = years or self._metric_years(key_metrics)
        findings: list[str] = []

        for year in target_years:
            assets = self._amount_for_year(key_metrics, "assets", year)
            liabilities = self._amount_for_year(key_metrics, "liabilities", year)
            equity = self._amount_for_year(key_metrics, "equity", year)

            if assets is None or liabilities is None:
                continue

            if equity is None:
                equity = assets - liabilities

            if equity >= 0:
                findings.append(
                    f"En {year}, la structure financiere parait saine: actifs {self._format_amount(assets)}, passifs {self._format_amount(liabilities)}, actif net {self._format_amount(equity)}."
                )
            else:
                findings.append(
                    f"En {year}, la structure financiere semble fragile: les passifs {self._format_amount(liabilities)} depassent les actifs {self._format_amount(assets)}, soit un actif net negatif de {self._format_amount(abs(equity))}."
                )

        if findings:
            return " ".join(findings)
        return None

    def _answer_from_metrics(self, question: str, result: dict[str, Any]) -> tuple[str | None, str | None]:
        normalized_question = self._normalize(question)
        years = re.findall(r"(?:19|20)\d{2}", question)
        key_metrics = result.get("structured_extraction", {}).get("financial_data", {}).get("key_metrics", {})

        for metric_name, aliases in self.metric_aliases.items():
            if not any(alias in normalized_question for alias in aliases):
                continue

            values = key_metrics.get(metric_name, {})
            if not isinstance(values, dict) or not values:
                continue

            if years:
                answers = [f"{year}: {values.get(year)}" for year in years if values.get(year) is not None]
                if answers:
                    metric_label = self.metric_labels.get(metric_name, metric_name.replace("_", " ").title())
                    return f"{metric_label} -> " + ", ".join(answers), metric_name
            else:
                answers = [f"{year}: {value}" for year, value in values.items() if value is not None]
                if answers:
                    metric_label = self.metric_labels.get(metric_name, metric_name.replace("_", " ").title())
                    return f"{metric_label} -> " + ", ".join(answers), metric_name

        return None, None

    def _answer_from_line_item(self, question: str, result: dict[str, Any]) -> str | None:
        target_tokens = self._line_item_tokens(question)
        if not target_tokens:
            return None

        structured_hit = self._match_structured_line_item(target_tokens, result)
        if structured_hit:
            label, values = structured_hit
            if len(values) == 1:
                year, amount = values[0]
                return f"{label} -> {year}: {amount}"
            rendered = ", ".join(f"{year}: {amount}" for year, amount in values)
            return f"{label} -> {rendered}"

        text_hit = self._match_text_line_item(target_tokens, result.get("text", ""))
        if text_hit:
            label = text_hit["label"]
            primary = text_hit["primary_amount"]
            alternate = text_hit.get("alternate_amount")
            if alternate and alternate != primary:
                return (
                    f"Le montant le plus probable pour `{label}` est {primary}. "
                    f"Une autre valeur proche ({alternate}) apparait dans la meme zone OCR, "
                    "mais elle semble provenir d'une ligne fusionnee."
                )
            return f"{label} -> {primary}"

        return None

    def _line_item_tokens(self, question: str) -> list[str]:
        normalized = self._normalize(question)
        stopwords = {
            "quel", "quelle", "quels", "quelles", "est", "le", "la", "les", "de", "du", "des", "payer",
            "paye", "paye?", "payé", "payee", "montant", "combien", "pour", "en", "au", "aux", "dans",
            "sur", "une", "un", "et", "d", "l", "il", "elle", "societe", "société",
        }
        tokens = [token for token in re.findall(r"[a-zA-Zà-ÿ]+", normalized) if len(token) >= 3 and token not in stopwords]
        return tokens[:4]

    def _match_structured_line_item(self, target_tokens: list[str], result: dict[str, Any]) -> tuple[str, list[tuple[str, str]]] | None:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        best: tuple[float, str, list[tuple[str, str]]] | None = None

        for statement in statements.values():
            for item in statement.get("line_items", []):
                label = str(item.get("label", "")).strip()
                if not label:
                    continue
                normalized_label = self._normalize(label)
                overlap = sum(token in normalized_label for token in target_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(len(target_tokens), 1)
                values = [(str(year), str(value)) for year, value in item.get("values", {}).items() if value is not None]
                if not values:
                    continue
                if best is None or score > best[0]:
                    best = (score, label, values)

        if best and best[0] >= 0.5:
            return best[1], best[2]
        return None

    def _match_text_line_item(self, target_tokens: list[str], text: str) -> dict[str, str] | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        best: tuple[float, dict[str, str]] | None = None

        for index, line in enumerate(lines):
            normalized_line = self._normalize(line)
            overlap = sum(token in normalized_line for token in target_tokens)
            if overlap == 0:
                continue

            amounts_in_line = self._extract_amount_strings(line)
            previous_amounts = self._extract_amount_strings(lines[index - 1]) if index > 0 else []
            score = overlap / max(len(target_tokens), 1)

            if previous_amounts and not self._normalize(lines[index - 1]).strip(" ;:.-"):
                previous_amounts = []

            if previous_amounts and not self._contains_letters(lines[index - 1]):
                primary = previous_amounts[-1]
                alternate = amounts_in_line[0] if amounts_in_line else None
                candidate = {
                    "label": line.split(primary)[0].strip() if primary in line else line,
                    "primary_amount": primary,
                    "alternate_amount": alternate or primary,
                }
                score += 0.35
            elif amounts_in_line:
                candidate = {
                    "label": line,
                    "primary_amount": amounts_in_line[0],
                }
            else:
                continue

            if best is None or score > best[0]:
                best = (score, candidate)

        return best[1] if best and best[0] >= 0.5 else None

    def _prioritize_metric_contexts(self, contexts: list[dict[str, Any]], metric_name: str | None) -> list[dict[str, Any]]:
        if not metric_name:
            return contexts

        aliases = self.metric_aliases.get(metric_name, [])
        priority = []
        others = []
        for item in contexts:
            normalized_text = self._normalize(item.get("text", ""))
            if any(alias in normalized_text for alias in aliases):
                priority.append(item)
            else:
                others.append(item)
        return priority + others

    def _prioritize_line_item_contexts(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        priority = []
        others = []
        for item in contexts:
            if item.get("doc_type") in {"statement_line", "raw_text"}:
                priority.append(item)
            else:
                others.append(item)
        return priority + others

    def _prioritize_reasoning_contexts(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        priority = []
        others = []
        for item in contexts:
            if item.get("doc_type") in {"key_metric", "risk_indicator", "statement_line"}:
                priority.append(item)
            else:
                others.append(item)
        return priority + others

    def _answer_with_hf(self, question: str, contexts: list[dict[str, Any]]) -> str | None:
        context_text = "\n".join(f"- {item['text']}" for item in contexts[:6])
        if not context_text.strip():
            return None

        prompt = (
            "Answer the question using only the supplied context from a financial statement. "
            "If the value is missing, say that it was not found. Quote years and line item labels when possible.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_text}"
        )

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
                        {"role": "system", "content": "Answer in French using only the provided context."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 350,
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"]).strip()
        except Exception:
            return None

    def _answer_locally(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "Je n'ai pas trouve de passage pertinent dans l'etat financier traite."

        best = contexts[0]["text"]
        if len(contexts) == 1:
            return f"Passage le plus pertinent:\n{best}"

        supporting = "\n".join(f"- {item['text']}" for item in contexts[:3])
        return f"Je n'ai pas de reponse calculee certaine pour cette question, mais voici les passages les plus pertinents:\n{supporting}"

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.lower()).strip()

    def _metric_years(self, key_metrics: dict[str, Any]) -> list[str]:
        years: list[str] = []
        for values in key_metrics.values():
            if isinstance(values, dict):
                for year in values:
                    if year not in years:
                        years.append(year)
        return years

    def _amount_for_year(self, key_metrics: dict[str, Any], metric_name: str, year: str) -> float | None:
        values = key_metrics.get(metric_name, {})
        if not isinstance(values, dict):
            return None
        return self._parse_amount(values.get(year))

    def _parse_amount(self, raw_value: Any) -> float | None:
        if raw_value is None:
            return None
        text = str(raw_value).strip()
        if not text:
            return None

        negative = text.startswith("(") and text.endswith(")")
        cleaned = (
            text.replace("(", "")
            .replace(")", "")
            .replace("$", "")
            .replace("CAD", "")
            .replace("USD", "")
            .replace("EUR", "")
            .replace("MAD", "")
            .replace("\u00a0", " ")
            .strip()
        )
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = cleaned.replace(",", ".")

        try:
            value = float(cleaned)
        except ValueError:
            return None
        return -value if negative else value

    def _format_amount(self, value: float) -> str:
        return f"{int(round(value)):,}".replace(",", " ")

    def _extract_amount_strings(self, text: str) -> list[str]:
        pattern = r"\(?-?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})?\)?|\(?-?\d{4,}(?:[.,]\d{2})?\)?"
        return [match.strip() for match in re.findall(pattern, text) if any(char.isdigit() for char in match)]

    def _contains_letters(self, text: str) -> bool:
        return bool(re.search(r"[A-Za-zÀ-ÿ]", text))

    def _is_public_or_nonprofit_statement(self, result: dict[str, Any]) -> bool:
        text = self._normalize(result.get("text", ""))
        markers = [
            "gouvernement",
            "government",
            "ministere",
            "ministère",
            "fondation",
            "foundation",
            "non audite",
            "non audité",
            "revenus gagnes pour le compte du gouvernement",
        ]
        return any(marker in text for marker in markers)

    def _statement_values(self, result: dict[str, Any], aliases: list[str]) -> dict[str, str]:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        income_statement = statements.get("income_statement", {})
        best_values: dict[str, str] = {}
        best_score = 0.0

        for item in income_statement.get("line_items", []):
            label = self._normalize(str(item.get("label", "")))
            if not label:
                continue
            score = max((SequenceMatcher(None, label, alias).ratio() for alias in aliases), default=0.0)
            if score > best_score:
                best_score = score
                best_values = {str(year): str(value) for year, value in item.get("values", {}).items() if value is not None}

        return best_values if best_score >= 0.45 else {}
