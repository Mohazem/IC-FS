from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

import requests

from src.document_platform.config import AppConfig
from src.document_platform.services.rag_chat import FinancialRAGService


class FinancialChatAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.rag = FinancialRAGService(config)
        self.max_steps = 4
        self.tools = {
            "get_key_metrics": self._tool_get_key_metrics,
            "get_statement_titles": self._tool_get_statement_titles,
            "get_statement_lines": self._tool_get_statement_lines,
            "find_section_items": self._tool_find_section_items,
            "analyze_section_amounts": self._tool_analyze_section_amounts,
            "find_line_item": self._tool_find_line_item,
            "search_context": self._tool_search_context,
        }

    def answer(self, question: str, result: dict[str, Any]) -> dict[str, Any]:
        question = question.strip()
        if not question:
            return {
                "answer": "Pose une question sur l'etat financier pour lancer l'analyse.",
                "contexts": [],
                "mode": "empty",
            }

        if not self.config.hf_token:
            fallback = self.rag.answer(question, result)
            fallback["mode"] = "agent_fallback_rag_hf_not_configured"
            fallback["tools_used"] = ["fallback_rag"]
            return fallback

        tool_contexts: list[dict[str, Any]] = []
        tool_trace: list[str] = []
        messages = self._build_messages(question, result)
        last_tool_action: str | None = None
        last_tool_result: dict[str, Any] | None = None

        for _step in range(self.max_steps):
            plan = self._call_agent(messages)
            if not plan:
                break

            action = str(plan.get("action", "")).strip()
            action_input = plan.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {}

            if action == "final":
                answer = str(plan.get("final_answer", "")).strip()
                if answer and not self._is_weak_final_answer(answer):
                    return {
                        "answer": answer,
                        "contexts": tool_contexts[:8],
                        "mode": "hf_llm_agent",
                        "tools_used": tool_trace,
                    }
                if last_tool_action and last_tool_result:
                    local_answer = self._build_local_answer_from_tool(last_tool_action, last_tool_result)
                    if local_answer:
                        return {
                            "answer": local_answer,
                            "contexts": tool_contexts[:8],
                            "mode": "hf_llm_agent_localized_final",
                            "tools_used": tool_trace,
                        }
                break

            tool = self.tools.get(action)
            if tool is None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(plan, ensure_ascii=True),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool error: unknown action. Use only one of "
                            f"{', '.join(sorted(self.tools))} or final."
                        ),
                    }
                )
                continue

            try:
                tool_result = tool(result, **action_input)
            except TypeError as exc:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(plan, ensure_ascii=True),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool error: {exc}. Adjust the action_input keys and try again.",
                    }
                )
                continue
            tool_trace.append(action)
            last_tool_action = action
            last_tool_result = tool_result
            tool_contexts.extend(tool_result.get("contexts", []))
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(plan, ensure_ascii=True),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Tool result:\n" + tool_result["content"],
                }
            )

        if last_tool_action and last_tool_result:
            local_answer = self._build_local_answer_from_tool(last_tool_action, last_tool_result)
            if local_answer:
                return {
                    "answer": local_answer,
                    "contexts": tool_contexts[:8],
                    "mode": "hf_llm_agent_localized_fallback",
                    "tools_used": tool_trace,
                }

        fallback = self.rag.answer(question, result)
        fallback["mode"] = "agent_fallback_rag_after_llm"
        fallback["tools_used"] = tool_trace or ["fallback_rag"]
        contexts = tool_contexts[:5] + fallback.get("contexts", [])
        fallback["contexts"] = self._dedupe_contexts(contexts)[:8]
        return fallback

    def _build_messages(self, question: str, result: dict[str, Any]) -> list[dict[str, str]]:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        statement_titles = []
        for statement_key, statement in statements.items():
            title = statement.get("title", statement_key.replace("_", " ").title())
            columns = ", ".join(statement.get("columns", []))
            statement_titles.append(f"- {statement_key}: {title}" + (f" | colonnes: {columns}" if columns else ""))

        key_metrics = result.get("structured_extraction", {}).get("financial_data", {}).get("key_metrics", {})
        metric_names = ", ".join(sorted(key_metrics)) if key_metrics else "none"
        available_tools = "\n".join(
            [
                "- get_key_metrics(metric_names?: list[str]) -> use for high-level metrics such as revenue, expenses, net_income, assets, liabilities, equity",
                "- get_statement_titles() -> list detected statements before choosing a statement-specific tool",
                "- get_statement_lines(statement_name?: str, limit?: int) -> inspect a full statement when you need nearby line items",
                "- find_section_items(section_label: str, limit?: int) -> use for questions asking to list, detail, compose, or enumerate a section",
                "- analyze_section_amounts(section_label: str, limit?: int) -> use for questions asking the largest, highest, smallest, or lowest item inside one section",
                "- find_line_item(label: str, year?: str) -> use for questions asking the amount of one specific item such as loyer, inventaire, encaisse, passifs, revenus",
                "- search_context(query: str, limit?: int) -> use when labels are unclear or the first tool was insufficient",
                "- final(final_answer: str) -> final answer in French based only on tool results",
            ]
        )
        system_prompt = (
            "You are a financial statement chat agent. "
            "You must answer in French using only tool results from this session. "
            "Do not invent numbers, labels, or years. "
            "Think step by step, but return JSON only. "
            "Use at most one tool per turn, then wait for the tool result. "
            "When you have enough evidence, return action=final.\n\n"
            "Valid JSON examples:\n"
            '{"action":"find_section_items","action_input":{"section_label":"actifs a court terme","limit":6},"final_answer":""}\n'
            '{"action":"analyze_section_amounts","action_input":{"section_label":"depenses","limit":6},"final_answer":""}\n'
            '{"action":"final","action_input":{},"final_answer":"Les actifs a court terme sont ..."}\n\n'
            f"Available tools:\n{available_tools}"
        )
        user_prompt = (
            f"Question utilisateur: {question}\n\n"
            "Contexte disponible:\n"
            f"- Key metrics disponibles: {metric_names}\n"
            f"- Statements detectes:\n{chr(10).join(statement_titles) if statement_titles else '- none'}\n"
            "Commence par choisir le meilleur outil."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_agent(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        try:
            response = requests.post(
                f"{self.config.hf_base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.hf_model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 450,
                    "response_format": {"type": "json_object"},
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            content = str(payload["choices"][0]["message"]["content"]).strip()
            return self._normalize_plan(self._parse_json_object(content))
        except Exception:
            return None

    def _tool_get_key_metrics(self, result: dict[str, Any], metric_names: list[str] | None = None, **_: Any) -> dict[str, Any]:
        key_metrics = result.get("structured_extraction", {}).get("financial_data", {}).get("key_metrics", {})
        normalized_filter = {self._normalize(name) for name in (metric_names or []) if str(name).strip()}
        selected: dict[str, Any] = {}
        contexts: list[dict[str, Any]] = []

        for metric_name, values in key_metrics.items():
            if normalized_filter and self._normalize(metric_name) not in normalized_filter:
                continue
            selected[metric_name] = values
            if isinstance(values, dict):
                rendered = ", ".join(f"{year}: {value}" for year, value in values.items() if value is not None)
                if rendered:
                    contexts.append(
                        {
                            "text": f"{metric_name} | {rendered}",
                            "source": "structured_key_metrics",
                            "doc_type": "key_metric",
                        }
                    )

        return {
            "content": json.dumps(selected or key_metrics, ensure_ascii=True, indent=2),
            "contexts": contexts,
        }

    def _tool_get_statement_titles(self, result: dict[str, Any], **_: Any) -> dict[str, Any]:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        rows = []
        contexts = []
        for statement_key, statement in statements.items():
            row = {
                "statement_name": statement_key,
                "title": statement.get("title", statement_key.replace("_", " ").title()),
                "columns": statement.get("columns", []),
                "page": statement.get("page"),
                "line_item_count": len(statement.get("line_items", [])),
            }
            rows.append(row)
            contexts.append(
                {
                    "text": f"{row['title']} | columns: {', '.join(row['columns'])}",
                    "source": statement_key,
                    "doc_type": "statement_title",
                }
            )

        return {"content": json.dumps(rows, ensure_ascii=True, indent=2), "contexts": contexts}

    def _tool_get_statement_lines(
        self,
        result: dict[str, Any],
        statement_name: str | None = None,
        limit: int = 12,
        **_: Any,
    ) -> dict[str, Any]:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        selected_key, selected_statement = self._resolve_statement(statements, statement_name)
        if not selected_statement:
            return {
                "content": json.dumps({"error": "statement_not_found", "statement_name": statement_name}, ensure_ascii=True),
                "contexts": [],
            }

        rows = []
        contexts = []
        for item in selected_statement.get("line_items", [])[: max(1, limit)]:
            row = {
                "label": item.get("label"),
                "values": item.get("values", {}),
            }
            rows.append(row)
            contexts.append(
                {
                    "text": f"{selected_statement.get('title', selected_key)} | {row['label']} | {self._render_values(row['values'])}",
                    "source": selected_key,
                    "doc_type": "statement_line",
                }
            )

        return {"content": json.dumps(rows, ensure_ascii=True, indent=2), "contexts": contexts}

    def _tool_find_section_items(
        self,
        result: dict[str, Any],
        section_label: str,
        limit: int = 8,
        **_: Any,
    ) -> dict[str, Any]:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        best = self._find_structured_section_items(section_label, statements, limit=limit)
        if best:
            return {
                "content": json.dumps(best, ensure_ascii=True, indent=2),
                "contexts": best["contexts"],
            }

        text_match = self._find_text_section_items(section_label, result.get("text", ""), limit=limit)
        if text_match:
            contexts = [
                {
                    "text": f"{text_match['section_label']} | {item}",
                    "source": "document_text",
                    "doc_type": "section_item",
                }
                for item in text_match["items"]
            ]
            payload = {
                "statement_name": "document_text",
                "statement_title": "Document text",
                "section_label": text_match["section_label"],
                "items": text_match["items"],
            }
            return {"content": json.dumps(payload, ensure_ascii=True, indent=2), "contexts": contexts}

        return {
            "content": json.dumps({"error": "section_not_found", "section_label": section_label}, ensure_ascii=True),
            "contexts": [],
        }

    def _tool_find_line_item(
        self,
        result: dict[str, Any],
        label: str,
        year: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        statements = result.get("structured_extraction", {}).get("financial_data", {}).get("financial_statements", {}).get("statements", {})
        best = self._find_line_item_match(label, statements)
        if best:
            payload = {
                "statement_name": best["statement_name"],
                "statement_title": best["statement_title"],
                "label": best["label"],
                "values": best["values"] if not year else {year: best["values"].get(year)},
            }
            contexts = [
                {
                    "text": f"{best['statement_title']} | {best['label']} | {self._render_values(best['values'])}",
                    "source": best["statement_name"],
                    "doc_type": "statement_line",
                }
            ]
            return {"content": json.dumps(payload, ensure_ascii=True, indent=2), "contexts": contexts}

        fallback = self.rag.answer(label, result)
        payload = {
            "fallback_answer": fallback.get("answer"),
            "contexts": fallback.get("contexts", []),
        }
        return {"content": json.dumps(payload, ensure_ascii=True, indent=2), "contexts": fallback.get("contexts", [])}

    def _tool_analyze_section_amounts(
        self,
        result: dict[str, Any],
        section_label: str,
        limit: int = 8,
        **_: Any,
    ) -> dict[str, Any]:
        items = self._extract_section_amount_items(section_label, result.get("text", ""), limit=limit)
        if not items:
            return {
                "content": json.dumps({"error": "section_amounts_not_found", "section_label": section_label}, ensure_ascii=True),
                "contexts": [],
            }

        ranked = sorted(items, key=lambda item: item["amount_value"], reverse=True)
        contexts = [
            {
                "text": f"{section_label} | {item['label']} | {item['amount_text']}",
                "source": "document_text",
                "doc_type": "section_amount_item",
            }
            for item in ranked
        ]
        payload = {
            "section_label": section_label,
            "items": ranked,
            "largest_item": ranked[0],
            "smallest_item": ranked[-1],
        }
        return {"content": json.dumps(payload, ensure_ascii=True, indent=2), "contexts": contexts}

    def _tool_search_context(
        self,
        result: dict[str, Any],
        query: str,
        limit: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        contexts = self.rag._retrieve_context(query, result)[: max(1, limit)]
        return {
            "content": json.dumps(contexts, ensure_ascii=True, indent=2),
            "contexts": contexts,
        }

    def _resolve_statement(
        self,
        statements: dict[str, Any],
        statement_name: str | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if not statements:
            return None, None
        if not statement_name:
            first_key = next(iter(statements))
            return first_key, statements[first_key]

        target = self._normalize(statement_name)
        best_key = None
        best_score = 0.0
        for statement_key, statement in statements.items():
            candidates = [statement_key, statement.get("title", "")]
            score = max((self._similarity(target, self._normalize(str(candidate))) for candidate in candidates if str(candidate).strip()), default=0.0)
            if score > best_score:
                best_key = statement_key
                best_score = score

        if best_key and best_score >= 0.45:
            return best_key, statements[best_key]
        return None, None

    def _find_structured_section_items(
        self,
        target: str,
        statements: dict[str, Any],
        limit: int,
    ) -> dict[str, Any] | None:
        normalized_target = self._normalize(target)
        best: tuple[float, dict[str, Any]] | None = None

        for statement_key, statement in statements.items():
            line_items = statement.get("line_items", [])
            for index, item in enumerate(line_items):
                label = str(item.get("label", "")).strip()
                if not label:
                    continue
                score = self._similarity(self._normalize(label), normalized_target)
                if score < 0.55:
                    continue

                items = []
                contexts = []
                for probe in line_items[index + 1 :]:
                    next_label = str(probe.get("label", "")).strip()
                    if not next_label:
                        continue
                    normalized_next = self._normalize(next_label)
                    if self._is_section_stop(normalized_next):
                        break
                    values = probe.get("values", {})
                    rendered_line = next_label
                    if values:
                        rendered_line = f"{next_label} | {self._render_values(values)}"
                    items.append(rendered_line)
                    contexts.append(
                        {
                            "text": f"{statement.get('title', statement_key)} | {rendered_line}",
                            "source": statement_key,
                            "doc_type": "statement_line",
                        }
                    )
                    if len(items) >= limit:
                        break

                if items:
                    payload = {
                        "statement_name": statement_key,
                        "statement_title": statement.get("title", statement_key),
                        "section_label": label,
                        "items": items,
                        "contexts": contexts,
                    }
                    if best is None or score > best[0]:
                        best = (score, payload)

        return best[1] if best else None

    def _find_text_section_items(self, target: str, text: str, limit: int) -> dict[str, Any] | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalized_target = self._normalize(target)
        best_index = -1
        best_label = ""
        best_score = 0.0

        for index, line in enumerate(lines):
            normalized_line = self._normalize(line)
            score = self._similarity(normalized_line, normalized_target)
            if score > best_score:
                best_score = score
                best_index = index
                best_label = line

        if best_score < 0.5 or best_index < 0:
            return None

        items: list[str] = []
        for line in lines[best_index + 1 :]:
            normalized_line = self._normalize(line)
            if self._is_section_stop(normalized_line):
                break
            if re.fullmatch(r"(19|20)\d{2}", line):
                continue
            items.append(line)
            if len(items) >= limit:
                break

        items = self._consolidate_text_section_items(items)
        if not items:
            return None
        return {"section_label": best_label, "items": items}

    def _extract_section_amount_items(self, target: str, text: str, limit: int) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalized_target = self._normalize(target)
        best_index = -1
        best_score = 0.0

        for index, line in enumerate(lines):
            score = self._similarity(self._normalize(line), normalized_target)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index < 0 or best_score < 0.5:
            return []

        candidates: list[str] = []
        for line in lines[best_index + 1 :]:
            normalized_line = self._normalize(line)
            if self._is_section_stop(normalized_line):
                break
            if re.fullmatch(r"(19|20)\d{2}", line):
                continue
            candidates.append(line)
            if len(candidates) >= limit * 4:
                break

        parsed: list[dict[str, Any]] = []
        index = 0
        while index < len(candidates):
            current = candidates[index].strip()
            next_item = candidates[index + 1].strip() if index + 1 < len(candidates) else ""

            if self._is_amount_only(current):
                if next_item and not self._is_amount_only(next_item) and not self._looks_like_total(next_item):
                    amount_value = self._parse_amount(current)
                    if amount_value is not None:
                        parsed.append(
                            {
                                "label": next_item,
                                "amount_text": current,
                                "amount_value": amount_value,
                            }
                        )
                    index += 2
                    continue
                index += 1
                continue

            embedded_amounts = self._extract_amount_strings(current)
            if embedded_amounts and not self._looks_like_total(current):
                amount_text = embedded_amounts[-1]
                label = current.replace(amount_text, "").strip(" |-:")
                amount_value = self._parse_amount(amount_text)
                if label and amount_value is not None:
                    parsed.append(
                        {
                            "label": label,
                            "amount_text": amount_text,
                            "amount_value": amount_value,
                        }
                    )
            index += 1

        deduped: list[dict[str, Any]] = []
        seen = set()
        for item in parsed:
            key = (self._normalize(item["label"]), item["amount_text"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _find_line_item_match(self, label: str, statements: dict[str, Any]) -> dict[str, Any] | None:
        target = self._normalize(label)
        best: tuple[float, dict[str, Any]] | None = None

        for statement_key, statement in statements.items():
            for item in statement.get("line_items", []):
                raw_label = str(item.get("label", "")).strip()
                if not raw_label:
                    continue
                score = self._similarity(self._normalize(raw_label), target)
                if score < 0.45:
                    continue
                payload = {
                    "statement_name": statement_key,
                    "statement_title": statement.get("title", statement_key),
                    "label": raw_label,
                    "values": item.get("values", {}),
                }
                if best is None or score > best[0]:
                    best = (score, payload)

        return best[1] if best else None

    def _parse_json_object(self, raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        if "```" in content:
            parts = [part.strip() for part in content.replace("```json", "```").split("```") if part.strip()]
            if parts:
                content = parts[0]

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start : end + 1]

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("agent_response_not_dict")
        return parsed

    def _normalize_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(plan)
        action = str(normalized.get("action", "")).strip()
        action_input = normalized.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}

        if action == "tool_name_or_final":
            candidate = str(action_input.pop("key", "")).strip()
            if candidate:
                action = candidate

        if action in {"find_section_items", "find_line_item"} and "label" in action_input and "section_label" not in action_input:
            action_input["section_label"] = action_input.pop("label")

        if action == "find_line_item":
            for key in ["section_label", "item_name", "line_item", "query"]:
                if key in action_input and "label" not in action_input:
                    action_input["label"] = action_input.pop(key)
                    break

        if action == "analyze_section_amounts":
            for key in ["label", "query", "item_name"]:
                if key in action_input and "section_label" not in action_input:
                    action_input["section_label"] = action_input.pop(key)
                    break

        if action == "search_context" and "section_label" in action_input and "query" not in action_input:
            action_input["query"] = action_input.pop("section_label")

        normalized["action"] = action
        normalized["action_input"] = action_input
        if "final_answer" not in normalized:
            normalized["final_answer"] = ""
        return normalized

    def _is_weak_final_answer(self, answer: str) -> bool:
        cleaned = answer.strip().lower()
        if not cleaned:
            return True
        weak_markers = [
            "...",
            "sont...",
            "sont ...",
            "comprennent...",
            "comprennent ...",
            "sont:",
            "sont :",
        ]
        if cleaned in weak_markers:
            return True
        if cleaned.endswith("..."):
            return True
        if len(cleaned) < 18:
            return True
        return False

    def _build_local_answer_from_tool(self, action: str, tool_result: dict[str, Any]) -> str | None:
        try:
            payload = json.loads(tool_result.get("content", "{}"))
        except Exception:
            return None

        if action == "find_section_items" and isinstance(payload, dict):
            section_label = str(payload.get("section_label", "")).strip()
            items = payload.get("items", [])
            if section_label and isinstance(items, list) and items:
                rendered = "\n".join(f"- {item}" for item in items)
                return f"{section_label} :\n{rendered}"

        if action == "find_line_item" and isinstance(payload, dict):
            label = str(payload.get("label", "")).strip()
            values = payload.get("values", {})
            if label and isinstance(values, dict):
                rendered = self._render_values(values)
                if rendered:
                    return f"{label} -> {rendered}"

        if action == "analyze_section_amounts" and isinstance(payload, dict):
            largest_item = payload.get("largest_item")
            if isinstance(largest_item, dict):
                label = str(largest_item.get("label", "")).strip()
                amount_text = str(largest_item.get("amount_text", "")).strip()
                if label and amount_text:
                    return f"La ligne la plus elevee de la section est `{label}` avec un montant de {amount_text}."

        if action == "get_key_metrics" and isinstance(payload, dict):
            lines = []
            for metric_name, values in payload.items():
                if isinstance(values, dict):
                    rendered = self._render_values(values)
                    if rendered:
                        lines.append(f"- {metric_name.replace('_', ' ').title()} -> {rendered}")
            if lines:
                return "\n".join(lines)

        if action == "search_context" and isinstance(payload, list) and payload:
            best = payload[0]
            if isinstance(best, dict):
                text = str(best.get("text", "")).strip()
                if text:
                    return text

        return None

    def _render_values(self, values: dict[str, Any]) -> str:
        return ", ".join(f"{year}: {value}" for year, value in values.items() if value is not None)

    def _dedupe_contexts(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique = []
        seen = set()
        for item in contexts:
            key = (item.get("source"), item.get("text"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _normalize(self, value: str) -> str:
        value = value.lower()
        value = value.replace("Ã ", "a").replace("Ã¢", "a").replace("Ã¤", "a")
        value = value.replace("Ã©", "e").replace("Ã¨", "e").replace("Ãª", "e").replace("Ã«", "e")
        value = value.replace("Ã®", "i").replace("Ã¯", "i")
        value = value.replace("Ã´", "o").replace("Ã¶", "o")
        value = value.replace("Ã¹", "u").replace("Ã»", "u").replace("Ã¼", "u")
        value = value.replace("Ã§", "c")
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        token_overlap = len(set(a.split()).intersection(set(b.split()))) / max(len(set(b.split())), 1)
        fuzzy = SequenceMatcher(None, a, b).ratio()
        return max(fuzzy, token_overlap)

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
        ]
        return any(normalized_line.startswith(token) for token in stop_tokens)

    def _looks_like_total(self, value: str) -> bool:
        normalized = self._normalize(value)
        return normalized.startswith("total") or "surplus" in normalized or "deficit" in normalized

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

    def _parse_amount(self, raw_value: str | None) -> float | None:
        if not raw_value:
            return None
        cleaned = str(raw_value).replace("$", "").replace("\u00a0", " ").strip()
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        cleaned = cleaned.replace("(", "").replace(")", "")
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = cleaned.replace(",", ".")
        try:
            value = float(cleaned)
        except ValueError:
            return None
        return -value if negative else value

    def _extract_amount_strings(self, text: str) -> list[str]:
        pattern = r"\(?-?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})?\)?|\(?-?\d{4,}(?:[.,]\d{2})?\)?"
        return [match.strip() for match in re.findall(pattern, text) if any(char.isdigit() for char in match)]
