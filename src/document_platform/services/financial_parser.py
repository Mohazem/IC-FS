from __future__ import annotations

import io
import re
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from statistics import median
from typing import Any
from unicodedata import normalize


class FinancialStatementParser:
    def __init__(self) -> None:
        self.statement_schema = {
            "balance_sheet": {
                "title": "État de la situation financière",
                "keywords": ["etat de la situation financiere", "statement of financial position", "balance sheet"],
            },
            "income_statement": {
                "title": "État des résultats",
                "keywords": ["etat des resultats", "etat des recettes et depenses", "statement of operations", "income statement"],
            },
            "net_debt_statement": {
                "title": "État de la variation de la dette nette",
                "keywords": ["etat de la variation de la dette nette", "net debt"],
            },
            "cash_flow_statement": {
                "title": "État des flux de trésorerie",
                "keywords": ["etat des flux de tresorerie", "cash flow"],
            },
        }
        self.metric_schema = {
            "revenue": {
                "section": "income_statement",
                "aliases": [
                    "total des recettes",
                    "total des revenus",
                    "total revenues",
                    "revenus",
                    "recettes",
                    "sales",
                ],
                "positive_tokens": {"total", "recettes", "revenus", "revenue", "sales"},
                "negative_tokens": {"depenses", "expenses", "passif", "actif", "net"},
            },
            "expenses": {
                "section": "income_statement",
                "aliases": [
                    "total des depenses",
                    "total expenses",
                    "depenses",
                    "charges",
                    "expenses",
                ],
                "positive_tokens": {"total", "depenses", "charges", "expenses", "couts"},
                "negative_tokens": {"recettes", "revenus", "passif", "actif", "net"},
            },
            "net_income": {
                "section": "income_statement",
                "aliases": [
                    "excedent",
                    "surplus",
                    "deficit",
                    "excedent deficit",
                    "surplus deficit",
                    "net income",
                    "resultat net",
                ],
                "positive_tokens": {"excedent", "surplus", "deficit", "resultat", "net"},
                "negative_tokens": {"debut", "fin", "actif"},
            },
            "assets": {
                "section": "balance_sheet",
                "aliases": [
                    "total de l actif",
                    "total de lactif",
                    "total assets",
                    "actif",
                    "assets",
                ],
                "positive_tokens": {"total", "actif", "assets"},
                "negative_tokens": {"passif", "net", "revenus", "depenses"},
            },
            "liabilities": {
                "section": "balance_sheet",
                "aliases": [
                    "total du passif",
                    "total liabilities",
                    "passif",
                    "liabilities",
                ],
                "positive_tokens": {"total", "passif", "liabilities"},
                "negative_tokens": {"actif", "net", "revenus", "depenses"},
            },
            "equity": {
                "section": "balance_sheet",
                "aliases": [
                    "actif net",
                    "total de l actif net",
                    "total de lactif net",
                    "total equity",
                    "equity",
                ],
                "positive_tokens": {"actif", "net", "equity"},
                "negative_tokens": {"passif et", "debut", "revenus", "depenses"},
            },
        }

    def extract_financial_statements(
        self,
        file_name: str,
        file_bytes: bytes | None,
    ) -> dict[str, Any] | None:
        if not file_bytes or not file_name.lower().endswith(".pdf"):
            return None

        pages = self._native_text_pages(file_bytes)
        if not pages:
            return None

        statements: dict[str, Any] = {}
        detected_years: list[str] = []
        camelot_tables = self._extract_camelot_statement_tables(file_bytes, pages)

        for page in pages:
            for statement_key, segment_text in self._extract_statement_segments(page["text"]):
                parsed = camelot_tables.get((statement_key, page["page"])) or self._parse_native_statement_page(segment_text, statement_key, page["page"])
                if not parsed["line_items"]:
                    continue

                statements[statement_key] = parsed
                for year in parsed["columns"]:
                    if year not in detected_years:
                        detected_years.append(year)

        if not statements:
            return None

        return {
            "years": detected_years,
            "statements": statements,
        }

    def _extract_camelot_statement_tables(
        self,
        file_bytes: bytes,
        pages: list[dict[str, Any]],
    ) -> dict[tuple[str, int], dict[str, Any]]:
        try:
            import camelot  # type: ignore
        except Exception:
            return {}

        page_statements: list[tuple[int, str]] = []
        for page in pages:
            for statement_key, _segment_text in self._extract_statement_segments(page["text"]):
                page_statements.append((page["page"], statement_key))

        if not page_statements:
            return {}

        results: dict[tuple[str, int], dict[str, Any]] = {}
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(file_bytes)
            temp_path = handle.name

        try:
            for page_number, statement_key in page_statements:
                try:
                    tables = camelot.read_pdf(temp_path, pages=str(page_number), flavor="stream")
                except Exception:
                    continue

                parsed = self._parse_camelot_tables(tables, statement_key, page_number)
                if parsed["line_items"]:
                    results[(statement_key, page_number)] = parsed
        finally:
            try:
                import os

                os.unlink(temp_path)
            except Exception:
                pass

        return results

    def _parse_camelot_tables(self, tables: Any, statement_key: str, page_number: int) -> dict[str, Any]:
        columns: list[str] = []
        line_items: list[dict[str, Any]] = []

        for table in tables:
            try:
                records = table.df.fillna("").values.tolist()
            except Exception:
                continue

            table_columns, table_items = self._camelot_table_to_line_items(records)
            if len(table_columns) > len(columns):
                columns = table_columns
            line_items.extend(table_items)

        line_items = self._consolidate_statement_line_items(line_items, columns)
        return {
            "title": self.statement_schema[statement_key]["title"],
            "page": page_number,
            "columns": columns,
            "line_items": line_items,
        }

    def _camelot_table_to_line_items(self, rows: list[list[Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        cleaned_rows = []
        for row in rows:
            values = [re.sub(r"\s+", " ", str(cell or "").replace("\n", " ")).strip() for cell in row]
            if any(values):
                cleaned_rows.append(values)

        if not cleaned_rows:
            return [], []

        data_start = None
        for index, row in enumerate(cleaned_rows):
            amount_cells = sum(1 for cell in row[1:] if self._extract_inline_amounts(cell))
            if row and row[0] and amount_cells >= 2:
                data_start = index
                break

        if data_start is None:
            return [], []

        header_rows = cleaned_rows[:data_start]
        total_columns = max(len(row) for row in cleaned_rows)
        columns: list[str] = []
        for column_index in range(1, total_columns):
            header_text = " ".join(row[column_index] for row in header_rows if column_index < len(row) and row[column_index]).strip()
            normalized_header = self._normalize_camelot_header(header_text)
            if normalized_header:
                columns.append(normalized_header)

        if len(columns) < 2:
            return [], []

        line_items: list[dict[str, Any]] = []
        for row in cleaned_rows[data_start:]:
            label = row[0].strip() if row else ""
            amounts: list[str] = []
            for cell in row[1:]:
                amounts.extend(self._extract_inline_amounts(cell))

            if not label and not amounts:
                continue
            if not label:
                label = "Valeur"

            values: dict[str, str] = {}
            for index, amount in enumerate(amounts[: len(columns)]):
                values[columns[index]] = amount

            if values:
                line_items.append({"label": label, "values": values})

        return columns, line_items

    def _normalize_camelot_header(self, header_text: str) -> str | None:
        if not header_text:
            return None

        lowered = self._normalize(header_text)
        year_match = re.search(r"(19|20)\d{2}", header_text)
        if not year_match:
            return None

        year = year_match.group(0)
        if "prevu" in lowered or "prévu" in lowered or "resultats prevus" in lowered:
            return f"Prévu {year}"
        return year

    def extract_period_metrics(
        self,
        file_name: str,
        file_bytes: bytes | None,
        default_year: str | None = None,
    ) -> dict[str, dict[str, str | None]] | None:
        if not file_bytes or not file_name.lower().endswith(".pdf"):
            return None

        statement_bundle = self.extract_financial_statements(file_name=file_name, file_bytes=file_bytes)
        derived_from_tables = self._derive_metrics_from_statements(statement_bundle)
        if derived_from_tables:
            return derived_from_tables

        rows = self._ocr_rows(file_bytes)
        if not rows:
            return None

        years = self._detect_document_years(rows, default_year)
        metrics: dict[str, dict[str, str | None]] = {
            "revenue": {year: None for year in years},
            "expenses": {year: None for year in years},
            "net_income": {year: None for year in years},
            "assets": {year: None for year in years},
            "liabilities": {year: None for year in years},
            "equity": {year: None for year in years},
        }

        page_context = self._build_page_context(rows, years)
        candidates = self._collect_candidates(rows, page_context)

        for metric, metric_candidates in candidates.items():
            best = self._select_best_candidate(metric_candidates)
            if best:
                self._apply_amounts(metrics[metric], best["amounts"], best["year_order"])

        self._derive_net_income(metrics)
        self._derive_equity(metrics)
        return metrics

    def _derive_metrics_from_statements(self, statement_bundle: dict[str, Any] | None) -> dict[str, dict[str, str | None]] | None:
        if not statement_bundle:
            return None

        years = statement_bundle.get("years", [])
        if not years:
            return None

        metrics: dict[str, dict[str, str | None]] = {
            "revenue": {year: None for year in years},
            "expenses": {year: None for year in years},
            "net_income": {year: None for year in years},
            "assets": {year: None for year in years},
            "liabilities": {year: None for year in years},
            "equity": {year: None for year in years},
        }

        statements = statement_bundle.get("statements", {})
        balance_sheet = statements.get("balance_sheet", {})
        income_statement = statements.get("income_statement", {})

        self._fill_metric_from_line_items(
            metrics["liabilities"],
            balance_sheet.get("line_items", []),
            ["total des passifs", "total du passif", "total liabilities"],
        )
        self._fill_metric_from_line_items(
            metrics["equity"],
            balance_sheet.get("line_items", []),
            ["situation financiere nette", "actif net", "total de l actif net"],
        )
        self._fill_metric_from_line_items(
            metrics["revenue"],
            income_statement.get("line_items", []),
            ["total des revenus nets", "total des revenus", "total revenues", "revenus nets"],
        )
        self._fill_metric_from_line_items(
            metrics["expenses"],
            income_statement.get("line_items", []),
            ["total des charges", "total des depenses", "total expenses"],
        )

        total_financial_assets = self._collect_line_item_values(
            balance_sheet.get("line_items", []),
            ["total des actifs financiers", "total financial assets"],
        )
        total_non_financial_assets = self._collect_line_item_values(
            balance_sheet.get("line_items", []),
            ["total des actifs non financiers", "total non financial assets"],
        )
        for year in years:
            financial_assets = self._parse_amount(total_financial_assets.get(year))
            non_financial_assets = self._parse_amount(total_non_financial_assets.get(year))
            if financial_assets is not None and non_financial_assets is not None:
                metrics["assets"][year] = self._format_amount(financial_assets + non_financial_assets)

        self._derive_net_income(metrics)
        self._derive_equity(metrics)

        if any(value is not None for metric_values in metrics.values() for value in metric_values.values()):
            return metrics
        return None

    def _fill_metric_from_line_items(self, target: dict[str, str | None], line_items: list[dict[str, Any]], aliases: list[str]) -> None:
        values = self._collect_line_item_values(line_items, aliases)
        for year, value in values.items():
            if year in target and value is not None:
                target[year] = value

    def _collect_line_item_values(self, line_items: list[dict[str, Any]], aliases: list[str]) -> dict[str, str | None]:
        best_values: dict[str, str | None] = {}
        best_score = -1.0

        for item in line_items:
            label = self._normalize(item.get("label", ""))
            if not label:
                continue
            score = max((SequenceMatcher(None, label, self._normalize(alias)).ratio() for alias in aliases), default=0.0)
            if score > best_score:
                best_score = score
                best_values = {year: value for year, value in item.get("values", {}).items()}

        return best_values if best_score >= 0.45 else {}

    def _native_text_pages(self, file_bytes: bytes) -> list[dict[str, Any]]:
        try:
            import fitz  # type: ignore
        except Exception:
            return []

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages: list[dict[str, Any]] = []
        for page_index in range(doc.page_count):
            pages.append({"page": page_index + 1, "text": doc[page_index].get_text("text")})
        return pages

    def _match_statement_type(self, text: str) -> str | None:
        normalized = self._normalize(text)
        for statement_key, definition in self.statement_schema.items():
            if any(normalized.startswith(keyword) for keyword in definition["keywords"]):
                return statement_key
        return None

    def _extract_statement_segments(self, text: str) -> list[tuple[str, str]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        markers: list[tuple[int, str]] = []

        for index, line in enumerate(lines):
            statement_key = self._match_statement_type(line)
            if statement_key:
                markers.append((index, statement_key))

        segments: list[tuple[str, str]] = []
        for marker_index, (start_index, statement_key) in enumerate(markers):
            end_index = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(lines)
            segment_text = "\n".join(lines[start_index:end_index]).strip()
            if segment_text:
                segments.append((statement_key, segment_text))

        return segments

    def _parse_native_statement_page(self, text: str, statement_key: str, page_number: int) -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        columns = self._extract_statement_columns(lines, statement_key)
        line_items = self._extract_statement_line_items(lines, columns)
        return {
            "title": self.statement_schema[statement_key]["title"],
            "page": page_number,
            "columns": columns,
            "line_items": line_items,
        }

    def _extract_statement_columns(self, lines: list[str], statement_key: str) -> list[str]:
        header_lines = lines[:18]
        header_text = " ".join(header_lines)
        years = re.findall(r"\b(?:19|20)\d{2}\b", header_text)

        if statement_key == "income_statement" and len(years) >= 3 and "resultats prevus" in self._normalize(header_text):
            return [f"Prévu {years[0]}", years[1], years[2]]

        ordered: list[str] = []
        for year in years:
            if year not in ordered:
                ordered.append(year)
        return ordered[:3]

    def _extract_statement_line_items(self, lines: list[str], columns: list[str]) -> list[dict[str, Any]]:
        line_items: list[dict[str, Any]] = []
        label_buffer: list[str] = []

        for line in lines:
            normalized = self._normalize(line)
            if self._skip_statement_line(normalized):
                continue

            amounts = self._extract_inline_amounts(line)
            if amounts:
                split_index = self._first_amount_index(line)
                label_fragment = line[:split_index].strip() if split_index is not None else ""
                label_parts = [part for part in label_buffer + ([label_fragment] if label_fragment else []) if part]
                label = " ".join(label_parts).strip()
                label = re.sub(r"\s+", " ", label)

                if not label:
                    label = "Valeur"

                values = {}
                effective_columns = columns[: len(amounts)] if columns else []
                if not effective_columns:
                    effective_columns = [f"value_{index + 1}" for index in range(len(amounts))]
                for index, amount in enumerate(amounts):
                    if index < len(effective_columns):
                        values[effective_columns[index]] = amount

                line_items.append(
                    {
                        "label": label,
                        "values": values,
                    }
                )
                label_buffer = []
            else:
                label_buffer.append(line)

        return self._consolidate_statement_line_items(line_items, columns)

    def _consolidate_statement_line_items(self, line_items: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
        if not line_items:
            return []

        consolidated: list[dict[str, Any]] = []
        expected_columns = columns or []

        for item in line_items:
            label = str(item.get("label", "")).strip()
            values = dict(item.get("values", {}))

            if self._is_header_like_line_item(label, values, expected_columns):
                continue

            if consolidated and self._is_value_continuation(label, values, expected_columns):
                previous = consolidated[-1]
                self._append_continuation_values(previous, values, expected_columns)
                continue

            consolidated.append(
                {
                    "label": label,
                    "values": values,
                }
            )

        return [item for item in consolidated if item.get("label") and any(value is not None for value in item.get("values", {}).values())]

    def _is_header_like_line_item(self, label: str, values: dict[str, str], columns: list[str]) -> bool:
        normalized_label = self._normalize(label)
        if not values:
            return False

        value_texts = [str(value).strip() for value in values.values() if value is not None]
        if not value_texts:
            return False

        if columns and all(value in columns for value in value_texts):
            return True

        return (
            any(token in normalized_label for token in ["etat de la situation financiere", "etat des resultats", "etat des flux de tresorerie"])
            and all(re.fullmatch(r"(?:19|20)\d{2}", value) for value in value_texts)
        )

    def _is_value_continuation(self, label: str, values: dict[str, str], columns: list[str]) -> bool:
        if not columns or not values:
            return False
        if self._normalize(label) != "valeur":
            return False
        return len(values) == 1

    def _append_continuation_values(self, target_item: dict[str, Any], continuation_values: dict[str, str], columns: list[str]) -> None:
        target_values = target_item.setdefault("values", {})
        if not continuation_values:
            return

        target_column_order = [column for column in columns if column in target_values]
        next_columns = [column for column in columns if column not in target_values]
        continuation_amounts = [value for value in continuation_values.values()]

        for index, amount in enumerate(continuation_amounts):
            if index < len(next_columns):
                target_values[next_columns[index]] = amount

        if not target_column_order and len(target_values) == 1 and len(columns) >= 1:
            only_value = next(iter(target_values.values()))
            target_values.clear()
            target_values[columns[0]] = only_value
            for index, amount in enumerate(continuation_amounts):
                if index + 1 < len(columns):
                    target_values[columns[index + 1]] = amount

    def _skip_statement_line(self, normalized: str) -> bool:
        if not normalized:
            return True
        if normalized in {"en dollars", "passifs", "actifs", "revenus", "charges", "activites de fonctionnement", "activites d investissement en immobilisations"}:
            return False
        return any(
            token in normalized
            for token in [
                "les notes complementaires font partie integrante",
                "original signe par",
                "ottawa canada",
                "note ",
                "information sectorielle",
            ]
        )

    def _extract_inline_amounts(self, line: str) -> list[str]:
        pattern = r"\(?-?\d{1,3}(?:[ \u00A0]\d{3})+(?:[.,]\d{2})?\)?|\(?-?\d{4,}(?:[.,]\d{2})?\)?"
        return [match.strip() for match in re.findall(pattern, line)]

    def _first_amount_index(self, line: str) -> int | None:
        pattern = r"\(?-?\d{1,3}(?:[ \u00A0]\d{3})+(?:[.,]\d{2})?\)?|\(?-?\d{4,}(?:[.,]\d{2})?\)?"
        match = re.search(pattern, line)
        if not match:
            return None
        return match.start()

    def _detect_document_years(self, rows: list[dict[str, Any]], fallback_year: str | None) -> list[str]:
        counts: dict[str, int] = defaultdict(int)
        positions: dict[str, list[float]] = defaultdict(list)

        for row in rows:
            for year_text, center in self._extract_year_positions([row]):
                counts[year_text] += 1
                positions[year_text].append(center)

        if counts:
            ordered = sorted(
                counts.keys(),
                key=lambda year: (
                    -counts[year],
                    -int(year),
                ),
            )
            if len(ordered) > 1:
                return sorted(ordered[:2], key=int, reverse=True)
            return ordered[:1]

        if fallback_year and fallback_year.isdigit():
            return [fallback_year]
        return ["current_period"]

    def _ocr_rows(self, file_bytes: bytes) -> list[dict[str, Any]]:
        try:
            import fitz  # type: ignore
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except Exception:
            return []

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        words: list[dict[str, Any]] = []

        for page_index in range(doc.page_count):
            pix = doc[page_index].get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            data = pytesseract.image_to_data(image, lang="eng+fra", output_type=pytesseract.Output.DICT)

            for i, text in enumerate(data["text"]):
                text = (text or "").strip()
                conf = str(data["conf"][i])
                if not text or conf == "-1":
                    continue

                words.append(
                    {
                        "page": page_index,
                        "text": text,
                        "x0": int(data["left"][i]),
                        "x1": int(data["left"][i]) + int(data["width"][i]),
                        "top": int(data["top"][i]),
                        "bottom": int(data["top"][i]) + int(data["height"][i]),
                    }
                )

        rows_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for word in sorted(words, key=lambda item: (item["page"], item["top"], item["x0"])):
            page_rows = rows_by_page[word["page"]]
            target_row = None
            word_mid = (word["top"] + word["bottom"]) / 2

            for row in page_rows:
                row_mid = (row["top"] + row["bottom"]) / 2
                if abs(word_mid - row_mid) <= 18:
                    target_row = row
                    break

            if target_row is None:
                target_row = {
                    "page": word["page"],
                    "top": word["top"],
                    "bottom": word["bottom"],
                    "words": [],
                }
                page_rows.append(target_row)

            target_row["words"].append(word)
            target_row["top"] = min(target_row["top"], word["top"])
            target_row["bottom"] = max(target_row["bottom"], word["bottom"])

        visual_rows: list[dict[str, Any]] = []
        for page in sorted(rows_by_page):
            for row in sorted(rows_by_page[page], key=lambda item: item["top"]):
                row["words"].sort(key=lambda item: item["x0"])
                row["label_words"] = [word for word in row["words"] if not self._looks_numeric(word["text"])]
                row["label"] = self._normalize(" ".join(word["text"] for word in row["label_words"]))
                row["amount_groups"] = self._extract_amount_groups(row["words"])
                visual_rows.append(row)

        return visual_rows

    def _build_page_context(self, rows: list[dict[str, Any]], years: list[str]) -> dict[int, dict[str, Any]]:
        rows_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_page[row["page"]].append(row)

        context: dict[int, dict[str, Any]] = {}
        for page, page_rows in rows_by_page.items():
            anchors = self._build_page_anchors(page_rows)
            year_positions = self._extract_year_positions(page_rows)
            ordered_years = self._resolve_year_order(years, anchors, year_positions)
            section = self._detect_section(page_rows)
            context[page] = {
                "anchors": anchors,
                "year_order": ordered_years,
                "section": section,
            }
        return context

    def _build_page_anchors(self, rows: list[dict[str, Any]]) -> list[float]:
        pairs: list[list[float]] = []
        for row in rows:
            if len(row["amount_groups"]) >= 2:
                pairs.append(sorted(group["center"] for group in row["amount_groups"][:2]))

        if not pairs:
            return []

        first_centers = [pair[0] for pair in pairs]
        second_centers = [pair[1] for pair in pairs if len(pair) > 1]
        if not first_centers or not second_centers:
            return []
        return [median(first_centers), median(second_centers)]

    def _extract_year_positions(self, rows: list[dict[str, Any]]) -> list[tuple[str, float]]:
        positions: list[tuple[str, float]] = []
        for row in rows:
            for word in row["words"]:
                text = word["text"].strip()
                if re.fullmatch(r"(19|20)\d{2}", text):
                    positions.append((text, (word["x0"] + word["x1"]) / 2))
        return positions

    def _resolve_year_order(self, fallback_years: list[str], anchors: list[float], year_positions: list[tuple[str, float]]) -> list[str]:
        if len(anchors) < 2:
            return fallback_years

        mapped: list[tuple[float, str]] = []
        for year_text, position in year_positions:
            nearest_anchor = min(anchors, key=lambda anchor: abs(anchor - position))
            mapped.append((nearest_anchor, year_text))

        unique: dict[float, str] = {}
        for anchor, year_text in mapped:
            unique.setdefault(anchor, year_text)

        if len(unique) >= 2:
            ordered = [unique[anchor] for anchor in sorted(unique)]
            if len(ordered) >= 2:
                return ordered[:2]

        return fallback_years

    def _detect_section(self, rows: list[dict[str, Any]]) -> str:
        balance_score = 0
        income_score = 0

        for row in rows:
            label = row["label"]
            if any(token in label for token in ["bilan", "situation financiere", "actif", "passif"]):
                balance_score += 1
            if any(token in label for token in ["etat des recettes", "etat des resultats", "revenus", "depenses", "surplus", "deficit"]):
                income_score += 1

        return "income_statement" if income_score > balance_score else "balance_sheet"

    def _collect_candidates(self, rows: list[dict[str, Any]], page_context: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for index, row in enumerate(rows):
            if not row["label"]:
                continue

            section = page_context.get(row["page"], {}).get("section", "balance_sheet")
            anchors = page_context.get(row["page"], {}).get("anchors", [])
            year_order = page_context.get(row["page"], {}).get("year_order", [])

            merged_amounts = self._merged_amounts(rows, index)
            for metric_name, metric_def in self.metric_schema.items():
                score = self._metric_match_score(
                    label=row["label"],
                    row=row,
                    section=section,
                    metric_name=metric_name,
                    metric_def=metric_def,
                )
                if score <= 0 or not merged_amounts:
                    continue

                normalized_amounts = self._normalize_amounts_to_years(merged_amounts, anchors, year_order)
                if not normalized_amounts:
                    continue

                candidates[metric_name].append(
                    {
                        "page": row["page"],
                        "score": score,
                        "label": row["label"],
                        "amounts": normalized_amounts,
                        "year_order": year_order,
                        "is_total_like": self._is_total_like(row["label"]),
                    }
                )

            section_total_candidate = self._infer_section_total_candidate(rows, index, page_context.get(row["page"], {}))
            if section_total_candidate:
                candidates[section_total_candidate["metric"]].append(section_total_candidate)

        return candidates

    def _infer_section_total_candidate(
        self,
        rows: list[dict[str, Any]],
        index: int,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        row = rows[index]
        label = row["label"]
        section = context.get("section", "balance_sheet")
        anchors = context.get("anchors", [])
        year_order = context.get("year_order", [])

        if section != "income_statement":
            return None
        if not year_order:
            return None

        metric: str | None = None
        if label in {"recette", "recettes", "revenue", "revenus"}:
            metric = "revenue"
        elif label in {"depenses", "expenses", "charges"}:
            metric = "expenses"
        else:
            return None

        best_amounts: list[dict[str, Any]] = []
        best_score = -1
        for probe in range(index + 1, min(index + 8, len(rows))):
            probe_row = rows[probe]
            if probe_row["page"] != row["page"]:
                break
            probe_label = probe_row["label"]
            if probe_label and probe != index + 1 and any(token in probe_label for token in ["recette", "revenu", "depense", "expense", "surplus", "deficit"]):
                break

            amounts = self._merged_amounts(rows, probe)
            if not amounts:
                continue

            parsed_values = [self._parse_amount(item["text"]) for item in amounts]
            parsed_values = [value for value in parsed_values if value is not None]
            if not parsed_values:
                continue

            score = max(parsed_values)
            if len(amounts) >= 2:
                score += 5000
            if self._is_total_like(probe_label):
                score += 2000
            if score > best_score:
                best_score = score
                best_amounts = amounts

        if not best_amounts:
            return None

        normalized_amounts = self._normalize_amounts_to_years(best_amounts, anchors, year_order)
        if not normalized_amounts:
            return None

        return {
            "metric": metric,
            "page": row["page"],
            "score": 140,
            "label": f"inferred_{metric}_total",
            "amounts": normalized_amounts,
            "year_order": year_order,
            "is_total_like": True,
        }

    def _merged_amounts(self, rows: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
        row = rows[index]
        amounts = list(row["amount_groups"])
        if len(amounts) >= 2:
            return amounts

        if index + 1 >= len(rows):
            return amounts

        next_row = rows[index + 1]
        if next_row["page"] != row["page"] or next_row["top"] - row["bottom"] > 40:
            return amounts

        next_label = next_row["label"]
        if next_row["amount_groups"] and next_label in {"l exercice", "pour l exercice"}:
            return list(next_row["amount_groups"])

        if not amounts and next_row["amount_groups"] and next_label == "":
            return list(next_row["amount_groups"])

        return amounts

    def _metric_match_score(
        self,
        label: str,
        row: dict[str, Any],
        section: str,
        metric_name: str,
        metric_def: dict[str, Any],
    ) -> float:
        positive_hits = sum(1 for token in metric_def["positive_tokens"] if token in label)
        negative_hits = sum(1 for token in metric_def["negative_tokens"] if token in label)
        alias_similarity = max((SequenceMatcher(None, label, alias).ratio() for alias in metric_def["aliases"]), default=0.0)

        score = alias_similarity * 100
        score += positive_hits * 12
        score -= negative_hits * 20

        if section == metric_def["section"]:
            score += 15
        else:
            score -= 40

        if self._is_total_like(label):
            score += 18

        if len(row["amount_groups"]) >= 2:
            score += 12
        elif len(row["amount_groups"]) == 1:
            score += 4

        if metric_name == "assets" and ("passif et" in label or "actif net" in label):
            score -= 80
        if metric_name == "liabilities" and "actif net" in label:
            score -= 60
        if metric_name == "equity" and "debut de lexercice" in label:
            score -= 70
        if metric_name == "net_income" and "debut de lexercice" in label:
            score -= 70

        return score

    def _normalize_amounts_to_years(
        self,
        amounts: list[dict[str, Any]],
        anchors: list[float],
        year_order: list[str],
    ) -> list[dict[str, Any]]:
        if not amounts or not year_order:
            return []

        normalized: list[dict[str, Any]] = []
        if len(amounts) >= 2:
            ordered = sorted(amounts[:2], key=lambda item: item["center"])
            for idx, amount in enumerate(ordered):
                if idx < len(year_order):
                    normalized.append({"year": year_order[idx], "text": amount["text"]})
            return normalized

        if len(anchors) >= 2:
            amount = amounts[0]
            nearest_anchor_index = min(range(len(anchors)), key=lambda idx: abs(amount["center"] - anchors[idx]))
            if nearest_anchor_index < len(year_order):
                return [{"year": year_order[nearest_anchor_index], "text": amount["text"]}]

        return [{"year": year_order[0], "text": amounts[0]["text"]}]

    def _select_best_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                item["score"],
                len(item["amounts"]),
                item["is_total_like"],
                -item["page"],
            ),
        )

    def _apply_amounts(self, target: dict[str, str | None], amounts: list[dict[str, Any]], year_order: list[str]) -> None:
        del year_order
        for amount in amounts:
            year = amount["year"]
            if year in target:
                target[year] = amount["text"]

    def _derive_equity(self, metrics: dict[str, dict[str, str | None]]) -> None:
        assets = metrics.get("assets", {})
        liabilities = metrics.get("liabilities", {})
        equity = metrics.get("equity", {})

        for year in equity:
            asset_value = self._parse_amount(assets.get(year))
            liability_value = self._parse_amount(liabilities.get(year))
            if asset_value is None or liability_value is None:
                continue
            derived = asset_value - liability_value
            parsed_existing = self._parse_amount(equity.get(year))
            if parsed_existing is None or abs(parsed_existing - derived) > 100:
                equity[year] = self._format_amount(derived)

    def _derive_net_income(self, metrics: dict[str, dict[str, str | None]]) -> None:
        revenue = metrics.get("revenue", {})
        expenses = metrics.get("expenses", {})
        net_income = metrics.get("net_income", {})

        for year in net_income:
            revenue_value = self._parse_amount(revenue.get(year))
            expenses_value = self._parse_amount(expenses.get(year))
            if revenue_value is None or expenses_value is None:
                continue
            derived = revenue_value - expenses_value
            parsed_existing = self._parse_amount(net_income.get(year))
            if parsed_existing is None or abs(parsed_existing - derived) > 100:
                net_income[year] = self._format_amount(derived)

    def _extract_amount_groups(self, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        numeric_words = [word for word in words if self._looks_numeric(word["text"])]
        if not numeric_words:
            return []

        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []
        last_x1: int | None = None

        for word in numeric_words:
            if last_x1 is None or word["x0"] - last_x1 <= 90:
                current_group.append(word)
            else:
                groups.append(current_group)
                current_group = [word]
            last_x1 = word["x1"]

        if current_group:
            groups.append(current_group)

        amounts: list[dict[str, Any]] = []
        for group in groups:
            amount = self._clean_amount(" ".join(item["text"] for item in group))
            if any(char.isdigit() for char in amount):
                parsed_amount = self._parse_amount(amount)
                label_context = self._normalize(" ".join(item["text"] for item in words if not self._looks_numeric(item["text"])))
                if parsed_amount is not None and 1900 <= parsed_amount <= 2100:
                    if any(token in label_context for token in ["decembre", "mars", "exercice", "terminant", "au"]):
                        continue
                x0 = min(item["x0"] for item in group)
                x1 = max(item["x1"] for item in group)
                amounts.append({"text": amount, "x0": x0, "x1": x1, "center": (x0 + x1) / 2})

        if not amounts:
            raw_text = " ".join(item["text"] for item in words)
            match = re.search(r"(\(?\d{1,3}(?:\s\d{3})+\)?|\(?\d{4,}\)?)\s*[$€£]?", raw_text)
            if match:
                amount = self._clean_amount(match.group(1))
                parsed_amount = self._parse_amount(amount)
                label_context = self._normalize(" ".join(item["text"] for item in words if not self._looks_numeric(item["text"])))
                if not (parsed_amount is not None and 1900 <= parsed_amount <= 2100 and any(token in label_context for token in ["decembre", "mars", "exercice", "terminant", "au"])):
                    numeric_positions = [item for item in words if any(char.isdigit() for char in item["text"])]
                    if numeric_positions:
                        x0 = min(item["x0"] for item in numeric_positions)
                        x1 = max(item["x1"] for item in numeric_positions)
                        amounts.append({"text": amount, "x0": x0, "x1": x1, "center": (x0 + x1) / 2})

        return amounts[:2]

    def _is_total_like(self, label: str) -> bool:
        return any(token in label for token in ["total", "surplus", "excedent", "deficit", "net income"])

    def _looks_numeric(self, value: str) -> bool:
        cleaned = value.strip()
        return bool(
            re.fullmatch(
                r"[$€£(]?(?:[\dOo]{1,3}(?:[\s.,][\dOo]{3})+|[\dOo]{1,3}|[\dOo]{4,})(?:[.,]\d{2})?[)$€£]?",
                cleaned,
            )
        )

    def _normalize(self, value: str) -> str:
        normalized = normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _clean_amount(self, value: str) -> str:
        cleaned = value.replace("O", "0").replace("o", "0").replace("Â", "")
        cleaned = cleaned.replace("€", "$").replace("£", "$")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\(\s+", "(", cleaned)
        cleaned = re.sub(r"\s+\)", ")", cleaned)
        return cleaned

    def _parse_amount(self, value: str | None) -> int | None:
        if not value:
            return None
        cleaned = self._clean_amount(value)
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        digits = re.sub(r"[^\d]", "", cleaned)
        if not digits:
            return None
        amount = int(digits)
        return -amount if negative else amount

    def _format_amount(self, value: int) -> str:
        sign = "-" if value < 0 else ""
        return f"{sign}{abs(value):,}".replace(",", " ")
