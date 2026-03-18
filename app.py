from pathlib import Path

import streamlit as st

from src.document_platform.config import AppConfig
from src.document_platform.pipeline import DocumentPipeline


st.set_page_config(page_title="Document Processing Platform", page_icon=":page_facing_up:", layout="wide")


def render_summary(result: dict) -> None:
    st.subheader("Resume")
    left, right = st.columns(2)
    with left:
        st.metric("Chunks indexes", result["indexing"]["indexed_chunks"])
        st.metric("Issues metier", len(result["business_rules"]["issues"]))
    with right:
        st.metric("OCR utilise", "Oui" if result["ocr"]["used"] else "Non")
        st.metric("Pages", result["extraction"]["page_count"])


def render_financial_table(financial_data: dict) -> None:
    key_metrics = financial_data.get("key_metrics", {})
    if not key_metrics:
        st.info("Aucune metrique financiere extraite.")
        return

    years: list[str] = []
    for values in key_metrics.values():
        if isinstance(values, dict):
            for year in values:
                if year not in years:
                    years.append(year)

    if not years:
        st.info("Aucune periode detectee pour les metriques financieres.")
        return

    metric_labels = {
        metric_name: metric_name.replace("_", " ").title()
        for metric_name in key_metrics
    }

    table_rows = []
    for year in years:
        row = {"Year": year}
        for metric_name, label in metric_labels.items():
            values = key_metrics.get(metric_name, {})
            if isinstance(values, dict):
                row[label] = values.get(year)
            else:
                row[label] = values if year == years[0] else None
        table_rows.append(row)

    st.subheader("Tableau des metriques")
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_financial_statements(financial_data: dict) -> None:
    statements = financial_data.get("financial_statements", {}).get("statements", {})
    if not statements:
        return

    st.subheader("Etats financiers detailles")
    for statement_key, statement in statements.items():
        title = statement.get("title", statement_key.replace("_", " ").title())
        page = statement.get("page")
        columns = statement.get("columns", [])
        line_items = statement.get("line_items", [])
        if not line_items:
            continue

        rows = []
        for item in line_items:
            row = {"Line Item": item.get("label")}
            values = item.get("values", {})
            for column in columns:
                row[column] = values.get(column)
            for column, value in values.items():
                if column not in row:
                    row[column] = value
            rows.append(row)

        st.markdown(f"**{title}**" + (f" - page {page}" if page else ""))
        st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Solution FS")
    st.caption("Extraction et structuration des donnees financieres pour PDF, images, Excel et CSV")

    config = AppConfig.from_env()
    pipeline = DocumentPipeline(config)

    with st.sidebar:
        st.header("Configuration")
        force_ocr = st.toggle("Forcer OCR", value=False)
        extraction_mode_label = st.radio(
            "Mode d'analyse",
            options=["Rapide (Local)", "Enrichi (Hugging Face)"],
            index=0,
            help="Rapide utilise uniquement Python et les regles locales. Enrichi active Hugging Face pour une structuration plus semantique.",
        )
        store_in_qdrant = st.toggle("Indexer dans Qdrant", value=True)
        st.write(f"Dossier data: `{config.data_dir}`")
        st.write(f"SQLite: `{config.sqlite_path}`")
        st.write(f"Hugging Face: `{config.hf_base_url}`")
        st.write(f"Modele HF: `{config.hf_model}`")
        st.write(f"Qdrant: `{config.qdrant_url}`")
        if extraction_mode_label == "Rapide (Local)":
            st.caption("Mode recommande pour la vitesse et la stabilite.")
        else:
            st.caption("Mode utile pour enrichir les libelles et la structuration via Hugging Face.")

    upload = st.file_uploader("Chargez un document", type=["pdf", "txt", "md", "csv", "xlsx", "xls", "png", "jpg", "jpeg"])
    manual_text = st.text_area("Ou collez directement un texte", height=220)

    if st.button("Traiter le document", type="primary"):
        if not upload and not manual_text.strip():
            st.warning("Ajoutez un fichier ou du texte avant de lancer le traitement.")
            return

        file_bytes = upload.getvalue() if upload else None
        file_name = upload.name if upload else "manual_input.txt"

        with st.spinner("Traitement en cours..."):
            result = pipeline.run(
                file_name=file_name,
                file_bytes=file_bytes,
                manual_text=manual_text,
                force_ocr=force_ocr,
                extraction_mode="huggingface" if extraction_mode_label == "Enrichi (Hugging Face)" else "local",
                store_in_qdrant=store_in_qdrant,
            )

        render_summary(result)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Texte", "Structure", "Finance", "Regles", "Indexation", "Stockage"])

        with tab1:
            st.text_area("Texte extrait", value=result["text"], height=320)

        with tab2:
            st.json(result["structured_extraction"])

        with tab3:
            financial_data = result["structured_extraction"].get("financial_data", {})
            render_financial_table(financial_data)
            render_financial_statements(financial_data)
            st.json(financial_data)

        with tab4:
            st.json(result["business_rules"])

        with tab5:
            st.json(result["indexing"])

        with tab6:
            st.json(result["storage"])
            json_path = Path(result["storage"]["json_path"])
            if json_path.exists():
                st.download_button(
                    label="Telecharger le JSON",
                    data=json_path.read_text(encoding="utf-8"),
                    file_name=json_path.name,
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
