# Document Processing Architecture

Architecture cible mise en place :

Streamlit UI
-> Python processing
-> PDF/Text extraction (`pdfplumber` / `pymupdf`)
-> OCR scans (`Tesseract`)
-> Extraction structuree (`Hugging Face` ou mode Python local)
-> Regles metier (Python)
-> Embeddings + indexation (`Qdrant`)
-> Stockage resultats (SQLite / JSON)

## Demarrage local Python

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Demarrage Docker Compose

```powershell
docker compose up -d --build
```

## Endpoints

- Streamlit: `http://localhost:8501`
- Qdrant API: `http://localhost:6333`

## Modele Hugging Face par defaut

- Generation structuree: `katanemo/Arch-Router-1.5B:hf-inference`
- Embeddings: `intfloat/multilingual-e5-large`
