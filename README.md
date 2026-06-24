# AI-Powered Reference Verification Tool

A web-based artifact built for a BSc Business Analytics thesis (University of Amsterdam) on AI-powered academic reference verification. Given a thesis PDF, it extracts the reference list and verifies each entry's existence using a fixed six-step LLM cascade, streaming per-reference results to the browser as they complete.

## What it does

1. **Extraction** — `extract_references.py` locates the references section in the uploaded PDF (via PyMuPDF), then uses an LLM (GPT-4.1) to parse it into structured reference records (authors, title, year, DOI, URL, journal, reference type).
2. **Verification** — `verify_pipeline.py` runs each reference through a six-step cascade:
   - Step 1: Website check (binary — live URL resolution + LLM title/description comparison)
   - Step 2: DOI check (CrossRef → arXiv via Semantic Scholar, Levenshtein + LLM fallback)
   - Steps 3–5: Title cascade (CrossRef → OpenAlex → Semantic Scholar, top-5 candidates, LLM comparison)
   - Step 6: Non-DOI URL fallback
   
   Each reference resolves to **Verified**, **Uncertain**, or **Unverified**.
3. **Interface** — `app.py` (Flask) serves the frontend and streams progress via Server-Sent Events (SSE) as each reference is processed, so results appear incrementally rather than after a single long wait.

## Features

- Drag-and-drop multi-PDF upload
- Live per-reference progress via SSE
- Summary counts (Verified / Uncertain / Unverified) with a Web/URL filter
- Per-reference result cards with reasoning
- Side-by-side comparison panel: the student's original citation against the matched knowledge-base or web record, plus an in-browser PDF viewer jumping to the references page

## Requirements

- Python 3.10+
- A `.env` file (not included — see below) with your LLM proxy credentials:
  ```
  OPENAI_API_KEY=...
  OPENAI_API_BASE=...
  OPENALEX_API_KEY=...
  ```
- Dependencies:
  ```
  flask
  openai
  python-dotenv
  pymupdf
  python-Levenshtein
  beautifulsoup4
  requests
  ```

## Setup

```bash
git clone https://github.com/JeremyFragnay/AI-Powered-Reference-Verification-Tool.git
cd AI-Powered-Reference-Verification-Tool
pip install flask openai python-dotenv pymupdf python-Levenshtein beautifulsoup4 requests
```

Create a `.env` file in the project root with the credentials listed above, then run:

```bash
python app.py
```

Open **http://127.0.0.1:5050** in your browser. (macOS reserves port 5000 for AirPlay; override with the `PORT` environment variable if needed.)

## Usage

1. Drag a thesis PDF onto the upload area (or click to browse).
2. Watch references populate and verify in real time.
3. Filter by status or source type, and click any reference to inspect the matched evidence side by side with the original citation.

## Project context

Built as the individual artifact for a thesis on reference-existence verification, evaluated against the CheckIfExist benchmark (Abbonato, 2026). See the accompanying thesis for full methodology, evaluation results, and related work.

## Notes

- `.env` is gitignored and never committed — each user supplies their own credentials.
- Knowledge bases used: CrossRef, OpenAlex, Semantic Scholar.
