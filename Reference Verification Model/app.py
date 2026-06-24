"""Local web interface for the reference verification pipeline.

Flow:
  1. POST /verify  — accepts a PDF upload.
  2. extract_references.extract_references_text() pulls the references section,
     then parse_references() turns it into structured reference dicts.
  3. verify_pipeline.verify_references() runs the verification cascade.
  4. Per-reference status updates stream back to the browser via Server-Sent
     Events (SSE) as each reference reaches a terminal status.

Run:
    /opt/miniconda3/envs/minai/bin/python app.py
    open http://127.0.0.1:5050

Port defaults to 5050 (macOS reserves 5000 for AirPlay/ControlCenter).
Override with the PORT environment variable.

Requires a populated .env (UvA LiteLLM proxy creds) next to this file — the
same .env used by extract_references.py and verify_pipeline.py.
"""

import json
import os
import queue
import tempfile
import threading
import time
import uuid

from flask import Flask, Response, request, send_from_directory, send_file
from openai import OpenAI
from dotenv import load_dotenv

# Reuse the extractor's reusable functions WITHOUT modifying it.
from extract_references import extract_references_text, parse_references
# Reuse the verification pipeline as-is.
from verify_pipeline import verify_references

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

app = Flask(__name__, static_folder="static", static_url_path="")

# ---- PDF retention for the in-browser viewer -------------------------------
# Verified PDFs used to be deleted the moment a run finished. The viewer
# needs them to stick around afterwards, so each upload gets a random token
# mapping to its temp-file path. Entries older than _PDF_TTL_SECONDS are
# swept lazily on the next /verify call rather than via a background thread.
_pdf_store: dict = {}
_PDF_TTL_SECONDS = 3600  # 1 hour — generous for a single local review session


def _cleanup_pdf_store():
    now = time.time()
    expired = [t for t, v in _pdf_store.items() if now - v["created"] > _PDF_TTL_SECONDS]
    for t in expired:
        entry = _pdf_store.pop(t, None)
        if entry:
            try:
                os.unlink(entry["path"])
            except OSError:
                pass


def _sse(event: str, payload: dict) -> str:
    """Format a dict as a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _is_doi_link(url) -> bool:
    if not url:
        return False
    u = str(url).lower()
    return "doi.org" in u or u.startswith("10.")


def _public_ref(ref: dict) -> dict:
    """Trim a reference down to the fields the frontend renders.

    has_doi / is_web are computed from the ORIGINAL reference fields (not the
    verification result) so the frontend filters are stable from the moment a
    reference is extracted, before verification fills in the url.

    matched_candidate is the KB (or live web page) record the pipeline
    compared the reference against, when a match was found — used by the
    frontend's side-by-side comparison view so an assessor can sanity-check
    the LLM's verdict against the original evidence.

    student_url is the link the student themselves provided in the
    reference, as originally extracted — separate from `url`, which is the
    resolved/matched link the pipeline found (DOI redirect, matched KB
    record, or the checked website itself). Showing both lets an assessor
    confirm the candidate is really the same source as what the student
    cited, not just a same-titled lookalike.
    """
    v = ref.get("verification") or {}
    orig_url = ref.get("url")
    return {
        "id":      ref.get("id"),
        "title":   ref.get("title"),
        "authors": ref.get("authors"),
        "year":    ref.get("year"),
        "doi":     ref.get("doi"),
        "journal": ref.get("journal"),
        "student_url": orig_url,
        "status":  ref.get("status") or v.get("label"),
        "source":  v.get("source"),
        "reason":  v.get("reason"),
        "checks":  v.get("checks"),
        "url":     v.get("url"),
        "matched_candidate": v.get("matched_candidate"),
        "has_doi": bool(ref.get("doi")),
        "is_web":  ref.get("reference_type") == "website"
                   or (bool(orig_url) and not _is_doi_link(orig_url)),
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/pdf/<token>")
def get_pdf(token):
    """Serve a retained upload back to the browser for the side-by-side viewer."""
    entry = _pdf_store.get(token)
    if not entry or not os.path.exists(entry["path"]):
        return Response(status=404)
    return send_file(
        entry["path"],
        mimetype="application/pdf",
        as_attachment=False,
        download_name=entry.get("original_name") or "thesis.pdf",
    )


@app.route("/verify", methods=["POST"])
def verify():
    """Accept a PDF and stream verification progress as SSE."""
    _cleanup_pdf_store()

    uploaded = request.files.get("pdf")
    if uploaded is None or uploaded.filename == "":
        return Response(
            _sse("error", {"message": "No PDF uploaded."}),
            mimetype="text/event-stream",
        )

    # Persist the upload to a temp file so PyMuPDF can open it by path.
    suffix = os.path.splitext(uploaded.filename)[1] or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    uploaded.save(tmp.name)
    tmp.close()
    original_name = uploaded.filename

    # Registered now (not after verification) so the viewer can serve the PDF
    # even if verification fails partway through.
    pdf_token = uuid.uuid4().hex
    _pdf_store[pdf_token] = {
        "path": tmp.name,
        "original_name": original_name,
        "created": time.time(),
    }

    def stream():
        events: "queue.Queue" = queue.Queue()
        SENTINEL = object()

        def worker():
            try:
                # -- Extraction -------------------------------------------------
                events.put(_sse("phase", {"message": f"Extracting references from {original_name}…"}))
                references_text, ref_page = extract_references_text(tmp.name)
                if references_text is None:
                    events.put(_sse("error", {"message": "No references section found in this PDF."}))
                    return

                client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_API_BASE"),
                )
                events.put(_sse("phase", {"message": "Parsing references with the LLM…"}))
                references = parse_references(references_text, client)

                if not references:
                    events.put(_sse("error", {"message": "Reference section found, but no references could be parsed."}))
                    return

                events.put(_sse("extracted", {
                    "total": len(references),
                    "references": [_public_ref({**r, "verification": {}}) for r in references],
                    "pdf_token": pdf_token,
                    "pdf_page": ref_page + 1,  # PDF.js/browser viewers use 1-based #page=N
                }))

                # -- Verification (per-reference progress) ----------------------
                events.put(_sse("phase", {"message": "Verifying references…"}))

                def on_progress(ref, step):
                    events.put(_sse("progress", {"step": step, "reference": _public_ref(ref)}))

                verify_references(references, progress_callback=on_progress)

                events.put(_sse("done", {
                    "references": [_public_ref(r) for r in references],
                }))
            except Exception as e:  # surface any failure to the browser
                events.put(_sse("error", {"message": f"{type(e).__name__}: {e}"}))
            finally:
                # The retained PDF is no longer unlinked here — it stays on
                # disk so the side-by-side viewer can fetch it via
                # GET /pdf/<token> after the run finishes. _cleanup_pdf_store()
                # sweeps it (and any other stale entries) on a later /verify call.
                events.put(SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = events.get()
            if item is SENTINEL:
                break
            yield item

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable proxy buffering if present
    }
    return Response(stream(), mimetype="text/event-stream", headers=headers)


if __name__ == "__main__":
    # threaded=True so the SSE stream and its worker thread coexist.
    # macOS reserves 5000 for AirPlay/ControlCenter, so default to 5050.
    port = int(os.getenv("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
