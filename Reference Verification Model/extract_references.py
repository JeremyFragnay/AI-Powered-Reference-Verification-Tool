"""Extract and parse bibliographic references from thesis PDFs.

Opens each PDF, locates the references section, sends it to an
OpenAI-compatible LLM for structured parsing, and writes one JSON
file per PDF.
"""

import argparse
import json
import os
import re

import fitz
from openai import OpenAI
from dotenv import load_dotenv


# Number of reference entries sent to the LLM per API call. Long reference
# lists are split into batches of this size so the model does not truncate
# its output. It is safe to tune this value.
CHUNK_SIZE = 30

extraction_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

# Reference section detection
pattern = re.compile(
    r"^[ \t]*"
    r"(?:\d+[.)]?[ \t]+)?"         
    r"(ref{1,2}erences?"
    r"|bibliography"
    r"|works?\s+cited"
    r"|reference\s+list"
    r"|list\s+of\s+references"
    r"|literature\s+cited)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


# LLM system prompt
prompt = """
You are a bibliographic reference parser with deep knowledge of all academic citation formats including APA, MLA, IEEE, Harvard, Chicago, and Vancouver across all academic disciplines and languages.

You will be given a block of raw text containing the references section of an academic thesis.
Extract the following fields from each reference and return a JSON object of the form {"references": [ ... ]}:

- id (a unique identifier for each reference, you can use a simple incremental integer starting from 1)
- authors (a list of author names as they appear in the reference. Example: ["last_name1, initial1", "last_name2, initial2"].
- title
- year
- doi
- url
- journal (the journal or venue name as written in the reference, if one is given — regardless of reference_type. Use null if no journal/venue name appears in the reference text. Do not invent a journal name, and do not substitute the publisher, conference name, or book title for it.)
- reference_type

Reference types:
- journal_article
- conference_paper
- book
- website
- report
- organizational document
- preprint
- other (if it does not fit any of the above categories)

Rules:
- Extract EVERY reference in the input. Do not summarize, abbreviate, or include only a sample. Do not add any commentary before or after the JSON.
- Do not invent information. PRESERVE THE ORIGINAL TEXT as much as possible, even if it contains errors. If you are unsure about a field, use null.
- References may include websites
- If author missing, use null
- If DOI missing, use null
- If no journal name is given in the reference text, use null — do not substitute the publisher, conference name, or book title
- Store the DOI string, not the full URL. Only store DOI if it is valid and starts with "10.".
- The raw string MIGHT include Appendix or unrelated sections. Ignore those if you think they are not valid references.
- Ignore likely page numbers, section headings, or other non-reference text that may be present in the raw string.



"""

def extract_references_text(pdf_path):
    """Open a PDF, extract all text, and return the references section.

    Returns (references_text, page_index) where references_text is the text
    after the FIRST references-style heading and page_index is the 0-based
    PDF page that heading was found on (for jumping a viewer straight to the
    references section). Returns (None, None) if no such heading is found.
    """
    doc = fitz.open(pdf_path)

    page_starts = []  # cumulative char offset at which each page's text begins
    text = ""
    for page in doc:
        page_starts.append(len(text))
        text += page.get_text()

    start_pos = int(len(text) * 0.6) # start searching the last 40% of the document
    tail_text = text[start_pos:]

    matches = list(pattern.finditer(tail_text))

    if not matches:
        return None, None   # fail loud

    match_offset = start_pos + matches[0].end()
    references_text = text[match_offset:]

    page_index = 0
    for i, page_start in enumerate(page_starts):
        if page_start <= match_offset:
            page_index = i
        else:
            break

    return references_text, page_index


def split_references(references_text):
    """Split the references block into individual entries.

    APA-style entries begin with a line like "Surname, Initial." — split
    on newlines that are followed by that opener. Page footers and
    headings won't match the opener, so they get absorbed into adjacent
    entries (the prompt already tells the model to ignore such text).
    """
    entries = re.split(r"\n(?=[A-ZÀ-Ý][^\n]*?,\s+[A-ZÀ-Ý]\.)", references_text)
    return [entry.strip() for entry in entries if entry.strip()]


def batch_entries(entries, chunk_size=CHUNK_SIZE):
    """Yield successive groups of chunk_size entries."""
    for i in range(0, len(entries), chunk_size):
        yield entries[i:i + chunk_size]


def parse_references(references_text, client):
    """Parse the references text into a list of structured references.

    The text is split into individual entries and processed in batches
    of CHUNK_SIZE, with one API call per batch, so the model does not
    truncate long reference lists. The per-batch results are concatenated
    and the id field is renumbered globally from 1.
    """
    entries = split_references(references_text)

    all_references = []
    for batch_num, batch in enumerate(batch_entries(entries), start=1):
        batch_text = "\n\n".join(batch)

        response = client.chat.completions.create(

            model="gpt-4.1",

            response_format={
                "type": "json_object"
            },

            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": 'References: ' + batch_text
                }
            ],

            temperature=0,
            max_tokens=8000
        )

        extraction_usage["input_tokens"]  += response.usage.prompt_tokens
        extraction_usage["output_tokens"] += response.usage.completion_tokens
        extraction_usage["calls"]         += 1
        
        batch_references = parse_llm_json(response.choices[0].message.content)
        if batch_references is None:
            batch_references = []

        print(f"  Batch {batch_num}: sent {len(batch)} entries, "
              f"returned {len(batch_references)} references")

        all_references.extend(batch_references)

    # Each batch restarts ids at 1, so renumber globally from 1.
    for new_id, reference in enumerate(all_references, start=1):
        reference["id"] = new_id

    return all_references


def parse_llm_json(raw_text):
    """Strip ```json fences from the LLM response and decode the JSON.

    Uses raw_decode() so any trailing prose or notes after the JSON do
    not cause an "Extra data" error. Accepts either a bare list or an
    object of the form {"references": [...]}.
    """
    if not raw_text or not raw_text.strip():
        print("Empty LLM response, nothing to parse.")
        return None

    cleaned = re.sub(r"```json|```", "", raw_text).strip()

    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned)

    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)
        return None

    if isinstance(data, dict):
        return data["references"]
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Extract and parse bibliographic references from thesis PDFs."
    )
    parser.add_argument("--input", required=True, help="Folder containing PDF files")
    parser.add_argument("--output", required=True, help="Folder to write JSON files into")
    args = parser.parse_args()

    # Load credentials from the .env file next to this script
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    os.makedirs(args.output, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(".pdf"))

    for filename in pdf_files:
        pdf_path = os.path.join(args.input, filename)
        print(f"Processing {filename} ...")

        try:
            references_text, _ = extract_references_text(pdf_path)

            if references_text is None:
                print(f"  WARNING: no references section found in {filename}, skipping.")
                continue

            references = parse_references(references_text, client)

            output_name = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(args.output, output_name)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(references, f, indent=2, ensure_ascii=False)

            print(f"  Wrote {output_path}")

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            continue


if __name__ == "__main__":
    main()
