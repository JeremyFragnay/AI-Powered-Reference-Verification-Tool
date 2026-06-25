"""Extract and parse bibliographic references from thesis PDFs.

Opens each PDF, locates the references section, sends it to an
OpenAI-compatible LLM for structured parsing, and writes one JSON
file per PDF.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Generator

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

# Number of reference entries sent to the LLM per API call. Long reference
# lists are split into batches of this size so the model does not truncate
# its output. It is safe to tune this value.
CHUNK_SIZE: int = 30

# Global tracking dictionary for LLM usage and API metrics
extraction_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

# Regular expression to identify standard academic reference section headings.
# Matches optional numbering, common titles (e.g., References, Bibliography), 
# and optional trailing colons.
pattern: re.Pattern = re.compile(
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

# Deeply specialized system prompt instructing the LLM on field formatting,
# accepted reference types, data cleaning constraints, and JSON structural layout.
prompt: str = """
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


def extract_references_text(pdf_path: str) -> Tuple[Optional[str], Optional[int]]:
    """Opens a PDF, extracts all text, and locates the references section.

    Looks for the reference section starting from the last 40% of the document's 
    compiled text block to avoid false positives in the table of contents or introduction.

    Args:
        pdf_path: The filesystem path to the target PDF file.

    Returns:
        A tuple containing:
            - references_text (str or None): The raw text appearing after the first 
              matching reference heading, or None if no heading was found.
            - page_index (int or None): The 0-based page number where the reference 
              section begins, or None if not found.
    """
    doc = fitz.open(pdf_path)

    page_starts: List[int] = []  # Tracks cumulative character offset per page
    text: str = ""
    for page in doc:
        page_starts.append(len(text))
        text += page.get_text()

    # The standard assumption is that bibliography/references live in the final 40% of a thesis
    start_pos = int(len(text) * 0.6)
    tail_text = text[start_pos:]

    matches = list(pattern.finditer(tail_text))

    if not matches:
        return None, None   # Fail loud if no reference section header is detected

    # Grab the offset of the first matched heading and isolate the subsequent text
    match_offset = start_pos + matches[0].end()
    references_text = text[match_offset:]

    # Map the character position back to its specific 0-based PDF page index
    page_index = 0
    for i, page_start in enumerate(page_starts):
        if page_start <= match_offset:
            page_index = i
        else:
            break

    return references_text, page_index


def split_references(references_text: str) -> List[str]:
    """Splits the raw reference block into a list of individual entries.

    Designed primarily for APA-style entries which typically begin with a newline
    followed by an uppercase Surname patterns (e.g., "Surname, I."). Non-standard
    lines like running headers or page footers are safely swallowed into adjacent records.

    Args:
        references_text: The long raw text block extracted from the PDF reference section.

    Returns:
        A list of cleaned individual reference string elements.
    """
    # Split on newlines that are immediately followed by an uppercase letter pattern ("Surname, I.")
    entries = re.split(r"\n(?=[A-ZÀ-Ý][^\n]*?,\s+[A-ZÀ-Ý]\.)", references_text)
    return [entry.strip() for entry in entries if entry.strip()]


def batch_entries(entries: List[str], chunk_size: int = CHUNK_SIZE) -> Generator[List[str], None, None]:
    """Yields successive chunks of elements from the entries list.

    Args:
        entries: The list of items to break up.
        chunk_size: The maximum size of each generated chunk.

    Yields:
        A sublist slice of the specified chunk_size.
    """
    for i in range(0, len(entries), chunk_size):
        yield entries[i:i + chunk_size]


def parse_references(references_text: str, client: OpenAI) -> List[Dict[str, Any]]:
    """Chunks reference text, runs LLM parsing routines, and combines the responses.

    Splits the text into individual items and processes them in batches of CHUNK_SIZE
    to prevent prompt window/token limits from cutting off long outputs. Combines 
    the result sets and maps unified globally-incremented IDs to the collection.

    Args:
        references_text: The total raw text of the isolated references section.
        client: An initialized OpenAI client instance.

    Returns:
        A list of structurally structured dictionaries, where each dict represents a single reference.
    """
    entries = split_references(references_text)
    all_references: List[Dict[str, Any]] = []
    
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
                    "content": f'References:\n{batch_text}'
                }
            ],
            temperature=0,
            max_tokens=8000
        )

        # Update global usage tracking metrics
        extraction_usage["input_tokens"] += response.usage.prompt_tokens
        extraction_usage["output_tokens"] += response.usage.completion_tokens
        extraction_usage["calls"] += 1
        
        batch_references = parse_llm_json(response.choices[0].message.content)
        if batch_references is None:
            batch_references = []

        print(f"  Batch {batch_num}: sent {len(batch)} entries, "
              f"returned {len(batch_references)} references")

        all_references.extend(batch_references)

    # Renumber the identifiers sequentially across the unified batch list
    for new_id, reference in enumerate(all_references, start=1):
        reference["id"] = new_id

    return all_references


def parse_llm_json(raw_text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Cleans Markdown formatting indicators and decodes raw JSON content.

    Employs JSONDecoder().raw_decode to parse the raw text object safely even if the
    LLM adds trailing thoughts, notes, or chat pleasantries outside the JSON block.

    Args:
        raw_text: The string response returned by the LLM completion API.

    Returns:
        A list of dictionaries representing individual references, or None if extraction fails.
    """
    if not raw_text or not raw_text.strip():
        print("Empty LLM response, nothing to parse.")
        return None

    # Strip out any markdown ```json backtick enclosures if present
    cleaned = re.sub(r"```json|```", "", raw_text).strip()

    try:
        data, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)
        return None

    # Handle cases where the response is either wrapped in a "references" object or returns as a bare list
    if isinstance(data, dict):
        return data.get("references", [])
    return data


def main() -> None:
    """CLI orchestration execution entry point.

    Handles arguments, initializes target directories, loads environmental vars,
    and runs the parsing routine iteratively through found PDF records.
    """
    parser = argparse.ArgumentParser(
        description="Extract and parse bibliographic references from thesis PDFs."
    )
    parser.add_argument("--input", required=True, help="Folder containing PDF files")
    parser.add_argument("--output", required=True, help="Folder to write JSON files into")
    args = parser.parse_args()

    # Locate and load the environment configurations (.env) relative to the script location
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

    os.makedirs(args.output, exist_ok=True)

    # Gather and sort all PDF targets in the defined path
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

            # Define destination JSON layout path scheme
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
