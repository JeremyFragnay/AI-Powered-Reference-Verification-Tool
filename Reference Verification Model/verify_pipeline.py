"""Reference verification pipeline (importable module).

Verifies each reference through a cascade of deterministic and LLM-assisted checks:
1. Website check — ping the URL
2. DOI check — CrossRef, then arXiv via S2; three-stage title verification
3. Title search cascade — CrossRef -> OpenAlex -> Semantic Scholar, each with LLM comparator
4. Final resolution — unresolved references labelled Unverified (or Uncertain, for non-website fallback paths pending manual review)

Design principle: the LLM compares retrieved candidates against the query; it
never uses its own parametric knowledge.
"""

# -- Setup -------------------------------------------------------------------
import json, time, requests, os, random
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from urllib.parse import quote, urlparse, urlunparse
from Levenshtein import ratio as lev_ratio

EMAIL       = "jeremy.fragnay@student.uva.nl"
CR_DELAY    = 1.5   # CrossRef polite pool pause between calls
DOI_TITLE_THRESHOLD = 0.8  # Levenshtein ratio — same as CheckIfExist and evaluation script

# Load credentials from the .env file next to this module (UvA LiteLLM proxy).
# Mirrors extract_references.py so behaviour is identical regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)


def title_matches(query_title, retrieved_title):
    """Returns True if normalised Levenshtein ratio >= DOI_TITLE_THRESHOLD."""
    if not query_title or not retrieved_title:
        return False
    return lev_ratio(query_title.lower().strip(), retrieved_title.lower().strip()) >= DOI_TITLE_THRESHOLD


# -- Step 1: Website check ---------------------------------------------------
# For website-type references, stream the first ~64 KB of HTML, extract
# <title> and <meta name="description">, then use the LLM to judge whether
# the page content matches the reference title.
# check_website() has exactly two possible outcomes:
#   Verified   -> page is live AND LLM confirms title/description match.
#   Unverified -> anything else (dead link after retries, connection error
#                 after retries, 403, other HTTP >=400, no metadata to compare,
#                 or LLM judges no match / LLM call failed after retries).
# Transient failures (HTTP 500/502/503/504, connection errors, timeouts) are
# retried inside _extract_page_metadata before falling through to Unverified.

COMPARATOR_PROMPT_WEB = """You are a reference verification assistant. Your only job is to determine whether a web page's metadata matches the reference title provided.

Do not use your own knowledge. Only compare the metadata provided.

MATCHING RULE: Return match: true if the page title or description clearly refers to the same document, article, or resource as the reference title. Minor wording differences, truncation, or site-name suffixes (e.g. "| Reuters", "- Wikipedia") do not prevent a match. Return match: false if the page is clearly unrelated, a generic homepage, a 404 page, or a search results page.

Respond in JSON only, no other text:
{
  "match": true or false,
  "note": "one-sentence reason" or null
}"""


def _extract_page_metadata(url: str, max_bytes: int = 65536, retries: int = 2, retry_delay: float = 3.0) -> dict:
    """Stream the first max_bytes of a URL and extract <title> and <meta description>.
    Retries on transient server errors (500, 502, 503, 504) and on connection
    errors (DNS failure, refused connection, timeout) — both may be transient."""
    headers = {"User-Agent": f"ThesisVerifier ({EMAIL})"}
    RETRYABLE = {500, 502, 503, 504}

    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=15, allow_redirects=True,
                              headers=headers) as r:
                if r.status_code == 403:
                    return {"status_code": 403, "title": "", "description": ""}
                if r.status_code in RETRYABLE and attempt < retries:
                    print(f"  Website returned {r.status_code} — retrying ({attempt + 1}/{retries})...")
                    time.sleep(retry_delay)
                    continue
                if r.status_code >= 400:
                    return {"status_code": r.status_code, "title": "", "description": ""}
                chunk = b""
                for data in r.iter_content(chunk_size=1024):
                    chunk += data
                    if len(chunk) >= max_bytes:
                        break
                status_code = r.status_code

            soup = BeautifulSoup(chunk.decode("utf-8", errors="ignore"), "html.parser")

            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            desc_tag = (
                soup.find("meta", attrs={"name": "description"}) or
                soup.find("meta", attrs={"property": "og:description"})
            )
            description = (desc_tag.get("content") or "").strip() if desc_tag else ""

            return {"status_code": status_code, "title": title, "description": description}

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < retries:
                print(f"  Connection error ({type(e).__name__}) — retrying ({attempt + 1}/{retries})...")
                time.sleep(retry_delay)
                continue
            return {"status_code": None, "title": "", "description": f"Connection error after {retries + 1} attempts."}
        except Exception as e:
            return {"status_code": None, "title": "", "description": str(e)}

    return {"status_code": None, "title": "", "description": "Max retries exceeded."}


def check_website(url, ref_title: str = ""):
    if not url.startswith("http"):
        url = "https://" + url
    parsed = urlparse(url)
    encoded = parsed._replace(path=quote(parsed.path, safe="/-._~!$&'()*+,;=:@"))
    url = urlunparse(encoded)

    meta = _extract_page_metadata(url)
    status_code = meta["status_code"]

    # --- Liveness failures (after retries in _extract_page_metadata) ---
    # No page content was retrieved, so there is nothing to show as a
    # comparison candidate.
    if status_code is None:
        return "Unverified", "Web", "Connection error after retries — domain may not exist.", None
    if status_code == 403:
        return "Unverified", "Web", "HTTP 403 — server refused request; resource may exist but access is restricted.", None
    if status_code >= 400:
        return "Unverified", "Web", f"HTTP {status_code} — page may have moved.", None

    # --- Page is live: semantic check ---
    page_title = meta["title"]
    page_desc  = meta["description"]
    page_candidate = {
        "title":       page_title or None,
        "authors":     None,
        "year":        None,
        "journal":     None,
        "doi":         None,
        "description": page_desc or None,
    }

    # If no reference title provided, fall back to pure liveness.
    if not ref_title or (not page_title and not page_desc):
        return "Unverified", "Web", f"HTTP {status_code} — page live; no metadata extracted for semantic check.", page_candidate

    user_msg = (
        f"Reference title: {ref_title}\n\n"
        f"Page <title>: {page_title}\n"
        f"Page <meta description>: {page_desc}"
    )
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": COMPARATOR_PROMPT_WEB},
                    {"role": "user",   "content": user_msg}
                ],
                temperature=0,
                max_tokens=100
            )
            llm_usage["input_tokens"]  += response.usage.prompt_tokens
            llm_usage["output_tokens"] += response.usage.completion_tokens
            llm_usage["calls"]         += 1
            result = json.loads(response.choices[0].message.content)
            if result.get("match"):
                note = result.get("note") or f"Page title: '{page_title}'"
                return "Verified", "Web", note, page_candidate
            else:
                note = result.get("note") or f"Page title '{page_title}' does not match reference title."
                return "Unverified", "Web", note, page_candidate
        except Exception as e:
            time.sleep(2 ** attempt)

    # LLM failed — fall back to liveness only
    return "Unverified", "Web", f"HTTP {status_code} — LLM semantic check failed after retries; manual review required.", page_candidate


# -- LLM comparator (Steps 2-5) ---------------------------------------------
# GPT-4.1 compares retrieved candidates against the query reference.
# Used as LLM fallback in Step 2 when Levenshtein fails, and as primary
# comparator in the title search cascade (Steps 3-5).
# Does NOT use the model's own knowledge — KB match is the only arbiter.

llm_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

# Used in Step 2 (DOI check): prefix matching allowed because a DOI-resolved
# title may omit subtitles from the reference as stored in the KB.
COMPARATOR_PROMPT_DOI = """You are a reference verification assistant. Your only job is to determine whether a candidate record retrieved via DOI lookup is the same publication as the query reference.

Do not use your own knowledge. Only compare the metadata provided.

You will receive:
- The query title (from the thesis reference list)
- The candidate's chapter/article title as stored in CrossRef
- Optionally: the candidate's container title (book or journal name), document type, authors, year, and publisher

MATCHING RULES — apply in order:
1. If the candidate title is substantially the same as the query title, return match: true.
   Differences in capitalisation, punctuation, articles, or minor word order do NOT prevent a match.
2. If the candidate title matches the beginning of the query title or vice versa, treat as match: true —
   publishers frequently omit subtitles.
3. If the document type is "book-chapter" (or similar), ALSO check whether the query title matches
   the container title (the book name). If it does, return match: true — the student likely cited the
   book title rather than the chapter title, which is a citation inaccuracy but NOT a fabrication.
4. Edition markers (e.g. "1st ed.", "2nd edition") absent from the candidate should be ignored if
   the rest of the title matches.

EXAMPLES OF MATCHES:
- Query: "Corporate Innovation: Disruptive Thinking in Organizations (1st ed.)" → Candidate title: "Corporate Innovation" → match: true
- Query: "Research methods for business students (8th ed.)" → Candidate title: "Research Methods for Business Students" → match: true
- Query: "Digital Transformation: A roadmap for billion-dollar organizations" → Candidate type: book-chapter, container: "Digital Transformation in Business" → match: true (query title matches container)

EXAMPLES OF NON-MATCHES:
- Query: "Blockchain technology overview" → Candidate: "An Overview of Blockchain Technology: Architecture, Consensus, and Future Trends" (no container match either) → match: false

When match is true, only report the title check. Add a note if there is a difference worth flagging (e.g. student cited book title instead of chapter title).
When match is false, return null for checks.

Respond in JSON only, no other text:
{
  "match": true or false,
  "checks": {
    "title": { "status": "match|mismatch", "note": "..." or null }
  } or null
}"""

# Used in Steps 3-5 (title search cascade): prefix matching not applied.
# Candidates are retrieved by title query so a partial title match is not
# sufficient evidence — the full title must be substantially the same.
COMPARATOR_PROMPT_TITLE = """You are a reference verification assistant. Your only job is to determine whether any candidate paper retrieved from an academic database is the same publication as the query reference.

Do not use your own knowledge. Only compare the metadata provided.

MATCHING RULE: If the candidate title is substantially the same as the query title (allowing for minor capitalisation, punctuation, or word-order differences), return match: true. Author names, year, and journal are context only — do NOT use them to override a title match.

When match is true, fill in the checks object using only the matched candidate's metadata:
- title: always "match" for a match, with a note if there are any noteworthy differences (e.g. student cited book title instead of chapter title, or edition markers differ).
- authors: "match" if every query author is present in the candidate, regardless of how each name is formatted. "partial" ONLY if one or more query authors are genuinely absent from the candidate's author list (the family name does not appear at all) while at least one other query author is present — note which family name(s) are missing. "mismatch" if the first author's family name does not appear at all — note it. A difference in name format (initials vs. spelled-out given names, "Surname, I." vs "Surname, Initial") is NEVER grounds for "partial" or "mismatch" on its own — only a missing family name is. Omit this key entirely if no author data is available.
  NAME MATCHING RULE: matching is done on family names; given-name format is irrelevant to match status. An author counts as present if the family name appears on both sides, regardless of whether the given name is an initial, spelled out in full, or absent. "M." matches "Manlio" or "Marco"; this is a format difference, not a missing author, and must NOT be flagged as partial or mismatch. This applies symmetrically: it does not matter which side (query or candidate) has the initial and which has the full name. Reference lists formatted as "Bresciani, S.; Ferraris, A.; Del Giudice, M." must be parsed as three distinct authors (split on ";"), each compared by family name only — do not treat this format as unparseable or as a single author, and do not penalize it for using initials.
- year: "match" if identical. "mismatch" if different — note both years. Omit this key entirely if year is missing.
- journal: only evaluate if a query journal name is provided AND the candidate has a journal/venue name. "match" if they refer to the same journal (abbreviations, "&" vs "and", and minor punctuation differences do not prevent a match). "mismatch" if they clearly refer to different journals — note both names. Omit this key entirely if either the query journal or the candidate journal is missing.

IMPORTANT: Omit any key entirely if its status would be unknown — do not include unknown or null-status fields in the checks object.

When match is false, return null for checks.

Respond in JSON only, no other text:
{
  "match": true or false,
  "matched_candidate": 1, 2, 3, 4, 5 or null,
  "checks": {
    "title":   { "status": "match|mismatch", "note": "..." or null },
    "authors": { "status": "match|partial|mismatch", "note": "..." or null },
    "year":    { "status": "match|mismatch", "note": "..." or null },
    "journal": { "status": "match|mismatch", "note": "..." or null }
  } or null
}"""

def llm_compare(ref, candidates, prompt=COMPARATOR_PROMPT_TITLE):
    if not candidates:
        return {"match": False, "matched_candidate": None,
                "confidence": "high", "reason": None}

    def _format_candidate(i, c):
        line = f"  {i+1}. Title: {c['title']} | Year: {c['year']} | Authors: {c['authors']}"
        if c.get("journal"):
            line += f" | Journal: {c['journal']}"
        # Append CrossRef extra metadata when present (used by COMPARATOR_PROMPT_DOI)
        extras = []
        if c.get("_doc_type"):
            extras.append(f"Type: {c['_doc_type']}")
        if c.get("_container_title"):
            extras.append(f"Container: {c['_container_title']}")
        if c.get("_authors_str"):
            extras.append(f"Authors (full): {c['_authors_str']}")
        if c.get("_publisher"):
            extras.append(f"Publisher: {c['_publisher']}")
        if extras:
            line += " | " + " | ".join(extras)
        return line

    candidates_text = "\n".join(
        _format_candidate(i, c) for i, c in enumerate(candidates)
    )
    user_message = (
        f"Query reference:\n"
        f"- Title: {ref.get('title')}\n"
        f"- Authors: {ref.get('authors')}\n"
        f"- Year: {ref.get('year')}\n"
        f"- Journal: {ref.get('journal')}\n\n"
        f"Candidates:\n{candidates_text}"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": user_message}
                ],
                temperature=0,
                max_tokens=200
            )
            usage = response.usage
            llm_usage["input_tokens"]  += usage.prompt_tokens
            llm_usage["output_tokens"] += usage.completion_tokens
            llm_usage["calls"]         += 1
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt + 1}/3): {e}")
            time.sleep(wait)

    print(f"  LLM failed after 3 attempts for: {ref.get('title', '')[:60]}")
    return {"match": False, "matched_candidate": None,
            "confidence": "high", "reason": None}


def _get_matched_url(candidates, matched_candidate_idx):
    """Extract and normalise the DOI URL from the matched candidate.
    CrossRef returns bare DOIs; OpenAlex returns full URIs already.
    Returns None if no DOI is available for the matched candidate.
    """
    idx = (matched_candidate_idx or 1) - 1
    if not (0 <= idx < len(candidates)):
        idx = 0
    doi = candidates[idx].get("doi") if candidates else None
    if not doi:
        return None
    if doi.startswith("http"):
        return doi  # OpenAlex already provides a full URI
    return f"https://doi.org/{doi}"


def _apply_doi_mismatch_downgrade(ref, label, reason):
    """If the reference's own cited DOI resolved to a real record but that
    record's title did NOT match the reference (set in Step 2 / Step 2b as
    _doi_mismatch on ref["verification"]), and the title cascade
    independently found and verified the correct paper by title alone,
    downgrade Verified to Uncertain rather than letting it pass silently.

    The underlying reference is real (hence the title-cascade match), but
    the student cited a DOI that points to a different work — a genuine
    citation defect distinct from "reference doesn't exist," and worth
    flagging for human review rather than auto-clearing as Verified.

    Existing Uncertain results (e.g. from an author/journal mismatch in
    the cascade) get the DOI-mismatch note appended rather than replaced,
    since both issues are independently true and worth surfacing.

    Returns the (possibly adjusted) (label, reason) tuple. No-op if the
    reference never had a DOI mismatch recorded.
    """
    doi_mismatch = ref["verification"].get("_doi_mismatch")
    if not doi_mismatch:
        return label, reason

    note = (f"Cited DOI ({doi_mismatch}) resolves to a different paper; "
            f"this reference was located independently by title.")
    if label == "Verified":
        return "Uncertain", note
    if label == "Uncertain":
        return label, f"{reason} {note}" if reason else note
    return label, reason


def _cascade_outcome(ref, checks, candidate_journal=None):
    """Given LLM checks for a title-cascade title match, decide whether the
    reference should be labelled Verified or Uncertain.

    A title match alone is no longer sufficient. Any one of the following
    downgrades the result to Uncertain:
      - authors check is "partial" (some but not all query authors found
        in the candidate record) OR "mismatch" (none of the query authors
        appear in the candidate record at all). A full mismatch is at least
        as strong a signal as a partial one and must not be waved through
        as Verified just because the title happened to line up.
      - journal presence is asymmetric: the candidate (KB) record has a
        journal name but the student's reference does not, OR the
        student's reference gives a journal but the matched candidate
        record has none. Either way, this side cannot be confirmed.
      - journal check is "mismatch" (both sides give a journal name, but
        the names clearly differ)

    candidate_journal is the journal name from the matched KB candidate
    (already resolved by the caller), independent of whatever the LLM
    checks dict says — it is needed to detect the asymmetric-missing case
    even when the LLM had nothing to compare (so never ran a journal check).

    As a side effect, this also fills in checks["journal"] when the LLM
    didn't already set one, so the comparison UI always shows a clear
    match/mismatch verdict for the asymmetric case rather than a neutral
    "info" row that looks like nothing is wrong.

    `reason` is intentionally a short summary that POINTS AT the relevant
    checklist row rather than restating its note text verbatim — the note
    is already shown inline in the per-field checklist in the UI, so
    repeating it here just duplicates the same sentence twice on screen.

    Returns (label, reason) where reason is None when label is "Verified".
    """
    reasons = []

    authors_check = checks.get("authors") or {}
    if authors_check.get("status") == "partial":
        reasons.append("Author list only partially matches the candidate (see Authors check above).")
    elif authors_check.get("status") == "mismatch":
        reasons.append("Author list does not match the candidate (see Authors check above).")

    ref_journal = ref.get("journal")

    if "journal" not in checks:
        if candidate_journal and not ref_journal:
            checks["journal"] = {
                "status": "mismatch",
                "note": f"Candidate has a journal ('{candidate_journal}'); student's reference gives none."
            }
            reasons.append("Journal presence mismatch (see Journal check above).")
        elif ref_journal and not candidate_journal:
            checks["journal"] = {
                "status": "mismatch",
                "note": f"Student's reference gives a journal ('{ref_journal}'); candidate has none."
            }
            reasons.append("Journal presence mismatch (see Journal check above).")
        elif candidate_journal:
            # Both sides present but the LLM didn't return a journal verdict
            # (shouldn't normally happen) — neutral info, not an assertion
            # of a match we never actually checked.
            checks["journal"] = {"status": "info", "note": candidate_journal}
    else:
        journal_check = checks["journal"]
        if journal_check.get("status") == "mismatch":
            reasons.append("Journal does not match the candidate (see Journal check above).")

    if reasons:
        return "Uncertain", " ".join(reasons)
    return "Verified", None


# -- Step 2: DOI check -------------------------------------------------------
# CrossRef -> arXiv (via S2).
#
# Three-stage title verification per DOI hit:
#   1. Levenshtein >= 0.8             -> Verified (fast, no LLM call)
#   2. Levenshtein fails, LLM matches -> Verified (handles edge cases)
#   3. Both fail                      -> set _doi_mismatch flag, return None
#      The ref stays Pending and falls through to the title search cascade.
#      If cascade finds the title -> Verified (wrong DOI noted in reason)
#      If cascade finds nothing   -> Unverified
#
# DOI not found in any KB -> stays Pending (no flag set)

def _doi_title_result(ref, retrieved_title, doi, extra_metadata=None):
    """Three-stage title check for a resolved DOI.
    Returns (label, source, checks, candidate) on match, or a None tuple +
    sets _doi_mismatch flag on the ref when both Levenshtein and LLM fail.
    source_label is inferred from the doi prefix.

    extra_metadata (optional dict) may contain:
      - container_title (str): book or journal name from CrossRef
      - doc_type        (str): CrossRef type, e.g. "book-chapter"
      - authors         (str): formatted author string (first 3, for the LLM)
      - authors_list    (list): full author name list, for the comparison UI
      - year            (str/int)
      - publisher       (str)
    Forwarded to the LLM so it can apply book-chapter matching rules.

    candidate is a flat snapshot of the matched KB record (title, authors,
    year, journal, doi) for side-by-side display — independent of `checks`,
    which only records the title verdict (DOI checks are title-only).
    """
    extra_metadata = extra_metadata or {}

    # Derive a human-readable source label from the DOI
    doi_clean = doi.strip().removeprefix("https://doi.org/").lower()
    if "10.48550/arxiv" in doi_clean or "arxiv.org" in doi_clean:
        source_label = "arXiv_DOI"
    else:
        source_label = "CrossRef_DOI"

    # A DOI hit confirms the title only. No note is attached here — DOI
    # checks are shown as a bare ✓ Title with no trailing sentence in the
    # comparison UI (unlike the title-search cascade, where notes explain
    # what was actually compared).
    checks = {
        "title": {
            "status": "match",
            "note": None
        }
    }

    candidate = {
        "title":   retrieved_title,
        "authors": extra_metadata.get("authors_list") or None,
        "year":    extra_metadata.get("year"),
        "journal": extra_metadata.get("container_title") or None,
        "doi":     doi.strip().removeprefix("https://doi.org/"),
    }

    # Stage 1: Levenshtein against chapter/article title
    if title_matches(ref.get("title", ""), retrieved_title):
        return "Verified", source_label, checks, candidate

    # Stage 1b: Levenshtein against container title (catches cases where the
    # student cited the book title rather than the individual chapter title).
    container_title = extra_metadata.get("container_title", "")
    if container_title and title_matches(ref.get("title", ""), container_title):
        return "Verified", source_label, checks, candidate

    # Stage 2: LLM fallback — pass full CrossRef metadata so it can apply
    # book-chapter matching rules (query title may match container, not chapter).
    llm_candidate = {
        "title": retrieved_title,
        "authors": [],
        "year": extra_metadata.get("year"),
        "_container_title": container_title,
        "_doc_type": extra_metadata.get("doc_type", ""),
        "_authors_str": extra_metadata.get("authors", ""),
        "_publisher": extra_metadata.get("publisher", ""),
    }
    result = llm_compare(ref, [llm_candidate], prompt=COMPARATOR_PROMPT_DOI)
    if result["match"]:
        return "Verified", source_label, checks, candidate

    # Stage 3: both failed — DOI resolves but title is genuinely different.
    # Store the mismatched DOI so title search steps can annotate their result.
    ref["verification"]["_doi_mismatch"] = doi
    return None, None, None, None


def check_crossref(doi, ref):
    doi_clean = doi.strip().removeprefix("https://doi.org/")
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi_clean}",
                         params={"mailto": EMAIL}, timeout=15)
        if r.status_code == 200:
            msg = r.json()["message"]
            retrieved_title = (msg.get("title") or [""])[0]
            full_authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in msg.get("author", [])
            ]
            # Extract additional metadata for LLM fallback
            extra_metadata = {
                "container_title": (msg.get("container-title") or [""])[0],
                "doc_type":        msg.get("type", ""),
                "authors":         ", ".join(full_authors[:3]),
                "authors_list":    full_authors or None,
                "year":            (
                                       (msg.get("published-print") or msg.get("published-online") or {})
                                       .get("date-parts", [[None]])[0][0]
                                   ),
                "publisher":       msg.get("publisher", ""),
            }
            return _doi_title_result(ref, retrieved_title, doi, extra_metadata)
        return None, None, None, None
    except Exception as e:
        print(f"  CrossRef DOI error: {e}")
        return None, None, None, None


def check_arxiv(doi_or_url, ref):
    """Accepts any arXiv identifier format — DOI, abs URL, PDF URL, or bare ID."""
    match = re.search(r'(?:10\.48550/arxiv\.|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)', 
                      doi_or_url.lower())
    if not match:
        return None, None, None, None
    arxiv_id = match.group(1)

    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}",
                headers=headers, params={"fields": "title,authors,year,venue,externalIds"}, timeout=15
            )
            if r.status_code == 200:
                data = r.json()
                retrieved_title = data.get("title", "")
                extra_metadata = {
                    "container_title": data.get("venue") or "",
                    "authors_list":    [a.get("name") for a in data.get("authors", [])] or None,
                    "year":            data.get("year"),
                }
                return _doi_title_result(ref, retrieved_title, doi_or_url, extra_metadata)
            if r.status_code == 429:
                backoff = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"  S2 429 on arXiv:{arxiv_id} — backing off {backoff}s (attempt {attempt + 1}/3)")
                time.sleep(backoff)
                continue
            return None, None, None, None
        except Exception as e:
            print(f"  arXiv lookup error: {e}")
            return None, None, None, None
    print(f"  arXiv check failed after 3 attempts for: {arxiv_id}")
    return None, None, None, None


# -- Step 3: CrossRef title search ------------------------------------------

def search_crossref_title(title, rows=5):
    params = {
        "query.title": title,
        "rows": rows,
        "mailto": EMAIL,
        "select": "title,author,published,DOI,container-title"
    }
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    for attempt in range(3):
        try:
            r = requests.get("https://api.crossref.org/works", params=params, timeout=15)
            if r.status_code == 200:
                results = []
                for item in r.json()["message"].get("items", []):
                    authors = [
                        f"{a.get('family', '')} {a.get('given', '')}".strip()
                        for a in item.get("author", [])
                    ]
                    pub = item.get("published", {})
                    year = pub.get("date-parts", [[None]])[0][0]
                    results.append({
                        "title":   (item.get("title") or [""])[0],
                        "authors": authors,
                        "year":    year,
                        "doi":     item.get("DOI"),
                        "journal": (item.get("container-title") or [""])[0] or None
                    })
                return results
            if r.status_code in RETRYABLE_STATUS and attempt < 2:
                wait = 2 ** attempt
                print(f"  CrossRef {r.status_code} — retrying in {wait}s (attempt {attempt + 1}/3)...")
                time.sleep(wait)
                continue
            print(f"  CrossRef returned {r.status_code} for: {title[:60]}")
            return []
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = 2 ** attempt
            print(f"  CrossRef {type(e).__name__} — retrying in {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
        except Exception as e:
            print(f"  CrossRef title error: {e}")
            return []
    print(f"  CrossRef failed after 3 attempts for: {title[:60]}")
    return []


# -- Step 4: OpenAlex title search ------------------------------------------

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")  # https://openalex.org/settings/api

def search_openalex(title, per_page=5):
    params = {
        "search": title,
        "per-page": per_page,
        "select": "title,authorships,publication_year,doi,primary_location"
    }
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    for attempt in range(3):
        try:
            r = requests.get("https://api.openalex.org/works", params=params,
                             headers={"User-Agent": f"ThesisVerifier ({EMAIL})"},
                             timeout=10)
            if r.status_code == 200:
                results = []
                for item in r.json().get("results", []):
                    authors = [
                        a["author"].get("display_name", "")
                        for a in item.get("authorships", [])
                    ]
                    primary = item.get("primary_location") or {}
                    source  = primary.get("source") or {}
                    results.append({
                        "title":   item.get("title"),
                        "authors": authors,
                        "year":    item.get("publication_year"),
                        "doi":     item.get("doi"),
                        "journal": source.get("display_name") or None
                    })
                return results
            if r.status_code in RETRYABLE_STATUS and attempt < 2:
                if r.status_code == 429 and r.headers.get("Retry-After"):
                    wait = float(r.headers["Retry-After"])
                else:
                    wait = (5 * (2 ** attempt)) + random.uniform(0, 2)  # 5-7s, 10-12s
                print(f"  OpenAlex {r.status_code} — retrying in {wait:.1f}s (attempt {attempt + 1}/3)...")
                time.sleep(wait)
                continue
            print(f"  OpenAlex returned {r.status_code} for: {title[:60]}")
            return []
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            wait = 2 ** attempt
            print(f"  OpenAlex {type(e).__name__} — retrying in {wait}s (attempt {attempt + 1}/3)...")
            time.sleep(wait)
        except Exception as e:
            print(f"  OpenAlex error: {e}")
            return []
    print(f"  OpenAlex failed after 3 attempts for: {title[:60]}")
    return []


# -- Step 5: Semantic Scholar title search ----------------------------------

def search_semantic_scholar(title, top_k=5):
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    for attempt in range(3):
        wait = 2 ** attempt  # 1s, 2s, 4s between attempts
        time.sleep(wait)
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": title, "limit": top_k, "fields": "title,authors,year,externalIds,venue"},
                headers=headers, timeout=15
            )
            if r.status_code == 200:
                return [
                    {
                        "title":   p.get("title"),
                        "authors": [a["name"] for a in p.get("authors", [])],
                        "year":    p.get("year"),
                        "doi":     p.get("externalIds", {}).get("DOI"),
                        "journal": p.get("venue") or None
                    }
                    for p in r.json().get("data", [])
                ]
            if r.status_code in (429, 500, 503):
                backoff = 5 * (2 ** attempt)
                print(f"  S2 {r.status_code} — backing off {backoff}s (attempt {attempt + 1}/3)")
                time.sleep(backoff)
                continue
            print(f"  S2 returned {r.status_code} for: {title[:60]}")
            return []
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            backoff = 5 * (2 ** attempt)
            print(f"  S2 {type(e).__name__} — retrying in {backoff}s (attempt {attempt + 1}/3)...")
            time.sleep(backoff)
            continue
        except Exception as e:
            print(f"  S2 error: {e}")
            return []
    print(f"  S2 failed after 3 attempts for: {title[:60]}")
    return []


def verify_references(references: list[dict], progress_callback=None) -> list[dict]:
    """Verify a list of references through the full cascade.

    Mutates and returns the same list, adding to each reference:
      * verification : the full notebook detail dict (label/source/reason/url/matched_candidate)
      * status       : "Verified" | "Uncertain" | "Unverified"  (== verification label)

    progress_callback (optional): called as progress_callback(ref, step) each
    time a reference reaches a terminal status, where `step` names the stage
    that resolved it. Purely observational — it never influences a decision.
    """
    # Fresh cost accounting per run (matches a fresh notebook execution).
    llm_usage["input_tokens"] = 0
    llm_usage["output_tokens"] = 0
    llm_usage["calls"] = 0

    for ref in references:
        ref["verification"] = {"label": "Pending", "source": None, "reason": None, "matched_candidate": None}

    # Observability: after each step, fire the callback for any reference that
    # newly left "Pending". Does not touch the verification flow.
    _emitted = set()

    def _emit(step):
        if progress_callback is None:
            return
        for ref in references:
            if ref["verification"]["label"] != "Pending" and id(ref) not in _emitted:
                _emitted.add(id(ref))
                progress_callback(ref, step)

    # -- Step 1: Website check ----------------------------------------------
    for ref in references:
        if ref.get("reference_type") == "website":
            url = ref.get("url") or ""
            if not url:
                ref["verification"] = {
                    "label": "Unverified", "source": "Web",
                    "reason": "No URL found.", "url": None,
                    "matched_candidate": None
                }
            elif "doi.org" in url.lower() or url.lower().startswith("10."):
                pass  # DOI URL — fall through to Step 2
            else:
                label, source, reason, candidate = check_website(url, ref_title=ref.get("title", ""))
                ref["verification"] = {
                    "label": label, "source": source,
                    "reason": reason, "url": url,
                    "matched_candidate": candidate
                }
            time.sleep(0.2)
    _emit("Website check")

    # -- Step 2: DOI check + arXiv URL check --------------------------------
    for ref in references:
        if ref["verification"]["label"] != "Pending":
            continue
        doi = ref.get("doi")
        url = ref.get("url") or ""
        label, source, checks, candidate = None, None, None, None

        if doi:
            label, source, checks, candidate = check_crossref(doi, ref)
            time.sleep(CR_DELAY)

            if label is None and "_doi_mismatch" not in ref["verification"]:
                label, source, checks, candidate = check_arxiv(doi, ref)

            if label is not None:
                ref["verification"] = {
                    "label":  label,
                    "source": source,
                    "checks": checks,
                    # DOI-resolved matches are not shown side-by-side — a DOI hit
                    # is treated as definitive on its own, no candidate snapshot.
                    "matched_candidate": None,
                    "url":    f"https://doi.org/{doi.strip().removeprefix('https://doi.org/')}"
                }

        elif "arxiv.org" in url.lower():
            # No DOI but URL points to arXiv — look up via S2 arXiv proxy.
            label, source, checks, candidate = check_arxiv(url, ref)
            if label is not None:
                ref["verification"] = {
                    "label":  label,
                    "source": source,
                    "checks": checks,
                    "matched_candidate": None,
                    "url":    url
                }

        # label is None + no _doi_mismatch -> not found -> stays Pending
        # label is None + _doi_mismatch set -> DOI found but title mismatch -> stays Pending
        # both fall through to title search cascade
    _emit("DOI check")

    # -- Step 3: CrossRef title search --------------------------------------
    for ref in references:
        if ref["verification"]["label"] != "Pending":
            continue
        candidates = search_crossref_title(ref.get("title", ""))
        result = llm_compare(ref, candidates)
        if result["match"]:
            checks = result.get("checks") or {}
            idx = (result.get("matched_candidate") or 1) - 1
            if not (0 <= idx < len(candidates)):
                idx = 0
            candidate_journal = candidates[idx].get("journal") if candidates else None
            matched_url = _get_matched_url(candidates, result.get("matched_candidate"))
            label, reason = _cascade_outcome(ref, checks, candidate_journal)
            label, reason = _apply_doi_mismatch_downgrade(ref, label, reason)
            ref["verification"] = {
                "label":  label,
                "source": "CrossRef_title",
                "checks": checks,
                "reason": reason,
                "matched_candidate": candidates[idx],
                "url":    matched_url
            }
    _emit("CrossRef title search")

    # -- Step 4: OpenAlex title search --------------------------------------
    for ref in references:
        if ref["verification"]["label"] != "Pending":
            continue
        time.sleep(1.0) 
        candidates = search_openalex(ref.get("title", ""))
        result = llm_compare(ref, candidates)
        if result["match"]:
            checks = result.get("checks") or {}
            idx = (result.get("matched_candidate") or 1) - 1
            if not (0 <= idx < len(candidates)):
                idx = 0
            candidate_journal = candidates[idx].get("journal") if candidates else None
            matched_url = _get_matched_url(candidates, result.get("matched_candidate"))
            label, reason = _cascade_outcome(ref, checks, candidate_journal)
            label, reason = _apply_doi_mismatch_downgrade(ref, label, reason)
            ref["verification"] = {
                "label":  label,
                "source": "OpenAlex_title",
                "checks": checks,
                "reason": reason,
                "matched_candidate": candidates[idx],
                "url":    matched_url
            }
    _emit("OpenAlex title search")

    # -- Step 5: Semantic Scholar title search ------------------------------
    for ref in references:
        if ref["verification"]["label"] != "Pending":
            continue
        time.sleep(1.5)
        candidates = search_semantic_scholar(ref.get("title", ""))
        result = llm_compare(ref, candidates)
        if result["match"]:
            checks = result.get("checks") or {}
            idx = (result.get("matched_candidate") or 1) - 1
            if not (0 <= idx < len(candidates)):
                idx = 0
            candidate_journal = candidates[idx].get("journal") if candidates else None
            matched_url = _get_matched_url(candidates, result.get("matched_candidate"))
            label, reason = _cascade_outcome(ref, checks, candidate_journal)
            label, reason = _apply_doi_mismatch_downgrade(ref, label, reason)
            ref["verification"] = {
                "label":  label,
                "source": "SemanticScholar_title",
                "checks": checks,
                "reason": reason,
                "matched_candidate": candidates[idx],
                "url":    matched_url
            }
    _emit("Semantic Scholar title search")

    # -- Step 6: Final resolution -------------------------------------------
    # Resolution order for still-Pending references:
    #   1. URL contains DOI -> Unverified (DOI failed all checks; note if the
    #      DOI resolved but its KB title differed from the reference title)
    #   2. Non-DOI URL present -> check_website(); Verified or Unverified (no Uncertain)
    #   3. Nothing -> Unverified
    for ref in references:
        if ref["verification"]["label"] != "Pending":
            ref["verification"].pop("_doi_mismatch", None)
            if "url" not in ref["verification"]:
                ref["verification"]["url"] = None
            if "matched_candidate" not in ref["verification"]:
                ref["verification"]["matched_candidate"] = None
            continue

        url = ref.get("url")
        doi_mismatch = ref["verification"].pop("_doi_mismatch", None)
        if url:
            is_doi_url = "doi.org" in url.lower() or url.lower().startswith("10.")
            if is_doi_url:
                if doi_mismatch:
                    unverified_reason = "DOI resolves but KB title differs from reference title; no title match found in cascade."
                else:
                    unverified_reason = "DOI URL did not resolve and no title match found in any knowledge base."
                ref["verification"] = {
                    "label":  "Unverified",
                    "source": None,
                    "reason": unverified_reason,
                    "matched_candidate": None,
                    "url":    url
                }
            else:
                label, _, reason, candidate = check_website(url, ref_title=ref.get("title", ""))
                if label == "Verified":
                    ref["verification"] = {
                        "label":  "Verified",
                        "source": "Web_fallback",
                        "reason": reason,
                        "matched_candidate": candidate,
                        "url":    url
                    }
                else:
                    ref["verification"] = {
                        "label":  "Unverified",
                        "source": "Web_fallback",
                        "reason": f"Not found in academic databases. URL present ({reason}).",
                        "matched_candidate": candidate,
                        "url":    url
                    }
            time.sleep(0.2)
            continue

        ref["verification"] = {
            "label":  "Unverified",
            "source": None,
            "reason": "No match found across all knowledge bases.",
            "matched_candidate": None,
            "url":    None
        }
    _emit("Final resolution")

    # Expose the requested top-level contract alongside the full detail dict.
    for ref in references:
        ref["status"] = ref["verification"]["label"]

    return references
