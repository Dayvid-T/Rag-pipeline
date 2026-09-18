"""
Reference generation: extract bibliographic metadata from a document with
the LLM, then format it as IEEE / APA with plain string rules so the output
is consistent and unit-testable.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from google.genai import types

from src.generation.generator import _get_client, generate_with_retry
from src.retrieval.hybrid_search import get_source_text

# How much of the document the extractor sees. Bibliographic details live
# at the top of almost every document; this keeps the call cheap.
EXTRACT_CHARS = 6000

_cache: Dict[str, Dict] = {}


def forget(source: str) -> None:
    _cache.pop(source, None)


# --- metadata extraction ----------------------------------------------------

def build_extraction_prompt(source: str, text: str) -> str:
    return (
        "Extract bibliographic metadata for the document below so it can be "
        "cited. Return ONLY a JSON object with these keys:\n"
        '  "title": the document\'s title. If none is stated, write a short '
        "descriptive title from its content (e.g. \"ECE 485 Assignment 5\").\n"
        '  "authors": list of {"given": "...", "family": "..."} for named '
        "authors, in order. Empty list if none are stated - never guess.\n"
        '  "year": 4-digit publication year as an integer, or null if unknown.\n'
        '  "venue": journal, publisher, institution, course, or website name; '
        "null if unknown.\n"
        '  "document_type": one of "article", "book", "report", "thesis", '
        '"course_material", "webpage", "other".\n'
        '  "url": a URL if one is stated, else null.\n'
        '  "doi": a DOI if one is stated, else null.\n\n'
        f"Filename: {source}\n\n"
        f"Document text (beginning):\n{text}\n\n"
        "JSON:"
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _clean_str(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def normalize_metadata(raw: Dict, source: str) -> Dict:
    """Coerce the LLM's JSON into a predictable shape with safe fallbacks."""
    authors = []
    for author in raw.get("authors") or []:
        if not isinstance(author, dict):
            continue
        given = _clean_str(author.get("given")) or ""
        family = _clean_str(author.get("family")) or ""
        if family or given:
            authors.append({"given": given, "family": family})

    year = raw.get("year")
    if isinstance(year, str) and year.strip().isdigit():
        year = int(year)
    if not isinstance(year, int) or not 1000 <= year <= 2999:
        year = None

    title = _clean_str(raw.get("title")) or Path(source).stem.replace("_", " ").strip() or source

    return {
        "title": title.rstrip("."),
        "authors": authors,
        "year": year,
        "venue": _clean_str(raw.get("venue")),
        "document_type": _clean_str(raw.get("document_type")) or "other",
        "url": _clean_str(raw.get("url")),
        "doi": _clean_str(raw.get("doi")),
    }


def parse_extraction(text: str, source: str) -> Dict:
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Metadata extractor returned non-JSON output: {text!r}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata extractor returned non-object JSON: {text!r}")
    return normalize_metadata(raw, source)


def extract_metadata(source: str, text: str) -> Dict:
    response = generate_with_retry(
        _get_client(),
        build_extraction_prompt(source, text[:EXTRACT_CHARS]),
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    return parse_extraction(response.text, source)


# --- formatting -----------------------------------------------------------------

def initials(given: str) -> str:
    """'John Michael' -> 'J. M.'; handles hyphenated names ('Jean-Luc' -> 'J.-L.')."""
    parts = []
    for word in given.split():
        letters = [p[0].upper() + "." for p in word.split("-") if p]
        parts.append("-".join(letters))
    return " ".join(parts)


def _ieee_author(author: Dict) -> str:
    ini = initials(author["given"])
    return f"{ini} {author['family']}".strip() if ini else author["family"]


def _apa_author(author: Dict) -> str:
    ini = initials(author["given"])
    if author["family"] and ini:
        return f"{author['family']}, {ini}"
    return author["family"] or ini


def _join_ieee_authors(authors: List[Dict]) -> str:
    names = [_ieee_author(a) for a in authors]
    if len(names) > 6:
        return f"{names[0]} et al."
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _join_apa_authors(authors: List[Dict]) -> str:
    names = [_apa_author(a) for a in authors]
    if len(names) == 1:
        return names[0]
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", . . . " + names[-1]


def format_ieee(meta: Dict) -> str:
    parts = []
    if meta["authors"]:
        parts.append(_join_ieee_authors(meta["authors"]))
    parts.append(f'"{meta["title"]},"')
    if meta.get("venue"):
        parts.append(meta["venue"])
    if meta.get("year"):
        parts.append(str(meta["year"]))

    text = ", ".join(parts)
    # The title segment already ends with a comma inside the quotes, so
    # collapse the `," ,` the join produced.
    text = text.replace(',", ', '," ')
    if text.endswith(',"'):
        text = text[:-2] + '."'
    else:
        text += "."

    if meta.get("doi"):
        text += f" doi: {meta['doi']}."
    elif meta.get("url"):
        text += f" [Online]. Available: {meta['url']}"
    return "[1] " + text


def format_apa(meta: Dict) -> str:
    year = f"({meta['year']})." if meta.get("year") else "(n.d.)."
    title = meta["title"]
    if not title.endswith(("?", "!", ".")):
        title += "."

    if meta["authors"]:
        head = f"{_join_apa_authors(meta['authors'])} {year} {title}"
    else:
        head = f"{title} {year}"

    tail = []
    if meta.get("venue"):
        venue = meta["venue"]
        tail.append(venue if venue.endswith(".") else venue + ".")
    if meta.get("doi"):
        doi = meta["doi"]
        tail.append(doi if doi.startswith("http") else f"https://doi.org/{doi}")
    elif meta.get("url"):
        tail.append(meta["url"])

    return " ".join([head, *tail])


# --- public entry point ------------------------------------------------------------

def cite(source: str) -> Optional[Dict]:
    """Return {'source', 'metadata', 'ieee', 'apa'} or None if not indexed."""
    if source in _cache:
        return _cache[source]

    text = get_source_text(source, max_chars=EXTRACT_CHARS)
    if not text:
        return None

    meta = extract_metadata(source, text)
    result = {
        "source": source,
        "metadata": meta,
        "ieee": format_ieee(meta),
        "apa": format_apa(meta),
    }
    _cache[source] = result
    return result
