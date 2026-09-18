"""
Tests for src.citation.citation. The LLM is mocked; formatting is pure.
"""

import pytest

import src.citation.citation as citation_module
from src.citation.citation import (
    cite,
    format_apa,
    format_ieee,
    initials,
    normalize_metadata,
    parse_extraction,
)


def _meta(**overrides):
    base = {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": [
            {"given": "Kaiming", "family": "He"},
            {"given": "Xiangyu", "family": "Zhang"},
        ],
        "year": 2016,
        "venue": "IEEE Conference on Computer Vision and Pattern Recognition",
        "document_type": "article",
        "url": None,
        "doi": None,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def clear_cache():
    citation_module._cache.clear()


# --- initials -----------------------------------------------------------------

def test_initials():
    assert initials("John Michael") == "J. M."
    assert initials("Jean-Luc") == "J.-L."
    assert initials("") == ""


# --- IEEE ---------------------------------------------------------------------

def test_ieee_two_authors():
    assert format_ieee(_meta()) == (
        '[1] K. He and X. Zhang, "Deep Residual Learning for Image Recognition," '
        "IEEE Conference on Computer Vision and Pattern Recognition, 2016."
    )


def test_ieee_three_authors_uses_serial_and():
    meta = _meta(authors=_meta()["authors"] + [{"given": "Shaoqing", "family": "Ren"}])
    assert format_ieee(meta).startswith("[1] K. He, X. Zhang, and S. Ren, ")


def test_ieee_more_than_six_authors_et_al():
    meta = _meta(authors=[{"given": f"A{i}", "family": f"Author{i}"} for i in range(7)])
    assert format_ieee(meta).startswith("[1] A. Author0 et al., ")


def test_ieee_no_authors_no_year():
    meta = _meta(authors=[], year=None, venue=None)
    assert format_ieee(meta) == '[1] "Deep Residual Learning for Image Recognition."'


def test_ieee_with_doi_and_url_prefers_doi():
    meta = _meta(doi="10.1109/CVPR.2016.90", url="https://example.com")
    assert format_ieee(meta).endswith("2016. doi: 10.1109/CVPR.2016.90.")


def test_ieee_with_url_only():
    meta = _meta(url="https://example.com/paper.pdf")
    assert format_ieee(meta).endswith("2016. [Online]. Available: https://example.com/paper.pdf")


# --- APA ----------------------------------------------------------------------

def test_apa_two_authors():
    assert format_apa(_meta()) == (
        "He, K., & Zhang, X. (2016). Deep Residual Learning for Image Recognition. "
        "IEEE Conference on Computer Vision and Pattern Recognition."
    )


def test_apa_single_author_no_year():
    meta = _meta(authors=[{"given": "Kaiming", "family": "He"}], year=None, venue=None)
    assert format_apa(meta) == "He, K. (n.d.). Deep Residual Learning for Image Recognition."


def test_apa_no_authors_title_first():
    meta = _meta(authors=[], venue="University of Victoria")
    assert format_apa(meta) == (
        "Deep Residual Learning for Image Recognition. (2016). University of Victoria."
    )


def test_apa_doi_becomes_link():
    meta = _meta(doi="10.1109/CVPR.2016.90")
    assert format_apa(meta).endswith("https://doi.org/10.1109/CVPR.2016.90")


def test_apa_title_keeps_question_mark():
    meta = _meta(title="Is Attention All You Need?", authors=[], venue=None)
    assert format_apa(meta) == "Is Attention All You Need? (2016)."


# --- extraction parsing ---------------------------------------------------------

def test_parse_extraction_strips_fence_and_normalizes():
    text = (
        '```json\n{"title": "  ECE 485  Assignment 5 ", "authors": [{"given": "A", "family": "B"}, '
        '"junk"], "year": "2026", "venue": null, "document_type": "course_material"}\n```'
    )
    meta = parse_extraction(text, "Assignment5.pdf")
    assert meta["title"] == "ECE 485 Assignment 5"
    assert meta["authors"] == [{"given": "A", "family": "B"}]
    assert meta["year"] == 2026
    assert meta["venue"] is None
    assert meta["document_type"] == "course_material"


def test_normalize_falls_back_to_filename_title():
    meta = normalize_metadata({"title": None, "year": 99}, "_my_notes.pdf")
    assert meta["title"] == "my notes"
    assert meta["year"] is None
    assert meta["authors"] == []


def test_parse_extraction_rejects_non_json():
    with pytest.raises(ValueError):
        parse_extraction("not json", "x.pdf")


# --- cite -----------------------------------------------------------------------

def test_cite_returns_none_for_unknown_source(monkeypatch):
    monkeypatch.setattr(citation_module, "get_source_text", lambda s, max_chars=None: "")
    assert cite("missing.pdf") is None


def test_cite_extracts_once_and_caches(monkeypatch):
    calls = []

    def fake_extract(source, text):
        calls.append((source, text))
        return _meta()

    monkeypatch.setattr(citation_module, "get_source_text", lambda s, max_chars=None: "some text")
    monkeypatch.setattr(citation_module, "extract_metadata", fake_extract)

    first = cite("paper.pdf")
    second = cite("paper.pdf")

    assert first is second
    assert calls == [("paper.pdf", "some text")]
    assert first["ieee"].startswith("[1] K. He and X. Zhang")
    assert first["apa"].startswith("He, K., & Zhang, X. (2016)")
