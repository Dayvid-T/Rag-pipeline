"""
Tests for src.ingestion.loader.

Deliberately does NOT depend on backend/data/ - that folder is gitignored
and developer-specific (it won't exist on a fresh checkout or in CI), so
every test builds its own throwaway files via pytest's built-in `tmp_path`
fixture instead. The .pdf tests fake out PyMuPDF entirely: we're testing
*our* loop/accumulation/dict-building logic, not whether PyMuPDF can
correctly parse a real PDF (that's its own test suite's job).
"""

import pytest

from src.ingestion.loader import load_documents, parse_document


class FakePage:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text


class FakePdf:
    def __init__(self, pages):
        self._pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._pages)


def _fake_pdf_open(monkeypatch, texts):
    monkeypatch.setattr(
        "src.ingestion.loader.pymupdf.open",
        lambda *args, **kwargs: FakePdf([FakePage(t) for t in texts]),
    )


# --- parse_document ---------------------------------------------------------

def test_parse_document_txt_decodes_utf8():
    result = parse_document("math.txt", "αi ≥ 0".encode("utf-8"))
    assert result == {"text": "αi ≥ 0", "source": "math.txt"}


def test_parse_document_pdf_joins_pages(monkeypatch):
    _fake_pdf_open(monkeypatch, ["page one. ", "page two."])

    result = parse_document("report.PDF", b"%PDF-1.4 fake content")

    assert result == {"text": "page one. page two.", "source": "report.PDF"}


def test_parse_document_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document("image.png", b"\x89PNG")


def test_parse_document_wraps_pdf_parse_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("cannot open broken document")

    monkeypatch.setattr("src.ingestion.loader.pymupdf.open", broken)

    with pytest.raises(ValueError, match="Could not read PDF"):
        parse_document("broken.pdf", b"not a pdf")


# --- load_documents ---------------------------------------------------------

def test_load_documents_empty_folder(tmp_path):
    """An empty directory should return an empty list, not raise."""
    result = load_documents(str(tmp_path))
    assert result == []


def test_load_documents_reads_txt_file(tmp_path):
    (tmp_path / "notes.txt").write_text("hello from a test file")

    result = load_documents(str(tmp_path))

    assert len(result) == 1
    assert result[0]["text"] == "hello from a test file"
    assert result[0]["source"] == "notes.txt"


def test_load_documents_ignores_other_extensions(tmp_path):
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "README.md").write_text("skip me")
    (tmp_path / ".gitkeep").write_text("")

    result = load_documents(str(tmp_path))

    assert len(result) == 1
    assert result[0]["source"] == "notes.txt"


def test_load_documents_reads_pdf_file(tmp_path, monkeypatch):
    _fake_pdf_open(monkeypatch, ["page one. ", "page two."])
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake content")

    result = load_documents(str(tmp_path))

    assert len(result) == 1
    assert result[0]["text"] == "page one. page two."
    assert result[0]["source"] == "report.pdf"


def test_load_documents_handles_mixed_files(tmp_path, monkeypatch):
    _fake_pdf_open(monkeypatch, ["pdf text"])
    (tmp_path / "a.txt").write_text("txt text")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake content")
    (tmp_path / "c.md").write_text("ignored")

    result = load_documents(str(tmp_path))

    sources = {d["source"] for d in result}
    assert sources == {"a.txt", "b.pdf"}
    assert len(result) == 2
