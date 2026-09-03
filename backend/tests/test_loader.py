"""
Tests for src.ingestion.loader.load_documents.

Deliberately does NOT depend on backend/data/ - that folder is gitignored
and developer-specific (it won't exist on a fresh checkout or in CI), so
every test builds its own throwaway files via pytest's built-in `tmp_path`
fixture instead. The .pdf tests fake out pypdf's PdfReader entirely: we're
testing *our* loop/accumulation/dict-building logic, not whether pypdf can
correctly parse a real PDF (that's pypdf's own test suite's job).
"""

from src.ingestion.loader import load_documents


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
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("page one. "), FakePage("page two.")]

    monkeypatch.setattr("src.ingestion.loader.PdfReader", FakeReader)

    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake content")

    result = load_documents(str(tmp_path))

    assert len(result) == 1
    assert result[0]["text"] == "page one. page two."
    assert result[0]["source"] == "report.pdf"


def test_load_documents_handles_mixed_files(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return "pdf text"

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage()]

    monkeypatch.setattr("src.ingestion.loader.PdfReader", FakeReader)

    (tmp_path / "a.txt").write_text("txt text")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake content")
    (tmp_path / "c.md").write_text("ignored")

    result = load_documents(str(tmp_path))

    sources = {d["source"] for d in result}
    assert sources == {"a.txt", "b.pdf"}
    assert len(result) == 2
