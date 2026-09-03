"""
Tests for src.ingestion.loader.chunk_documents.
"""

from src.ingestion.loader import chunk_documents


def test_chunk_documents_empty_list():
    assert chunk_documents([]) == []


def test_chunk_documents_short_text_stays_one_chunk():
    docs = [{"text": "a short document", "source": "notes.txt"}]

    result = chunk_documents(docs)

    assert len(result) == 1
    assert result[0]["text"] == "a short document"
    assert result[0]["source"] == "notes.txt"
    assert result[0]["chunk_id"] == "notes.txt::0"


def test_chunk_documents_splits_long_text():
    long_text = "word " * 300  # well over the 500-character chunk_size

    result = chunk_documents([{"text": long_text, "source": "big.txt"}])

    assert len(result) > 1
    assert all(c["source"] == "big.txt" for c in result)
    # chunk_ids are sequential per document, starting at 0
    assert [c["chunk_id"] for c in result] == [f"big.txt::{i}" for i in range(len(result))]


def test_chunk_documents_resets_chunk_id_per_document():
    long_text = "word " * 300
    docs = [
        {"text": long_text, "source": "one.txt"},
        {"text": long_text, "source": "two.txt"},
    ]

    result = chunk_documents(docs)

    first_doc_chunks = [c for c in result if c["source"] == "one.txt"]
    second_doc_chunks = [c for c in result if c["source"] == "two.txt"]

    assert first_doc_chunks[0]["chunk_id"] == "one.txt::0"
    assert second_doc_chunks[0]["chunk_id"] == "two.txt::0"
