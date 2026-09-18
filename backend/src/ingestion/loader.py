from typing import List, Dict
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def parse_document(name: str, data: bytes) -> Dict:
    """Turn raw file bytes into {'text', 'source'}; raises ValueError if unusable."""
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        text = data.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                text = "".join(page.get_text() for page in pdf)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Could not read PDF '{name}': {e}") from e
    else:
        raise ValueError(f"Unsupported file type '{suffix}'; only .pdf and .txt are supported")
    return {"text": text, "source": name}


def load_documents(path: str) -> List[Dict]:
    folder = Path(path)
    return [
        parse_document(f.name, f.read_bytes())
        for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Split loaded documents into retrievable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}::{i}",
            })
    return chunks
