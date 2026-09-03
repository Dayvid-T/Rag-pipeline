"""
Document ingestion: load source documents and split them into retrievable
chunks.

Phase: 2 (Build Core Retrieval Logic, Locally)
Why it exists: unstructured documents (PDFs, text files) are too large to
hand to an embedding model or an LLM whole. This module is responsible for
turning raw files in data/ into a list of clean, appropriately-sized text
chunks, ready for embedding.

TODO (build-out steps):
  1. load_documents(path): read all files from a directory (start with .pdf
     and .txt).
  2. chunk_documents(docs): split into overlapping chunks (try ~500 tokens
     with ~50 token overlap as a starting point, then tune).
  3. Return a list of dicts: {"text": ..., "source": ..., "chunk_id": ...}
     so retrieval results can always be traced back to their source file.
"""

from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader




def load_documents(path: str) -> List[Dict]:

    folder = Path(path)
    documents = []
    for f in folder.iterdir():
        if f.suffix.lower() == ".txt":
            #this is the palce holder
            documents.append({"text": f.read_text(), "source": f.name})

        elif f.suffix.lower() == ".pdf":
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append({"text": text, "source": f.name})
    return documents


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Split loaded documents into retrievable chunks. Not yet implemented."""
    raise NotImplementedError("Phase 2: implement chunking")
