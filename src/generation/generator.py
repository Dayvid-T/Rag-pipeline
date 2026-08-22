"""
Generation layer: turn a user question + retrieved context into a grounded
answer using an LLM.

Phase: 2 (Build Core Retrieval Logic, Locally)
Why it exists: this is the "G" in RAG - it takes what retrieval found and
asks the model to answer using only that material, so answers stay
grounded in your actual documents instead of the model's memorized (and
possibly wrong or outdated) knowledge.

TODO (build-out steps):
  1. build_prompt(query, retrieved_chunks): assemble a prompt that clearly
     separates instructions, retrieved context, and the user's question.
  2. generate_answer(query, retrieved_chunks): call the LLM API and return
     the answer, ideally with the source chunks it used cited alongside it.
"""

from typing import List, Dict


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """Assemble the grounded prompt sent to the LLM. Not yet implemented."""
    raise NotImplementedError("Phase 2: implement prompt construction")


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> Dict:
    """Call the LLM and return {'answer': ..., 'sources': [...]}. Not yet implemented."""
    raise NotImplementedError("Phase 2: implement answer generation")
