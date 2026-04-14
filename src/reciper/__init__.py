"""Minimal companion implementation for the RECIPER paper."""

from .benchmark import main as benchmark_main
from .data import DEFAULT_CORPUS_PATH, DEFAULT_QA_PATH, load_jsonl, load_rag_documents
from .retrieval import (
    BM25Retriever,
    DEFAULT_DENSE_MODEL,
    DenseRetriever,
    RECIPERRetriever,
    RetrievalHit,
    build_retriever,
    collapse_to_paper_ids,
)

__all__ = [
    "BM25Retriever",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_QA_PATH",
    "DenseRetriever",
    "RECIPERRetriever",
    "RetrievalHit",
    "benchmark_main",
    "build_retriever",
    "collapse_to_paper_ids",
    "load_jsonl",
    "load_rag_documents",
]
