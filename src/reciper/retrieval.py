from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Any

import numpy as np

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None


TOKEN_PATTERN = re.compile(r"\b\w+\b")
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_ALIASES = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "e5-large-v2": "intfloat/e5-large-v2",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def resolve_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def uses_e5_format(model_name: str) -> bool:
    return "e5" in model_name.lower()


@dataclass
class RetrievalHit:
    doc_id: str
    title: str
    score: float
    text: str
    metadata: dict[str, Any]


def collapse_to_paper_ids(hits: list[RetrievalHit]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        paper_id = str(hit.metadata.get("paper_id", "")).strip()
        if paper_id and paper_id not in seen:
            ranked.append(paper_id)
            seen.add(paper_id)
    return ranked


class DenseEncoder:
    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for dense retrieval.")
        self.model_name = resolve_model_name(model_name)
        try:
            self.encoder = SentenceTransformer(self.model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load HuggingFace model '{self.model_name}'. "
                "Download it once with network access or use BM25 instead."
            ) from exc

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.encoder.encode(
                [self._format_passage(text) for text in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self.encoder.encode(
                [self._format_query(query)],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
        )

    def _format_passage(self, text: str) -> str:
        if uses_e5_format(self.model_name):
            return f"passage: {text}"
        return text

    def _format_query(self, query: str) -> str:
        if uses_e5_format(self.model_name):
            return f"query: {query}"
        return query


class BM25Retriever:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.tokens = [tokenize(doc["text"]) for doc in documents]
        if BM25Okapi is not None:
            self.engine = BM25Okapi(self.tokens)
        else:
            self.engine = None
            self.doc_freq = Counter()
            for terms in self.tokens:
                self.doc_freq.update(set(terms))
            self.avgdl = sum(len(terms) for terms in self.tokens) / max(len(self.tokens), 1)

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
        n_docs = max(len(self.tokens), 1)
        scores: list[float] = []
        for terms in self.tokens:
            term_counts = Counter(terms)
            doc_len = len(terms) or 1
            score = 0.0
            for token in query_tokens:
                freq = term_counts[token]
                if not freq:
                    continue
                idf = log((n_docs - self.doc_freq[token] + 0.5) / (self.doc_freq[token] + 0.5) + 1.0)
                denom = freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(self.avgdl, 1e-6))
                score += idf * ((freq * 2.5) / denom)
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        scores = self.engine.get_scores(query_tokens) if self.engine is not None else self._fallback_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [self._to_hit(index, float(score)) for index, score in ranked]

    def _to_hit(self, index: int, score: float) -> RetrievalHit:
        document = self.documents[index]
        return RetrievalHit(
            doc_id=document["doc_id"],
            title=document["title"],
            score=score,
            text=document["text"],
            metadata=document["metadata"],
        )


class DenseRetriever:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        model_name: str = DEFAULT_DENSE_MODEL,
        encoder: DenseEncoder | None = None,
    ):
        self.documents = documents
        # The public companion repository keeps the implementation compact,
        # but these are the dense backbones used in the paper-facing release:
        # - BAAI/bge-small-en-v1.5
        # - BAAI/bge-large-en-v1.5
        # - intfloat/e5-large-v2
        self.encoder = encoder or DenseEncoder(model_name)
        self.model_name = self.encoder.model_name
        self.doc_embeddings = self.encoder.encode_passages([document["text"] for document in documents])

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        query_embedding = self.encoder.encode_query(query)
        scores = self.doc_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [self._to_hit(int(index), float(scores[index])) for index in ranked_indices]

    def _to_hit(self, index: int, score: float) -> RetrievalHit:
        document = self.documents[index]
        return RetrievalHit(
            doc_id=document["doc_id"],
            title=document["title"],
            score=score,
            text=document["text"],
            metadata=document["metadata"],
        )


class RECIPERRetriever:
    def __init__(
        self,
        paragraph_documents: list[dict[str, Any]],
        recipe_documents: list[dict[str, Any]],
        model_name: str = DEFAULT_DENSE_MODEL,
    ):
        self.encoder = DenseEncoder(model_name)
        self.model_name = self.encoder.model_name
        self.paragraph_retriever = DenseRetriever(paragraph_documents, encoder=self.encoder)
        self.recipe_retriever = DenseRetriever(recipe_documents, encoder=self.encoder)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        candidate_k = max(top_k * 4, 20)
        paragraph_hits = self.paragraph_retriever.search(query, top_k=candidate_k)
        recipe_hits = self.recipe_retriever.search(query, top_k=candidate_k)
        return self._fuse_and_rerank(query, paragraph_hits, recipe_hits, top_k=top_k)

    def _fuse_and_rerank(
        self,
        query: str,
        paragraph_hits: list[RetrievalHit],
        recipe_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        fused_scores: dict[str, float] = {}
        best_hit_by_paper: dict[str, RetrievalHit] = {}
        query_terms = set(tokenize(query))

        for rank, hit in enumerate(paragraph_hits, start=1):
            paper_id = str(hit.metadata.get("paper_id", "")).strip()
            fused_scores[paper_id] = fused_scores.get(paper_id, 0.0) + 0.65 / (60 + rank)
            best_hit_by_paper[paper_id] = self._pick_representative(best_hit_by_paper.get(paper_id), hit)

        for rank, hit in enumerate(recipe_hits, start=1):
            paper_id = str(hit.metadata.get("paper_id", "")).strip()
            fused_scores[paper_id] = fused_scores.get(paper_id, 0.0) + 0.35 / (60 + rank)
            best_hit_by_paper[paper_id] = self._pick_representative(best_hit_by_paper.get(paper_id), hit)

        reranked: list[RetrievalHit] = []
        for paper_id, fused_score in fused_scores.items():
            hit = best_hit_by_paper[paper_id]
            lexical_bonus = self._lexical_bonus(query_terms, hit)
            reranked.append(
                RetrievalHit(
                    doc_id=hit.doc_id,
                    title=hit.title,
                    score=fused_score + lexical_bonus,
                    text=hit.text,
                    metadata=hit.metadata,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def _pick_representative(self, current: RetrievalHit | None, candidate: RetrievalHit) -> RetrievalHit:
        if current is None:
            return candidate
        return candidate if candidate.score > current.score else current

    def _lexical_bonus(self, query_terms: set[str], hit: RetrievalHit) -> float:
        title_terms = set(tokenize(hit.title))
        text_terms = set(tokenize(hit.text))
        overlap = len(query_terms & (title_terms | text_terms))
        title_overlap = len(query_terms & title_terms)
        return overlap * 0.01 + title_overlap * 0.02


def build_retriever(
    method: str,
    documents: list[dict[str, Any]] | None = None,
    model_name: str = DEFAULT_DENSE_MODEL,
    paragraph_documents: list[dict[str, Any]] | None = None,
    recipe_documents: list[dict[str, Any]] | None = None,
) -> BM25Retriever | DenseRetriever | RECIPERRetriever:
    if method == "bm25":
        if documents is None:
            raise ValueError("documents are required for BM25 retrieval")
        return BM25Retriever(documents)
    if method == "dense":
        if documents is None:
            raise ValueError("documents are required for dense retrieval")
        return DenseRetriever(documents, model_name=model_name)
    if method == "reciper":
        if paragraph_documents is None or recipe_documents is None:
            raise ValueError("paragraph and recipe documents are required for RECIPER retrieval")
        return RECIPERRetriever(
            paragraph_documents=paragraph_documents,
            recipe_documents=recipe_documents,
            model_name=model_name,
        )
    raise ValueError(f"Unsupported retrieval method: {method}")
