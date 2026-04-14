from __future__ import annotations

import argparse

from .data import DEFAULT_CORPUS_PATH, build_snippet, load_rag_documents
from .retrieval import DEFAULT_DENSE_MODEL, build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal RECIPER retrieval demo.")
    parser.add_argument("--query", required=True, help="Query text for retrieval.")
    parser.add_argument("--method", choices=["bm25", "dense", "reciper"], default="reciper")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stream", choices=["combined", "paragraph", "recipe"], default="combined")
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--model-name", default=DEFAULT_DENSE_MODEL)
    args = parser.parse_args()

    if args.method == "reciper":
        paragraph_documents = load_rag_documents(args.corpus_path, stream="paragraph")
        recipe_documents = load_rag_documents(args.corpus_path, stream="recipe")
        retriever = build_retriever(
            "reciper",
            model_name=args.model_name,
            paragraph_documents=paragraph_documents,
            recipe_documents=recipe_documents,
        )
        document_count = len(paragraph_documents) + len(recipe_documents)
        stream_name = "dual-view"
    else:
        documents = load_rag_documents(args.corpus_path, stream=args.stream)
        retriever = build_retriever(args.method, documents=documents, model_name=args.model_name)
        document_count = len(documents)
        stream_name = args.stream

    hits = retriever.search(args.query, top_k=args.top_k)

    print(f"Loaded {document_count} documents from {args.corpus_path}")
    print(f"Stream: {stream_name}")
    print(f"Method: {args.method}")
    if args.method != "bm25":
        print(f"Model: {args.model_name}")
    print(f"Query: {args.query}")
    print("")

    for rank, hit in enumerate(hits, start=1):
        print(f"[{rank}] {hit.title or hit.doc_id}")
        print(f"score={hit.score:.4f}")
        if hit.metadata.get("paper_id"):
            print(f"paper_id={hit.metadata['paper_id']}")
        if hit.metadata.get("stream"):
            print(f"stream={hit.metadata['stream']}")
        if hit.metadata.get("heading"):
            print(f"heading={hit.metadata['heading']}")
        if hit.metadata.get("doi"):
            print(f"doi={hit.metadata['doi']}")
        elif hit.metadata.get("arxiv_id"):
            print(f"arxiv_id={hit.metadata['arxiv_id']}")
        print(build_snippet(hit.text))
        print("")


if __name__ == "__main__":
    main()
