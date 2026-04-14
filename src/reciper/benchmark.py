from __future__ import annotations

import argparse
import json

from .data import DEFAULT_CORPUS_PATH, DEFAULT_QA_PATH, load_jsonl, load_rag_documents
from .metrics import mrr, ndcg_at_k, recall_at_k
from .retrieval import DEFAULT_DENSE_MODEL, build_retriever, collapse_to_paper_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal RECIPER retrieval benchmark.")
    parser.add_argument("--method", choices=["bm25", "dense", "reciper"], default="reciper")
    parser.add_argument("--stream", choices=["combined", "paragraph", "recipe"], default="combined")
    parser.add_argument("--model-name", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS_PATH))
    parser.add_argument("--qa-path", default=str(DEFAULT_QA_PATH))
    parser.add_argument("--limit", type=int, default=0, help="Optional number of QA rows to evaluate.")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    qa_rows = load_jsonl(args.qa_path)
    if args.limit > 0:
        qa_rows = qa_rows[: args.limit]

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
    else:
        documents = load_rag_documents(args.corpus_path, stream=args.stream)
        retriever = build_retriever(args.method, documents=documents, model_name=args.model_name)
        document_count = len(documents)

    metrics = {f"Recall@{k}": 0.0 for k in args.top_k}
    metrics.update({f"nDCG@{k}": 0.0 for k in args.top_k})
    metrics["MRR"] = 0.0

    for row in qa_rows:
        hits = retriever.search(row["question"], top_k=max(args.top_k))
        ranked_ids = collapse_to_paper_ids(hits)
        relevant_ids = [row["paper_id"]]
        for k in args.top_k:
            metrics[f"Recall@{k}"] += recall_at_k(ranked_ids, relevant_ids, k)
            metrics[f"nDCG@{k}"] += ndcg_at_k(ranked_ids, relevant_ids, k)
        metrics["MRR"] += mrr(ranked_ids, relevant_ids)

    total = max(len(qa_rows), 1)
    for key in metrics:
        metrics[key] = round(metrics[key] / total * 100.0, 2)

    payload = {
        "method": args.method,
        "model_name": args.model_name if args.method != "bm25" else "bm25",
        "stream": args.stream if args.method != "reciper" else "dual-view",
        "document_count": document_count,
        "qa_count": len(qa_rows),
        "metrics_percent": metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
