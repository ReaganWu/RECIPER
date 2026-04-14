# RECIPER

Companion repository for the paper [RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering](https://arxiv.org/abs/2604.11229).

RECIPER is a retrieval pipeline for materials question answering. It indexes two complementary views of the same paper collection:

- paragraph-level evidence from the full paper text
- compact procedure-oriented recipe summaries

The public repository keeps the code path small and readable while preserving the main paper-facing ingredients: the RAG corpus, the QA dataset, dense backbones, and a minimal dual-view RECIPER retriever.

## Paper

- Title: `RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering`
- arXiv: `2604.11229`
- URL: `https://arxiv.org/abs/2604.11229`
- DOI: `https://doi.org/10.48550/arXiv.2604.11229`

Paper summary:
RECIPER targets procedure-oriented retrieval in materials science, where synthesis evidence is often distributed across long papers and is not always well captured by paragraph-only dense retrieval. The method combines paragraph retrieval and recipe-summary retrieval, followed by lightweight lexical reranking. In the paper, this dual-view setup improves early-rank retrieval over paragraph-only dense retrieval across multiple dense backbones.

Paper result snapshot:

- average gain of `+3.73` in `Recall@1`
- average gain of `+2.85` in `nDCG@10`
- average gain of `+3.13` in `MRR`
- with `BGE-large-en-v1.5`, the paper reports `86.82%` `Recall@1`, `97.07%` `Recall@5`, and `97.85%` `Recall@10`

## Repository contents

- `data/rag_database/rag_database.json`: structured paper corpus
- `data/rag_database/qa_dataset.jsonl`: QA dataset used for retrieval evaluation
- `src/reciper/demo.py`: interactive retrieval demo
- `src/reciper/benchmark.py`: minimal QA benchmark entrypoint
- `src/reciper/retrieval.py`: BM25, dense, and minimal RECIPER dual-view retrieval

This is a compact companion repo, not the full internal experimentation workspace. The goal is to make the paper artifact easy to read and easy to run.

## Dataset summary

- `343` papers in `rag_database.json`
- `12,162` paragraph chunks
- `544` recipe chunks
- `1,024` QA pairs

The retrieval code supports three views:

- `paragraph`
- `recipe`
- `combined`

The `reciper` method uses a dual-view setup over paragraph and recipe streams.

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

Run the paper-style dual-view retriever:

```bash
python -m src.reciper.demo \
  --query "self-healing coating corrosion" \
  --method reciper \
  --model-name bge-large-en-v1.5 \
  --top-k 5
```

Run paragraph-only dense retrieval:

```bash
python -m src.reciper.demo \
  --query "self-healing coating corrosion" \
  --method dense \
  --stream paragraph \
  --model-name bge-large-en-v1.5 \
  --top-k 5
```

Run a minimal QA benchmark:

```bash
python -m src.reciper.benchmark \
  --method reciper \
  --model-name bge-large-en-v1.5 \
  --limit 100
```

The benchmark reports `Recall@k`, `nDCG@k`, and `MRR` in percentage form over the provided QA pairs.

## Dense backbones

The public code path supports the main dense backbones used in this release:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`

Short aliases also work:

- `bge-small-en-v1.5`
- `bge-large-en-v1.5`
- `e5-large-v2`

For `e5-large-v2`, the expected `query:` and `passage:` prefixes are added automatically.

## Local smoke-test resource notes

The following numbers are local CPU smoke-test measurements intended only as rough operational reference. They are not paper result claims. The dense build time includes model loading and document encoding on `combined[:2048]`.

| Method / model | Corpus slice | Build time | Search time | Peak RSS memory |
| --- | --- | --- | --- | --- |
| BM25 | `combined[:2048]` | 0.052 s | 0.011 s | 729.3 MB |
| BAAI/bge-small-en-v1.5 | `combined[:2048]` | 14.820 s | 0.020 s | 5507.7 MB |
| BAAI/bge-large-en-v1.5 | `combined[:2048]` | 37.875 s | 0.022 s | 5483.0 MB |
| intfloat/e5-large-v2 | `combined[:2048]` | 24.729 s | 0.027 s | 5486.8 MB |

## Notes

- The first dense run downloads the requested backbone from HuggingFace.
- `reciper` in this repository is a compact reference implementation of the dual-view idea described in the paper.
- This repository focuses on retrieval and dataset release; it does not include the full original automation stack.

## Citation

```bibtex
@article{wu2026reciper,
  title={RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering},
  author={Wu, Zhuoyu and Ou, Wenhui and Tan, Pei-Sze and Fang, Wenqi and Rajanala, Sailaja and Phan, Rapha{\"e}l C.-W.},
  journal={arXiv preprint arXiv:2604.11229},
  year={2026},
  doi={10.48550/arXiv.2604.11229},
  url={https://arxiv.org/abs/2604.11229}
}
```
