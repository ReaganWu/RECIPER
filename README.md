---
license: mit
language:
  - en
task_categories:
  - question-answering
pretty_name: RECIPER
size_categories:
  - 1K<n<10K
tags:
  - materials-science
  - information-retrieval
  - retrieval-augmented-generation
  - scientific-literature
  - procedure-oriented-qa
---

# RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering

[![Paper](https://img.shields.io/badge/Paper-arXiv%202604.11229-blue)](https://arxiv.org/abs/2604.11229)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ReaganWZY/RECIPER)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official dataset and reference implementation of RECIPER**

> **RECIPER: A Dual-View Retrieval Pipeline for Procedure-Oriented Materials Question Answering**
>
> Zhuoyu Wu, Wenhui Ou, Pei-Sze Tan, Wenqi Fang, Sailaja Rajanala, and Raphaël C.-W. Phan

RECIPER is a retrieval pipeline for procedure-oriented materials question answering. It indexes two complementary views of the same scientific paper collection:

1. paragraph-level evidence from full paper text
2. compact procedure-oriented recipe summaries

The public release keeps the code path small and readable while preserving the main paper-facing artifacts: the RAG corpus, the QA dataset, dense retrieval backbones, and a minimal dual-view RECIPER retriever.

## Highlights

- Dual-view retrieval over paragraph and recipe streams
- Procedure-oriented materials science QA benchmark with 1,024 questions
- Structured RAG corpus with 343 papers, 12,162 paragraph chunks, and 544 recipe chunks
- Compact BM25, dense, and RECIPER retrieval implementations
- Hugging Face dataset release for direct reuse
- Paper-aligned evaluation entrypoint with `Recall@k`, `nDCG@k`, and `MRR`

Paper-reported reference numbers:

- Average gain of `+3.73` in `Recall@1`
- Average gain of `+2.85` in `nDCG@10`
- Average gain of `+3.13` in `MRR`
- With `BGE-large-en-v1.5`: `86.82%` `Recall@1`, `97.07%` `Recall@5`, and `97.85%` `Recall@10`

## Dataset

The release contains a structured materials-science retrieval corpus and a QA benchmark.

| File | Format | Rows | Description |
| --- | --- | ---: | --- |
| `data/rag_database/rag_database.json` | JSON array | 343 papers | Paper metadata, section paragraphs, extracted entities, and procedure-oriented recipe summaries |
| `data/rag_database/qa_dataset.jsonl` | JSON Lines | 1,024 QA pairs | Retrieval evaluation questions with answers and gold `paper_id` labels |

Dataset summary:

- Papers: `343`
- Sections: `2,763`
- Paragraph chunks: `12,162`
- Recipe chunks: `544`
- QA pairs: `1,024`
- Language: English
- Domain: materials science literature

The retrieval code supports three document streams:

- `paragraph`
- `recipe`
- `combined`

The `reciper` method uses a dual-view setup over paragraph and recipe streams.

## Hugging Face Usage

Load the released dataset directly from Hugging Face:

```python
from datasets import load_dataset

repo_id = "ReaganWZY/RECIPER"

qa = load_dataset(
    "json",
    data_files=f"hf://datasets/{repo_id}/data/rag_database/qa_dataset.jsonl",
    split="train",
)

corpus = load_dataset(
    "json",
    data_files=f"hf://datasets/{repo_id}/data/rag_database/rag_database.json",
    split="train",
)

print(qa[0])
print(corpus[0]["paper_id"])
```

Dataset page:

```text
https://huggingface.co/datasets/ReaganWZY/RECIPER
```

## Installation

```bash
git clone https://github.com/ReaganWu/RECIPER.git
cd RECIPER
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

## Model Backbones

The public code path supports the main dense backbones used in this release:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-large-en-v1.5`
- `intfloat/e5-large-v2`

Short aliases also work:

- `bge-small-en-v1.5`
- `bge-large-en-v1.5`
- `e5-large-v2`

For `e5-large-v2`, the expected `query:` and `passage:` prefixes are added automatically.

## Data Schema

Each paper record in `rag_database.json` contains:

- `paper_id`: stable local paper identifier, such as `paper_0237`
- `title`: paper title
- `abstract`: paper abstract
- `metadata`: source metadata, including authors, DOI or arXiv identifier when available, source URL fields, and download timestamp
- `sections`: paper sections with `heading` and `paragraphs_with_entities`
- `recipes`: procedure-oriented synthesis or experiment summaries used by the recipe retrieval stream

Each QA row in `qa_dataset.jsonl` contains:

- `question`: natural-language materials science question
- `answer`: reference answer
- `topic`: broad question topic, such as `synthesis`, `properties`, or `characterization`
- `paper_id`: gold source paper identifier in `rag_database.json`
- `paper_title`: title of the gold source paper
- `source`: DOI or arXiv source URL
- `source_type`: `doi` or `arxiv`

For local Python usage:

```python
from src.reciper.data import load_json, load_jsonl, load_rag_documents

corpus = load_json("data/rag_database/rag_database.json")
qa = load_jsonl("data/rag_database/qa_dataset.jsonl")
documents = load_rag_documents(stream="combined")
```

## Repository Structure

```text
RECIPER/
  data/rag_database/
    rag_database.json      # structured retrieval corpus
    qa_dataset.jsonl       # paper-level QA benchmark
  src/reciper/
    data.py                # dataset loading and document construction
    retrieval.py           # BM25, dense, and RECIPER retrievers
    demo.py                # interactive retrieval demo
    benchmark.py           # compact QA benchmark
    metrics.py             # Recall, nDCG, and MRR
  CITATION.cff
  LICENSE
  README.md
  requirements.txt
```

## Local Smoke-Test Notes

The following numbers are local CPU smoke-test measurements intended only as rough operational reference. They are not paper result claims. The dense build time includes model loading and document encoding on `combined[:2048]`.

| Method / model | Corpus slice | Build time | Search time | Peak RSS memory |
| --- | --- | --- | --- | --- |
| BM25 | `combined[:2048]` | 0.052 s | 0.011 s | 729.3 MB |
| BAAI/bge-small-en-v1.5 | `combined[:2048]` | 14.820 s | 0.020 s | 5507.7 MB |
| BAAI/bge-large-en-v1.5 | `combined[:2048]` | 37.875 s | 0.022 s | 5483.0 MB |
| intfloat/e5-large-v2 | `combined[:2048]` | 24.729 s | 0.027 s | 5486.8 MB |

## Intended Use and Limitations

RECIPER is intended for research on retrieval, retrieval-augmented generation, scientific question answering, and procedure-oriented evidence retrieval in materials science. It can be used to evaluate whether retrieval systems find the correct source paper for a question and whether paragraph-level and recipe-level views provide complementary evidence.

The dataset is not intended to be used as a substitute for reading the original papers, as a source of medical, safety-critical, or manufacturing instructions, or as a complete representation of the materials science literature.

Known limitations:

- Coverage is limited to the paper collection used for the RECIPER artifact.
- Entity labels are model-derived and may contain extraction errors.
- Recipe summaries are compact views of procedures and may omit details present in the original papers.
- QA labels identify source papers for retrieval evaluation, not exhaustive evidence spans.

## Data Sources

The corpus is derived from publicly reachable scientific literature sources identified by DOI or arXiv links. The dataset includes structured metadata, paper text passages, extracted entities, recipe summaries, and QA pairs prepared for the RECIPER paper artifact.

Users should cite the original papers where appropriate and should verify source licensing and downstream redistribution requirements for their own use case.

## License

This repository is released under the MIT License. See `LICENSE`.

If you redistribute derived versions of the data, keep the source metadata and citation information so users can trace records back to the original publications.

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

## Acknowledgements

This release uses NumPy, rank-bm25, Sentence Transformers, Hugging Face Datasets, and Hugging Face Hub.
