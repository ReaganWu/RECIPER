from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = ROOT / "data" / "rag_database" / "rag_database.json"
DEFAULT_QA_PATH = ROOT / "data" / "rag_database" / "qa_dataset.jsonl"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_recipe_text(recipe: Any) -> str:
    if isinstance(recipe, str):
        return recipe.strip()
    if not isinstance(recipe, dict):
        return ""
    text = str(recipe.get("text", "") or recipe.get("recipe", "")).strip()
    if text:
        return text
    fields: list[str] = []
    material_name = str(recipe.get("material_name", "")).strip()
    synthesis_method = str(recipe.get("synthesis_method", "")).strip()
    key_points = str(recipe.get("key_points", "")).strip()
    if material_name:
        fields.append(f"Material: {material_name}")
    if synthesis_method:
        fields.append(f"Method: {synthesis_method}")
    for step in recipe.get("steps", []):
        if not isinstance(step, dict):
            continue
        description = str(step.get("description", "")).strip()
        if description:
            fields.append(description)
    if key_points:
        fields.append(f"Key points: {key_points}")
    return " ".join(fields).strip()


def load_rag_documents(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    stream: str = "combined",
) -> list[dict[str, Any]]:
    corpus = load_json(corpus_path)
    documents: list[dict[str, Any]] = []
    include_paragraphs = stream in {"combined", "paragraph"}
    include_recipes = stream in {"combined", "recipe"}

    for paper in corpus:
        paper_id = _clean_text(paper.get("paper_id", ""))
        title = _clean_text(paper.get("title", ""))
        abstract = _clean_text(paper.get("abstract", ""))
        metadata = paper.get("metadata", {}) if isinstance(paper.get("metadata", {}), dict) else {}

        if include_paragraphs:
            for section in paper.get("sections", []):
                heading = str(section.get("heading", "")).strip()
                for offset, paragraph in enumerate(section.get("paragraphs_with_entities", [])):
                    text = str(paragraph.get("text", "")).strip() if isinstance(paragraph, dict) else ""
                    if not text:
                        continue
                    documents.append(
                        {
                            "doc_id": f"{paper_id}::paragraph::{heading or 'section'}::{offset}",
                            "paper_id": paper_id,
                            "title": title,
                            "text": text,
                            "metadata": {
                                "paper_id": paper_id,
                                "heading": heading,
                                "stream": "paragraph",
                                "abstract": abstract,
                                "source": _clean_text(metadata.get("source", "")),
                                "doi": _clean_text(metadata.get("doi", "")),
                                "arxiv_id": _clean_text(metadata.get("arxiv_id", "")),
                            },
                        }
                    )

        if include_recipes:
            for offset, recipe in enumerate(paper.get("recipes", [])):
                text = _build_recipe_text(recipe)
                if not text:
                    continue
                documents.append(
                    {
                        "doc_id": f"{paper_id}::recipe::{offset}",
                        "paper_id": paper_id,
                        "title": title,
                        "text": text,
                        "metadata": {
                            "paper_id": paper_id,
                            "heading": "recipe",
                            "stream": "recipe",
                            "abstract": abstract,
                            "source": _clean_text(metadata.get("source", "")),
                            "doi": _clean_text(metadata.get("doi", "")),
                            "arxiv_id": _clean_text(metadata.get("arxiv_id", "")),
                        },
                    }
                )

    return documents


def build_snippet(text: str, max_chars: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
