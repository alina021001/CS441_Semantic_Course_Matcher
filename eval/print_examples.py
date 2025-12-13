# eval/print_examples.py
"""
Print qualitative examples for three models:

  1) Bi-encoder baseline (base encoder)
  2) Bi-encoder distilled (student encoder)
  3) Bi-encoder + Cross-encoder rerank

For each selected query, we show:
  - the query text
  - gold course_ids
  - top-5 results from each model, with a [G] marker if it is a gold course.

Project layout:

  CS441-FP/
    search.py
    rerank.py
    index/course_index_meta.json
    models/course_embeddings.npy
    models/student_biencoder_distilled/
    data/evaluation_set.json
    eval/print_examples.py   <-- this file

Run from project root:

  cd /media/yee2y/KESU/CS441/CS441-FP
  source .venv/bin/activate
  python eval/print_examples.py
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------
# Resolve project root and fix import path
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from search import CourseRecommender
from rerank import CrossEncoderReranker

EVAL_SET_PATH = ROOT_DIR / "data" / "evaluation_set.json"


@dataclass
class EvalItem:
    query_id: int
    query_text: str
    gold_course_ids: List[str]


def load_eval_set(path: Path = EVAL_SET_PATH) -> List[EvalItem]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation set not found at {path}. "
            f"Please make sure data/evaluation_set.json exists."
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items: List[EvalItem] = []
    for obj in raw:
        items.append(
            EvalItem(
                query_id=int(obj["query_id"]),
                query_text=str(obj["query_text"]),
                gold_course_ids=[str(cid) for cid in obj.get("gold_course_ids", [])],
            )
        )
    return items


def pick_examples(items: List[EvalItem], num_examples: int = 5, seed: int = 42) -> List[EvalItem]:
    """Randomly pick some queries for qualitative analysis."""
    if len(items) <= num_examples:
        return items
    rng = random.Random(seed)
    return rng.sample(items, num_examples)


def print_result_block(
    name: str,
    results: List[Dict[str, Any]],
    gold_set: set,
) -> None:
    """Pretty-print one model's top-k results for a single query."""
    print(f"  -- {name} (top-{len(results)}) --")
    for rank, r in enumerate(results, start=1):
        cid = str(r.get("course_id", ""))
        title = r.get("title", "")
        subj = r.get("subject", "")
        lvl = r.get("course_number", "")
        score = r.get("score", None)
        ce_score = r.get("ce_score", None)

        if ce_score is not None:
            score_str = f"ce_score={ce_score:.3f}"
        elif score is not None:
            score_str = f"score={score:.3f}"
        else:
            score_str = ""

        marker = "[G]" if cid in gold_set else "   "
        print(f"    {marker} {rank}. {cid} | {title} | {subj} | level={lvl} | {score_str}")
    print()


def main() -> None:
    # 1) Load evaluation queries
    eval_items = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(eval_items)} evaluation queries from {EVAL_SET_PATH}.\n")

    # 2) Initialize models
    print("Loading base bi-encoder recommender...")
    base_rec = CourseRecommender()

    print("Loading distilled bi-encoder recommender...")
    student_model_path = ROOT_DIR / "models" / "student_biencoder_distilled"
    distilled_rec = CourseRecommender(
        encoder_model_name_or_path=str(student_model_path),
        embeddings_path=None,
        rebuild_embeddings=True,
    )

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(base_recommender=base_rec)

    # 3) Randomly pick a few example queries
    examples = pick_examples(eval_items, num_examples=5, seed=42)

    print("=== Qualitative Examples (Base vs Distilled vs Rerank) ===\n")

    for item in examples:
        qid = item.query_id
        query = item.query_text
        gold_set = set(item.gold_course_ids)

        print(f"=== Query {qid} ===")
        print(f"Text : {query}")
        print(f"Gold : {', '.join(sorted(gold_set)) if gold_set else '(none)'}\n")

        # Get top-5 from each model
        base_results = base_rec.recommend_courses(query_text=query, k=5)
        distilled_results = distilled_rec.recommend_courses(query_text=query, k=5)
        rerank_results = reranker.search(query_text=query, k=5, pool_size=30)

        print_result_block("Bi-encoder baseline", base_results, gold_set)
        print_result_block("Bi-encoder distilled", distilled_results, gold_set)
        print_result_block("Bi-encoder + CE rerank", rerank_results, gold_set)

        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
