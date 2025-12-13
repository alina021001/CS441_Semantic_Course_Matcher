# eval/eval_retrieval.py
"""
Evaluate course retrieval and reranking on a small labeled evaluation set.

We compare:
  1) Bi-encoder baseline (base encoder + precomputed embeddings)
  2) Bi-encoder distilled (student_biencoder_distilled, fresh embeddings)
  3) Bi-encoder + Cross-encoder rerank (two-stage)

Project layout:

  CS441-FP/
    search.py
    rerank.py
    index/course_index_meta.json
    models/course_embeddings.npy
    models/student_biencoder_distilled/
    data/evaluation_set.json
    eval/eval_retrieval.py   <-- this file

Run from project root:

  cd /media/yee2y/KESU/CS441/CS441-FP
  source .venv/bin/activate
  python eval/eval_retrieval.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------
# Resolve project root and fix import path
# ---------------------------------------------------------------------
# This file lives in CS441-FP/eval/, so project root is parent of parent.
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from search import CourseRecommender
from rerank import CrossEncoderReranker

# Path to evaluation set JSON (under data/)
EVAL_SET_PATH = ROOT_DIR / "data" / "evaluation_set.json"

# Top-k values we care about
K_LIST = [1, 3, 5, 10]


@dataclass
class EvalItem:
    query_id: int
    query_text: str
    gold_course_ids: List[str]


def load_eval_set(path: Path = EVAL_SET_PATH) -> List[EvalItem]:
    """Load evaluation set from JSON."""
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


def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    """Recall@k = (# of gold items in top-k) / (# of gold items)."""
    if not gold_ids:
        return 0.0
    topk = pred_ids[:k]
    hits = len(set(topk) & set(gold_ids))
    return hits / float(len(gold_ids))


def mrr_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    """
    Mean Reciprocal Rank at k for a single query:
      1 / rank_of_first_hit   if any gold in top-k
      0                        otherwise
    """
    topk = pred_ids[:k]
    gold_set = set(gold_ids)
    for idx, cid in enumerate(topk):
        if cid in gold_set:
            # rank is 1-based
            return 1.0 / float(idx + 1)
    return 0.0


def evaluate_method(
    name: str,
    eval_items: List[EvalItem],
    retrieval_fn: Callable[[str, int], List[Dict[str, Any]]],
    k_list: List[int] = K_LIST,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate a retrieval method on all queries.

    Args:
        name: name of the method, used only for printing.
        eval_items: list of EvalItem with queries and gold course_ids.
        retrieval_fn: function that takes (query_text, k_max) and returns
                      a ranked list of course dicts with 'course_id' keys.
        k_list: list of k values for which to compute Recall@k and MRR@k.

    Returns:
        A nested dict:
        {
          "recall": {k: avg_recall_at_k},
          "mrr": {k: avg_mrr_at_k}
        }
    """
    print(f"\n=== Evaluating method: {name} ===")

    # Prepare accumulators
    recall_sums = {k: 0.0 for k in k_list}
    mrr_sums = {k: 0.0 for k in k_list}
    num_queries = 0

    for item in eval_items:
        q = item.query_text
        gold = item.gold_course_ids

        # Skip queries with no gold labels
        if not gold:
            continue

        # We always retrieve up to max(k_list)
        k_max = max(k_list)
        results = retrieval_fn(q, k_max)

        pred_ids = [str(r.get("course_id", "")) for r in results]

        num_queries += 1

        for k in k_list:
            r_k = recall_at_k(pred_ids, gold, k)
            mrr_k = mrr_at_k(pred_ids, gold, k)
            recall_sums[k] += r_k
            mrr_sums[k] += mrr_k

    if num_queries == 0:
        print("No queries with gold labels found in evaluation set.")
        return {"recall": {k: 0.0 for k in k_list}, "mrr": {k: 0.0 for k in k_list}}

    # Compute averages
    recall_avg = {k: recall_sums[k] / float(num_queries) for k in k_list}
    mrr_avg = {k: mrr_sums[k] / float(num_queries) for k in k_list}

    # Pretty print
    print(f"Number of evaluated queries: {num_queries}")
    print("\nRecall@k:")
    for k in k_list:
        print(f"  Recall@{k}: {recall_avg[k]:.3f}")

    print("\nMRR@k:")
    for k in k_list:
        print(f"  MRR@{k}: {mrr_avg[k]:.3f}")

    print()
    return {"recall": recall_avg, "mrr": mrr_avg}


def make_retrieval_fn(recommender: CourseRecommender):
    """Wrap CourseRecommender into a retrieval_fn."""

    def _fn(query_text: str, k_max: int) -> List[Dict[str, Any]]:
        return recommender.recommend_courses(
            query_text=query_text,
            k=k_max,
            min_level=None,
            max_level=None,
            subject_whitelist=None,
            exclude_title_keywords=None,
        )

    return _fn


def make_rerank_retrieval(reranker: CrossEncoderReranker, pool_size: int = 30):
    """
    Wrap CrossEncoderReranker into a retrieval_fn.

    It first retrieves 'pool_size' candidates with the bi-encoder,
    then returns the top-k_max reranked results.
    """

    def _fn(query_text: str, k_max: int) -> List[Dict[str, Any]]:
        return reranker.search(
            query_text=query_text,
            k=k_max,
            pool_size=pool_size,
            min_level=None,
            max_level=None,
            subject_whitelist=None,
            exclude_title_keywords=None,
        )

    return _fn


def main() -> None:
    # 1) Load evaluation set
    eval_items = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(eval_items)} evaluation queries from {EVAL_SET_PATH}.")

    # 2) Initialize base recommender (uses default encoder + precomputed embeddings)
    print("Loading base bi-encoder recommender...")
    base_recommender = CourseRecommender()

    # 3) Initialize distilled recommender (student encoder, recompute embeddings)
    print("Loading distilled bi-encoder recommender...")
    student_model_path = ROOT_DIR / "models" / "student_biencoder_distilled"
    distilled_recommender = CourseRecommender(
        encoder_model_name_or_path=str(student_model_path),
        embeddings_path=None,          # do not load base embeddings
        rebuild_embeddings=True,       # recompute embeddings with student
    )

    # 4) Initialize reranker (built on top of base recommender)
    print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(base_recommender=base_recommender)

    # 5) Build retrieval functions
    baseline_fn = make_retrieval_fn(base_recommender)
    distilled_fn = make_retrieval_fn(distilled_recommender)
    rerank_fn = make_rerank_retrieval(reranker, pool_size=30)

    # 6) Evaluate all three methods
    baseline_metrics = evaluate_method(
        name="Bi-encoder baseline (base encoder)",
        eval_items=eval_items,
        retrieval_fn=baseline_fn,
        k_list=K_LIST,
    )

    distilled_metrics = evaluate_method(
        name="Bi-encoder distilled (student encoder)",
        eval_items=eval_items,
        retrieval_fn=distilled_fn,
        k_list=K_LIST,
    )

    rerank_metrics = evaluate_method(
        name="Bi-encoder + Cross-encoder rerank",
        eval_items=eval_items,
        retrieval_fn=rerank_fn,
        k_list=K_LIST,
    )

    # 7) Print a compact comparison summary
    print("=== Summary (average over all queries) ===")
    header = (
        "k\t"
        "Recall_base\tRecall_distill\tRecall_rerank\t"
        "MRR_base\tMRR_distill\tMRR_rerank"
    )
    print(header)
    for k in K_LIST:
        rb = baseline_metrics["recall"][k]
        rs = distilled_metrics["recall"][k]
        rr = rerank_metrics["recall"][k]
        mb = baseline_metrics["mrr"][k]
        ms = distilled_metrics["mrr"][k]
        mr = rerank_metrics["mrr"][k]
        print(
            f"{k}\t"
            f"{rb:.3f}\t\t{rs:.3f}\t\t{rr:.3f}\t\t"
            f"{mb:.3f}\t\t{ms:.3f}\t\t{mr:.3f}"
        )


if __name__ == "__main__":
    main()


