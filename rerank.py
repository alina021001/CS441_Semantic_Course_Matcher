# rerank.py
"""
Reranking module for the course search system.

We use a two-stage architecture:
  1. Bi-encoder (in search.py) retrieves a pool of candidate courses.
  2. Cross-encoder reranks these candidates with a more accurate
     relevance score conditioned on (query, course_text) pairs.

This improves the quality of the top-k results without having to
encode the entire catalog with a heavy model.
"""

from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from search import CourseRecommender  # reuse your existing retrieval code


class CrossEncoderReranker:
    """
    Cross-encoder reranker on top of the existing CourseRecommender.

    Usage:
        base = CourseRecommender()
        reranker = CrossEncoderReranker(base)
        results = reranker.search(
            query_text="I want a senior-level course on VLSI",
            k=5,
            pool_size=30,
            min_level=400,
            max_level=499,
            subject_whitelist=["ECE"],
        )
    """

    def __init__(
        self,
        base_recommender: Optional[CourseRecommender] = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Args:
            base_recommender: An instance of CourseRecommender (from search.py).
                              If None, we create a new one.
            model_name: HuggingFace / sentence-transformers cross-encoder model.
        """
        self.base = base_recommender or CourseRecommender()
        self.model = CrossEncoder(model_name)

    def search(
        self,
        query_text: str,
        k: int = 5,
        pool_size: int = 30,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
        subject_whitelist: Optional[List[str]] = None,
        exclude_title_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Two-stage search:
          1) Use bi-encoder to retrieve a pool of `pool_size` candidates.
          2) Rerank these candidates with a cross-encoder.

        Args:
            query_text: user query in English.
            k: number of final results to return.
            pool_size: number of candidates from the bi-encoder stage.
            min_level, max_level, subject_whitelist, exclude_title_keywords:
                Same semantics as in CourseRecommender.recommend_courses.

        Returns:
            Top-k candidate courses with an additional 'ce_score' (cross-encoder score).
        """
        # 1) Initial candidates from bi-encoder
        candidates = self.base.recommend_courses(
            query_text=query_text,
            k=pool_size,
            min_level=min_level,
            subject_whitelist=subject_whitelist,
            max_level=max_level,
            exclude_title_keywords=exclude_title_keywords,
        )

        if not candidates:
            return []

        # 2) Prepare (query, text) pairs for the cross-encoder
        pairs = []
        for c in candidates:
            text = c.get("embedding_text") or c.get("raw_description") or ""
            pairs.append((query_text, text))

        # 3) Predict relevance scores
        scores = self.model.predict(pairs)  # higher = more relevant
        scores = np.array(scores)

        # 4) Sort by cross-encoder score and take top-k
        topk_idx = np.argsort(-scores)[:k]
        reranked_results: List[Dict[str, Any]] = []
        for idx in topk_idx:
            item = candidates[idx].copy()
            item["ce_score"] = float(scores[idx])
            reranked_results.append(item)

        return reranked_results

    @staticmethod
    def pretty_print_results(results: List[Dict[str, Any]], header: str) -> None:
        """
        Small helper for quick CLI demos.
        """
        print(header)
        if not results:
            print("  (no results)\n")
            return

        for r in results:
            cid = r.get("course_id", "N/A")
            title = r.get("title", "N/A")
            level = r.get("course_number", "N/A")
            subj = r.get("subject", "N/A")
            score = r.get("ce_score", 0.0)

            print(f"{cid} | {title} | level={level} | subject={subj} | ce_score={score:.3f}")
            preview = (r.get("embedding_text") or "")[:140].replace("\n", " ")
            print("   ", preview, "...\n")


if __name__ == "__main__":
    """
    Simple command-line demo for the reranker.

    Run:
        source .venv/bin/activate
        python rerank.py
    """
    base = CourseRecommender()
    reranker = CrossEncoderReranker(base)

    # Example 1: VLSI
    q1 = (
        "I am looking for a senior-level ECE course on VLSI, "
        "CMOS circuits, and integrated chip design."
    )
    res_vlsi = reranker.search(
        query_text=q1,
        k=5,
        pool_size=30,
        min_level=400,
        max_level=499,
        subject_whitelist=["ECE"],
        exclude_title_keywords=[
            "Individual Study",
            "Seminar",
            "Reading Group",
            "Internship",
            "Honors",
        ],
    )
    reranker.pretty_print_results(res_vlsi, "\n=== Cross-encoder reranked VLSI courses ===")

    # Example 2: ML / data
    q2 = (
        "I would like to take an upper-level course about machine learning, "
        "data science, or data analysis with programming."
    )
    res_ml = reranker.search(
        query_text=q2,
        k=5,
        pool_size=30,
        min_level=300,
        max_level=499,
        subject_whitelist=None,
        exclude_title_keywords=[
            "Individual Study",
            "Seminar",
            "Reading Group",
            "Internship",
            "Honors",
        ],
    )
    reranker.pretty_print_results(res_ml, "\n=== Cross-encoder reranked ML / data courses ===")

