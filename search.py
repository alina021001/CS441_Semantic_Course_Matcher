# search.py
from pathlib import Path
from typing import List, Optional, Dict, Any

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path("models")
INDEX_DIR = Path("index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class CourseRecommender:
    """
    A simple semantic search engine over course descriptions.

    It loads:
      - pre-computed course embeddings (NumPy array)
      - metadata for each course (JSON list of dicts)

    And exposes:
      - recommend_courses(): semantic retrieval with hard filters
    """

    def __init__(self):
        meta_path = INDEX_DIR / "course_index_meta.json"
        emb_path = MODEL_DIR / "course_embeddings.npy"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: List[Dict[str, Any]] = json.load(f)

        self.embeddings = np.load(emb_path)

        # SentenceTransformer will embed user queries
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _apply_hard_filters(
        self,
        min_level: Optional[int],
        subject_whitelist: Optional[List[str]],
        max_level: Optional[int] = None,
        exclude_title_keywords: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Build a boolean mask over all courses based on hard constraints:

        - min_level / max_level: course_number range filter
        - subject_whitelist: list of allowed subjects (e.g., ["ECE", "Computer Science"])
        - exclude_title_keywords: list of keywords that should NOT appear in the course title
          (e.g., ["Individual Study", "Seminar", "Reading Group"])
        """
        n = len(self.meta)
        mask = np.ones(n, dtype=bool)

        # Course level range filter: course_number in [min_level, max_level]
        course_numbers = np.array([
            m.get("course_number") if isinstance(m.get("course_number"), int) else -1
            for m in self.meta
        ])

        if min_level is not None:
            mask &= (course_numbers >= min_level)
        if max_level is not None:
            mask &= (course_numbers <= max_level)

        # Subject whitelist filter
        if subject_whitelist is not None:
            subjects = np.array([m.get("subject") for m in self.meta])
            mask &= np.isin(subjects, subject_whitelist)

        # Exclude some course “types” based on title keywords
        if exclude_title_keywords:
            titles = np.array([m.get("title") or "" for m in self.meta])
            lowered = np.char.lower(titles.astype(str))
            for kw in exclude_title_keywords:
                kw_lower = kw.lower()
                # np.char.find returns -1 if not found; >= 0 means the keyword is present
                contains_kw = np.char.find(lowered, kw_lower) >= 0
                mask &= ~contains_kw

        return mask

    def recommend_courses(
        self,
        query_text: str,
        k: int = 5,
        min_level: Optional[int] = None,
        subject_whitelist: Optional[List[str]] = None,
        max_level: Optional[int] = None,
        exclude_title_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: given a natural language query (in English),
        return top-k most relevant courses after applying hard filters.

        Args:
            query_text: user query in English (e.g., "I want a senior-level course on VLSI and CMOS design")
            k: number of results to return
            min_level: minimum course number (e.g., 400 for senior-level)
            subject_whitelist: list of allowed subjects (e.g., ["ECE", "Computer Science"])
            max_level: maximum course number (e.g., 499 to exclude grad-level courses)
            exclude_title_keywords: list of title keywords to exclude, e.g.:
                ["Individual Study", "Seminar", "Reading Group"]

        Returns:
            A list of course dicts, each augmented with a "similarity" score.
        """
        # 1. Encode the query
        query_emb = self.model.encode([query_text], normalize_embeddings=True)

        # 2. Apply hard filters to the course pool
        mask = self._apply_hard_filters(
            min_level=min_level,
            subject_whitelist=subject_whitelist,
            max_level=max_level,
            exclude_title_keywords=exclude_title_keywords,
        )
        filtered_embeddings = self.embeddings[mask]
        if filtered_embeddings.shape[0] == 0:
            return []

        # 3. Compute cosine similarity in the filtered subset
        sims = cosine_similarity(query_emb, filtered_embeddings)[0]  # shape: (num_filtered,)

        # 4. Select top-k indices within the filtered subset
        topk_idx_local = np.argsort(-sims)[:k]
        filtered_indices = np.where(mask)[0]

        # 5. Map back to global indices and attach metadata + similarity
        results: List[Dict[str, Any]] = []
        for local_idx in topk_idx_local:
            global_idx = filtered_indices[local_idx]
            course_info = self.meta[global_idx].copy()
            course_info["similarity"] = float(sims[local_idx])
            results.append(course_info)

        return results

    @staticmethod
    def pretty_print_results(results: List[Dict[str, Any]], header: str) -> None:
        """
        Helper for quick command-line demos.
        """
        print(header)
        if not results:
            print("  (no results found)\n")
            return

        for r in results:
            cid = r.get("course_id", "N/A")
            title = r.get("title", "N/A")
            level = r.get("course_number", "N/A")
            subj = r.get("subject", "N/A")
            score = r.get("similarity", 0.0)

            print(f"{cid} | {title} | level={level} | subject={subj} | score={score:.3f}")
            emb_text_preview = (r.get("embedding_text") or "")[:140].replace("\n", " ")
            print("   ", emb_text_preview, "...\n")


if __name__ == "__main__":
    """
    Simple English-only demo.

    Run this with:
        source .venv/bin/activate
        python search.py
    """

    recommender = CourseRecommender()

    # Example 1: VLSI / CMOS senior-level ECE course
    query1 = (
        "I am looking for a senior-level ECE course on VLSI, "
        "CMOS circuits, and integrated chip design."
    )

    results_vlsi = recommender.recommend_courses(
        query_text=query1,
        k=5,
        min_level=400,           # senior-level and above
        max_level=499,           # exclude graduate (500-level) courses
        subject_whitelist=["ECE"],   # restrict to ECE courses only
        exclude_title_keywords=[
            "Individual Study",
            "Seminar",
            "Reading Group",
            "Internship",
            "Honors",
        ],
    )

    recommender.pretty_print_results(results_vlsi, "\n=== Recommended VLSI / CMOS Courses ===")

    # Example 2: data / machine learning course across multiple departments
    query2 = (
        "I would like to take an upper-level course about machine learning, "
        "data science, or data analysis with programming."
    )

    # Note: subjects depend on your JSON data.
    # For example, CS might have subject="Computer Science",
    # ECE might be "ECE", BIOE might be "Bioengineering", etc.
    results_ml = recommender.recommend_courses(
        query_text=query2,
        k=5,
        min_level=300,   # 300-level and above
        max_level=499,   # exclude graduate-level
        subject_whitelist=None,  # allow all subjects for now
        exclude_title_keywords=[
            "Individual Study",
            "Seminar",
            "Reading Group",
            "Internship",
            "Honors",
        ],
    )

    recommender.pretty_print_results(results_ml, "\n=== Recommended Data / ML Courses ===")


