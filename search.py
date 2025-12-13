# search.py
"""
CourseRecommender: bi-encoder semantic search over course catalog.

This version supports:
  - Default encoder: sentence-transformers/all-MiniLM-L6-v2
  - Optional student encoder: models/student_biencoder_distilled
  - Optional re-building of course embeddings when using a new encoder.

Artifacts expected:

  index/course_index_meta.json   # produced by build_index.py
  models/course_embeddings.npy   # (optional) default embeddings for base model

Usage example (baseline):

  from search import CourseRecommender
  rec = CourseRecommender()
  results = rec.recommend_courses("VLSI CMOS design course", k=5)

Usage example (distilled model):

  rec_student = CourseRecommender(
      encoder_model_name_or_path="models/student_biencoder_distilled",
      embeddings_path=None,          # do not use precomputed base embeddings
      rebuild_embeddings=True        # recompute embeddings with student
  )
  results = rec_student.recommend_courses("VLSI CMOS design course", k=5)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_META_PATH = ROOT_DIR / "index" / "course_index_meta.json"
DEFAULT_EMBEDDINGS_PATH = ROOT_DIR / "models" / "course_embeddings.npy"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class CourseRecord:
    idx: int
    course_id: str
    title: str
    subject: Optional[str]
    course_number: Optional[int]
    raw_description: Optional[str]
    embedding_text: str
    extra: Dict[str, Any]


class CourseRecommender:
    def __init__(
        self,
        index_meta_path: Path = DEFAULT_META_PATH,
        embeddings_path: Optional[Path] = DEFAULT_EMBEDDINGS_PATH,
        encoder_model_name_or_path: str = DEFAULT_ENCODER,
        rebuild_embeddings: bool = False,
    ) -> None:
        """
        Args:
            index_meta_path: JSON with list of courses and embedding_text.
            embeddings_path: .npy file with precomputed course embeddings.
                             If None, embeddings will always be recomputed.
            encoder_model_name_or_path: HuggingFace / local path for encoder.
            rebuild_embeddings: If True, recompute embeddings even if
                                embeddings_path exists.
        """
        self.index_meta_path = Path(index_meta_path)
        self.embeddings_path = Path(embeddings_path) if embeddings_path is not None else None
        self.encoder_model_name_or_path = encoder_model_name_or_path
        self.rebuild_embeddings = rebuild_embeddings

        self.records: List[CourseRecord] = []
        self.embeddings: np.ndarray

        self._load_metadata()
        self._load_or_build_embeddings()

    # ------------------------------------------------------------------
    # Loading metadata and embeddings
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        if not self.index_meta_path.exists():
            raise FileNotFoundError(
                f"Course index metadata not found at {self.index_meta_path}. "
                f"Please run build_index.py first."
            )

        with self.index_meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        records: List[CourseRecord] = []
        for i, obj in enumerate(data):
            course_id = str(obj.get("course_id", ""))
            title = str(obj.get("title", ""))

            subject = obj.get("subject")
            course_number = obj.get("course_number")
            raw_description = obj.get("raw_description")

            # Prefer embedding_text; if missing, fallback to title + description
            embedding_text = obj.get("embedding_text")
            if not embedding_text:
                parts = [title]
                if raw_description:
                    parts.append(raw_description)
                embedding_text = ". ".join(parts)

            # Keep any extra fields
            extra = {
                k: v
                for k, v in obj.items()
                if k
                not in {
                    "course_id",
                    "title",
                    "subject",
                    "course_number",
                    "raw_description",
                    "embedding_text",
                }
            }

            records.append(
                CourseRecord(
                    idx=i,
                    course_id=course_id,
                    title=title,
                    subject=subject,
                    course_number=course_number,
                    raw_description=raw_description,
                    embedding_text=embedding_text,
                    extra=extra,
                )
            )

        self.records = records

    def _load_or_build_embeddings(self) -> None:
        """
        Load precomputed embeddings if appropriate, otherwise recompute
        using the specified encoder model (and optionally save).
        """
        # Always initialize encoder here
        self.model = SentenceTransformer(self.encoder_model_name_or_path)

        if (
            self.embeddings_path is not None
            and self.embeddings_path.exists()
            and not self.rebuild_embeddings
        ):
            # Fast path: load existing embeddings
            self.embeddings = np.load(self.embeddings_path)
            # Assume they are already normalized
            return

        # Otherwise, recompute embeddings from embedding_text
        texts = [rec.embedding_text for rec in self.records]
        print(
            f"Building embeddings with encoder={self.encoder_model_name_or_path} "
            f"for {len(texts)} courses..."
        )
        emb = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = emb

        # Optionally save embeddings for faster reuse
        if self.embeddings_path is not None:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.embeddings_path, self.embeddings)
            print(f"Saved embeddings to {self.embeddings_path}")

    # ------------------------------------------------------------------
    # Recommendation API
    # ------------------------------------------------------------------

    def recommend_courses(
        self,
        query_text: str,
        k: int = 5,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
        subject_whitelist: Optional[List[str]] = None,
        exclude_title_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k courses for a query using cosine similarity.

        Args:
            query_text: user query in natural language.
            k: number of results to return.
            min_level: minimum course_number (e.g., 400 for 400-level).
            max_level: maximum course_number (e.g., 499 for 400-level).
            subject_whitelist: list of allowed subjects (e.g., ["ECE", "Computer Science"]).
            exclude_title_keywords: lowercase keywords; if present in title, course is excluded.

        Returns:
            A list of dicts, each containing course info and similarity score.
        """
        if not self.records:
            return []

        # Encode query
        q_emb = self.model.encode(
            query_text, convert_to_numpy=True, normalize_embeddings=True
        )

        num_courses = len(self.records)
        scores = np.zeros(num_courses, dtype=np.float32)

        # Precompute mask for hard filters
        mask = np.ones(num_courses, dtype=bool)

        # Level filter
        if min_level is not None or max_level is not None:
            levels = np.array(
                [
                    rec.course_number if rec.course_number is not None else -1
                    for rec in self.records
                ]
            )
            if min_level is not None:
                mask &= levels >= min_level
            if max_level is not None:
                mask &= levels <= max_level

        # Subject whitelist
        if subject_whitelist is not None:
            subjects = np.array(
                [rec.subject if rec.subject is not None else "" for rec in self.records],
                dtype=object,
            )
            allowed = set(subject_whitelist)
            mask &= np.array([subj in allowed for subj in subjects])

        # Exclude certain keywords in title
        if exclude_title_keywords:
            lowered_keywords = [kw.lower() for kw in exclude_title_keywords]
            title_mask = []
            for rec in self.records:
                title_l = rec.title.lower()
                if any(kw in title_l for kw in lowered_keywords):
                    title_mask.append(False)
                else:
                    title_mask.append(True)
            mask &= np.array(title_mask, dtype=bool)

        # If nothing passes filters, early return empty list
        if not mask.any():
            return []

        # Compute cosine similarity for masked courses
        valid_indices = np.where(mask)[0]
        valid_embs = self.embeddings[valid_indices]  # (M, D)

        # embeddings and query are unit-normalized, so cosine = dot product
        scores_subset = valid_embs @ q_emb.astype(np.float32)
        for idx_local, idx_global in enumerate(valid_indices):
            scores[idx_global] = scores_subset[idx_local]

        # Get top-k indices over all courses (scores for masked ones are 0 or -inf equivalent)
        # To avoid masked-but-zero confusion, we sort only valid_indices.
        k_eff = min(k, len(valid_indices))
        top_local = np.argsort(-scores_subset)[:k_eff]
        top_indices = valid_indices[top_local]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            rec = self.records[idx]
            score = float(scores[idx])
            item: Dict[str, Any] = {
                "course_id": rec.course_id,
                "title": rec.title,
                "subject": rec.subject,
                "course_number": rec.course_number,
                "raw_description": rec.raw_description,
                "embedding_text": rec.embedding_text,
                "score": score,
            }
            # merge extra fields
            item.update(rec.extra)
            results.append(item)

        return results


# ----------------------------------------------------------------------
# Simple CLI demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic course search demo")
    parser.add_argument("query", type=str, help="Query text")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--student",
        action="store_true",
        help="Use distilled student bi-encoder (models/student_biencoder_distilled)",
    )
    args = parser.parse_args()

    if args.student:
        model_path = ROOT_DIR / "models" / "student_biencoder_distilled"
        rec = CourseRecommender(
            encoder_model_name_or_path=str(model_path),
            embeddings_path=None,
            rebuild_embeddings=True,
        )
        print(f"Using student model from {model_path}")
    else:
        rec = CourseRecommender()
        print(f"Using base encoder: {DEFAULT_ENCODER}")

    results = rec.recommend_courses(args.query, k=args.k)
    print(f"\nTop {len(results)} results:")
    for r in results:
        cid = r.get("course_id")
        title = r.get("title")
        subj = r.get("subject")
        lvl = r.get("course_number")
        score = r.get("score")
        print(f"{cid} | {title} | {subj} | level={lvl} | score={score:.3f}")



