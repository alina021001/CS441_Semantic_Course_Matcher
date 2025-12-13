# distill.py
"""
Distill a bi-encoder student model from a cross-encoder teacher
for the course recommendation task.

Pipeline:

  1) Load evaluation queries from data/evaluation_set.json
  2) For each query, retrieve a candidate pool using the existing
     CourseRecommender (bi-encoder baseline).
  3) Use a cross-encoder teacher to score (query, course_text) pairs.
  4) Normalize teacher scores and build training pairs for a
     SentenceTransformer bi-encoder student.
  5) Fine-tune the student model and save it under:
        models/student_biencoder_distilled

After training, you can modify search.py to load the distilled model
instead of the original one and re-run eval/eval_retrieval.py to
compare performance.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Resolve project root and import local modules
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from search import CourseRecommender  # type: ignore

# Teacher & student models
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader


EVAL_SET_PATH = ROOT_DIR / "data" / "evaluation_set.json"
STUDENT_OUTPUT_DIR = ROOT_DIR / "models" / "student_biencoder_distilled"

# How many candidates per query to distill over
CANDIDATE_POOL_SIZE = 40

# Basic hyperparameters
BATCH_SIZE = 16
EPOCHS = 2


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


def build_distillation_pairs(
    recommender: CourseRecommender,
    teacher: CrossEncoder,
    eval_items: List[EvalItem],
    pool_size: int = CANDIDATE_POOL_SIZE,
) -> List[InputExample]:
    """
    For each query, retrieve a candidate pool with the bi-encoder,
    then score each (query, course_text) pair with the cross-encoder
    teacher. Convert scores to [0,1] and return InputExample list.
    """
    print("Building distillation dataset...")

    # 1) Collect all (query, course_text) pairs
    pairs: List[Tuple[str, str]] = []
    pair_meta: List[Dict] = []  # keep track of (query_id, course_id) if needed

    for item in tqdm(eval_items, desc="Collecting candidates"):
        q = item.query_text

        # Use bi-encoder recommender to get candidate pool
        candidates = recommender.recommend_courses(
            query_text=q,
            k=pool_size,
            min_level=None,
            max_level=None,
            subject_whitelist=None,
            exclude_title_keywords=None,
        )

        for c in candidates:
            cid = str(c.get("course_id", ""))
            text = c.get("embedding_text") or c.get("raw_description") or c.get("title") or ""
            if not text:
                continue

            pairs.append((q, text))
            pair_meta.append(
                {
                    "query_id": item.query_id,
                    "course_id": cid,
                }
            )

    if not pairs:
        raise RuntimeError("No (query, course) pairs collected for distillation.")

    print(f"Total candidate pairs collected: {len(pairs)}")

    # 2) Use cross-encoder teacher to score all pairs
    print("Scoring pairs with cross-encoder teacher...")
    teacher_scores = teacher.predict(pairs, batch_size=BATCH_SIZE)
    teacher_scores = list(map(float, teacher_scores))

    # 3) Normalize teacher scores to [0,1] for cosine similarity loss
    s_min = min(teacher_scores)
    s_max = max(teacher_scores)
    if s_max == s_min:
        # Degenerate case: all scores equal; just set everything to 1.0
        print("Warning: teacher scores are constant; using label=1.0 for all pairs.")
        labels = [1.0 for _ in teacher_scores]
    else:
        labels = [(s - s_min) / (s_max - s_min) for s in teacher_scores]

    # 4) Build InputExample list
    train_examples: List[InputExample] = []
    for (q, text), label in zip(pairs, labels):
        train_examples.append(InputExample(texts=[q, text], label=label))

    print(f"Built {len(train_examples)} training examples for distillation.")
    return train_examples


def train_student(
    train_examples: List[InputExample],
    output_dir: Path = STUDENT_OUTPUT_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> SentenceTransformer:
    """
    Fine-tune a SentenceTransformer bi-encoder student using the
    distillation pairs and cosine similarity loss.
    """
    print("Initializing student bi-encoder model...")
    student = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(student)

    warmup_steps = int(0.1 * len(train_dataloader)) if len(train_dataloader) > 0 else 0

    print(
        f"Starting training: {len(train_examples)} examples, "
        f"{len(train_dataloader)} batches/epoch, epochs={epochs}"
    )
    student.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    student.save(str(output_dir))
    print(f"Student model saved to: {output_dir}")

    return student


def main() -> None:
    # 1) Load evaluation queries as distillation queries
    eval_items = load_eval_set(EVAL_SET_PATH)
    print(f"Loaded {len(eval_items)} evaluation queries from {EVAL_SET_PATH}.")

    # 2) Initialize base recommender (bi-encoder) and teacher (cross-encoder)
    print("Loading course index and base bi-encoder recommender...")
    recommender = CourseRecommender()

    print("Loading cross-encoder teacher model...")
    teacher = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # 3) Build distillation training pairs
    train_examples = build_distillation_pairs(
        recommender=recommender,
        teacher=teacher,
        eval_items=eval_items,
        pool_size=CANDIDATE_POOL_SIZE,
    )

    # 4) Train student bi-encoder model
    train_student(
        train_examples=train_examples,
        output_dir=STUDENT_OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    print("Distillation finished. You can now update search.py to use the "
          "distilled model from models/student_biencoder_distilled and "
          "re-run eval/eval_retrieval.py to compare performance.")


if __name__ == "__main__":
    main()
