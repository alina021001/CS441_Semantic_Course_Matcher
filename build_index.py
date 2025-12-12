# build_index.py
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
INDEX_DIR = Path("index")

MODEL_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# 你现在有的所有学院文件
COURSE_FILES = [
    ("ECE",  DATA_DIR / "ece.json"),
    ("BIOE", DATA_DIR / "bioe_courses.json"),
    ("CS",   DATA_DIR / "cs_courses.json"),
    ("CEE",  DATA_DIR / "cee_courses.json"),
    ("ME",   DATA_DIR / "me_courses.json"),
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def flatten_json_list(obj):
    """
    把像 [ {...}, {...}, [ {...}, {...} ] ] 这种结构展平成
    [ {...}, {...}, {...}, {...} ]
    """
    if not isinstance(obj, list):
        return [obj]

    flat = []
    for x in obj:
        if isinstance(x, list):
            flat.extend(x)
        else:
            flat.append(x)
    return flat


def ensure_course_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保每条记录都有 course_number(int)。
    - 如果原本就有，尝试转成 int；
    - 如果没有，就从 course_id 里抽数字，比如 'ME 330' -> 330；
    - 抽不到数字的就设为 -1（过滤时自然会被排除）。
    """
    if "course_number" not in df.columns:
        df["course_number"] = np.nan

    def _extract_num(row):
        n = row.get("course_number", None)
        if pd.notna(n):
            try:
                return int(n)
            except Exception:
                pass
        cid = row.get("course_id", "")
        if isinstance(cid, str):
            m = re.search(r"(\d+)", cid)
            if m:
                return int(m.group(1))
        return -1

    df["course_number"] = df.apply(_extract_num, axis=1).astype(int)
    return df


def load_courses():
    dfs = []

    for short_name, path in COURSE_FILES:
        if not path.exists():
            print(f"[WARN] file not found: {path} (skip)")
            continue

        print(f"Loading {short_name} from {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data = flatten_json_list(raw)  # 关键：展开顶层 list 里的 list
        df = pd.DataFrame(data)

        # 补 course_number
        df = ensure_course_number(df)

        # 加一个 global_id，方便以后区分哪门课
        df["global_id"] = f"{short_name}_" + df.index.astype(str)

        dfs.append(df)

    if not dfs:
        raise RuntimeError("No course files loaded. Check COURSE_FILES and data/ directory.")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def main():
    print("Loading courses...")
    df = load_courses()

    if "embedding_text" not in df.columns:
        raise RuntimeError("No 'embedding_text' column found in data.")

    # 只保留有 embedding_text 的课程
    df = df[df["embedding_text"].notnull()].reset_index(drop=True)

    print(f"Total courses with embedding_text: {len(df)}")

    print(f"Loading model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = df["embedding_text"].tolist()
    print("Encoding courses...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # 保存向量
    np.save(MODEL_DIR / "course_embeddings.npy", embeddings)
    print("Saved embeddings to models/course_embeddings.npy")

    # 保存 meta 信息
    keep_cols = [
        "global_id",
        "course_id",
        "course_number",
        "subject",
        "title",
        "embedding_text",
        "raw_description",
        "prerequisites",
        "topics",
        "associated_term",
        "campus",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    meta = df[keep_cols].to_dict(orient="records")
    with open(INDEX_DIR / "course_index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved index metadata to index/course_index_meta.json")


if __name__ == "__main__":
    main()


