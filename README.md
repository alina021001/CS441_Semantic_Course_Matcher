# CS 441 Final Project – Planning Form (Semantic Course Matcher)

## Project Title

**Semantic Course Matcher: Natural Language Course Search and Recommendation System**

---

## What is the goal of your project?

The goal of this project is to build a proof-of-concept natural-language–based course search and recommendation system that allows students to describe their learning goals in free-form text and retrieve the most semantically relevant university courses. The system demonstrates how embedding-based semantic retrieval can outperform traditional keyword search and enable more intuitive course exploration.

---

## Group Members

- **Zishuo Yu** `<zishuoy2@illinois.edu>` 
- **Peiling(Alina Wu)** `<peiling4@illinois.edu>`    
- **Yuqing Zhang** `<yuqing20@illinois.edu>`   
- **Liyang Wan** `<liyangw4@illinois.edu>`

---

## Describe the machine learning formulation

### What is the input (features)?

The model takes two types of text inputs:

1. **User Query**  
   - A free-form natural-language description of the student’s learning goal.

2. **Course Text Data**  
   - For every course, we use:
     - course title  
     - course description  
     - topics/keywords  
   - These fields are merged into a single `embedding_text` field and encoded using a pretrained sentence-transformer model.  
   - The set of all course embeddings forms the retrieval index.

---

### What is the output (target)?

- The output is a **ranked list of courses**, sorted by semantic similarity to the user query.  
- For evaluation, the target is whether the ground-truth relevant courses appear in the **Top-K** results for each query.

---

### How is the performance of the model measured?

We use standard information-retrieval metrics:

- **Recall@K / Top-K Accuracy** – fraction of queries for which at least one relevant course appears in the top-K results.  
- **Mean Reciprocal Rank (MRR@K)** – averages the reciprocal rank of the first relevant course, rewarding systems that rank correct courses higher.

These metrics directly measure the quality of the ranking produced by our system.

---

## Describe your data sources

We collect our own course and query data rather than using any curated ML benchmark.

- **Course Corpus (Retrieval Index)**
  - Scraped ~500 course JSON records from multiple UIUC departments:
    - BIOE, CEE, ECE, ME, and CS.
  - Each record includes:
    - course id / number / subject  
    - title  
    - description (`raw_description`)  
    - topics/keywords  
    - `embedding_text` (concatenation of title + description + topics)
  - All `embedding_text` fields are encoded with a pretrained SentenceTransformer model to build the vector index for retrieval.

- **Evaluation Set (Manually Labeled Queries)**
  - We created **30+ realistic English natural-language queries** that reflect how students actually describe their learning goals.
  - For each query, we **manually annotated one or more correct course IDs** as ground-truth matches.
  - These labeled queries are stored in `evaluation_set.json` and are used exclusively for evaluation.

---

### What data will be used for training?

This project does **not** involve supervised model training:

- We use a **pretrained sentence-embedding model** (SentenceTransformer) for both queries and courses.
- The scraped UIUC course corpus (~500 courses) serves as the **retrieval corpus / index**, not a supervised training set.
- Therefore, we do not have or need a traditional “training set” of labeled examples for optimization.

---

### What data will be used for validation? (comparing approaches, models, and hyperparameters)

- Our system does **not** perform supervised training or extensive hyperparameter tuning.  
- As a result, we do **not** use a separate validation set.  
- This is a common setup for unsupervised semantic retrieval systems based on pretrained embeddings.

(If we later introduce trainable components, we could split the labeled queries into validation and test portions, but the current project design does not require this.)

---

### What data will be used for testing? (final evaluation)

- The **test set** consists of the **30 manually labeled natural-language queries** in `evaluation_set.json`.  
- Each query has one or more ground-truth relevant course IDs.  
- These queries are used **only for evaluation**, never for training or validation.  
- We compute **Recall@K** and **MRR@K** on this set to compare different retrieval / reranking configurations.

---

## What machine learning approaches or models will you compare?

We focus on comparing different **semantic retrieval architectures** rather than training new models from scratch:

1. **Bi-Encoder Retrieval (Baseline)**
   - Uses a pretrained SentenceTransformer to encode the user query and each course independently into the same embedding space.
   - Retrieves top-K candidate courses via cosine similarity on the embedding vectors.
   - Fast and scalable; serves as our main baseline.

2. **Cross-Encoder Reranker (Two-Stage Retrieval)**
   - For the top-K candidates from the bi-encoder, we apply a cross-encoder model that jointly encodes the *(query, course)* pair.
   - Produces a refined relevance score that captures token-level interactions.
   - Used only on a small candidate set to balance accuracy and efficiency.

By comparing the **bi-encoder baseline** and the **bi-encoder + cross-encoder reranking pipeline**, we evaluate how much accuracy we gain from more expressive semantic modeling while keeping the system practical for interactive course search.
