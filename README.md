# Semantic Codebase Search Engine

Search any Python codebase using natural language instead of keywords. Point it at a repo, ask "where is rate limiting handled?", and get back the exact function — file, line range, and relevance score — instead of grepping for guesses.

Runs fully locally. No code ever leaves your machine, and no external LLM API calls are made at query time.

---

## How it works

```
Codebase (local files / repo)
        ↓
Chunking — tree-sitter AST parsing, function-level granularity
        ↓
Embedding — CodeBERT (fine-tuned), mean pooling, L2 normalized
        ↓
Vector Index — FAISS (cosine similarity)
        ↓
FastAPI  →  POST /search  →  ranked results (file, line range, function, score)
```

Each function/method in the codebase is parsed into its own semantic chunk (with class context, decorators, and async functions all handled correctly), embedded into a 768-dim vector space using a CodeBERT model **fine-tuned specifically for code retrieval**, and indexed with FAISS for millisecond search.

## Results

CodeBERT was fine-tuned on CodeSearchNet (Python) using a custom contrastive training loop (symmetric InfoNCE loss, in-batch negatives). Evaluated on the full 10,030-example held-out test set:

| Model | MRR@10 | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Zero-shot CodeBERT | 0.0025 | — | — | — |
| **Fine-tuned (this project)** | **0.82** | 0.74 | 0.92 | 0.94 |

**NOTE:** The zero-shot CodeBERT here uses mean pooling instead of the standard CLS pooling.

---

## Project structure

```
app/                    Core application
├── chunker.py          tree-sitter based AST parsing → function-level chunks
├── embedder.py         CodeBERT embedding + mean pooling
├── indexer.py          FAISS index build + search
├── codebase_indexer.py Walks a directory, chunks + embeds + indexes every file
├── models.py           Pydantic request/response schemas
└── main.py             FastAPI app (lifespan-based startup indexing, /search endpoint)

Fine_Tuning/            Training pipeline
├── dataset_preparer.py Loads + filters CodeSearchNet, strips docstring leakage
├── train.py            Custom contrastive training loop
├── evaluate.py          MRR@10 / Recall@k evaluation
└── helper.py            Shared utilities (pooling, etc.)

download_model.py
model/                  Fine-tuned model weights (not committed — see Setup below)
notebooks/               Experimentation notebooks
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the fine-tuned model

The trained model isn't committed to this repo (weights are large and don't belong in git). It's hosted on HuggingFace Hub instead:

```bash
python download_model.py
```

This pulls the fine-tuned checkpoint into `model/`.

### 3. Set the codebase to index

```bash
# PowerShell
$env:CODEBASE_PATH = "C:\path\to\the\repo\you\want\to\search"

# bash / macOS / Linux
export CODEBASE_PATH=/path/to/the/repo/you/want/to/search
```

### 4. Run the server

```bash
cd app
uvicorn main:app --reload
```

Indexing runs once at startup (chunking + embedding the whole codebase). Once it's done, open `http://127.0.0.1:8000/docs` for the interactive API, or hit `/search` directly:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "function that retries failed HTTP requests"}'
```

---

## Tech stack

* **Chunking:** tree-sitter (AST parsing) 
* **Embedding:** CodeBERT (`microsoft/codebert-base`, fine-tuned)
* **Search:** FAISS · **API:** FastAPI
* **Training:** PyTorch (custom contrastive loop, no `Trainer`)
* **Data:** CodeSearchNet (via HuggingFace `datasets`)

## Fine-tuning this yourself

See `Fine_Tuning/` — `dataset_preparer.py` handles loading, filtering, and de-leaking CodeSearchNet; `train.py` runs the contrastive training loop; `evaluate.py` computes MRR@10/Recall@k against a zero-shot baseline. 

## Known limitations

- Python only (tree-sitter grammar for other languages not yet wired in)
- Fine-tuned on ~25% of one epoch due to compute constraints — further training showed diminishing returns past this point, but hasn't been exhausted
- No incremental re-indexing yet on file changes — a full re-index is currently required to pick up codebase changes