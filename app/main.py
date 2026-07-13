import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from transformers import AutoTokenizer

from .models import SearchRequest, SearchResult
from .codebase_indexer import index_codebase

# Path to the codebase we want to index — set via environment variable,
# To run use this: `CODEBASE_PATH=/path/to/repo uvicorn main:app --reload`
CODEBASE_PATH = os.environ.get("CODEBASE_PATH")
MODEL_NAME = r".\model"  # Path to the local model directory
TOP_K = 5

if not CODEBASE_PATH:
    raise RuntimeError(
        "CODEBASE_PATH environment variable is not set. "
        "Run with: CODEBASE_PATH=/path/to/repo uvicorn main:app --reload"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    indexer, embedder = index_codebase(CODEBASE_PATH, tokenizer)

    # Stash them on app.state so the endpoint below can access them
    app.state.indexer = indexer
    app.state.embedder = embedder

    yield  

    # Shutdown

app = FastAPI(lifespan=lifespan)

@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    embedder = app.state.embedder
    indexer = app.state.indexer

    query_embedding = embedder.embed_query(request.query)
    results = indexer.search(query_embedding, k=TOP_K)

    return results

