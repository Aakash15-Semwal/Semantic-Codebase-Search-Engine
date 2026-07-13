from pathlib import Path

from .chunker import CodeChunker
from .embedder import CodeEmbedder
from .indexer import CodeIndexer

EXCLUDED_DIRS = {
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    "node_modules",
}

def find_python_files(root_path):
    """Recursively find every .py file under root_path."""
    python_files = []
    for file_path in Path(root_path).rglob("*.py"):
        # Skip files inside excluded directories
        if any(part in EXCLUDED_DIRS for part in file_path.parts):
            continue
        python_files.append(file_path)

    return python_files


def index_codebase(root_path, tokenizer, chunker=None, embedder=None, indexer=None):
    """
    Walks root_path, chunks every .py file, embeds all chunks, and builds
    one combined FAISS index + metadata store.

    Returns the ready-to-use CodeIndexer instance.
    """
    chunker = chunker or CodeChunker(tokenizer)
    embedder = embedder or CodeEmbedder()
    indexer = indexer or CodeIndexer()

    py_files = find_python_files(root_path)

    all_chunks = []
    for file_path in py_files:
        try:
            file_chunks = chunker.chunk_file(str(file_path))
            all_chunks.extend(file_chunks)
        except Exception as e:
            # Skip unindexable files (e.g., syntax errors, encoding issues) but log the error.
            print(f"Skipping {file_path}: {e}")

    if not all_chunks:
        raise ValueError(f"No chunks produced from {root_path} — check the path and file contents.")

    embedder.embed_chunks(all_chunks)
    indexer.build_index(all_chunks)

    return indexer, embedder

