import os
from semantic_units import extract_semantic_units
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path


model = SentenceTransformer("all-MiniLM-L6-v2")

EXCLUDED_DIRS = {".venv", "__pycache__", ".git"}

def iter_py_files(root: Path):
    """Yield only .py files, skipping excluded directories like .venv and __pycache__."""
    for path in root.rglob("*.py"):
        if not any(part in EXCLUDED_DIRS for part in path.parts):
            yield path


def build_vector_db(directory: Path):
    all_units = []
    embeddings = []

    for file_path in iter_py_files(directory):  # only safe .py files
        semantic_units = extract_semantic_units(file_path)

        for unit in semantic_units:
            emb = model.encode(unit["code"])
            unit["embedding"] = emb.tolist()
            all_units.append(unit)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError("No Python files found to index!")

    # Build FAISS index
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))

    return index, all_units
