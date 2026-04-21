#!/usr/bin/env python3

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# pip install numpy tqdm rank-bm25 sentence-transformers faiss-cpu


INPUT_PATH = Path("data/processed/jsons/chunk_jsons/chunks.json")
OUTPUT_DIR = Path("data/processed/artifacts2")

FAISS_INDEX_PATH = OUTPUT_DIR / "faiss.index"
FAISS_META_PATH = OUTPUT_DIR / "faiss_meta.json"
BM25_PATH = OUTPUT_DIR / "bm25.pkl"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 8


# -----------------------------
# Load chunks
# -----------------------------
def load_chunks(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


# -----------------------------
# Only embed valid chunks
# -----------------------------
def get_retrieval_text(chunk):
    text = chunk.get("embedding_text")
    if text is None:
        return None
    text = text.strip()
    return text if text else None


# -----------------------------
# Embedding
# -----------------------------
def embed_texts(model, texts, batch_size=32):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


# -----------------------------
# BM25 tokenizer
# -----------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


# -----------------------------
# Build FAISS
# -----------------------------
def build_faiss(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    chunks = load_chunks(INPUT_PATH)

    indexed_chunks = []
    texts = []

    for chunk in chunks:
        t = get_retrieval_text(chunk)
        if t is None:
            continue

        indexed_chunks.append(chunk)
        texts.append(t)

    print(f"Total chunks: {len(chunks)}")
    print(f"Embedded chunks: {len(indexed_chunks)}")

    # Load model
    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    # -------------------------
    # 1. EMBEDDINGS
    # -------------------------
    vectors = embed_texts(model, texts, BATCH_SIZE)

    # -------------------------
    # 2. FAISS INDEX
    # -------------------------
    index = build_faiss(vectors)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # -------------------------
    # 3. METADATA (UPDATED)
    # -------------------------
    metadata = []

    for idx, chunk in enumerate(indexed_chunks):
        metadata.append({
            "id": idx,
            "chunk_id": chunk.get("chunk_id"),
            "chunk_type": chunk.get("chunk_type"),

            # STRUCTURE
            "chapter": chunk.get("chapter"),
            "section": chunk.get("section"),
            "citation": chunk.get("citation"),
            "node_ids": chunk.get("node_ids", [])
        })
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # -------------------------
    # 4. BM25 INDEX
    # -------------------------
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        chunk_ids = [chunk["chunk_id"] for chunk in indexed_chunks]
        pickle.dump({
            "bm25": bm25,
            "texts": texts,
            "chunk_ids": chunk_ids
        }, f)

    print("Phase 7 completed")
    print("FAISS + BM25 + Metadata ready")


if __name__ == "__main__":
    main()