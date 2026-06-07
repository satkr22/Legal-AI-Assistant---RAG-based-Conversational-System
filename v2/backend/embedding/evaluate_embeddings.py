#!/usr/bin/env python3

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"   # SAME query model
TOP_K = 5

BASE_PATH = "data/processed/artifacts"
LARGE_PATH = "data/processed/artifacts2"


# -----------------------------
# LOAD FAISS + META
# -----------------------------
def load_index(path):
    index = faiss.read_index(f"{path}/faiss.index")
    with open(f"{path}/faiss_meta.json", "r") as f:
        meta = json.load(f)
    return index, meta


# -----------------------------
# QUERY EMBEDDING (BGE format)
# -----------------------------
def embed_query(model, query):
    prefix = "Represent this sentence for searching relevant passages: "
    q = prefix + query
    vec = model.encode([q], normalize_embeddings=True)
    return np.array(vec, dtype=np.float32)


# -----------------------------
# SEARCH
# -----------------------------
def search(index, meta, query_vec):
    scores, ids = index.search(query_vec, TOP_K)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        results.append({
            "score": float(score),
            "chunk_id": meta[idx]["chunk_id"],
            "citation": meta[idx].get("citation"),
            "section": meta[idx].get("section")
        })
    return results


# -----------------------------
# PRINT COMPARISON
# -----------------------------
def print_comparison(query, base_results, large_results):
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)

    print("\n--- BASE MODEL ---")
    for r in base_results:
        print(f"{r['score']:.4f} | {r['chunk_id']} | {r['citation']}")

    print("\n--- LARGE MODEL ---")
    for r in large_results:
        print(f"{r['score']:.4f} | {r['chunk_id']} | {r['citation']}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    # Load model (query embedding)
    # model = SentenceTransformer(MODEL_NAME, device="cuda")
    
    base_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    large_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

    # Load both indexes
    base_index, base_meta = load_index(BASE_PATH)
    large_index, large_meta = load_index(LARGE_PATH)

    # Test queries
    queries = [
        "I accidently hit a car in road while reversing what could happen now ??"
    ]
    # # Test queries
    # queries = [
    #     "What happens if crime is committed outside India?",
    #     "Definition of person under this act",
    #     "Punishment for false evidence",
    #     "What is explanation under section 1",
    #     "Illustration of offence"
    # ]

    for query in queries:
        # qvec = embed_query(model, query)
        base_qvec = embed_query(base_model, query)
        large_qvec = embed_query(large_model, query)

        base_results = search(base_index, base_meta, base_qvec)
        large_results = search(large_index, large_meta, large_qvec)

        print_comparison(query, base_results, large_results)


if __name__ == "__main__":
    main()