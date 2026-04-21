from __future__ import annotations

"""Phase 9 — Hybrid Retrieval for legal RAG.

This version is built around the retrieval contract we settled on:

- Use the original Phase 8 query and each sub-query separately.
- Do not inject intent into retrieval text.
- Do not use target types in retrieval text.
- FAISS gets short natural-language text.
- BM25 gets sparse exact terms from the query, target values, and only a
  few strong concept words.
- Each query is retrieved and reranked independently.
- Then all query-specific results are merged.
- Finally, a global rerank using the original query is run on the merged set.

Critical indexing fix:
- FAISS result rows map only through faiss_meta.json.
- BM25 result rows map only through bm25.pkl chunk_ids.
- No fallback to self.chunks[idx] for FAISS/BM25 row mapping.
- Metadata length mismatches are validated early.

Output includes both:
- results_without_global_rerank
- results_with_global_rerank

Expected artifacts in --base-dir:
    - faiss.index
    - faiss_meta.json
    - bm25.pkl
    - chunks.json
    
    
    
    
    
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import math
import pickle
import re
from collections import defaultdict

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None
    CrossEncoder = None



EXPLANATORY_INTENTS = {
    "explanation",
    "reasoning",
    "hypothetical",
    "procedure",
    "comparison",
    "consequence",
    "legal_exception",
    "legal_condition",
    "legal_penalty",
    "case_application",
}


EXPLANATORY_TRIGGERS = (
    "what happens",
    "what will happen",
    "what should i do",
    "what can i do",
    "how",
    "why",
    "explain",
    "what is the effect",
    "what are the consequences",
)

BASE_CHUNK_TYPES = {"section", "subsection", "clause", "content"}
EXPLANATORY_CHUNK_TYPES = {"explanation", "illustration"}

GENERIC_QUERY_CONCEPTS = {
    "legal remedy",
    "legal issue",
    "conditional liability",
    "issue",
    "matter",
    "case",
    "thing",
    "person",
    "persons",
    "someone",
    "something",
    "general",
    "public",
    "legal",
}

BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by", "can", "could",
    "did", "do", "does", "for", "from", "got", "how", "i", "if", "in", "into", "is",
    "it", "me", "my", "of", "on", "or", "our", "should", "that", "the", "their",
    "then", "these", "they", "this", "to", "trying", "was", "we", "what", "when",
    "where", "which", "while", "who", "why", "with", "without", "would", "you", "your",
}

# Minimal legal expansions to help BM25 catch common paraphrases.
LEGAL_TOKEN_EXPANSIONS = {
    "injure": ["hurt"],
    "injured": ["hurt"],
    "injury": ["hurt"],
    "steal": ["theft"],
    "stole": ["theft"],
    "stolen": ["theft"],
    "thief": ["theft"],
    "phone": ["property"],
    "mobile": ["property"],
}

# Domain-aware lexical expansion for legal questions.
# These are intentionally small and precise so BM25 stays sparse.
LEGAL_CONTEXT_EXPANSION = {
    "hit": ["assault", "criminal force"],
    "beat": ["assault", "criminal force"],
    "beating": ["assault", "criminal force"],
    "hurt": ["causing hurt"],
    "harmed": ["hurt"],
    "harm": ["hurt"],
    "assault": ["criminal force"],
    "force": ["criminal force"],
    "strike": ["assault"],
    "struck": ["assault"],
}

# Tokens that are usually too generic to help legal retrieval on their own.
GENERIC_BM25_TERMS = {
    "case",
    "law",
    "legal",
    "matter",
    "person",
    "persons",
    "people",
    "someone",
    "something",
    "thing",
    "issue",
    "public",
    "section",
    "act",
    "provision",
    "rule",
    "code",
    "punishment",
}


# -----------------------------
# Helpers
# -----------------------------


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _safe_lower(text: Any) -> str:
    return str(text or "").strip().lower()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _unique_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        key = item if isinstance(item, (str, int, float, tuple)) else json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _normalize_scores_to_unit(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if math.isclose(mn, mx):
        return [1.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def _rank_scores(n: int) -> List[float]:
    return [1.0 / (i + 1) for i in range(n)]


def _normalize_confidence(conf: Any) -> Dict[str, float]:
    if isinstance(conf, dict):
        out: Dict[str, float] = {}
        for k, v in conf.items():
            if isinstance(v, (int, float)):
                out[str(k)] = float(v)
        return out
    if isinstance(conf, (int, float)):
        return {"overall": float(conf)}
    return {}


def _flatten_targets(targets: List[Dict[str, str]]) -> List[str]:
    vals: List[str] = []
    for t in targets or []:
        t_val = str(t.get("value", "")).strip()
        if not t_val:
            continue
        vals.append(t_val)
    return _unique_keep_order(vals)


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_phase8_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("results"), list):
        return [x for x in obj["results"] if isinstance(x, dict)]
    if "query" in obj and ("intent" in obj or "concepts" in obj or "sub_queries" in obj):
        return [obj]
    if isinstance(obj.get("result"), dict):
        return [obj]
    return [obj]


def _extract_phase8_result(item: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(item.get("result"), dict):
        return item["result"]
    return item


def _normalize_phrase(text: str) -> str:
    text = _normalize_space(text).lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return _normalize_space(text)


def _is_good_bm25_term(text: str) -> bool:
    phrase = _normalize_phrase(text)
    if not phrase or phrase in GENERIC_QUERY_CONCEPTS:
        return False
    toks = phrase.split()
    if not toks:
        return False
    if len(toks) > 5:
        return False
    if all(t in BM25_STOPWORDS for t in toks):
        return False
    return True


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    haystack = _safe_lower(text)
    return any(term in haystack for term in terms if term)


def _normalize_term_list(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        phrase = _normalize_phrase(value)
        if phrase and phrase not in out:
            out.append(phrase)
    return out


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Phase8Query:
    query: str
    intent: Dict[str, Any]
    targets: List[Dict[str, str]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    query_features: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    method: str = "rules"
    decomposition: Dict[str, Any] = field(default_factory=dict)
    sub_queries: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Phase8Query":
        
        sub_queries = data.get("sub_queries")
        if not sub_queries:
            sub_queries = data.get("decomposition", {}).get("sub_queries", [])
        
        return cls(
            query=str(data.get("query", "")).strip(),
            intent=dict(data.get("intent", {}) or {}),
            targets=list(data.get("targets", []) or []),
            concepts=list(data.get("concepts", []) or []),
            constraints=dict(data.get("constraints", {}) or {}),
            query_features=dict(data.get("query_features", {}) or {}),
            confidence=_normalize_confidence(data.get("confidence", {})),
            method=str(data.get("method", "rules")),
            decomposition=dict(data.get("decomposition", {}) or {}),
            # sub_queries=list(data.get("sub_queries", []) or []),
            sub_queries = list(sub_queries or []), 
            notes=list(data.get("notes", []) or []),
            )

    @property
    def primary_intent(self) -> str:
        return _safe_lower(self.intent.get("primary", "explanation"))

    @property
    def is_multi_hop(self) -> bool:
        return bool(self.query_features.get("is_multi_hop", False))

    @property
    def requires_reasoning(self) -> bool:
        return bool(self.query_features.get("requires_reasoning", False))

    @property
    def requires_exact_match(self) -> bool:
        return bool(self.query_features.get("requires_exact_match", False))

    @property
    def specificity(self) -> str:
        val = _safe_lower(self.constraints.get("specificity", "broad"))
        return val if val in {"broad", "narrow"} else "broad"

    @property
    def jurisdiction(self) -> Optional[str]:
        j = self.constraints.get("jurisdiction", None)
        if isinstance(j, str) and j.strip():
            return j.strip().lower()
        return None


@dataclass
class QueryVariant:
    name: str
    q8: Phase8Query
    weight: float
    source: str = "original"


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    query_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class ChunkRecord:
    chunk: Dict[str, Any]
    chunk_id: str
    chunk_type: str
    text: str
    embedding_text: str
    derived_context: str
    keywords: List[str]
    legal_concepts: List[str]
    parent_chunk_id: Optional[str]
    children_chunk_ids: List[str]
    references: List[Dict[str, Any]]
    explains_node_id: Optional[str]
    section_number: Optional[int]
    act: str
    chapter_title: Optional[str]

    @classmethod
    def from_dict(cls, chunk: Dict[str, Any]) -> "ChunkRecord":
        section = chunk.get("section", {}) or {}
        chapter = chunk.get("chapter", {}) or {}
        section_number = section.get("section_number")
        if isinstance(section_number, str) and section_number.isdigit():
            section_number = int(section_number)
        elif not isinstance(section_number, int):
            section_number = None

        return cls(
            chunk=chunk,
            chunk_id=str(chunk.get("chunk_id", "")),
            chunk_type=str(chunk.get("chunk_type", "")),
            text=str(chunk.get("text", "") or ""),
            embedding_text=str(chunk.get("embedding_text", "") or ""),
            derived_context=str(chunk.get("derived_context", "") or ""),
            keywords=[str(x).lower() for x in (chunk.get("keywords", []) or []) if str(x).strip()],
            legal_concepts=[str(x).lower() for x in (chunk.get("legal_concepts", []) or []) if str(x).strip()],
            parent_chunk_id=chunk.get("parent_chunk_id"),
            children_chunk_ids=list(chunk.get("children_chunk_ids", []) or []),
            references=list(chunk.get("references", []) or []),
            explains_node_id=chunk.get("explains_node_id"),
            section_number=section_number,
            act=str(chunk.get("act", "") or ""),
            chapter_title=chapter.get("chapter_title"),
        )

    @property
    def retrieval_text(self) -> str:
        parts = [self.embedding_text, self.derived_context, self.text]
        return "\n".join([p for p in parts if p])

    @property
    def citation(self) -> str:
        if self.section_number is not None:
            return f"Section {self.section_number}"
        return self.chunk_id


# -----------------------------
# Retriever
# -----------------------------


class Phase9HybridRetriever:
    def __init__(
        self,
        base_dir: str | Path,
        embed_model_name: str = "BAAI/bge-large-en-v1.5",
        rerank_model_name: str = "BAAI/bge-reranker-large",
        top_k_faiss: int = 20,
        top_k_bm25: int = 20,
        final_k: int = 5,
        graph_depth: int = 1,
        graph_max_extra: int = 4,
        enable_graph: bool = True,
        enable_rerank: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.embed_model_name = embed_model_name
        self.rerank_model_name = rerank_model_name
        self.top_k_faiss = top_k_faiss
        self.top_k_bm25 = top_k_bm25
        self.final_k = final_k
        self.graph_depth = graph_depth
        self.graph_max_extra = graph_max_extra
        self.enable_graph = enable_graph
        self.enable_rerank = enable_rerank
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        chunks_path = self.base_dir / "chunks.json"
        meta_path = self.base_dir / "faiss_meta.json"
        bm25_path = self.base_dir / "bm25.pkl"
        faiss_path = self.base_dir / "faiss.index"

        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks file: {chunks_path}")
        if not bm25_path.exists():
            raise FileNotFoundError(f"Missing BM25 file: {bm25_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing FAISS metadata file: {meta_path}")
        if faiss is not None and not faiss_path.exists():
            raise FileNotFoundError(f"Missing FAISS index file: {faiss_path}")

        chunks_doc = _load_json(chunks_path)
        self.meta = _load_json(meta_path)

        if isinstance(chunks_doc, dict):
            self.chunks: List[Dict[str, Any]] = list(chunks_doc.get("chunks", []) or [])
        elif isinstance(chunks_doc, list):
            self.chunks = chunks_doc
        else:
            self.chunks = []

        self.faiss_chunk_ids = self._extract_index_chunk_ids(self.meta)

        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"] if isinstance(bm25_data, dict) and "bm25" in bm25_data else bm25_data
        self.bm25_chunk_ids = self._extract_bm25_chunk_ids(bm25_data)

        self.chunk_records: Dict[str, ChunkRecord] = {}
        self.node_to_chunk_id: Dict[str, str] = {}
        self.section_to_chunk_ids: Dict[int, List[str]] = defaultdict(list)
        self.parent_to_children: Dict[str, List[str]] = defaultdict(list)

        for chunk in self.chunks:
            rec = ChunkRecord.from_dict(chunk)
            if not rec.chunk_id:
                continue
            self.chunk_records[rec.chunk_id] = rec
            for node_id in (chunk.get("node_ids", []) or []):
                self.node_to_chunk_id[str(node_id)] = rec.chunk_id
            if rec.section_number is not None:
                self.section_to_chunk_ids[rec.section_number].append(rec.chunk_id)
            if rec.parent_chunk_id:
                self.parent_to_children[rec.parent_chunk_id].append(rec.chunk_id)

        self.faiss_index = None
        if faiss is not None:
            self.faiss_index = faiss.read_index(str(faiss_path))

        if self.faiss_chunk_ids and self.faiss_index is not None and self.faiss_index.ntotal != len(self.faiss_chunk_ids):
            raise ValueError(
                f"FAISS metadata mismatch: index has {self.faiss_index.ntotal} vectors but metadata has {len(self.faiss_chunk_ids)} chunk ids."
            )

        corpus_size = getattr(self.bm25, "corpus_size", None)
        if self.bm25_chunk_ids and isinstance(corpus_size, int) and corpus_size != len(self.bm25_chunk_ids):
            raise ValueError(
                f"BM25 metadata mismatch: corpus has {corpus_size} documents but metadata has {len(self.bm25_chunk_ids)} chunk ids."
            )

        if SentenceTransformer is not None:
            self.model = SentenceTransformer(self.embed_model_name)
        else:
            self.model = None

        if self.enable_rerank and CrossEncoder is not None:
            self.reranker_model = CrossEncoder(self.rerank_model_name)
        else:
            self.reranker_model = None

    def _extract_index_chunk_ids(self, meta_doc: Any) -> List[str]:
        if isinstance(meta_doc, list):
            out: List[str] = []
            for item in meta_doc:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("chunk_id", "")).strip()
                if cid:
                    out.append(cid)
            return out

        if not isinstance(meta_doc, dict):
            return []

        for key in ("chunk_ids", "index_chunk_ids", "indexed_chunk_ids"):
            val = meta_doc.get(key)
            if isinstance(val, list):
                out = [str(x).strip() for x in val if str(x).strip()]
                if out:
                    return out

        for key in ("index_to_chunk_id", "faiss_index_to_chunk_id"):
            mapping = meta_doc.get(key)
            if not isinstance(mapping, dict):
                continue
            ordered: List[Tuple[int, str]] = []
            for k, v in mapping.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                cid = str(v).strip()
                if cid:
                    ordered.append((idx, cid))
            if ordered:
                ordered.sort(key=lambda x: x[0])
                return [cid for _, cid in ordered]

        return []

    def _extract_bm25_chunk_ids(self, bm25_data: Any) -> List[str]:
        if isinstance(bm25_data, dict):
            for key in ("chunk_ids", "indexed_chunk_ids", "doc_ids", "document_ids"):
                val = bm25_data.get(key)
                if isinstance(val, list):
                    out = [str(x).strip() for x in val if str(x).strip()]
                    if out:
                        return out
        elif isinstance(bm25_data, list):
            out = [str(x).strip() for x in bm25_data if str(x).strip()]
            if out:
                return out
        return []

    # -----------------------------
    # Query construction
    # -----------------------------

    def _strong_query_concepts(self, q8: Phase8Query, max_terms: int = 2) -> List[str]:
        """Return the strongest legal concepts for retrieval grounding."""
        strong: List[str] = []
        for concept in q8.concepts or []:
            phrase = _normalize_phrase(str(concept))
            if not phrase:
                continue
            if phrase in GENERIC_QUERY_CONCEPTS or phrase in GENERIC_BM25_TERMS:
                continue
            if not _is_good_bm25_term(phrase):
                continue
            if phrase in strong:
                continue
            strong.append(phrase)
            if len(strong) >= max_terms:
                break
        return strong

    def _clean_concepts_for_bm25(self, q8: Phase8Query, max_terms: int = 4) -> List[str]:
        """Keep BM25 concepts sparse and avoid noisy legal boilerplate."""
        return self._strong_query_concepts(q8, max_terms=max_terms)

    def _bm25_terms_for_query(self, q8: Phase8Query) -> List[str]:
        """Build a sparse BM25 term list with legal grounding and contextual expansion."""
        terms: List[str] = []

        strong_concepts = self._strong_query_concepts(q8, max_terms=2)
        terms.extend(strong_concepts)
        terms.extend(_flatten_targets(q8.targets))

        tokens = _tokenize(q8.query)
        has_strong_anchor = bool(strong_concepts or terms)

        expanded: List[str] = []
        for token in tokens:
            if token in BM25_STOPWORDS:
                continue

            # Drop generic terms unless the query already has a strong legal anchor.
            if token in GENERIC_BM25_TERMS and token != "punishment":
                continue
            if token == "punishment" and not has_strong_anchor:
                continue

            expanded.append(token)
            expanded.extend(LEGAL_CONTEXT_EXPANSION.get(token, []))
            expanded.extend(LEGAL_TOKEN_EXPANSIONS.get(token, []))

        for token in expanded:
            term = _normalize_phrase(token)
            if not term or term in BM25_STOPWORDS:
                continue
            if term in GENERIC_BM25_TERMS and term != "punishment":
                continue
            if term not in terms:
                terms.append(term)
            if len(terms) >= 10:
                break

        return _unique_keep_order(terms[:10])

    def build_faiss_text(self, q8: Phase8Query) -> str:
        # Short semantic text, but grounded with the top legal concepts.
        parts = [q8.query]
        parts.extend(self._strong_query_concepts(q8, max_terms=2))
        parts.extend(_flatten_targets(q8.targets))
        return _normalize_space(" ".join(_unique_keep_order(p for p in parts if p)))

    def build_bm25_tokens(self, q8: Phase8Query) -> List[str]:
        # Sparse exact terms only.
        return _tokenize(" ".join(self._bm25_terms_for_query(q8)))

    def _build_rerank_query(self, q8: Phase8Query) -> str:
        """Use the original query plus the top legal concepts for reranking."""
        parts = [q8.query]
        parts.extend(self._strong_query_concepts(q8, max_terms=2))
        parts.extend(_flatten_targets(q8.targets))
        return _normalize_space(" ".join(_unique_keep_order(p for p in parts if p)))

    def _build_query_variants(self, q8: Phase8Query) -> List[QueryVariant]:
        variants: List[QueryVariant] = [QueryVariant(name="original", q8=q8, weight=1.0, source="original")]
        seen = {q8.query.strip().lower()}

        for i, item in enumerate(q8.sub_queries or [], start=1):
            if not isinstance(item, dict):
                continue
            sub_q8 = Phase8Query.from_dict(item)
            nq = sub_q8.query.strip().lower()
            if not nq or nq in seen:
                continue
            seen.add(nq)
            variants.append(QueryVariant(name=f"sub_query_{i}", q8=sub_q8, weight=0.9, source="decomposed"))

        return variants

    # -----------------------------
    # Retrieval primitives
    # -----------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.model is None:
            return None
        query = f"Represent this legal query for retrieving relevant statutory chunks: {text}"
        emb = self.model.encode([query], normalize_embeddings=True)
        if np is not None:
            return emb[0].astype("float32").tolist()
        return emb[0].tolist()

    def _faiss_search(self, search_text: str, top_k: int) -> List[Tuple[str, float]]:
        if self.faiss_index is None:
            return []
        vec = self._embed(search_text)
        if vec is None:
            return []
        q = np.array([vec], dtype="float32") if np is not None else vec
        scores, indices = self.faiss_index.search(q, top_k)  # type: ignore[arg-type]

        out: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            chunk_id = self._chunk_id_from_faiss_index(int(idx))
            if not chunk_id:
                continue
            out.append((chunk_id, float(score)))
        return out

    def _chunk_id_from_faiss_index(self, index: int) -> Optional[str]:
        # The indexed subset order is authoritative.
        if 0 <= index < len(self.faiss_chunk_ids):
            return self.faiss_chunk_ids[index]

        # Metadata lookups for older/alternate layouts.
        if isinstance(self.meta, dict):
            for key in ("index_to_chunk_id", "faiss_index_to_chunk_id"):
                mapping = self.meta.get(key)
                if isinstance(mapping, dict):
                    val = mapping.get(str(index), mapping.get(index))
                    if val:
                        return str(val)
            for key in ("chunk_ids", "index_chunk_ids", "indexed_chunk_ids"):
                val = self.meta.get(key)
                if isinstance(val, list) and 0 <= index < len(val):
                    cid = str(val[index]).strip()
                    if cid:
                        return cid

        return None

    def _bm25_search(self, query_tokens: List[str], top_k: int) -> List[Tuple[str, float]]:
        if not query_tokens:
            return []
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception:
            return []

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        out: List[Tuple[str, float]] = []
        for idx, score in ranked:
            if 0 <= idx < len(self.bm25_chunk_ids):
                out.append((self.bm25_chunk_ids[idx], float(score)))
        return out

    def _collect_neighbors(self, chunk_id: str, depth: int = 1, max_extra: int = 4) -> List[str]:
        if depth <= 0 or max_extra <= 0:
            return []

        seen = {chunk_id}
        frontier = [chunk_id]
        out: List[str] = []

        for _ in range(depth):
            next_frontier: List[str] = []
            for cid in frontier:
                rec = self.chunk_records.get(cid)
                if not rec:
                    continue

                neighbors = set()
                if rec.parent_chunk_id:
                    neighbors.add(rec.parent_chunk_id)
                neighbors.update(rec.children_chunk_ids)
                neighbors.update(self.parent_to_children.get(cid, []))

                for ref in rec.references:
                    if not isinstance(ref, dict):
                        continue
                    if ref.get("target_chunk_id"):
                        neighbors.add(str(ref["target_chunk_id"]))
                    if ref.get("target_id"):
                        mapped = self.node_to_chunk_id.get(str(ref["target_id"]))
                        if mapped:
                            neighbors.add(mapped)
                    if ref.get("target_section") is not None:
                        sec = ref.get("target_section")
                        if isinstance(sec, str) and sec.isdigit():
                            sec = int(sec)
                        if isinstance(sec, int):
                            neighbors.update(self.section_to_chunk_ids.get(sec, []))

                for nid in neighbors:
                    if nid in seen:
                        continue
                    seen.add(nid)
                    out.append(nid)
                    next_frontier.append(nid)
                    if len(out) >= max_extra:
                        return out
            frontier = next_frontier
            if not frontier:
                break
        return out

    # -----------------------------
    # Companion attachment
    # -----------------------------

    def _should_attach_explanatory_material(self, q8: Phase8Query) -> bool:
        if q8.primary_intent in EXPLANATORY_INTENTS:
            return True
        if q8.requires_reasoning or q8.is_multi_hop:
            return True
        q = q8.query.lower()
        return any(trigger in q for trigger in EXPLANATORY_TRIGGERS)

    def _is_explanatory_chunk(self, rec: ChunkRecord) -> bool:
        return rec.chunk_type in EXPLANATORY_CHUNK_TYPES

    def _is_base_legal_chunk(self, rec: ChunkRecord) -> bool:
        return rec.chunk_type in BASE_CHUNK_TYPES

    def _resolve_node_or_chunk(self, target_id: str) -> Optional[str]:
        if not target_id:
            return None
        if target_id in self.chunk_records:
            return target_id
        return self.node_to_chunk_id.get(target_id)

    def _attach_hit(self, hits: Dict[str, RetrievalHit], chunk_id: str, source: str, score: float, reason: str) -> None:
        if not chunk_id:
            return
        hit = hits.get(chunk_id)
        if hit is None:
            hit = RetrievalHit(chunk_id=chunk_id, score=0.0)
            hits[chunk_id] = hit
        hit.source_scores[source] = max(hit.source_scores.get(source, 0.0), score)
        hit.query_scores[source] = max(hit.query_scores.get(source, 0.0), score)
        hit.reasons.append(reason)
        hit.score = max(hit.score, score)

    def _attach_anchor_chunks(self, hits: Dict[str, RetrievalHit]) -> Dict[str, RetrievalHit]:
        updated = dict(hits)
        for chunk_id, hit in list(hits.items()):
            rec = self.chunk_records.get(chunk_id)
            if not rec or not self._is_explanatory_chunk(rec):
                continue

            anchor_id: Optional[str] = None
            if rec.explains_node_id:
                anchor_id = self.node_to_chunk_id.get(str(rec.explains_node_id))
            if not anchor_id and rec.parent_chunk_id:
                anchor_id = rec.parent_chunk_id

            anchor_id = self._resolve_node_or_chunk(anchor_id or "")
            if not anchor_id or anchor_id in updated:
                continue

            anchor_rec = self.chunk_records.get(anchor_id)
            if not anchor_rec:
                continue

            new_score = max(0.75 * hit.score, hit.score - 0.10)
            self._attach_hit(updated, anchor_id, "anchor", new_score, f"anchor_of:{chunk_id}")

            if anchor_rec.chunk_type == "section" and rec.explains_node_id:
                precise = self.node_to_chunk_id.get(str(rec.explains_node_id))
                if precise and precise not in updated:
                    precise_score = max(0.78 * hit.score, hit.score - 0.06)
                    self._attach_hit(updated, precise, "anchor_precise", precise_score, f"precise_anchor_of:{chunk_id}")

        return updated

    def _attach_explanatory_companions(self, hits: Dict[str, RetrievalHit], q8: Phase8Query) -> Dict[str, RetrievalHit]:
        if not self._should_attach_explanatory_material(q8):
            return hits

        updated = dict(hits)
        for chunk_id, hit in list(hits.items()):
            rec = self.chunk_records.get(chunk_id)
            if not rec or not self._is_base_legal_chunk(rec):
                continue

            companion_ids: List[str] = []

            for child_id in rec.children_chunk_ids:
                child_rec = self.chunk_records.get(child_id)
                if child_rec and self._is_explanatory_chunk(child_rec):
                    companion_ids.append(child_id)

            for ref in rec.references:
                if not isinstance(ref, dict):
                    continue
                relation = _safe_lower(ref.get("relation"))
                if relation not in {"explains", "illustrates", "explanation"}:
                    continue
                target_id = ref.get("target_id") or ref.get("target_chunk_id")
                if not target_id:
                    continue
                resolved = self._resolve_node_or_chunk(str(target_id))
                if not resolved:
                    continue
                child_rec = self.chunk_records.get(resolved)
                if child_rec and self._is_explanatory_chunk(child_rec):
                    companion_ids.append(resolved)

            for comp_id in _unique_keep_order(companion_ids):
                if comp_id in updated:
                    continue
                comp_rec = self.chunk_records.get(comp_id)
                if not comp_rec:
                    continue
                comp_score = max(0.72 * hit.score, hit.score - 0.12)
                if comp_rec.chunk_type == "illustration":
                    comp_score = max(comp_score, 0.68 * hit.score)
                self._attach_hit(updated, comp_id, "explanatory_companion", comp_score, f"companion_of:{chunk_id}")

        return updated

    def _apply_retrieval_scoring_rules(self, q8: Phase8Query, hits: Dict[str, RetrievalHit]) -> Dict[str, RetrievalHit]:
        """
        Apply lightweight legal-domain heuristics before reranking.

        This keeps the architecture intact while nudging obvious matches
        (assault / criminal force / hurt) ahead of generic punishment-only
        or public-servant / hospital noise.
        """
        updated: Dict[str, RetrievalHit] = {}
        q_lower = q8.query.lower()
        has_force_context = any(term in q_lower for term in ("assault", "hurt", "force"))
        strong_concepts = self._strong_query_concepts(q8, max_terms=3)

        irrelevant_terms = ("public servant", "hospital", "judicial proceeding")

        for chunk_id, hit in hits.items():
            rec = self.chunk_records.get(chunk_id)
            if not rec:
                continue

            score = float(hit.score)
            reasons = list(hit.reasons)

            rec_concepts = _normalize_term_list(rec.legal_concepts)
            if strong_concepts and any(concept in rec_concepts for concept in strong_concepts):
                score *= 1.2
                reasons.append("concept_match_boost")

            rec_text = _safe_lower(rec.text)
            if _contains_any(rec_text, irrelevant_terms) and not has_force_context:
                score *= 0.6
                reasons.append("domain_penalty")

            if rec.chunk_type == "content":
                score *= 1.1
                reasons.append("content_boost")
            elif rec.chunk_type == "illustration":
                score *= 0.9
                reasons.append("illustration_penalty")

            hit.score = _clamp(score)
            hit.reasons = _unique_keep_order(reasons)
            updated[chunk_id] = hit

        return updated

    # -----------------------------
    # Scoring / reranking
    # -----------------------------
    # -----------------------------
    # Scoring / reranking
    # -----------------------------

    def _retrieve_for_variant(self, variant: QueryVariant) -> Dict[str, RetrievalHit]:
        q8 = variant.q8
        faiss_text = self.build_faiss_text(q8)
        bm25_tokens = self.build_bm25_tokens(q8)

        faiss_hits = self._faiss_search(faiss_text, self.top_k_faiss)
        bm25_hits = self._bm25_search(bm25_tokens, self.top_k_bm25)

        hits: Dict[str, RetrievalHit] = {}

        def add_hit(chunk_id: str, source: str, score: float, reason: str) -> None:
            if not chunk_id:
                return
            hit = hits.get(chunk_id)
            if hit is None:
                hit = RetrievalHit(chunk_id=chunk_id, score=0.0)
                hits[chunk_id] = hit
            hit.source_scores[source] = max(hit.source_scores.get(source, 0.0), score)
            hit.query_scores[variant.name] = max(hit.query_scores.get(variant.name, 0.0), score)
            hit.reasons.append(reason)
            hit.score = max(hit.score, score)

        for rank, (chunk_id, raw_score) in enumerate(faiss_hits):
            rank_score = _rank_scores(len(faiss_hits))[rank]
            faiss_score = max(rank_score, _clamp(float(raw_score)))
            add_hit(chunk_id, "faiss", faiss_score * variant.weight, f"faiss:{variant.name}")

        bm25_raw = [s for _, s in bm25_hits]
        bm25_norm = _normalize_scores_to_unit(bm25_raw)
        for rank, ((chunk_id, _raw), norm_score) in enumerate(zip(bm25_hits, bm25_norm)):
            bm25_score = max(_rank_scores(len(bm25_hits))[rank], norm_score)
            add_hit(chunk_id, "bm25", bm25_score * variant.weight, f"bm25:{variant.name}")

        hits = self._apply_retrieval_scoring_rules(q8, hits)
        return hits

    def _rerank(self, q8: Phase8Query, hits: Dict[str, RetrievalHit], limit: Optional[int] = None) -> List[RetrievalHit]:
        if limit is None:
            limit = self.final_k

        ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)
        ranked = ranked[: max(limit * 4, 20)]

        query_text = self._build_rerank_query(q8)
        if self.reranker_model is None or len(ranked) <= 1:
            return ranked[:limit]

        pairs: List[List[str]] = []
        keep: List[RetrievalHit] = []
        for hit in ranked:
            rec = self.chunk_records.get(hit.chunk_id)
            if not rec:
                continue
            pairs.append([query_text, rec.retrieval_text or rec.text])
            keep.append(hit)

        if not pairs:
            return ranked[:limit]

        try:
            raw_scores = self.reranker_model.predict(pairs)
        except Exception:
            return ranked[:limit]

        if np is not None:
            raw_scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        norm = _normalize_scores_to_unit([float(x) for x in raw_scores])
        for hit, rr in zip(keep, norm):
            hit.score = _clamp(0.4 * hit.score + 0.6 * rr)
            hit.reasons.append("rerank")

        ranked.sort(key=lambda h: h.score, reverse=True)
        return ranked[:limit]

    def _merge_variant_hits(self, per_variant: Dict[str, List[RetrievalHit]]) -> Dict[str, RetrievalHit]:
        merged: Dict[str, RetrievalHit] = {}
        for variant_name, hits in per_variant.items():
            for hit in hits:
                m = merged.get(hit.chunk_id)
                if m is None:
                    m = RetrievalHit(chunk_id=hit.chunk_id, score=0.0)
                    merged[hit.chunk_id] = m
                m.source_scores.update({f"{variant_name}:{k}": v for k, v in hit.source_scores.items()})
                m.query_scores[variant_name] = max(m.query_scores.get(variant_name, 0.0), hit.score)
                m.reasons.extend(hit.reasons)
                m.score = max(m.score, hit.score)

        for hit in merged.values():
            if len(hit.query_scores) > 1:
                hit.score = _clamp(hit.score * 1.15)
                hit.reasons.append("multi_query_support")

        return merged
    def _graph_expand(self, hits: Dict[str, RetrievalHit], q8: Phase8Query) -> Dict[str, RetrievalHit]:
        if not self.enable_graph:
            return hits
        if not (q8.is_multi_hop or q8.decomposition.get("needed") or len(q8.sub_queries or []) > 1):
            return hits

        expanded = dict(hits)
        seeds = [cid for cid, _ in sorted(((cid, h.score) for cid, h in hits.items()), key=lambda x: x[1], reverse=True)[: self.final_k]]
        added = 0

        for seed in seeds:
            neighbors = self._collect_neighbors(seed, depth=self.graph_depth, max_extra=self.graph_max_extra)
            for nid in neighbors:
                if nid in expanded:
                    continue
                rec = self.chunk_records.get(nid)
                if not rec:
                    continue
                base = hits[seed].score if seed in hits else 0.2
                score = _clamp(base * 0.82)
                expanded[nid] = RetrievalHit(
                    chunk_id=nid,
                    score=score,
                    source_scores={"graph": score},
                    query_scores={"graph": score},
                    reasons=[f"graph_from:{seed}"],
                )
                added += 1
                if added >= self.graph_max_extra:
                    return expanded
        return expanded

    def _format_results(self, hits: List[RetrievalHit]) -> List[Dict[str, Any]]:
        results = []
        for rank, hit in enumerate(hits, start=1):
            rec = self.chunk_records.get(hit.chunk_id)
            if not rec:
                continue
            results.append(
                {
                    "rank": rank,
                    "chunk_id": hit.chunk_id,
                    "score": round(float(hit.score), 6),
                    "chunk_type": rec.chunk_type,
                    "act": rec.act,
                    "chapter_title": rec.chapter_title,
                    "section_number": rec.section_number,
                    "citation": rec.citation,
                    "text": rec.text,
                    "derived_context": rec.derived_context,
                    "keywords": rec.keywords,
                    "legal_concepts": rec.legal_concepts,
                    "source_scores": hit.source_scores,
                    "query_scores": hit.query_scores,
                    "reasons": _unique_keep_order(hit.reasons),
                }
            )
        return results

    def _process_single_query(self, q8: Phase8Query, variant_name: str) -> List[RetrievalHit]:
        variant = QueryVariant(
            name=variant_name,
            q8=q8,
            weight=1.0 if variant_name == "original" else 0.9,
            source="original" if variant_name == "original" else "decomposed",
        )
        raw_hits = self._retrieve_for_variant(variant)
        raw_hits = self._attach_anchor_chunks(raw_hits)
        raw_hits = self._attach_explanatory_companions(raw_hits, q8)
        return self._rerank(q8, raw_hits, limit=self.final_k)

    # -----------------------------
    # Public API
    # -----------------------------

    def retrieve_one(self, phase8_item: Dict[str, Any]) -> Dict[str, Any]:
        item = _extract_phase8_result(phase8_item)
        q8 = Phase8Query.from_dict(item)
        if not q8.query:
            return {"query": "", "results": [], "meta": {"error": "empty query"}}

        variants = self._build_query_variants(q8)

        per_variant_reranked: Dict[str, List[RetrievalHit]] = {}
        for variant in variants:
            per_variant_reranked[variant.name] = self._process_single_query(variant.q8, variant.name)

        merged = self._merge_variant_hits(per_variant_reranked)
        merged = self._graph_expand(merged, q8)

        results_without_global_rerank = self._format_results(
            sorted(merged.values(), key=lambda h: h.score, reverse=True)[: self.final_k]
        )

        global_hits = self._rerank(q8, dict(merged), limit=self.final_k)
        results_with_global_rerank = self._format_results(global_hits)

        return {
            "query": q8.query,
            "phase8": item,
            "retrieval": {
                "variants": [
                    {
                        "name": v.name,
                        "source": v.source,
                        "weight": v.weight,
                        "query": v.q8.query,
                        "targets": _flatten_targets(v.q8.targets),
                        "concepts": v.q8.concepts,
                    }
                    for v in variants
                ],
                "per_query_results": {
                    name: self._format_results(hits)
                    for name, hits in per_variant_reranked.items()
                },
                "results_without_global_rerank": results_without_global_rerank,
                "results_with_global_rerank": results_with_global_rerank,
                "settings": {
                    "graph_enabled": self.enable_graph,
                    "rerank_enabled": self.reranker_model is not None,
                    "final_k": self.final_k,
                },
            },
        }

    def retrieve_many(self, phase8_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.retrieve_one(item) for item in phase8_items]


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 9 hybrid retrieval for legal RAG")
    p.add_argument("--phase8", required=True, help="Path to Phase 8 JSON output or cleaned_results.json")
    p.add_argument("--base-dir", required=True, help="Directory containing faiss.index, faiss_meta.json, bm25.pkl, chunks.json")
    p.add_argument("--output", default=None, help="Optional output JSON file")
    p.add_argument("--k", type=int, default=5, help="Final number of chunks to keep")
    p.add_argument("--top-k-faiss", type=int, default=20, help="FAISS candidate count per variant")
    p.add_argument("--top-k-bm25", type=int, default=20, help="BM25 candidate count per variant")
    p.add_argument("--graph-depth", type=int, default=1, help="Graph expansion depth")
    p.add_argument("--graph-max-extra", type=int, default=2, help="Max extra chunks from graph expansion")
    p.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5", help="SentenceTransformer embedding model")
    p.add_argument("--rerank-model", default="BAAI/bge-reranker-large", help="CrossEncoder reranker model")
    p.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    p.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    phase8_obj = _load_json(args.phase8)
    phase8_items = _normalize_phase8_items(phase8_obj)

    retriever = Phase9HybridRetriever(
        base_dir=args.base_dir,
        embed_model_name=args.embed_model,
        rerank_model_name=args.rerank_model,
        top_k_faiss=args.top_k_faiss,
        top_k_bm25=args.top_k_bm25,
        final_k=args.k,
        graph_depth=args.graph_depth,
        graph_max_extra=args.graph_max_extra,
        enable_graph=not args.no_graph,
        enable_rerank=not args.no_rerank,
    )

    output = retriever.retrieve_many(phase8_items)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(output)} retrieval item(s) to {out_path}")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




'''

python retrieval/rq.py --phase8 query_analysis/result_2_.json  --base-dir data/processed/artifacts2  --k 5  --output retrieval/output_2_.json  --no-graph

'''
