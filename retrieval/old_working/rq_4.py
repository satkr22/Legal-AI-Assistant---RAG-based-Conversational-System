from __future__ import annotations

"""Phase 9 — Hybrid Retrieval for legal RAG.

Retrieval contract
------------------
- Use the original Phase 8 query and each sub-query separately.
- Do not inject intent into retrieval text.
- FAISS gets an enriched natural-language query (original + top concepts).
- BM25 gets a cleaned, concept-prioritized sparse term list.
- Each query variant is retrieved and per-variant reranked independently.
- All variant results are then merged with a multi-query support boost.
- A final global rerank on the merged set uses the enriched query.

Design philosophy (v2 — fully generalized)
-------------------------------------------
- NO hardcoded legal domain terms, keywords, or category lists.
- Query weakness is detected from surface signals (generic tokens, short
  length, absence of substantive concepts) — not domain-specific patterns.
- Domain relevance is measured via concept-set overlap between the query and
  each retrieved chunk — adapts automatically to any legal topic.
- General vs. specialized clause distinction uses structural text signals
  (conditional markers, text length) — no topic-specific heuristics.
- BM25 query construction uses Phase 8 concepts as the primary signal,
  supplemented by meaningful query tokens after stopword/noise removal.

Critical indexing notes
-----------------------
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

# Concepts that carry no discriminative domain signal — too generic to ground retrieval.
# Used to filter q8.concepts before injecting them into queries.
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

# Universal English stopwords — removed from all BM25 term lists.
BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "before", "by", "can", "could",
    "did", "do", "does", "for", "from", "got", "how", "i", "if", "in", "into", "is",
    "it", "me", "my", "of", "on", "or", "our", "should", "that", "the", "their",
    "then", "these", "they", "this", "to", "trying", "was", "we", "what", "when",
    "where", "which", "while", "who", "why", "with", "without", "would", "you", "your",
}

# Generic surface tokens that appear across many unrelated legal sections and add
# no discriminative BM25 signal on their own. These are structural/procedural words,
# not domain concepts — their presence in a query dilutes precision.
# NOTE: No legal-domain specific keywords here. This list is purely about surface
# generality — words whose IDF across a legal corpus is near zero.
BM25_SURFACE_NOISE = {
    "punishment", "punish", "punished",       # appears in virtually every penal section
    "law", "legal", "illegal",                # structural labels, not domain concepts
    "case", "matter", "issue",                # generic legal discourse words
    "person", "persons", "someone", "anyone", # near-universal legal subjects
    "something", "anything",                  # placeholder references
    "general", "public",                      # governance-level words
    "crime", "criminal", "offense", "offence",# near-universal penal labels
    "act", "section", "provision", "clause",  # document-structural terms
}

# Tokens that, when present in a query, indicate it is semantically weak — it lacks
# domain grounding and will benefit from concept injection.
# These are surface-level question words and placeholder nouns, not domain terms.
WEAK_QUERY_TOKENS = {
    "punishment", "law", "legal", "illegal", "crime", "criminal",
    "offense", "offence", "happens", "happen", "case", "matter",
    "issue", "situation", "scenario", "provision", "section", "act",
}

# Conditional markers that signal a highly specific or exceptional clause.
# Clauses heavily loaded with these tend to be niche provisions rather than
# core definitions — used for the general-vs-specialized heuristic.
CONDITIONAL_MARKERS = {
    "if", "when", "unless", "except", "provided", "notwithstanding",
    "subject to", "in case", "where", "only if", "shall not", "may not",
}

# Maximum BM25 query terms — keeps the term list focused and avoids IDF dilution.
BM25_MAX_TERMS = 10

# Minimum fraction of query tokens that must be substantive (non-noise, non-stopword)
# for a query to be considered semantically strong enough without concept injection.
WEAK_QUERY_SUBSTANCE_THRESHOLD = 0.4

# Concept overlap scoring multiplier per matching concept (Task 2).
# e.g. 2 matching concepts → score * (1 + 0.1 * 2) = score * 1.2
CONCEPT_MATCH_MULTIPLIER = 0.1

# Score penalty factor for chunks with zero concept overlap with the query (Task 3).
ZERO_OVERLAP_PENALTY = 0.6

# Score boost for general/broad clauses (Task 4).
GENERAL_CLAUSE_BOOST = 1.05

# Score penalty for highly conditional/niche clauses (Task 4).
CONDITIONAL_CLAUSE_PENALTY = 0.9

# Minimum word count below which a chunk is considered definitionally short/broad.
GENERAL_CLAUSE_MAX_WORDS = 80

# Minimum conditional marker density to flag a clause as highly specialized.
# Ratio = (number of conditional marker tokens) / (total tokens in text)
CONDITIONAL_MARKER_DENSITY_THRESHOLD = 0.06

# Multi-query support boost multiplier (Task 6).
MULTI_QUERY_BOOST = 1.1

# Chunk-type scoring multipliers (Task 5 — kept from v1, domain-agnostic).
CONTENT_CHUNK_BOOST = 1.1
ILLUSTRATION_CHUNK_PENALTY = 0.9


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
    """Return True if *text* is a clean, non-trivial BM25 term."""
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


# -----------------------------------------------------------------------
# Task 1 — Generic query enrichment
# -----------------------------------------------------------------------

def _is_weak_query(query: str, concepts: List[str]) -> bool:
    """Detect whether a query lacks sufficient domain-specific grounding.

    A query is "weak" when it is dominated by generic surface tokens and has
    few or no substantive Phase 8 concepts attached. This is a
    domain-agnostic signal — it does not test for any particular legal topic.

    Detection criteria (any one suffices):
    1. Fewer than 40 % of non-stopword tokens are substantive
       (i.e., most tokens are in WEAK_QUERY_TOKENS or BM25_STOPWORDS).
    2. No non-generic Phase 8 concepts are available.
    3. The raw query is very short (≤ 3 content words).
    """
    tokens = _tokenize(query)
    content_tokens = [t for t in tokens if t not in BM25_STOPWORDS]
    if not content_tokens:
        return True

    # Fraction of content tokens that are substantive (not generic noise).
    substantive = [t for t in content_tokens if t not in WEAK_QUERY_TOKENS]
    substance_ratio = len(substantive) / len(content_tokens)

    # Check whether any non-generic concepts exist.
    good_concepts = [
        c for c in (concepts or [])
        if _normalize_phrase(c) not in GENERIC_QUERY_CONCEPTS
    ]

    if not good_concepts:
        return True  # No domain anchor at all — always weak.

    if len(content_tokens) <= 3:
        return True  # Too short to carry its own domain signal.

    return substance_ratio < WEAK_QUERY_SUBSTANCE_THRESHOLD


def enrich_query(query: str, concepts: List[str]) -> str:
    """Task 1: Return a semantically enriched query for FAISS retrieval.

    When the original query is weak (generic, short, concept-free), the top
    Phase 8 concepts are appended to ground the embedding in the correct legal
    domain. The function is fully domain-agnostic — it relies only on the
    concept list supplied by Phase 8, not any hardcoded topic keywords.

    Algorithm
    ---------
    1. Strip generic concepts from the candidate injection list.
    2. Detect query weakness via _is_weak_query().
    3. If weak — or even if not weak — always inject the top 2 substantive
       concepts for additional semantic grounding (they are deduplicated below).
    4. Normalize whitespace and deduplicate tokens while preserving order.

    Examples
    --------
    query="What is the punishment?", concepts=["theft", "fraud"]
    → "What is the punishment theft fraud"

    query="theft dishonest intent", concepts=["theft", "dishonest intention"]
    → "theft dishonest intent intention"  (concepts deduplicated at token level)
    """
    good_concepts = [
        c for c in (concepts or [])
        if _normalize_phrase(c) not in GENERIC_QUERY_CONCEPTS
    ]

    if not good_concepts:
        return _normalize_space(query)

    # Always inject the top 2 concepts; deduplication happens at token level.
    injection = good_concepts[:2] if not _is_weak_query(query, concepts) else good_concepts[:3]

    combined = query + " " + " ".join(injection)

    # Deduplicate tokens while preserving order (prevents redundancy when
    # concepts already appear verbatim in the query).
    seen_tokens: set = set()
    deduped_parts: List[str] = []
    for token in _normalize_space(combined).split():
        lower = token.lower()
        if lower not in seen_tokens:
            seen_tokens.add(lower)
            deduped_parts.append(token)

    return " ".join(deduped_parts)


# -----------------------------------------------------------------------
# Task 3 — Concept overlap measurement (domain-agnostic)
# -----------------------------------------------------------------------

def _compute_concept_overlap(
    query_concepts: set,
    chunk_concepts: List[str],
) -> int:
    """Count how many chunk legal_concepts match the query concept set.

    Both sides are lowercased. Multi-word concepts (e.g. "dishonest intention")
    are matched as whole strings, not individual tokens, so there is no
    accidental partial matching.

    Returns an integer ≥ 0; higher means more domain alignment.
    """
    if not query_concepts or not chunk_concepts:
        return 0
    return sum(1 for c in chunk_concepts if c in query_concepts)


# -----------------------------------------------------------------------
# Task 4 — General vs. Specialized clause heuristics (domain-agnostic)
# -----------------------------------------------------------------------

def _is_general_clause(text: str) -> bool:
    """Return True if the chunk text looks like a broad, definitional clause.

    Heuristics (no domain-specific keywords):
    - Short text (few words) → more likely a core definition.
    - Low density of conditional markers → not heavily qualified.
    """
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return False

    # Short chunks tend to be high-level definitions or statements.
    if word_count > GENERAL_CLAUSE_MAX_WORDS:
        return False

    # Even short chunks with heavy qualification are not truly "general".
    tokens_lower = {w.lower().strip(".,;:()[]") for w in words}
    conditional_hits = sum(
        1 for marker in CONDITIONAL_MARKERS
        if any(marker in " ".join(words[i:i+3]).lower() for i in range(len(words)))
    )
    density = conditional_hits / word_count
    return density < CONDITIONAL_MARKER_DENSITY_THRESHOLD


def _is_highly_conditional(text: str) -> bool:
    """Return True if the chunk is heavily qualified with conditional language.

    A high density of conditional markers (if, unless, provided, subject to …)
    indicates a specialized exception or niche provision rather than a general rule.
    Domain-agnostic: works for any legal corpus.
    """
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return False

    conditional_hits = sum(
        1 for marker in CONDITIONAL_MARKERS
        if any(marker in " ".join(words[i:i+3]).lower() for i in range(len(words)))
    )
    density = conditional_hits / word_count
    return density >= CONDITIONAL_MARKER_DENSITY_THRESHOLD


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

    def _clean_concepts_for_bm25(self, q8: Phase8Query, max_terms: int = 4) -> List[str]:
        terms: List[str] = []
        for concept in q8.concepts or []:
            phrase = _normalize_phrase(str(concept))
            if not phrase:
                continue
            if phrase in GENERIC_QUERY_CONCEPTS:
                continue
            if not _is_good_bm25_term(phrase):
                continue
            if phrase in terms:
                continue
            terms.append(phrase)
            if len(terms) >= max_terms:
                break
        return terms

    def _bm25_terms_for_query(self, q8: Phase8Query) -> List[str]:
        """Task 5: Build a clean, focused BM25 term list — fully domain-agnostic.

        Term priority (highest to lowest discriminative value):
        1. Target values from Phase 8 (section numbers, named entities).
        2. Phase 8 legal concepts (domain-specific, extracted by upstream NLP).
        3. Substantive query tokens after removing stopwords and surface noise.

        No hardcoded legal-domain expansions. Phase 8 concepts are the sole
        mechanism for domain enrichment — they are produced by the upstream
        concept extractor and cover any legal topic automatically.
        """
        terms: List[str] = []

        # 1. High-precision target values (e.g. "section 131", "IPC").
        terms.extend(_flatten_targets(q8.targets))

        # 2. Phase 8 concepts — the strongest discriminative signal.
        #    Use a generous cap (6) to preserve multi-word concepts.
        terms.extend(self._clean_concepts_for_bm25(q8, max_terms=6))

        # 3. Substantive query tokens — fill remaining budget.
        tokens = _tokenize(q8.query)
        for token in tokens:
            if len(terms) >= BM25_MAX_TERMS:
                break
            if token in BM25_STOPWORDS:
                continue
            if token in BM25_SURFACE_NOISE:
                # Skip high-frequency generic tokens that span all sections.
                continue
            if token not in terms:
                terms.append(token)

        return _unique_keep_order(terms)

    def build_faiss_text(self, q8: Phase8Query) -> str:
        """Task 1: Build an enriched FAISS query text using enrich_query().

        Delegates all enrichment logic to the module-level enrich_query()
        function, which is domain-agnostic and reusable outside this class.
        Target values are appended after enrichment for additional specificity.
        """
        enriched = enrich_query(q8.query, q8.concepts)

        # Append explicit target values (section numbers, named values).
        targets = _flatten_targets(q8.targets)
        if targets:
            enriched = _normalize_space(enriched + " " + " ".join(targets))

        return enriched

    def build_bm25_tokens(self, q8: Phase8Query) -> List[str]:
        # Sparse exact terms only.
        return _tokenize(" ".join(self._bm25_terms_for_query(q8)))

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

    # -----------------------------
    # Scoring / reranking
    # -----------------------------

    def _retrieve_for_variant(self, variant: QueryVariant) -> Dict[str, RetrievalHit]:
        """Retrieve candidates via FAISS + BM25, then apply domain-agnostic scoring.

        Post-retrieval adjustments (in order):
        Task 2 — Semantic concept boost:    proportional to concept overlap count.
        Task 3 — Domain relevance filter:   penalise zero-overlap chunks.
        Task 4 — Clause specificity:        boost general definitions, penalise niche.
        Task 5 — Chunk type priority:       content > illustration.

        None of these adjustments reference hardcoded domain keywords.
        """
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

        # Build the query concept set once — lowercased, generic concepts excluded.
        # This is the domain "fingerprint" of the query against which every chunk
        # will be compared. No topic-specific keywords needed.
        query_concepts: set = {
            _safe_lower(c)
            for c in (q8.concepts or [])
            if _normalize_phrase(c) not in GENERIC_QUERY_CONCEPTS
        }

        for chunk_id, hit in hits.items():
            rec = self.chunk_records.get(chunk_id)
            if rec is None:
                continue

            score = hit.score

            # ------------------------------------------------------------------
            # Task 2: Semantic concept boost — proportional to overlap count.
            #
            # Each matching concept between the query and the chunk adds a fixed
            # multiplier increment. Two matches → ×1.2, three → ×1.3, etc.
            # This is fully domain-agnostic: the concept set is whatever Phase 8
            # extracted (theft, fraud, defamation, murder … anything).
            # ------------------------------------------------------------------
            overlap = _compute_concept_overlap(query_concepts, rec.legal_concepts)
            if overlap > 0:
                score *= 1.0 + CONCEPT_MATCH_MULTIPLIER * overlap
                hit.reasons.append(f"concept_overlap:{overlap}")

            # ------------------------------------------------------------------
            # Task 3: Domain relevance filter — penalise zero-overlap chunks.
            #
            # If a chunk shares no concepts with the query at all, it is likely
            # from an unrelated legal domain. Apply a soft penalty so it drops
            # in rank without being fully excluded (it might still be retrieved
            # by FAISS for good embedding-level reasons).
            #
            # Only apply when the query itself has substantive concepts — if the
            # query has no concepts we have no domain fingerprint to filter by.
            # ------------------------------------------------------------------
            if query_concepts and overlap == 0:
                score *= ZERO_OVERLAP_PENALTY
                hit.reasons.append("zero_concept_overlap_penalty")

            # ------------------------------------------------------------------
            # Task 4: General vs. specialized clause balancing.
            #
            # Core definitions and short, unconditional provisions should rank
            # above heavily qualified niche clauses. This heuristic uses only
            # structural text signals — no domain keywords.
            # ------------------------------------------------------------------
            if _is_general_clause(rec.text):
                score *= GENERAL_CLAUSE_BOOST
                hit.reasons.append("general_clause_boost")
            elif _is_highly_conditional(rec.text):
                score *= CONDITIONAL_CLAUSE_PENALTY
                hit.reasons.append("conditional_clause_penalty")

            # ------------------------------------------------------------------
            # Task 5: Chunk-type priority.
            #
            # Primary statutory content should outrank illustrative examples.
            # Illustrations are valuable companions but must not displace text.
            # ------------------------------------------------------------------
            if rec.chunk_type == "content":
                score *= CONTENT_CHUNK_BOOST
                hit.reasons.append("content_type_boost")
            elif rec.chunk_type == "illustration":
                score *= ILLUSTRATION_CHUNK_PENALTY
                hit.reasons.append("illustration_type_penalty")

            hit.score = _clamp(score)

        return hits

    def _rerank(self, query_text: str, hits: Dict[str, RetrievalHit], limit: Optional[int] = None, q8: Optional["Phase8Query"] = None) -> List[RetrievalHit]:
        """Task 8: Rerank candidates using a concept-enriched query.

        The CrossEncoder receives the same enriched query that FAISS uses,
        ensuring consistent domain grounding across all retrieval stages.
        Score normalization is applied across all candidates before combining
        with the pre-rerank score (weighted 40/60 hybrid).
        """
        if limit is None:
            limit = self.final_k

        ranked = sorted(hits.values(), key=lambda h: h.score, reverse=True)
        ranked = ranked[: max(limit * 4, 20)]

        if self.reranker_model is None or len(ranked) <= 1:
            return ranked[:limit]

        # Task 8: Use the enriched query (same as FAISS) for reranking.
        # enrich_query() is domain-agnostic and reuses Phase 8 concepts — no
        # hardcoded terms required. Falls back to raw query if no q8 provided.
        if q8 is not None:
            rerank_query = enrich_query(q8.query, q8.concepts)
        else:
            rerank_query = query_text

        pairs: List[List[str]] = []
        keep: List[RetrievalHit] = []
        for hit in ranked:
            rec = self.chunk_records.get(hit.chunk_id)
            if not rec:
                continue
            pairs.append([rerank_query, rec.retrieval_text or rec.text])
            keep.append(hit)

        if not pairs:
            return ranked[:limit]

        try:
            raw_scores = self.reranker_model.predict(pairs)
        except Exception:
            return ranked[:limit]

        if np is not None:
            raw_scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        # Task 8: Normalize rerank scores across all candidates before combining.
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
            support = len(hit.query_scores)
            if support >= 2:
                # Task 6: Chunks corroborated by multiple query variants are more
                # likely to be genuinely relevant — apply a multiplicative boost.
                # The boost is defined by MULTI_QUERY_BOOST (default ×1.1) and
                # applied once regardless of how many variants agree, keeping
                # scoring simple and predictable.
                hit.score = _clamp(hit.score * MULTI_QUERY_BOOST)
                hit.reasons.append(f"multi_query_support:{support}")

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
        return self._rerank(q8.query, raw_hits, limit=self.final_k, q8=q8)

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

        global_hits = self._rerank(q8.query, dict(merged), limit=self.final_k, q8=q8)
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

python retrieval/rq_4.py --phase8 query_analysis/result_2_.json  --base-dir data/processed/artifacts2  --k 5  --output retrieval/output_2__2.json  --no-graph

'''
