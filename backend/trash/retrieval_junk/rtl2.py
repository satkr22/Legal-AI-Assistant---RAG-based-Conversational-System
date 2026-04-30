from __future__ import annotations

"""Phase 9 — Hybrid Retrieval for legal RAG.

Expected artifacts:
  - faiss.index
  - faiss_meta.json
  - bm25.pkl
  - chunks.json
  - Phase 8 result dict or result.json
  
  
  To run with graph expansion:
python retrieval/rtl2.py \
  --phase8 query_analysis/result2.json \
  --base-dir data/processed/artifacts2 \
  --k 5 \
  --details \
  --output retrieval/output2.json
  
  
  To run without graph expansion:
python retrieval/rtl2.py \
  --phase8 query_analysis/result2.json \
  --base-dir data/processed/artifacts2 \
  --k 5 \
  --details
  --output retrieval/output2.json
  -- no-graph
  
  
  

"""

import json
import math
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from sentence_transformers import CrossEncoder

# Optional dependencies. The code is written to fail gracefully if one is missing.
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


FINAL_INTENTS = {
    "definition", "explanation", "summary", "lookup", "comparison", 
    "reasoning", "hypothetical", "procedure", "eligibility", 
    "consequence", "legal_scope", "legal_exception", 
    "legal_condition", "legal_penalty", "case_application",
}

TARGET_TYPES = {
    "section", "subsection", "clause", "explanation", 
    "illustration", "definition", "concept", "act",
}


# ----------------------------
# Data structures
# ----------------------------

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
        return cls(
            query=str(data.get("query", "")).strip(),
            intent=dict(data.get("intent", {}) or {}),
            targets=list(data.get("targets", []) or []),
            concepts=list(data.get("concepts", []) or []),
            constraints=dict(data.get("constraints", {}) or {}),
            query_features=dict(data.get("query_features", {}) or {}),
            confidence=dict(data.get("confidence", {}) or {}),
            method=str(data.get("method", "rules")),
            decomposition=dict(data.get("decomposition", {}) or {}),
            sub_queries=list(data.get("sub_queries", []) or []),
            notes=list(data.get("notes", []) or []),
        )

    @property
    def primary_intent(self) -> str:
        primary = self.intent.get("primary", "explanation")
        return primary if primary in FINAL_INTENTS else "explanation"

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
        specificity = self.constraints.get("specificity", "broad")
        return specificity if specificity in {"broad", "narrow"} else "broad"

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


# ----------------------------
# Utility functions
# ----------------------------

def _sigmoid(x: float) -> float:
    """Safely converts raw logits to a 0-1 probability scale."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _safe_lower(text: Any) -> str:
    return str(text or "").strip().lower()

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

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

def _tokenize(text: str) -> List[str]:
    # Match the tokenizer from build_embeddings.py exactly 
    # This ensures "3(1)" becomes ["3", "1"] in both the index and the query
    return re.findall(r"[a-z0-9]+", text.lower())
def _norm_bm25_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    mn = min(scores)
    if math.isclose(mx, mn):
        return [1.0 if mx > 0 else 0.0 for _ in scores]
    if mx <= 0:
        return [0.0 for _ in scores]
    return [max(0.0, s / mx) for s in scores]

def _norm_list(values: Sequence[float], inverse: bool = False) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if math.isclose(mx, mn):
        return [1.0 for _ in values]
    if inverse:
        return [1.0 - ((v - mn) / (mx - mn)) for v in values]
    return [((v - mn) / (mx - mn)) for v in values]

def _flatten_targets(targets: List[Dict[str, str]]) -> List[str]:
    vals = []
    for t in targets:
        t_type = _safe_lower(t.get("type"))
        t_val = str(t.get("value", "")).strip()
        if not t_val:
            continue
        vals.append(t_val)
        if t_type == "act":
            vals.append(t_val.upper())
    return _unique_keep_order(vals)


# ----------------------------
# Main retriever
# ----------------------------

class Phase9HybridRetriever:
    def __init__(
        self,
        faiss_index_path: str,
        faiss_meta_path: str,
        bm25_path: str,
        chunks_path: str,
        embed_model_name: str = "BAAI/bge-large-en-v1.5",
        embedder: Optional[Callable[[str], List[float]]] = None,
        top_k_faiss: int = 20,
        top_k_bm25: int = 20,
        final_k: int = 5,
        graph_depth: int = 1,
        graph_max_extra: int = 4,
    ) -> None:
        self.faiss_index_path = faiss_index_path
        self.faiss_meta_path = faiss_meta_path
        self.bm25_path = bm25_path
        self.chunks_path = chunks_path
        self.embed_model_name = embed_model_name
        self.embedder = embedder
        self.top_k_faiss = top_k_faiss
        self.top_k_bm25 = top_k_bm25
        self.final_k = final_k
        self.graph_depth = graph_depth
        self.graph_max_extra = graph_max_extra

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        self.chunks_doc = self._load_json(self.chunks_path)
        self.meta = self._load_json(self.faiss_meta_path)
        self.chunks: List[Dict[str, Any]] = list(self.chunks_doc.get("chunks", []) or [])
        self.chunk_records: Dict[str, ChunkRecord] = {}
        self.node_to_chunk_id: Dict[str, str] = {}
        self.section_to_chunk_ids: Dict[int, List[str]] = defaultdict(list)
        self.act_to_chunk_ids: Dict[str, List[str]] = defaultdict(list)
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
            if rec.act:
                self.act_to_chunk_ids[_safe_lower(rec.act)].append(rec.chunk_id)
            if rec.parent_chunk_id:
                self.parent_to_children[rec.parent_chunk_id].append(rec.chunk_id)

        self.faiss_index = None
        if faiss is not None and os.path.exists(self.faiss_index_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)

        with open(self.bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]  # Extract the actual BM25Okapi object!

        self.model = None
        if self.embedder is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not available.")
            self.model = SentenceTransformer(self.embed_model_name)
            self.embedder = self._default_embedder
            
        self.reranker_model = CrossEncoder("BAAI/bge-reranker-large")

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data = {str(i): item for i, item in enumerate(data)}
        elif not isinstance(data, dict):
            raise ValueError(f"Expected JSON object or list in {path}")
        return data

    def _default_embedder(self, text: str) -> List[float]:
        assert self.model is not None
        query = f"Represent this legal query for retrieving relevant statutory chunks: {text}"
        emb = self.model.encode([query], normalize_embeddings=True)
        if np is None:
            return emb[0].tolist()  
        return emb[0].astype("float32").tolist()

    def build_search_text(self, q8: Phase8Query) -> str:
        parts: List[str] = [q8.query]
        parts.extend(q8.concepts)
        parts.extend(_flatten_targets(q8.targets))

        intent_boost_terms = {
            "definition": ["meaning", "defined", "definition"],
            "explanation": ["explain", "explanation", "details"],
            "legal_penalty": ["punishment", "penalty", "fine", "imprisonment"],
            "case_application": ["apply", "facts", "situation", "scenario"],
        }
        parts.extend(intent_boost_terms.get(q8.primary_intent, []))

        if q8.jurisdiction:
            parts.append(q8.jurisdiction)
        if q8.specificity == "narrow":
            parts.append("exact section")

        return _normalize_space(" ".join([p for p in parts if p]))

    def build_bm25_query_tokens(self, q8: Phase8Query) -> List[str]:
        parts = [q8.query]
        parts.extend(q8.concepts)
        parts.extend(_flatten_targets(q8.targets))
        parts.extend([q8.primary_intent])
        return _tokenize(" ".join(parts))

    def _build_query_variants(self, q8: Phase8Query) -> List[QueryVariant]:
        variants: List[QueryVariant] = [QueryVariant(name="original", q8=q8, weight=1.0, source="original")]
        seen_queries = {q8.query.strip().lower()}

        for idx, item in enumerate(q8.sub_queries, start=1):
            if not isinstance(item, dict):
                continue
            sub_q8 = Phase8Query.from_dict(item)
            normalized_query = sub_q8.query.strip().lower()
            if not normalized_query or normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            variants.append(
                QueryVariant(
                    name=f"sub_query_{idx}",
                    q8=sub_q8,
                    weight=0.85,
                    source="decomposed",
                )
            )
        return variants

    def _aggregate_variant_hits(
        self,
        variants: Sequence[QueryVariant],
        final_k: int,
    ) -> Dict[str, RetrievalHit]:
        aggregate: Dict[str, RetrievalHit] = {}
        per_variant_limit = max(final_k * 4, self.top_k_faiss, self.top_k_bm25)

        for variant in variants:
            semantic_text = self.build_search_text(variant.q8)
            bm25_tokens = self.build_bm25_query_tokens(variant.q8)

            faiss_hits = self._faiss_search(semantic_text, self.top_k_faiss)
            bm25_hits = self._bm25_search(bm25_tokens, self.top_k_bm25)
            merged = self._merge_hits(variant.q8, faiss_hits, bm25_hits)
            reranked = self._rerank(variant.q8, merged)[:per_variant_limit]

            for hit in reranked:
                weighted_score = _clamp(hit.score * variant.weight)
                current = aggregate.get(hit.chunk_id)
                if current is None:
                    current = RetrievalHit(chunk_id=hit.chunk_id, score=weighted_score)
                    aggregate[hit.chunk_id] = current

                current.source_scores.update({
                    f"{variant.name}:faiss": hit.source_scores.get("faiss", 0.0),
                    f"{variant.name}:bm25": hit.source_scores.get("bm25", 0.0),
                })
                current.query_scores[variant.name] = round(weighted_score, 4)
                current.reasons.extend([reason for reason in hit.reasons if reason not in current.reasons])
                current.reasons.append(f"matched_{variant.name}")

        for hit in aggregate.values():
            scores = list(hit.query_scores.values())
            if not scores:
                continue
            best = max(scores)
            avg = sum(scores) / len(scores)
            support_bonus = min(0.16, 0.04 * max(0, len(scores) - 1))
            hit.score = _clamp((best * 0.75) + (avg * 0.25) + support_bonus)

        return aggregate

    def retrieve(
        self,
        phase8_result: Dict[str, Any] | Phase8Query,
        final_k: Optional[int] = None,
        include_details: bool = False,
        graph_expand: bool = True,
    ) -> List[Dict[str, Any]]:
        q8 = phase8_result if isinstance(phase8_result, Phase8Query) else Phase8Query.from_dict(phase8_result)
        k = final_k or self.final_k
        variants = self._build_query_variants(q8)
        merged = self._aggregate_variant_hits(variants, k)
        reranked = self._rerank(q8, merged)

        if graph_expand:
            reranked = self._expand_graph(q8, reranked)
            reranked = self._rerank(q8, reranked)
        
        # Send top 15 to the heavier semantic reranker
        reranked = self._semantic_rerank(variants, reranked[:15])

        reranked = reranked[:k]

        if include_details:
            return [self._format_detailed_hit(hit, q8) for hit in reranked]
        return [{"chunk_id": hit.chunk_id, "score": round(hit.score, 4)} for hit in reranked]

    def _semantic_rerank(self, variants: Sequence[QueryVariant], hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not hits:
            return hits

        pairs = []
        pair_map: List[Tuple[RetrievalHit, QueryVariant]] = []
        valid_hits = []

        for hit in hits:
            rec = self.chunk_records.get(hit.chunk_id)
            if not rec:
                continue
            text = rec.retrieval_text
            valid_hits.append(hit)
            for variant in variants:
                pairs.append((variant.q8.query, text))
                pair_map.append((hit, variant))

        if not pairs:
            return hits

        logits = self.reranker_model.predict(pairs)
        per_hit_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

        for (hit, variant), logit in zip(pair_map, logits):
            norm_score = _sigmoid(float(logit)) * variant.weight
            per_hit_scores[hit.chunk_id][variant.name] = round(norm_score, 4)

        for hit in valid_hits:
            variant_scores = per_hit_scores.get(hit.chunk_id, {})
            if not variant_scores:
                continue

            best_variant = max(variant_scores, key=variant_scores.get)
            best_score = variant_scores[best_variant]
            avg_score = sum(variant_scores.values()) / len(variant_scores)
            support_bonus = min(0.10, 0.03 * max(0, len(variant_scores) - 1))
            semantic_score = _clamp((best_score * 0.70) + (avg_score * 0.30) + support_bonus)

            hit.score = _clamp((hit.score * 0.35) + (semantic_score * 0.65))
            hit.reasons.append(f"cross_encoder_best:{best_variant}={round(best_score, 4)}")
            hit.source_scores[f"cross_encoder:{best_variant}"] = round(best_score, 4)

        return sorted(valid_hits, key=lambda x: x.score, reverse=True)

    def _faiss_search(self, query_text: str, top_k: int) -> Dict[str, float]:
        if self.faiss_index is None:
            return {}

        qvec = self.embedder(query_text)
        q = np.asarray([qvec], dtype="float32") if np is not None else [qvec]

        try:
            distances, ids = self.faiss_index.search(q, top_k)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"FAISS search failed: {exc}") from exc

        distances = list(distances[0])
        ids = list(ids[0])

        metric_type = getattr(self.faiss_index, "metric_type", None)
        is_ip = faiss is not None and metric_type == getattr(faiss, "METRIC_INNER_PRODUCT", -1)

        raw_scores = distances
        norm_scores = _norm_list(raw_scores, inverse=not is_ip)

        out: Dict[str, float] = {}
        for idx, score in zip(ids, norm_scores):
            if idx is None or idx < 0 or idx >= len(self.meta):
                continue
            meta_item = self.meta.get(str(idx))
            if not meta_item:
                continue
            chunk_id = str(meta_item.get("chunk_id"))
            if chunk_id:
                out[chunk_id] = max(out.get(chunk_id, 0.0), float(score))
        return out

    def _bm25_search(self, query_tokens: List[str], top_k: int) -> Dict[str, float]:
        if not query_tokens or not hasattr(self.bm25, "get_scores"):
            return {}

        # Graceful fallback: token formats often break between indexing and querying
        try:
            scores = self.bm25.get_scores(query_tokens)
        except Exception:
            try:
                # Fallback 1: Space-joined string
                scores = self.bm25.get_scores(" ".join(query_tokens))
            except Exception:
                try:
                    # Fallback 2: Basic whitespace split
                    scores = self.bm25.get_scores(" ".join(query_tokens).split())
                except Exception:
                    return {} 

        scores = list(scores)
        if not scores:
            return {}

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        norm_scores = _norm_bm25_scores([scores[i] for i in top_idx])

        out: Dict[str, float] = {}
        for i, norm in zip(top_idx, norm_scores):
            if i >= len(self.meta):
                continue
            meta_item = self.meta.get(str(i))
            if not meta_item:
                continue
            chunk_id = str(meta_item.get("chunk_id"))
            if chunk_id:
                out[chunk_id] = max(out.get(chunk_id, 0.0), float(norm))
        return out

    def _merge_hits(
        self, q8: Phase8Query, faiss_hits: Dict[str, float], bm25_hits: Dict[str, float]
    ) -> Dict[str, RetrievalHit]:
        candidate_ids = set(faiss_hits) | set(bm25_hits)
        merged: Dict[str, RetrievalHit] = {}

        weights = self._intent_weights(q8)
        for cid in candidate_ids:
            fs = faiss_hits.get(cid, 0.0)
            bs = bm25_hits.get(cid, 0.0)
            base = weights["faiss"] * fs + weights["bm25"] * bs
            merged[cid] = RetrievalHit(
                chunk_id=cid, score=base, source_scores={"faiss": fs, "bm25": bs}, reasons=[]
            )
        return merged

    def _intent_weights(self, q8: Phase8Query) -> Dict[str, float]:
        intent = q8.primary_intent
        faiss_w, bm25_w = 0.55, 0.45

        if intent in {"lookup", "legal_penalty"}:
            bm25_w, faiss_w = 0.65, 0.35
        elif intent in {"definition", "explanation", "reasoning", "case_application", "consequence"}:
            faiss_w, bm25_w = 0.65, 0.35
        elif intent in {"comparison", "hypothetical"} or q8.is_multi_hop:
            faiss_w, bm25_w = 0.60, 0.40

        if q8.requires_exact_match:
            bm25_w += 0.10; faiss_w -= 0.10
        if q8.requires_reasoning:
            faiss_w += 0.05; bm25_w -= 0.05
        if q8.specificity == "narrow":
            bm25_w += 0.08; faiss_w -= 0.08

        faiss_w = _clamp(faiss_w, 0.10, 0.90)
        bm25_w = _clamp(bm25_w, 0.10, 0.90)
        total = faiss_w + bm25_w
        return {"faiss": faiss_w / total, "bm25": bm25_w / total}

    def _rerank(self, q8: Phase8Query, hits: Dict[str, RetrievalHit]) -> List[RetrievalHit]:
        query_terms = set(_tokenize(q8.query)) | set(_tokenize(" ".join(q8.concepts)))
        target_values = set(_tokenize(" ".join(_flatten_targets(q8.targets))))
        act_targets = {_safe_lower(t.get("value")) for t in q8.targets if _safe_lower(t.get("type")) == "act"}
        section_targets = []
        for t in q8.targets:
            if _safe_lower(t.get("type")) == "section":
                try:
                    section_targets.append(int(str(t.get("value")).strip()))
                except Exception:
                    pass
        
        hit_list = list(hits.values()) if isinstance(hits, dict) else hits
            
        for hit in hit_list:
            rec = self.chunk_records.get(hit.chunk_id)
            if rec is None:
                continue

            text = _safe_lower(rec.retrieval_text)
            keywords = set(_tokenize(" ".join(rec.keywords + rec.legal_concepts)))
            text_tokens = set(_tokenize(text))

            bonus = 0.0
            reasons = []

            # Drastically reduced heuristics to prevent Illustration bloat
            overlap = len(query_terms & text_tokens)
            if overlap:
                bonus += min(0.10, 0.02 * overlap) 

            if section_targets and rec.section_number in section_targets:
                bonus += 0.08  # Was 0.22
                reasons.append("section_exact")

            if act_targets:
                rec_act = _safe_lower(rec.act)
                if rec_act in act_targets or any(a in rec_act for a in act_targets):
                    bonus += 0.05  # Was 0.12

            if target_values & keywords:
                bonus += 0.05

            if q8.primary_intent == "definition" and rec.chunk_type in {"definition", "subsection"}:
                bonus += 0.05
            elif q8.primary_intent == "legal_penalty" and rec.chunk_type in {"subsection", "clause"}:
                bonus += 0.05

            if rec.references:
                bonus += 0.01  # Was 0.03
            if rec.explains_node_id:
                bonus += 0.01  # Was 0.02

            if q8.requires_exact_match and q8.targets:
                exact_targets = [str(t.get("value", "")).strip() for t in q8.targets if str(t.get("value", "")).strip()]
                if any(self._exact_text_match(text, et) for et in exact_targets):
                    bonus += 0.05

            hit.score = _clamp(hit.score + bonus)
            if reasons:
                hit.reasons.extend(reasons)

        ordered = sorted(hit_list, key=lambda h: h.score, reverse=True)
        return ordered

    @staticmethod
    def _exact_text_match(haystack: str, needle: str) -> bool:
        haystack = _normalize_space(haystack).lower()
        needle = _normalize_space(needle).lower()
        if not haystack or not needle:
            return False
        return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None

    def _expand_graph(self, q8: Phase8Query, ranked: List[RetrievalHit]) -> List[RetrievalHit]:
        if self.graph_depth <= 0:
            return ranked

        existing_ids = {h.chunk_id for h in ranked}
        extra: List[RetrievalHit] = []
        seen = set(existing_ids)

        seed_ids = [h.chunk_id for h in ranked[: min(len(ranked), self.final_k)]]
        frontier = list(seed_ids)

        for _ in range(self.graph_depth):
            next_frontier: List[str] = []
            for cid in frontier:
                rec = self.chunk_records.get(cid)
                if rec is None:
                    continue

                related_ids: List[str] = []
                if rec.parent_chunk_id:
                    related_ids.append(rec.parent_chunk_id)
                related_ids.extend(rec.children_chunk_ids)

                for ref in rec.references:
                    tgt = ref.get("target_id")
                    if tgt:
                        resolved = self._resolve_node_or_chunk_id(str(tgt))
                        if resolved:
                            related_ids.append(resolved)

                if rec.explains_node_id:
                    resolved = self._resolve_node_or_chunk_id(str(rec.explains_node_id))
                    if resolved:
                        related_ids.append(resolved)

                for rid in related_ids:
                    if rid in seen:
                        continue
                    seen.add(rid)
                    if self.chunk_records.get(rid):
                        extra.append(
                            RetrievalHit(
                                chunk_id=rid,
                                score=0.10, # Lowered graph expansion baseline score
                                source_scores={"graph": 1.0},
                                reasons=["graph_expansion"],
                            )
                        )
                        next_frontier.append(rid)
                        if len(extra) >= self.graph_max_extra:
                            break
                if len(extra) >= self.graph_max_extra:
                    break
            frontier = next_frontier

        if not extra:
            return ranked

        merged: Dict[str, RetrievalHit] = {h.chunk_id: h for h in ranked}
        for e in extra:
            if e.chunk_id not in merged:
                merged[e.chunk_id] = e
            else:
                merged[e.chunk_id].score = max(merged[e.chunk_id].score, e.score)
                merged[e.chunk_id].source_scores.update(e.source_scores)
                merged[e.chunk_id].query_scores.update(e.query_scores)
                merged[e.chunk_id].reasons.extend(e.reasons)

        return sorted(merged.values(), key=lambda h: h.score, reverse=True)

    def _resolve_node_or_chunk_id(self, maybe_id: str) -> Optional[str]:
        maybe_id = maybe_id.strip()
        if not maybe_id:
            return None
        if maybe_id in self.chunk_records:
            return maybe_id
        if maybe_id in self.node_to_chunk_id:
            return self.node_to_chunk_id[maybe_id]
        if maybe_id.startswith("chunk__") and maybe_id in self.chunk_records:
            return maybe_id
        return None

    def _format_detailed_hit(self, hit: RetrievalHit, q8: Phase8Query) -> Dict[str, Any]:
        rec = self.chunk_records.get(hit.chunk_id)
        payload = {
            "chunk_id": hit.chunk_id,
            "score": round(hit.score, 4),
            "source_scores": {k: round(v, 4) for k, v in hit.source_scores.items()},
            "query_scores": {k: round(v, 4) for k, v in hit.query_scores.items()},
            "reasons": hit.reasons,
        }
        if rec is not None:
            payload.update({
                "chunk_type": rec.chunk_type,
                "section_number": rec.section_number,
                "act": rec.act,
                "chapter_title": rec.chapter_title,
                "text": rec.text,
                "derived_context": rec.derived_context,
                "keywords": rec.keywords,
                "legal_concepts": rec.legal_concepts,
            })
        return payload

    @classmethod
    def from_default_paths(cls, base_dir: str = "/mnt/data", **kwargs: Any) -> "Phase9HybridRetriever":
        base = Path(base_dir)
        return cls(
            faiss_index_path=str(base / "faiss.index"),
            faiss_meta_path=str(base / "faiss_meta.json"),
            bm25_path=str(base / "bm25.pkl"),
            chunks_path=str(base / "chunks.json"),
            **kwargs,
        )

# ----------------------------
# Batch / CLI helpers
# ----------------------------

def load_phase8_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Phase 8 results must be a dict or a list of dicts")

def run_batch(
    phase8_results: List[Dict[str, Any]], retriever: Phase9HybridRetriever, 
    final_k: int = 5, include_details: bool = False, graph_expand: bool = True
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in phase8_results:
        q = str(item.get("query", "")).strip()
        hits = retriever.retrieve(item, final_k=final_k, include_details=include_details, graph_expand=graph_expand)
        out.append({"query": q, "results": hits})
    return out

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 9 Hybrid Retrieval")
    parser.add_argument("--phase8", required=False, help="Path to Phase 8 JSON results")
    parser.add_argument("--query", required=False, help="Single raw user query")
    parser.add_argument("--output", required=False, help="Optional output file path")
    parser.add_argument("--details", action="store_true", help="Return detailed candidate objects")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    parser.add_argument("--k", type=int, default=5, help="Final number of chunks to return")
    parser.add_argument("--base-dir", default="/mnt/data", help="Directory containing artifacts")
    args = parser.parse_args()

    retriever = Phase9HybridRetriever.from_default_paths(base_dir=args.base_dir, final_k=args.k)

    if args.phase8:
        phase8_results = load_phase8_results(args.phase8)
        output = run_batch(
            phase8_results, retriever, final_k=args.k, 
            include_details=args.details, graph_expand=not args.no_graph
        )
    elif args.query:
        raise SystemExit("Provide --phase8 <result.json>")
    else:
        raise SystemExit("Provide --phase8 <result.json>")

    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)

if __name__ == "__main__":
    main()
