from __future__ import annotations

"""Phase 8 - Hybrid query understanding for the legal RAG pipeline.

Design goals:
- Rules handle only structure and routing.
- LLM handles semantic understanding.
- Retrieval is lightweight grounding only.
- Output is compact JSON for Phase 9.
"""

import argparse
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

AMBIGUOUS_TERMS = {
    "act",
    "case",
    "crime",
    "details",
    "explain",
    "help",
    "issue",
    "law",
    "matter",
    "meaning",
    "penalty",
    "punishment",
    "section",
    "tell",
    "this",
    "that",
    "thing",
}

REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|subsection|sub-section|clause)\s*"
    r"(\d+[a-z]?)(?:\s*\(\s*([0-9a-z]+)\s*\))?",
    flags=re.IGNORECASE,
)

MULTI_INTENT_CONNECTOR_PATTERN = re.compile(r"\b(?:and|or|also|plus|along with)\b|[;,/]", re.IGNORECASE)
QUESTION_FRAME_PATTERN = re.compile(r"\b(?:what|how|why|when|where|which|whether)\b", re.IGNORECASE)
SUBQUERY_DEPENDENCY_PATTERN = re.compile(
    r"\b(?:it|they|them|he|she|him|her|this|that|these|those|such|same|former|latter)\b",
    re.IGNORECASE,
)
QUESTION_START_PATTERN = re.compile(
    r"^(?:what|how|why|when|where|which|whether|can|should|is|are|do|does|did)\b",
    re.IGNORECASE,
)

ACT_NAME = "BNS, 2023"
CANONICAL_INTENTS = (
    "definition",
    "explanation",
    "summary",
    "lookup",
    "comparison",
    "reasoning",
    "hypothetical",
    "procedure",
    "eligibility",
    "consequence",
    "legal scope",
    "legal exception",
    "legal condition",
    "legal penalty",
    "case application",
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_space(text).lower())


def _dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for item in items:
        key = item if isinstance(item, (str, int, float, tuple)) else json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        text = _normalize_space(str(item or ""))
        if text:
            out.append(text)
    return _dedupe_keep_order(out)


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    text = _normalize_space(raw)
    if not text:
        return {}
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _extract_exact_references(query: str) -> List[str]:
    refs: List[str] = []
    for match in REFERENCE_PATTERN.finditer(query or ""):
        base = match.group(1).upper()
        sub = match.group(2)
        refs.append(f"Section {base}{f'({sub})' if sub else ''}")
    return _dedupe_keep_order(refs)


def _extract_section_numbers(query: str) -> List[str]:
    numbers: List[str] = []
    for ref in _extract_exact_references(query):
        number = ref.replace("Section", "", 1).strip()
        if number:
            numbers.append(number)
    return _dedupe_keep_order(numbers)


def _normalize_section_number(value: Any) -> Optional[str]:
    text = _normalize_space(str(value or ""))
    if not text:
        return None
    return text


def _join_brief_list(items: Sequence[str], max_items: int = 3) -> str:
    cleaned = [item for item in (_normalize_space(x) for x in items) if item][:max_items]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _looks_like_fact_pattern(query: str) -> bool:
    lowered = _normalize_space(query).lower()
    if not lowered:
        return False
    actor_markers = ("my ", "me ", "i ", "someone", "a person", "person ", "neighbor", "thief", "attacker")
    event_markers = (
        "stole",
        "hit",
        "entered",
        "kidnapped",
        "forged",
        "threatened",
        "damaged",
        "abused",
        "tricked",
        "disappeared",
        "demanded",
        "commits",
        "commit",
        "causes",
        "dies",
    )
    ask_markers = (
        "what crime",
        "what offence",
        "what offense",
        "what offences",
        "what sections",
        "what all offences",
        "what happens",
        "what punishment",
        "what can i do",
        "may apply",
        "apply",
    )
    return (
        any(marker in lowered for marker in actor_markers)
        and any(marker in lowered for marker in event_markers)
        and any(marker in lowered for marker in ask_markers)
    )


def _query_requests_penalty(query: str) -> bool:
    lowered = _normalize_space(query).lower()
    return any(term in lowered for term in ("punishment", "penalty", "punished", "sentence", "fine", "imprisonment"))


def _canonicalize_intent_label(raw_label: Any, fallback: Optional[str] = None) -> str:
    label = _normalize_space(str(raw_label or "")).lower().replace("_", " ")
    if not label:
        return fallback or ""
    if label in CANONICAL_INTENTS:
        return label

    checks = [
        ("case application", ("case application", "apply facts", "fact pattern", "applicable offence", "applicable offense", "apply to scenario")),
        ("legal penalty", ("penalty", "punishment", "sentence", "fine", "imprisonment", "liable to punishment")),
        ("comparison", ("compare", "comparison", "difference", "distinguish", "versus", "vs", "between")),
        ("procedure", ("procedure", "steps", "process", "what should i do", "what can i do", "how to", "complaint", "report")),
        ("definition", ("definition", "define", "meaning", "what is", "what does")),
        ("summary", ("summary", "summar", "overview", "brief")),
        ("lookup", ("lookup", "section", "clause", "provision", "reference")),
        ("legal exception", ("exception", "defence", "defense", "exemption", "unless")),
        ("legal condition", ("condition", "if", "when", "where", "provided that")),
        ("legal scope", ("scope", "apply", "applies", "jurisdiction", "coverage")),
        ("eligibility", ("eligibility", "eligible", "qualification")),
        ("hypothetical", ("hypothetical", "suppose", "assume")),
        ("consequence", ("consequence", "effect", "result", "outcome", "happen")),
        ("reasoning", ("reasoning", "analyze", "interpret", "determine")),
        ("explanation", ("explain", "explanation", "clarify")),
    ]
    for intent, markers in checks:
        if any(marker in label for marker in markers):
            return intent
    return fallback or ""


def _derive_subquery_context(query: str, targets: Sequence[str], concepts: Sequence[str]) -> str:
    if targets:
        return _join_brief_list(targets, max_items=2)
    if concepts:
        return _join_brief_list(concepts, max_items=3)
    return _normalize_space(query.rstrip("?."))


def _mentions_context(text: str, targets: Sequence[str], concepts: Sequence[str]) -> bool:
    lowered = _normalize_space(text).lower()
    for item in list(targets) + list(concepts):
        marker = _normalize_space(item).lower()
        if marker and marker in lowered:
            return True
    return False


def _rewrite_subquery_from_context(context: str, primary_intent: str) -> str:
    context = _normalize_space(context.rstrip("?."))
    intent = primary_intent or "lookup"
    if not context:
        return f"What does {ACT_NAME} provide?"
    if context.lower().startswith("section "):
        return f"What does {context} of {ACT_NAME} provide?"
    templates = {
        "definition": f"What is the legal meaning of {context} under {ACT_NAME}?",
        "summary": f"What is the summary of the law on {context} under {ACT_NAME}?",
        "lookup": f"What does the law provide regarding {context} under {ACT_NAME}?",
        "comparison": f"What is the legal difference between {context} under {ACT_NAME}?",
        "reasoning": f"How does the law apply to {context} under {ACT_NAME}?",
        "hypothetical": f"How would the law apply to the situation involving {context} under {ACT_NAME}?",
        "procedure": f"What is the legal procedure related to {context} under {ACT_NAME}?",
        "eligibility": f"Who is legally eligible in matters involving {context} under {ACT_NAME}?",
        "consequence": f"What are the legal consequences related to {context} under {ACT_NAME}?",
        "legal scope": f"What is the legal scope of {context} under {ACT_NAME}?",
        "legal exception": f"What legal exceptions apply to {context} under {ACT_NAME}?",
        "legal condition": f"What legal conditions apply to {context} under {ACT_NAME}?",
        "legal penalty": f"What punishment applies to {context} under {ACT_NAME}?",
        "case application": f"What legal provisions apply to the facts involving {context} under {ACT_NAME}?",
        "explanation": f"What does the law say about {context} under {ACT_NAME}?",
    }
    return templates.get(intent, f"What does the law say about {context} under {ACT_NAME}?")


def _normalize_subquery_text(
    subquery: str,
    *,
    query: str,
    targets: Sequence[str],
    concepts: Sequence[str],
    primary_intent: str,
) -> str:
    sq = _normalize_space(subquery).strip("\"'")
    if not sq:
        return ""

    context = _derive_subquery_context(query, targets, concepts)
    has_context = _mentions_context(sq, targets, concepts)
    looks_dependent = bool(SUBQUERY_DEPENDENCY_PATTERN.search(sq)) and not has_context
    looks_too_short = len(_tokenize(sq)) < 5 and not has_context

    if re.fullmatch(r"section\s+\d+[a-z]?(?:\([0-9a-z]+\))?", sq, flags=re.IGNORECASE):
        sq = f"What does {sq} of {ACT_NAME} provide?"
    elif sq.lower().startswith("retrieve material"):
        sq = _rewrite_subquery_from_context(context, primary_intent)
    elif looks_dependent or looks_too_short:
        sq = _rewrite_subquery_from_context(context, primary_intent)
    elif not QUESTION_START_PATTERN.search(sq):
        if has_context:
            sq = f"What does the law say about {sq.rstrip('?.')} under {ACT_NAME}?"
        else:
            sq = _rewrite_subquery_from_context(context, primary_intent)
    elif ACT_NAME.lower() not in sq.lower() and not any(target.lower() in sq.lower() for target in targets):
        sq = sq.rstrip("?.")
        sq = f"{sq} under {ACT_NAME}?"

    if sq and sq[-1] not in {"?", "."}:
        sq += "?"
    return sq


def _normalize_subquery_objects(
    subqueries: Sequence[Any],
    *,
    query: str,
    targets: Sequence[str],
    concepts: Sequence[str],
    primary_intent: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in subqueries:
        if isinstance(item, dict):
            sq_text = _normalize_space(item.get("query", ""))
            sq_targets = _coerce_str_list(item.get("targets", []))
            sq_concepts = _coerce_str_list(item.get("concepts", []))
        else:
            sq_text = _normalize_subquery_text(
                str(item),
                query=query,
                targets=targets,
                concepts=concepts,
                primary_intent=primary_intent,
            )
            sq_targets = list(targets[:2])
            sq_concepts = list(concepts[:4])

        if not sq_text:
            continue

        out.append({
            "query": sq_text,
            "intent": {
                "primary": primary_intent,
                "secondary": []
            },
            "targets": sq_targets,
            "concepts": sq_concepts,
            "query_features": {
                "is_multi_hop": False,
                "requires_reasoning": False
            },
            "decomposition": {
                "needed": False,
                "sub_queries": []
            },
            "method": "llm_subquery",
            "notes": []
        })
    return out


@dataclass(frozen=True)
class ChunkHintRecord:
    chunk_id: str
    chunk_type: str
    section_number: Optional[str]
    section_title: str
    keywords: Tuple[str, ...]
    legal_concepts: Tuple[str, ...]
    semantic_summary: str
    embedding_text: str
    # subheading: str

    @classmethod
    def from_chunk(cls, chunk: Dict[str, Any]) -> "ChunkHintRecord":
        section = chunk.get("section", {}) or {}
        
        summary = _normalize_space(str(chunk.get("semantic_summary", "") or ""))
        section_title = _normalize_space(str(section.get("section_title", "") or ""))
        # subheading = _normalize_space(str(section.get("subheading", "") or ""))
        keywords = tuple(
            _normalize_space(str(item).lower())
            for item in (chunk.get("keywords") or [])
            if _normalize_space(str(item))
        )
        legal_concepts = tuple(
            _normalize_space(str(item).lower())
            for item in (chunk.get("legal_concepts") or [])
            if _normalize_space(str(item))
        )
        return cls(
            chunk_id=str(chunk.get("chunk_id", "") or ""),
            chunk_type=str(chunk.get("chunk_type", "") or ""),
            section_number=_normalize_section_number(section.get("section_number")),
            section_title=section_title,
            # subheading=subheading,
            keywords=keywords,
            legal_concepts=legal_concepts,
            semantic_summary=summary or section_title,
            embedding_text=_normalize_space(str(chunk.get("embedding_text", "") or "")) or summary or section_title,
        )

    @property
    def grounding_text(self) -> str:
        parts = [self.section_title, " ".join(self.legal_concepts), self.semantic_summary]
        return _normalize_space(" | ".join(part for part in parts if part))


@dataclass(frozen=True)
class HintHit:
    chunk_id: str
    section_number: Optional[str]
    legal_concepts: Tuple[str, ...]
    semantic_summary: str
    score: float
    method: str

    def to_prompt_line(self) -> str:
        label = ", ".join(self.legal_concepts[:3]) or self.semantic_summary or "legal provision"
        section = f"Section {self.section_number}" if self.section_number else "Unknown section"
        summary = self.semantic_summary or "No summary available."
        return f"- {label} ({section}): {summary}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_number": self.section_number,
            "legal_concepts": list(self.legal_concepts),
            "semantic_summary": self.semantic_summary,
            "score": round(self.score, 4),
            "method": self.method,
        }


class OpenAIClient:
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key)

    def generate_json(self, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return _safe_json_loads(response.choices[0].message.content or "{}")

    def embed(self, texts: Sequence[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        response = self.client.embeddings.create(model=model, input=list(texts))
        return [list(item.embedding) for item in response.data]


class HintRetriever:
    """Lightweight grounding only. No ranking stack from Phase 9 is used."""

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        embedding_provider: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        sentence_model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.records = [ChunkHintRecord.from_chunk(chunk) for chunk in chunks]
        self.records = [record for record in self.records if record.chunk_id]
        self.by_section_number: Dict[str, List[ChunkHintRecord]] = {}
        for record in self.records:
            if record.section_number:
                self.by_section_number.setdefault(record.section_number.upper(), []).append(record)

        self.embedding_provider = embedding_provider
        self.sentence_model_name = sentence_model_name
        self._sentence_model: Optional[Any] = None
        self._embedding_cache: Optional[List[List[float]]] = None
        self._embedding_lock = threading.Lock()

    def lookup_sections(self, section_numbers: Sequence[str]) -> List[ChunkHintRecord]:
        hits: List[ChunkHintRecord] = []
        for number in section_numbers:
            hits.extend(self.by_section_number.get(_normalize_space(number).upper(), []))
        hits.sort(key=lambda item: (0 if item.chunk_type == "section" else 1, item.chunk_id))
        return hits

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        top_k = max(3, min(5, int(top_k)))
        keyword_hits = self._keyword_overlap_hits(query, top_k=top_k)
        if self._keyword_overlap_is_weak(keyword_hits):
            embedding_hits = self._embedding_similarity_hits(query, top_k=3)
            if embedding_hits:
                return self._package_hits(embedding_hits, method="embedding_similarity")
        return self._package_hits(keyword_hits, method="keyword_overlap")

    def _keyword_overlap_hits(self, query: str, top_k: int) -> List[HintHit]:
        query_tokens = [token for token in _tokenize(query) if token not in QUERY_STOPWORDS]
        query_lower = _normalize_space(query).lower()
        if not query_tokens:
            return []

        scored: List[HintHit] = []
        for record in self.records:
            terms = set()
            for phrase in list(record.keywords) + list(record.legal_concepts):
                terms.update(_tokenize(phrase))
            if not terms:
                continue

            overlap = set(query_tokens) & terms
            phrase_hits = sum(
                1
                for phrase in list(record.keywords) + list(record.legal_concepts)
                if phrase and phrase in query_lower
            )
            lexical_score = len(overlap) / max(1, min(len(query_tokens), len(terms)))
            phrase_score = min(1.0, phrase_hits / max(1, len(record.keywords) + len(record.legal_concepts)))
            score = (0.7 * lexical_score) + (0.3 * phrase_score)
            if record.chunk_type == "section":
                score += 0.05
            if record.semantic_summary:
                score += 0.03
            score = _clamp(score)
            if score <= 0.0:
                continue

            scored.append(
                HintHit(
                    chunk_id=record.chunk_id,
                    section_number=record.section_number,
                    legal_concepts=record.legal_concepts,
                    semantic_summary=record.semantic_summary,
                    score=score,
                    method="keyword_overlap",
                )
            )

        return self._dedupe_sections(scored)[:top_k]

    def _keyword_overlap_is_weak(self, hits: Sequence[HintHit]) -> bool:
        if not hits:
            return True
        best = hits[0].score
        combined = sum(hit.score for hit in hits[:2])
        return best < 0.2 or combined < 0.4

    def _embedding_similarity_hits(self, query: str, top_k: int) -> List[HintHit]:
        vectors = self._embed_texts([query] + [record.grounding_text for record in self.records])
        if not vectors or len(vectors) != len(self.records) + 1:
            return []

        query_vec = vectors[0]
        chunk_vecs = vectors[1:]
        scored: List[HintHit] = []
        for record, vector in zip(self.records, chunk_vecs):
            score = _clamp((self._cosine_similarity(query_vec, vector) + 1.0) / 2.0)
            if record.chunk_type == "section":
                score += 0.03
            if score <= 0.0:
                continue
            scored.append(
                HintHit(
                    chunk_id=record.chunk_id,
                    section_number=record.section_number,
                    legal_concepts=record.legal_concepts,
                    semantic_summary=record.semantic_summary,
                    score=_clamp(score),
                    method="embedding_similarity",
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return self._dedupe_sections(scored)[:top_k]

    def _package_hits(self, hits: Sequence[HintHit], method: str) -> Dict[str, Any]:
        ordered = list(hits)
        relevance = 0.0
        if ordered:
            avg = sum(hit.score for hit in ordered) / len(ordered)
            relevance = _clamp((ordered[0].score * 0.6) + (avg * 0.4))

        return {
            "method": method if ordered else "none",
            "hint_relevance": round(relevance, 4),
            "items": [hit.to_dict() for hit in ordered],
            "prompt_block": (
                "Relevant legal hints:\n" + "\n".join(hit.to_prompt_line() for hit in ordered)
                if ordered
                else "Relevant legal hints:\n- none"
            ),
        }

    def _dedupe_sections(self, hits: Sequence[HintHit]) -> List[HintHit]:
        ordered = sorted(hits, key=lambda item: item.score, reverse=True)
        seen = set()
        out: List[HintHit] = []
        for hit in ordered:
            key = hit.section_number or hit.chunk_id
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
        return out

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self.embedding_provider is not None:
            with self._embedding_lock:
                if self._embedding_cache is None:
                    try:
                        self._embedding_cache = self.embedding_provider(
                            [record.grounding_text for record in self.records]
                        )
                    except Exception:
                        self._embedding_cache = []
                try:
                    query_vectors = self.embedding_provider(list(texts[:1]))
                except Exception:
                    return []
            if len(texts) == len(self.records) + 1:
                return list(query_vectors) + list(self._embedding_cache or [])
            return list(query_vectors)

        if SentenceTransformer is None:
            return []

        with self._embedding_lock:
            if self._sentence_model is None:
                try:
                    self._sentence_model = SentenceTransformer(self.sentence_model_name)
                except Exception:
                    self._sentence_model = False  # type: ignore[assignment]

            if self._sentence_model is False:  # type: ignore[comparison-overlap]
                return []

            if self._embedding_cache is None:
                try:
                    encoded = self._sentence_model.encode(  # type: ignore[union-attr]
                        [record.grounding_text for record in self.records],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    self._embedding_cache = [self._vector_to_list(item) for item in encoded]
                except Exception:
                    self._embedding_cache = []

            try:
                query_encoded = self._sentence_model.encode(  # type: ignore[union-attr]
                    list(texts[:1]),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception:
                return []

        if len(texts) == len(self.records) + 1:
            return [self._vector_to_list(query_encoded[0])] + list(self._embedding_cache or [])
        return [self._vector_to_list(item) for item in query_encoded]

    @staticmethod
    def _vector_to_list(vector: Any) -> List[float]:
        if hasattr(vector, "tolist"):
            converted = vector.tolist()
            if isinstance(converted, list):
                return [float(x) for x in converted]
        if isinstance(vector, (list, tuple)):
            return [float(x) for x in vector]
        return []

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


def load_chunks(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        chunks = data.get("chunks", []) or []
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    if isinstance(data, list):
        return [chunk for chunk in data if isinstance(chunk, dict)]
    return []


def analyze_rules(query: str) -> Dict[str, Any]:
    tokens = _tokenize(query)
    exact_references = _extract_exact_references(query)
    question_frames = len(QUESTION_FRAME_PATTERN.findall(query or ""))
    connector_hits = len(MULTI_INTENT_CONNECTOR_PATTERN.findall(query or ""))
    segments = [
        segment
        for segment in re.split(r"\?|;|,|\band\b|\bor\b|\balso\b", _normalize_space(query), flags=re.IGNORECASE)
        if len(_tokenize(segment)) >= 2
    ]

    pronoun_heavy = bool(tokens) and all(token in AMBIGUOUS_TERMS or token in QUERY_STOPWORDS for token in tokens)
    ambiguous = (
        (len(tokens) <= 3 and not exact_references)
        or (len(tokens) <= 5 and pronoun_heavy and not exact_references)
        or _normalize_space(query).lower() in {"help", "explain", "details", "what about this"}
    )

    multi_intent = (
        len(exact_references) >= 2
        or question_frames >= 2
        or (len(segments) >= 2 and connector_hits >= 1 and len(tokens) >= 8)
    )

    if len(tokens) >= 20 or (multi_intent and len(tokens) >= 10) or len(exact_references) >= 2:
        complexity = "complex"
    elif len(tokens) >= 8 or multi_intent or bool(exact_references):
        complexity = "medium"
    else:
        complexity = "simple"

    return {
        "multi_intent": bool(multi_intent),
        "ambiguous": bool(ambiguous),
        "exact_reference": bool(exact_references),
        "complexity": complexity,
        "query_length": len(tokens),
        "exact_references": exact_references,
    }


def route_query(rule_output: Dict[str, Any]) -> str:
    if rule_output.get("exact_reference") and not rule_output.get("ambiguous"):
        return "rules_only"
    return "llm"


def retrieve_hints(
    query: str,
    chunks: Sequence[Dict[str, Any]] | HintRetriever,
    top_k: int = 4,
    embedding_provider: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
) -> Dict[str, Any]:
    retriever = chunks if isinstance(chunks, HintRetriever) else HintRetriever(chunks, embedding_provider=embedding_provider)
    return retriever.retrieve(query, top_k=top_k)


def build_prompt(query: str, hints: Dict[str, Any], rule_output: Dict[str, Any]) -> str:
    schema = {
        "intent": {"primary": "", "secondary": []},
        "concepts": [],
        "targets": [],
        "query_features": {"is_multi_hop": False, "requires_reasoning": False},
        "decomposition": {
        "needed": False,
        "sub_queries": [
                {
                    "query": "",
                    "intent": {"primary": "", "secondary": []},
                    "targets": [],
                    "concepts": [],
                    "query_features": {
                        "is_multi_hop": False,
                        "requires_reasoning": False
                    },
                    "decomposition": {
                        "needed": False,
                        "sub_queries": []
                    }
                }
            ]
        },
    }
    prompt_block = hints.get("prompt_block") or "Relevant legal hints:\n- none"
    rule_signals = {
        key: value
        for key, value in rule_output.items()
        if key in {"multi_intent", "ambiguous", "exact_reference", "complexity"}
    }
    prompt = f"""
Analyze the legal query for Bharatiya Nyaya Sanhita, 2023.

Rules:
1. Identify the primary intent.
2. Identify any secondary intents.
3. Extract the legal concepts that matter for retrieval.
4. Detect whether the query is multi-hop.
5. Decide whether decomposition is needed.
6. If decomposition is needed, produce short standalone retrieval-friendly sub-queries atleat two or more if needed.
7. Use the hints only as grounding support. They are not final retrieval results.
8. Return strict JSON only and follow this exact schema:
{json.dumps(schema, ensure_ascii=False)}

Intent format requirements:
- intent.primary must be one or two words only.
- intent.secondary items must also be one or two words only.
- Use only labels from this set:
{json.dumps(list(CANONICAL_INTENTS), ensure_ascii=False)}

Sub-query requirements:
- Every sub-query must be independently understandable without reading the original query.
- Do not use dependent references like "this", "that", "they", "them", or "it" unless the noun is explicitly restated.
- Each sub-query should be a complete sentence or question.

Rule-layer structure signals:
{json.dumps(rule_signals, ensure_ascii=False)}

{prompt_block}

User query:
{query}
"""
    return prompt.strip()


def call_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    llm_client: Optional[OpenAIClient] = None,
) -> Dict[str, Any]:
    if llm_client is None:
        return {}
    return llm_client.generate_json(
        model=model,
        system_prompt="You are a legal query analyzer for BNS 2023.",
        user_prompt=prompt,
    )


def _normalize_output_shape(
    query: str,
    obj: Dict[str, Any],
    rule_output: Dict[str, Any],
    fallback_intent: str = "lookup",
) -> Dict[str, Any]:
    intent = obj.get("intent", {}) if isinstance(obj.get("intent"), dict) else {}
    query_features = obj.get("query_features", {}) if isinstance(obj.get("query_features"), dict) else {}
    decomposition = obj.get("decomposition", {}) if isinstance(obj.get("decomposition"), dict) else {}

    primary = _canonicalize_intent_label(intent.get("primary", ""), fallback=fallback_intent) or fallback_intent
    secondary = [
        label
        for label in (
            _canonicalize_intent_label(item, fallback=None)
            for item in _coerce_str_list(intent.get("secondary", []))
        )
        if label and label != primary
    ]
    secondary = _dedupe_keep_order(secondary)
    concepts = _coerce_str_list(obj.get("concepts", []))
    targets = _coerce_str_list(obj.get("targets", []))

    if _looks_like_fact_pattern(query) and primary in {"lookup", "definition", "explanation", "reasoning", "consequence"}:
        if primary != "case application":
            secondary = _dedupe_keep_order([primary] + secondary)
        primary = "case application"
    if _query_requests_penalty(query) and primary != "legal penalty" and "legal penalty" not in secondary:
        secondary = _dedupe_keep_order(secondary + ["legal penalty"])

    raw_subqueries = decomposition.get("sub_queries", [])
    needed = bool(decomposition.get("needed", False))

    if not needed:
        sub_queries = []
    else:
        sub_queries = _normalize_subquery_objects(
            raw_subqueries,
            query=query,
            targets=targets,
            concepts=concepts,
            primary_intent=primary,
        )

        if not sub_queries:
            fallback_contexts = list(targets[:4]) or list(concepts[:4])
            sub_queries = _normalize_subquery_objects(
                fallback_contexts,
                query=query,
                targets=targets,
                concepts=concepts,
                primary_intent=primary,
            )
    if needed and not sub_queries:
        fallback_contexts = list(targets[:4]) or list(concepts[:4])
        sub_queries = _normalize_subquery_objects(
            fallback_contexts,
            query=query,
            targets=targets,
            concepts=concepts,
            primary_intent=primary,
        )

    inferred_multi_hop = bool(
        query_features.get("is_multi_hop", False)
        or needed
        or len(sub_queries) > 1
        or rule_output.get("multi_intent", False)
    )
    inferred_requires_reasoning = bool(
        query_features.get("requires_reasoning", False)
        or rule_output.get("complexity") == "complex"
        or (needed and len(sub_queries) > 1)
    )

    normalized = {
        "query": query,
        "intent": {
            "primary": primary,
            "secondary": secondary,
        },
        "concepts": concepts,
        "targets": targets,
        "query_features": {
            "is_multi_hop": inferred_multi_hop,
            "requires_reasoning": inferred_requires_reasoning,
        },
        "decomposition": {
            "needed": bool(needed),
            "sub_queries": sub_queries,
        },
    }
    return normalized


def _build_rule_only_output(
    query: str,
    rule_output: Dict[str, Any],
    retriever: HintRetriever,
) -> Tuple[Dict[str, Any], float]:
    section_numbers = _extract_section_numbers(query)
    matched_records = retriever.lookup_sections(section_numbers)
    concepts = _dedupe_keep_order(
        concept
        for record in matched_records
        for concept in record.legal_concepts
        if concept
    )
    targets = [f"Section {number}" for number in section_numbers]
    decomposition_needed = len(targets) > 1

    output = {
        "query": query,
        "intent": {
            "primary": "lookup",
            "secondary": [],
        },
        "concepts": concepts[:8],
        "targets": targets,
        "query_features": {
            "is_multi_hop": decomposition_needed,
            "requires_reasoning": False,
        },
        "decomposition": {
            "needed": decomposition_needed,
            "sub_queries": (
                _normalize_subquery_objects(
                    targets,
                    query=query,
                    targets=targets,
                    concepts=concepts[:8],
                    primary_intent="lookup",
                )
                if decomposition_needed
                else []
            ),
        },
    }
    hint_relevance = 1.0 if matched_records else 0.7
    return output, hint_relevance


def _build_non_llm_fallback_output(
    query: str,
    rule_output: Dict[str, Any],
    hints: Dict[str, Any],
) -> Dict[str, Any]:
    hint_items = hints.get("items", []) if isinstance(hints.get("items"), list) else []
    concepts = _dedupe_keep_order(
        concept
        for item in hint_items
        for concept in _coerce_str_list(item.get("legal_concepts", []))
    )
    targets = _extract_exact_references(query)
    decomposition_needed = bool(rule_output.get("multi_intent")) or rule_output.get("complexity") == "complex"
    primary = "lookup" if targets else ("case application" if _looks_like_fact_pattern(query) else "explanation")
    secondary: List[str] = []
    if _query_requests_penalty(query) and primary != "legal penalty":
        secondary.append("legal penalty")
    fallback_contexts = list(targets[:4]) or list(concepts[:4])
    return {
        "query": query,
        "intent": {
            "primary": primary,
            "secondary": secondary,
        },
        "concepts": concepts[:8],
        "targets": targets,
        "query_features": {
            "is_multi_hop": decomposition_needed,
            "requires_reasoning": rule_output.get("complexity") == "complex",
        },
        "decomposition": {
            "needed": decomposition_needed,
            "sub_queries": (
                _normalize_subquery_objects(
                    fallback_contexts,
                    query=query,
                    targets=targets,
                    concepts=concepts[:8],
                    primary_intent=primary,
                )
                if decomposition_needed
                else []
            ),
        },
    }


def _score_rule_strength(rule_output: Dict[str, Any]) -> float:
    score = 0.2
    if rule_output.get("exact_reference"):
        score += 0.45
    if rule_output.get("multi_intent"):
        score += 0.15
    if rule_output.get("complexity") == "medium":
        score += 0.1
    if rule_output.get("complexity") == "complex":
        score += 0.2
    if rule_output.get("ambiguous"):
        score -= 0.25
    if int(rule_output.get("query_length", 0) or 0) >= 5:
        score += 0.05
    return _clamp(score)


def _score_llm_consistency(result: Dict[str, Any], route: str) -> float:
    if route == "rules_only":
        return 1.0

    score = 0.0
    if _normalize_space(result.get("intent", {}).get("primary", "")):
        score += 0.2
    if isinstance(result.get("intent", {}).get("secondary"), list):
        score += 0.1
    if isinstance(result.get("concepts"), list):
        score += 0.15
    if isinstance(result.get("targets"), list):
        score += 0.15
    features = result.get("query_features", {})
    if isinstance(features, dict) and isinstance(features.get("is_multi_hop"), bool):
        score += 0.15
    if isinstance(features, dict) and isinstance(features.get("requires_reasoning"), bool):
        score += 0.15
    decomposition = result.get("decomposition", {})
    if isinstance(decomposition, dict) and isinstance(decomposition.get("needed"), bool):
        score += 0.05
    if isinstance(decomposition, dict) and isinstance(decomposition.get("sub_queries"), list):
        score += 0.05
    if isinstance(decomposition, dict):
        needed = bool(decomposition.get("needed"))
        sub_queries = decomposition.get("sub_queries", [])
        if (needed and isinstance(sub_queries, list) and sub_queries) or (not needed and sub_queries == []):
            score += 0.05
    return _clamp(score)


def _score_structure_validity(result: Dict[str, Any]) -> float:
    checks = [
        isinstance(result.get("query"), str),
        isinstance(result.get("intent"), dict),
        isinstance(result.get("intent", {}).get("primary"), str),
        isinstance(result.get("intent", {}).get("secondary"), list),
        isinstance(result.get("concepts"), list),
        isinstance(result.get("targets"), list),
        isinstance(result.get("query_features"), dict),
        isinstance(result.get("query_features", {}).get("is_multi_hop"), bool),
        isinstance(result.get("query_features", {}).get("requires_reasoning"), bool),
        isinstance(result.get("decomposition"), dict),
        isinstance(result.get("decomposition", {}).get("needed"), bool),
        isinstance(result.get("decomposition", {}).get("sub_queries"), list),
    ]
    return round(sum(1.0 for ok in checks if ok) / len(checks), 4)


def compute_confidence(
    hint_relevance: float,
    llm_consistency: float,
    rule_strength: float,
    structure_validity: float,
) -> Dict[str, float]:
    overall = (
        0.4 * _clamp(hint_relevance)
        + 0.3 * _clamp(llm_consistency)
        + 0.2 * _clamp(rule_strength)
        + 0.1 * _clamp(structure_validity)
    )
    return {
        "hint_relevance": round(_clamp(hint_relevance), 4),
        "llm_consistency": round(_clamp(llm_consistency), 4),
        "rule_strength": round(_clamp(rule_strength), 4),
        "structure_validity": round(_clamp(structure_validity), 4),
        "overall": round(_clamp(overall), 4),
    }


def main_pipeline(
    user_query: str,
    chunks: Sequence[Dict[str, Any]] | str | Path,
    model: str = "gpt-4o-mini",
    enable_llm: bool = True,
    llm_client: Optional[OpenAIClient] = None,
    top_k_hints: int = 4,
) -> Dict[str, Any]:
    chunk_list = load_chunks(chunks) if isinstance(chunks, (str, Path)) else list(chunks)
    embedding_provider = llm_client.embed if llm_client is not None else None
    retriever = HintRetriever(chunk_list, embedding_provider=embedding_provider)
    llm_available = enable_llm and llm_client is not None

    rule_output = analyze_rules(user_query)
    route = route_query(rule_output)

    if route == "rules_only":
        result, hint_relevance = _build_rule_only_output(user_query, rule_output, retriever)
        llm_consistency = 1.0
    else:
        hints = retrieve_hints(user_query, retriever, top_k=top_k_hints)
        hint_relevance = float(hints.get("hint_relevance", 0.0) or 0.0)
        if llm_available:
            prompt = build_prompt(user_query, hints, rule_output)
            llm_raw = call_llm(prompt, model=model, llm_client=llm_client)
            result = _normalize_output_shape(
                query=user_query,
                obj=llm_raw,
                rule_output=rule_output,
                fallback_intent="explanation",
            )
        else:
            result = _build_non_llm_fallback_output(user_query, rule_output, hints)
        llm_consistency = _score_llm_consistency(result, route=route) if llm_available else 0.0

    structure_validity = _score_structure_validity(result)
    rule_strength = _score_rule_strength(rule_output)
    result["confidence"] = compute_confidence(
        hint_relevance=hint_relevance,
        llm_consistency=llm_consistency,
        rule_strength=rule_strength,
        structure_validity=structure_validity,
    )
    return result


class QueryAnalyzer:
    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        llm_client: Optional[OpenAIClient] = None,
        llm_model: str = "gpt-4o-mini",
        enable_llm: bool = True,
        top_k_hints: int = 4,
    ):
        self.chunks = list(chunks)
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.enable_llm = enable_llm
        self.top_k_hints = top_k_hints

    def analyze(self, query: str) -> Dict[str, Any]:
        return main_pipeline(
            user_query=query,
            chunks=self.chunks,
            model=self.llm_model,
            enable_llm=self.enable_llm,
            llm_client=self.llm_client,
            top_k_hints=self.top_k_hints,
        )


def build_analyzer(chunks_path: str, model: str, enable_llm: bool = True) -> QueryAnalyzer:
    llm_client: Optional[OpenAIClient] = None
    if enable_llm and os.getenv("OPENAI_API_KEY"):
        try:
            llm_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", ""))
        except Exception:
            llm_client = None
    return QueryAnalyzer(
        chunks=load_chunks(chunks_path),
        llm_client=llm_client,
        llm_model=model,
        enable_llm=enable_llm and llm_client is not None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 hybrid legal query analysis")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--output", help="Optional output file")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM analysis")
    args = parser.parse_args()

    analyzer = build_analyzer(args.chunks, model=args.model, enable_llm=not args.no_llm)
    result = analyzer.analyze(args.query)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()



# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "Someone hit me during an argument, what crime is this and what is the punishment?"     --output query_analysis/result_2.json



# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "What is the difference between wrongful gain and wrongful loss with example?"     --output query_analysis/result_2.json

# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "Can a doctor operate without consent in emergency situations legally?"     --output query_analysis/result_3.json

# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "What is section 303?"     --output query_analysis/result_5_.json


# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "What is the punishment for robbery?"     --output query_analysis/result_6_.json

# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "Someone hit me during an argument, but I had provoked them—what is the legal outcome?"     --output query_analysis/result_7_.json

# python query_analysis/aq_d.py --chunks data/processed/artifacts2/chunks.json     --query "Explain section 375."     --output query_analysis/result_8_.json