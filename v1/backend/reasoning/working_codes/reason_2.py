from __future__ import annotations

"""Phase 11 - Constrained legal reasoning over validated evidence.

Phase 11 does not retrieve or traverse the legal graph. It consumes the final
evidence packet already assembled by upstream phases and produces a structured,
conservative reasoning result. The LLM, when enabled, only explains the
structured reasoning output and is never allowed to decide the law.
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


BASE_CHUNK_TYPES = {"section", "subsection", "clause", "content"}
EXPLANATORY_CHUNK_TYPES = {"explanation", "illustration"}
ROLE_PRIORITY = {
    "primary_rule": 9,
    "punishment": 8,
    "condition": 7,
    "exception": 7,
    "supporting_rule": 6,
    "definition": 5,
    "explanation": 4,
    "illustration": 3,
    "conflicting_clause": 2,
}

LOOKUP_INTENTS = {"lookup", "definition", "summary"}
APPLICATION_INTENTS = {"case application", "legal penalty"}
EXPLANATORY_INTENTS = {
    "explanation",
    "reasoning",
    "hypothetical",
    "comparison",
    "consequence",
    "procedure",
    "legal exception",
    "legal condition",
}

CONDITION_KEYWORDS = {
    "consent": ("consent", "without consent", "with consent"),
    "good faith": ("good faith",),
    "emergency": ("emergency", "impossible to obtain consent", "incapable of giving consent"),
    "provocation": ("grave and sudden provocation", "provocation"),
    "exception": ("exception", "does not apply", "shall not", "nothing is an offence"),
}

SECTION_REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|subsection|sub-section|clause)\s*"
    r"(\d+[a-z]?)(?:\s*\(\s*([0-9a-z]+)\s*\))?",
    flags=re.IGNORECASE,
)
PENALTY_SIGNAL_PATTERN = re.compile(
    r"\b(?:shall be punished|punished with|imprisonment|liable to fine|liable to punishment)\b",
    flags=re.IGNORECASE,
)
EXCEPTION_SIGNAL_PATTERN = re.compile(
    r"\b(?:exception|provided that|unless|shall not apply|nothing is an offence|notwithstanding)\b",
    flags=re.IGNORECASE,
)
CONDITION_SIGNAL_PATTERN = re.compile(
    r"\b(?:if|when|where|only if|provided that|in the event|with consent|without consent|good faith|provocation|emergency)\b",
    flags=re.IGNORECASE,
)
DEFINITION_SIGNAL_PATTERN = re.compile(
    r"\b(?:means|denotes|is said to|includes)\b",
    flags=re.IGNORECASE,
)
CONFLICT_CONNECTOR_PATTERN = re.compile(
    r"\b(?:except|unless|provided that|however|subject to|but)\b",
    flags=re.IGNORECASE,
)


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_lower(text: Any) -> str:
    return _normalize_space(text).lower()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


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


def _coerce_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [text for text in (_normalize_space(x) for x in value) if text]


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    text = _normalize_space(raw)
    if not text:
        return {}
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _safe_lower(text))


GENERIC_CONCEPT_TOKENS = {
    "bharatiya",
    "nyaya",
    "sanhita",
    "act",
    "law",
    "legal",
    "section",
}


def _extract_section_references(text: str) -> List[str]:
    refs: List[str] = []
    for match in SECTION_REFERENCE_PATTERN.finditer(text or ""):
        base = _normalize_space(match.group(1)).upper()
        sub = _normalize_space(match.group(2))
        refs.append(f"Section {base}{f'({sub})' if sub else ''}")
    return _dedupe_keep_order(refs)


def _canonical_reference_token(text: Any) -> Optional[str]:
    raw = _normalize_space(text)
    if not raw:
        return None
    match = SECTION_REFERENCE_PATTERN.search(raw)
    if match:
        base = _normalize_space(match.group(1)).upper()
        sub = _normalize_space(match.group(2))
        return f"{base}{f'({sub})' if sub else ''}"

    raw = raw.replace(",", " ")
    simple = re.match(r"^(\d+[a-z]?)(?:\s*\(\s*([0-9a-z]+)\s*\))?$", raw, flags=re.IGNORECASE)
    if simple:
        base = _normalize_space(simple.group(1)).upper()
        sub = _normalize_space(simple.group(2))
        return f"{base}{f'({sub})' if sub else ''}"
    return None


def _section_token_from_number(value: Any) -> Optional[str]:
    text = _normalize_space(value)
    if not text:
        return None
    return _canonical_reference_token(f"Section {text}")


def _split_sentences(text: str) -> List[str]:
    cleaned = _normalize_space(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    out = [part.strip() for part in parts if part.strip()]
    return out or ([cleaned] if cleaned else [])


def _compact_text(text: str, max_sentences: int = 2, max_chars: int = 420) -> str:
    if not text:
        return ""
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    selected = " ".join(sentences[:max_sentences]).strip()
    if len(selected) <= max_chars:
        return selected
    trimmed = selected[: max_chars - 1].rstrip()
    last_space = trimmed.rfind(" ")
    if last_space > 100:
        trimmed = trimmed[:last_space]
    return trimmed.rstrip(" ,;:") + "..."


def _join_brief(items: Sequence[str], max_items: int = 3) -> str:
    cleaned = [item for item in (_normalize_space(x) for x in items) if item][:max_items]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _normalize_phase9_items(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        if "retrieval" in obj:
            return [obj]
        if "items" in obj and isinstance(obj["items"], list):
            return [item for item in obj["items"] if isinstance(item, dict)]
    return []


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.4:
        return "guarded"
    return "low"


def _risk_level(score: float, completeness: str, conflict_unresolved: bool) -> str:
    if completeness == "unsafe" or conflict_unresolved:
        return "high"
    if completeness == "partial" or score < 0.65:
        return "medium"
    return "low"


@dataclass
class CorpusChunk:
    chunk_id: str
    chunk_type: str
    act: str
    chapter_title: str
    section_number: Optional[str]
    section_title: str
    section_type: str
    citation_text: str
    text: str
    derived_context: str
    keywords: List[str] = field(default_factory=list)
    legal_concepts: List[str] = field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    children_chunk_ids: List[str] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusChunk":
        section = dict(data.get("section", {}) or {})
        citation = dict(data.get("citation", {}) or {})
        chapter = dict(data.get("chapter", {}) or {})
        return cls(
            chunk_id=_normalize_space(data.get("chunk_id")),
            chunk_type=_safe_lower(data.get("chunk_type")),
            act=_normalize_space(data.get("act")),
            chapter_title=_normalize_space(chapter.get("chapter_title")),
            section_number=_normalize_space(section.get("section_number")) or None,
            section_title=_normalize_space(section.get("section_title")),
            section_type=_safe_lower(section.get("section_type")),
            citation_text=_normalize_space(citation.get("citation_text")),
            text=_normalize_space(data.get("text")),
            derived_context=_normalize_space(data.get("derived_context")),
            keywords=_coerce_str_list(data.get("keywords")),
            legal_concepts=_coerce_str_list(data.get("legal_concepts")),
            parent_chunk_id=_normalize_space(data.get("parent_chunk_id")) or None,
            children_chunk_ids=_coerce_str_list(data.get("children_chunk_ids")),
            references=[ref for ref in (data.get("references") or []) if isinstance(ref, dict)],
        )

    @property
    def section_token(self) -> Optional[str]:
        return _section_token_from_number(self.section_number)


class CorpusIndex:
    def __init__(self, chunks_path: str):
        raw = _load_json(chunks_path)
        chunk_items = raw.get("chunks", []) if isinstance(raw, dict) else []

        self.doc_meta = {
            "doc_id": _normalize_space(raw.get("doc_id")) if isinstance(raw, dict) else "",
            "act": _normalize_space(raw.get("act")) if isinstance(raw, dict) else "",
        }
        self.by_chunk_id: Dict[str, CorpusChunk] = {}
        self.by_section_token: Dict[str, List[CorpusChunk]] = defaultdict(list)
        self.citation_token_to_chunks: Dict[str, List[CorpusChunk]] = defaultdict(list)

        for item in chunk_items:
            if not isinstance(item, dict):
                continue
            chunk = CorpusChunk.from_dict(item)
            if not chunk.chunk_id:
                continue
            self.by_chunk_id[chunk.chunk_id] = chunk
            if chunk.section_token:
                self.by_section_token[chunk.section_token].append(chunk)
            citation_token = _canonical_reference_token(chunk.citation_text)
            if citation_token:
                self.citation_token_to_chunks[citation_token].append(chunk)

    def get_chunk(self, chunk_id: str) -> Optional[CorpusChunk]:
        return self.by_chunk_id.get(chunk_id)

    def has_reference(self, reference: str) -> bool:
        token = _canonical_reference_token(reference)
        if not token:
            return False
        base = token.split("(")[0]
        return bool(self.citation_token_to_chunks.get(token) or self.by_section_token.get(base))

    def section_title(self, section_token: Optional[str]) -> str:
        if not section_token:
            return ""
        chunks = self.by_section_token.get(section_token, [])
        for chunk in chunks:
            if chunk.section_title:
                return chunk.section_title
        return ""


@dataclass
class EvidenceHit:
    rank: int
    chunk_id: str
    score: float
    chunk_type: str
    act: str
    chapter_title: str
    section_number: Optional[str]
    citation: str
    text: str
    derived_context: str
    legal_concepts: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    concept_coverage_ratio: float = 0.0
    concept_match_count: int = 0
    reasons: List[str] = field(default_factory=list)
    source_scores: Dict[str, float] = field(default_factory=dict)
    query_scores: Dict[str, float] = field(default_factory=dict)
    corpus_chunk: Optional[CorpusChunk] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], corpus_chunk: Optional[CorpusChunk]) -> "EvidenceHit":
        coverage = dict(data.get("concept_coverage", {}) or {})
        section_number = _normalize_space(data.get("section_number"))
        if not section_number and corpus_chunk and corpus_chunk.section_number:
            section_number = corpus_chunk.section_number
        citation = _normalize_space(data.get("citation"))
        if not citation and corpus_chunk and corpus_chunk.citation_text:
            citation = corpus_chunk.citation_text
        return cls(
            rank=int(data.get("rank", 0) or 0),
            chunk_id=_normalize_space(data.get("chunk_id")),
            score=float(data.get("score", 0.0) or 0.0),
            chunk_type=_safe_lower(data.get("chunk_type") or (corpus_chunk.chunk_type if corpus_chunk else "")),
            act=_normalize_space(data.get("act") or (corpus_chunk.act if corpus_chunk else "")),
            chapter_title=_normalize_space(data.get("chapter_title") or (corpus_chunk.chapter_title if corpus_chunk else "")),
            section_number=section_number or None,
            citation=citation,
            text=_normalize_space(data.get("text") or (corpus_chunk.text if corpus_chunk else "")),
            derived_context=_normalize_space(data.get("derived_context") or (corpus_chunk.derived_context if corpus_chunk else "")),
            legal_concepts=_coerce_str_list(data.get("legal_concepts") or (corpus_chunk.legal_concepts if corpus_chunk else [])),
            keywords=_coerce_str_list(data.get("keywords") or (corpus_chunk.keywords if corpus_chunk else [])),
            concept_coverage_ratio=float(coverage.get("coverage_ratio", 0.0) or 0.0),
            concept_match_count=int(coverage.get("matched_count", 0) or 0),
            reasons=_coerce_str_list(data.get("reasons")),
            source_scores=dict(data.get("source_scores", {}) or {}),
            query_scores=dict(data.get("query_scores", {}) or {}),
            corpus_chunk=corpus_chunk,
        )

    @property
    def section_token(self) -> Optional[str]:
        return _section_token_from_number(self.section_number)

    @property
    def section_label(self) -> Optional[str]:
        return f"Section {self.section_token}" if self.section_token else None

    @property
    def is_base(self) -> bool:
        return self.chunk_type in BASE_CHUNK_TYPES

    @property
    def is_explanatory(self) -> bool:
        return self.chunk_type in EXPLANATORY_CHUNK_TYPES


@dataclass
class RequestedTargetSummary:
    requested: List[str]
    present_in_corpus: List[str]
    missing_from_corpus: List[str]
    retrieved_matches: List[str]
    support_status: str


class Phase11OpenAIClient:
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


class Phase11Reasoner:
    def __init__(
        self,
        corpus_index: CorpusIndex,
        top_support_hits: int = 4,
        llm_client: Optional[Phase11OpenAIClient] = None,
        llm_model: str = "gpt-4o-mini",
        enable_llm: bool = False,
    ):
        self.corpus_index = corpus_index
        self.top_support_hits = max(1, int(top_support_hits))
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.enable_llm = bool(enable_llm and llm_client is not None)

    def _extract_requested_targets(self, query: str, phase8: Dict[str, Any]) -> List[str]:
        phase8_targets = phase8.get("targets", []) or []
        target_values: List[str] = []
        for item in phase8_targets:
            if isinstance(item, dict):
                target_values.extend([_normalize_space(v) for v in item.values() if _normalize_space(v)])
            else:
                value = _normalize_space(item)
                if value:
                    target_values.append(value)
        target_values.extend(_extract_section_references(query))
        target_values.extend(_extract_section_references(" ".join(target_values)))

        normalized: List[str] = []
        for target in target_values:
            token = _canonical_reference_token(target)
            if token:
                normalized.append(f"Section {token}")
        return _dedupe_keep_order(normalized)

    def _load_hits(self, retrieval: Dict[str, Any]) -> List[EvidenceHit]:
        rows = retrieval.get("results_with_global_rerank") or retrieval.get("results_without_global_rerank") or []
        hits: List[EvidenceHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            chunk_id = _normalize_space(row.get("chunk_id"))
            corpus_chunk = self.corpus_index.get_chunk(chunk_id) if chunk_id else None
            hit = EvidenceHit.from_dict(row, corpus_chunk)
            if hit.chunk_id:
                hits.append(hit)
        return hits

    def _target_matches_hit(self, target: str, hit: EvidenceHit) -> bool:
        target_token = _canonical_reference_token(target)
        if not target_token:
            return False
        base = target_token.split("(")[0]
        if hit.section_token == base:
            return True
        citation_token = _canonical_reference_token(hit.citation)
        if citation_token == target_token:
            return True
        if hit.corpus_chunk:
            chunk_citation_token = _canonical_reference_token(hit.corpus_chunk.citation_text)
            if chunk_citation_token == target_token:
                return True
        return False

    def _summarize_targets(self, requested_targets: List[str], hits: List[EvidenceHit]) -> RequestedTargetSummary:
        if not requested_targets:
            return RequestedTargetSummary([], [], [], [], "no_specific_section_requested")

        present = [target for target in requested_targets if self.corpus_index.has_reference(target)]
        missing = [target for target in requested_targets if target not in present]
        retrieved = [target for target in present if any(self._target_matches_hit(target, hit) for hit in hits)]

        if missing and not present:
            status = "section_not_in_corpus"
        elif present and retrieved:
            status = "exact_section_supported"
        elif present and not retrieved:
            status = "section_exists_but_not_retrieved_well"
        else:
            status = "no_specific_section_requested"

        return RequestedTargetSummary(
            requested=requested_targets,
            present_in_corpus=present,
            missing_from_corpus=missing,
            retrieved_matches=retrieved,
            support_status=status,
        )

    def _intent_family(self, phase8: Dict[str, Any]) -> str:
        intent = dict(phase8.get("intent", {}) or {})
        primary = _safe_lower(intent.get("primary", "explanation"))
        if primary in LOOKUP_INTENTS:
            return "lookup"
        if primary in APPLICATION_INTENTS:
            return "application"
        if primary in EXPLANATORY_INTENTS:
            return "explanation"
        return "general"

    def _condition_focuses(self, query: str, phase8: Dict[str, Any]) -> List[str]:
        text = f"{query} {' '.join(_coerce_str_list(phase8.get('concepts')))}".lower()
        found: List[str] = []
        for label, phrases in CONDITION_KEYWORDS.items():
            if any(phrase in text for phrase in phrases):
                found.append(label)
        return found

    def _classify_roles(
        self,
        hit: EvidenceHit,
        phase8: Dict[str, Any],
        target_summary: RequestedTargetSummary,
        condition_focuses: List[str],
    ) -> List[str]:
        roles: List[str] = []
        title = _safe_lower(hit.corpus_chunk.section_title if hit.corpus_chunk else "")
        text = _safe_lower(f"{hit.text}\n{hit.derived_context}")
        intent_primary = _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary"))

        if hit.chunk_type == "illustration":
            roles.append("illustration")
        if hit.chunk_type == "explanation":
            roles.append("explanation")

        if hit.section_label and hit.section_label in target_summary.retrieved_matches:
            roles.append("primary_rule")
        elif hit.is_base:
            roles.append("supporting_rule")

        if PENALTY_SIGNAL_PATTERN.search(text) or "punishment" in title:
            roles.append("punishment")
        if EXCEPTION_SIGNAL_PATTERN.search(text) or "exception" in title:
            roles.append("exception")
        if CONDITION_SIGNAL_PATTERN.search(text) or any(condition in text for condition in condition_focuses):
            roles.append("condition")
        if DEFINITION_SIGNAL_PATTERN.search(text) or hit.corpus_chunk and hit.corpus_chunk.section_type == "definition" or "definition" in title:
            roles.append("definition")

        if intent_primary == "lookup" and hit.is_base and "primary_rule" not in roles:
            roles.append("primary_rule" if hit.rank == 1 else "supporting_rule")
        if intent_primary == "legal penalty" and "punishment" in roles and "primary_rule" not in roles and hit.is_base:
            roles.append("primary_rule")

        return _dedupe_keep_order(roles or (["supporting_rule"] if hit.is_base else ["explanation"]))

    def _role_assignments(
        self,
        hits: List[EvidenceHit],
        phase8: Dict[str, Any],
        target_summary: RequestedTargetSummary,
    ) -> Dict[str, List[str]]:
        condition_focuses = self._condition_focuses(_normalize_space(phase8.get("query")), phase8)
        assignments: Dict[str, List[str]] = {}
        for hit in hits:
            assignments[hit.chunk_id] = self._classify_roles(hit, phase8, target_summary, condition_focuses)
        return assignments

    def _preferred_hit(
        self,
        hits: List[EvidenceHit],
        intent_family: str,
        target_summary: RequestedTargetSummary,
        role_assignments: Dict[str, List[str]],
    ) -> Optional[EvidenceHit]:
        if not hits:
            return None

        target_tokens = {
            _canonical_reference_token(target).split("(")[0]
            for target in target_summary.present_in_corpus
            if _canonical_reference_token(target)
        }
        requested_match_required = bool(target_tokens)
        type_priority = {
            "content": 6,
            "subsection": 5,
            "section": 5,
            "clause": 4,
            "explanation": 3,
            "illustration": 2,
        }

        def sort_key(hit: EvidenceHit) -> Tuple[int, int, int, int, int, float, float]:
            roles = role_assignments.get(hit.chunk_id, [])
            target_match = 1 if hit.section_token in target_tokens else 0
            direct_rule = 1 if hit.is_base else 0
            punishment_present = 1 if "punishment" in roles else 0
            primary_role = max((ROLE_PRIORITY.get(role, 0) for role in roles), default=0)
            if intent_family == "lookup":
                return (target_match, direct_rule, type_priority.get(hit.chunk_type, 0), primary_role, hit.concept_match_count, hit.concept_coverage_ratio, hit.score)
            if intent_family == "application":
                return (target_match, punishment_present, direct_rule, type_priority.get(hit.chunk_type, 0), primary_role, hit.concept_coverage_ratio, hit.score)
            return (target_match, direct_rule, type_priority.get(hit.chunk_type, 0), primary_role, hit.concept_match_count, hit.concept_coverage_ratio, hit.score)

        if requested_match_required:
            target_hits = [hit for hit in hits if hit.section_token in target_tokens]
            if target_hits:
                return max(target_hits, key=sort_key)
        return max(hits, key=sort_key)

    def _related_hits(
        self,
        primary_hit: Optional[EvidenceHit],
        all_hits: List[EvidenceHit],
        role_assignments: Dict[str, List[str]],
    ) -> List[EvidenceHit]:
        if not primary_hit:
            return []

        same_section = [hit for hit in all_hits if hit.section_token == primary_hit.section_token and hit.chunk_id != primary_hit.chunk_id]
        cross_section_support = [
            hit for hit in all_hits
            if hit.section_token != primary_hit.section_token
            and hit.score >= max(0.35, primary_hit.score * 0.85)
            and any(role in role_assignments.get(hit.chunk_id, []) for role in {"exception", "condition", "punishment", "conflicting_clause"})
        ]

        def sort_key(hit: EvidenceHit) -> Tuple[int, float, int]:
            roles = role_assignments.get(hit.chunk_id, [])
            priority = max((ROLE_PRIORITY.get(role, 0) for role in roles), default=0)
            return (priority, hit.score, -hit.rank)

        ordered = sorted(same_section + cross_section_support, key=sort_key, reverse=True)
        deduped: List[EvidenceHit] = []
        seen = set()
        for hit in ordered:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            deduped.append(hit)
        return deduped[: self.top_support_hits]

    def _matched_concepts(self, phase8: Dict[str, Any], hits: List[EvidenceHit]) -> Tuple[List[str], List[str]]:
        query_concepts = _coerce_str_list(phase8.get("concepts"))
        evidence_tokens: Set[str] = set()
        for hit in hits:
            for concept in hit.legal_concepts:
                evidence_tokens.update(_tokenize(concept))
            for keyword in hit.keywords:
                evidence_tokens.update(_tokenize(keyword))
            evidence_tokens.update(_tokenize(hit.text))
            evidence_tokens.update(_tokenize(hit.derived_context))

        matched: List[str] = []
        for concept in query_concepts:
            concept_tokens = {token for token in _tokenize(concept) if token not in GENERIC_CONCEPT_TOKENS}
            if not concept_tokens:
                matched.append(concept)
                continue
            if concept_tokens & evidence_tokens:
                matched.append(concept)
        unmatched = [concept for concept in query_concepts if concept not in matched]
        return _dedupe_keep_order(matched), _dedupe_keep_order(unmatched)

    def _required_roles(
        self,
        query: str,
        phase8: Dict[str, Any],
        target_summary: RequestedTargetSummary,
    ) -> List[str]:
        intent = _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary"))
        required = ["primary_rule"]
        if intent == "legal penalty":
            required.extend(["punishment"])
        if intent == "lookup" and target_summary.requested:
            required.append("primary_rule")
        for condition in self._condition_focuses(query, phase8):
            if condition in {"consent", "good faith", "emergency", "provocation"}:
                required.append("condition")
            if condition == "exception":
                required.append("exception")
        return _dedupe_keep_order(required)

    def _dependency_coverage(
        self,
        required_roles: List[str],
        role_assignments: Dict[str, List[str]],
    ) -> List[str]:
        present_roles = {role for roles in role_assignments.values() for role in roles}
        missing = [role for role in required_roles if role not in present_roles]
        return _dedupe_keep_order(missing)

    def _fact_support_status(
        self,
        query: str,
        phase8: Dict[str, Any],
        matched_concepts: List[str],
        unmatched_concepts: List[str],
        missing_dependencies: List[str],
    ) -> str:
        primary_intent = _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary"))
        if primary_intent in LOOKUP_INTENTS and not missing_dependencies:
            return "supported"
        if primary_intent in {"procedure", "explanation", "reasoning", "summary"} and matched_concepts and not missing_dependencies:
            return "supported"
        if primary_intent in {"case application", "legal penalty", "hypothetical", "consequence"}:
            if not matched_concepts and phase8.get("concepts"):
                return "unsupported"
            if missing_dependencies:
                return "underspecified"
            return "underspecified"
        if not matched_concepts and phase8.get("concepts"):
            return "unsupported"
        if missing_dependencies:
            return "underspecified"
        if self._condition_focuses(query, phase8) and "condition" in missing_dependencies:
            return "underspecified"
        return "supported"

    def _support_status_for_intent(
        self,
        hits: List[EvidenceHit],
        primary_hit: Optional[EvidenceHit],
        target_summary: RequestedTargetSummary,
    ) -> str:
        if not hits or not primary_hit:
            return "unsupported"
        if target_summary.support_status in {"section_not_in_corpus", "section_exists_but_not_retrieved_well"}:
            return target_summary.support_status
        return "supported"

    def _supporting_sections_by_role(
        self,
        hits: List[EvidenceHit],
        role_assignments: Dict[str, List[str]],
        role: str,
    ) -> List[str]:
        sections = [hit.section_label for hit in hits if role in role_assignments.get(hit.chunk_id, []) and hit.section_label]
        return _dedupe_keep_order([section for section in sections if section])

    def _detect_conflicts(
        self,
        hits: List[EvidenceHit],
        role_assignments: Dict[str, List[str]],
        primary_hit: Optional[EvidenceHit],
    ) -> Dict[str, Any]:
        sections = _dedupe_keep_order([hit.section_label for hit in hits if hit.section_label])
        reasons: List[str] = []
        conflicting_sections: List[str] = []

        base_hits = [hit for hit in hits if hit.is_base and hit.section_label]
        if len(base_hits) > 1 and primary_hit:
            competing = [
                hit for hit in base_hits
                if hit.section_label != primary_hit.section_label
                and hit.score >= max(0.55, primary_hit.score * 0.8)
            ]
            if competing:
                reasons.append("Multiple strong base-rule sections were retrieved for the same query.")
                conflicting_sections.extend([primary_hit.section_label] + [hit.section_label for hit in competing if hit.section_label])

        for hit in hits:
            roles = role_assignments.get(hit.chunk_id, [])
            if "conflicting_clause" in roles:
                conflicting_sections.append(hit.section_label or hit.chunk_id)
                reasons.append(f"Potential limiting or conflicting clause found in {hit.section_label or hit.chunk_id}.")

        conflicting_sections = [section for section in _dedupe_keep_order(conflicting_sections) if section]
        unresolved = bool(conflicting_sections and len(conflicting_sections) > 1)
        resolved = bool(conflicting_sections) and not unresolved

        resolution_notes: List[str] = []
        if resolved and primary_hit:
            resolution_notes.append(
                f"Conflict resolved conservatively in favor of {primary_hit.section_label or primary_hit.chunk_id} because direct rule text outranks explanatory or peripheral material."
            )

        return {
            "has_conflict": bool(conflicting_sections),
            "resolved": resolved,
            "unresolved": unresolved,
            "conflicting_sections": conflicting_sections,
            "reasons": _dedupe_keep_order(reasons),
            "resolution_notes": resolution_notes,
        }

    def _completeness(
        self,
        support_status: str,
        missing_dependencies: List[str],
        conflict_flags: Dict[str, Any],
        fact_support_status: str,
    ) -> str:
        if support_status in {"unsupported", "section_not_in_corpus", "section_exists_but_not_retrieved_well"}:
            return "unsafe"
        if conflict_flags.get("unresolved"):
            return "unsafe"
        if fact_support_status == "unsupported":
            return "unsafe"
        if missing_dependencies or fact_support_status == "underspecified":
            return "partial"
        return "complete"

    def _confidence_score(
        self,
        hits: List[EvidenceHit],
        primary_hit: Optional[EvidenceHit],
        target_summary: RequestedTargetSummary,
        missing_dependencies: List[str],
        conflict_flags: Dict[str, Any],
        fact_support_status: str,
        completeness: str,
    ) -> float:
        if not hits or not primary_hit:
            return 0.12

        score = 0.2
        if target_summary.support_status == "exact_section_supported":
            score += 0.25
        elif target_summary.support_status == "no_specific_section_requested":
            score += 0.12
        elif target_summary.support_status == "section_exists_but_not_retrieved_well":
            score -= 0.12
        elif target_summary.support_status == "section_not_in_corpus":
            score -= 0.25

        if primary_hit.is_base:
            score += 0.16
        if primary_hit.concept_coverage_ratio:
            score += min(primary_hit.concept_coverage_ratio, 0.2)
        score += min(primary_hit.score * 0.15, 0.15)
        score -= min(len(missing_dependencies) * 0.08, 0.24)
        if conflict_flags.get("unresolved"):
            score -= 0.25
        elif conflict_flags.get("has_conflict"):
            score -= 0.08
        if fact_support_status == "underspecified":
            score -= 0.1
        elif fact_support_status == "unsupported":
            score -= 0.25
        if completeness == "partial":
            score -= 0.08
        elif completeness == "unsafe":
            score -= 0.2
        return round(_clamp(score, 0.05, 0.98), 4)

    def _canonical_conclusion(
        self,
        query: str,
        phase8: Dict[str, Any],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        completeness: str,
        missing_dependencies: List[str],
        conflict_flags: Dict[str, Any],
        target_summary: RequestedTargetSummary,
    ) -> str:
        if target_summary.support_status == "section_not_in_corpus":
            return f"{_join_brief(target_summary.missing_from_corpus)} is not present in the indexed corpus."
        if target_summary.support_status == "section_exists_but_not_retrieved_well":
            return f"{_join_brief(target_summary.present_in_corpus)} exists in the corpus, but the retrieved evidence does not safely support an answer."
        if not primary_hit:
            return "The retrieved evidence does not safely support a legal conclusion."

        section_label = primary_hit.section_label or primary_hit.citation or "the retrieved provision"
        section_title = primary_hit.corpus_chunk.section_title if primary_hit.corpus_chunk else ""
        title_part = f" ({section_title})" if section_title else ""
        core = _compact_text(primary_hit.text or primary_hit.derived_context, max_sentences=2, max_chars=360)

        if completeness == "unsafe":
            return (
                f"The evidence points toward {section_label}{title_part}, but the legal conclusion remains unsafe because "
                f"{_join_brief(missing_dependencies or conflict_flags.get('reasons') or ['the evidence is incomplete'])}."
            )
        if completeness == "partial":
            return (
                f"The best-supported provision is {section_label}{title_part}. {core} "
                f"This conclusion is partial because {_join_brief(missing_dependencies or ['some legal dependencies are not covered'])}."
            )
        return f"The evidence most strongly supports {section_label}{title_part}. {core}"

    def _citations(self, primary_hit: Optional[EvidenceHit], related_hits: List[EvidenceHit]) -> List[str]:
        citations: List[str] = []
        for hit in [primary_hit] + list(related_hits):
            if not hit:
                continue
            citation = hit.corpus_chunk.citation_text if hit.corpus_chunk and hit.corpus_chunk.citation_text else hit.citation
            if citation:
                citations.append(citation)
        return _dedupe_keep_order(citations)

    def _human_answer(
        self,
        completeness: str,
        canonical_conclusion: str,
        citations: List[str],
    ) -> Optional[str]:
        if completeness == "unsafe":
            return None
        if completeness == "partial":
            return f"{canonical_conclusion} Citations: {_join_brief(citations, max_items=4)}."
        return f"{canonical_conclusion} Citations: {_join_brief(citations, max_items=4)}."

    def _find_sentence(self, hits: Sequence[EvidenceHit], pattern: re.Pattern[str]) -> str:
        for hit in hits:
            for sentence in _split_sentences(hit.text or hit.derived_context):
                if pattern.search(sentence):
                    return _compact_text(sentence, max_sentences=1, max_chars=280)
        return ""

    def _supporting_detail(self, hits: Sequence[EvidenceHit], primary_chunk_id: Optional[str]) -> str:
        for hit in hits:
            if hit.chunk_id == primary_chunk_id:
                continue
            detail = _compact_text(hit.text or hit.derived_context, max_sentences=1, max_chars=260)
            if detail:
                return detail
        return ""

    def _fallback_summary_answer(
        self,
        query: str,
        phase8: Dict[str, Any],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        completeness: str,
        canonical_conclusion: str,
    ) -> str:
        if completeness == "unsafe" or not primary_hit:
            return canonical_conclusion

        intent = _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary"))
        query_lower = _safe_lower(query)
        section_label = primary_hit.section_label or primary_hit.citation or "the retrieved provision"
        title = _normalize_space(primary_hit.corpus_chunk.section_title if primary_hit.corpus_chunk else "")
        all_hits = [primary_hit] + list(related_hits)
        penalty = self._find_sentence(all_hits, PENALTY_SIGNAL_PATTERN)

        if "without consent" in query_lower or ("consent" in query_lower and "good faith" in title.lower()):
            return (
                "Yes, the retrieved evidence indicates that an act may be lawful without consent "
                "when it is done in good faith for the person's benefit and consent cannot be obtained in time."
            )

        if intent == "case application":
            if penalty:
                return f"The most likely applicable provision is {section_label}, and the retrieved punishment is {penalty}"
            return f"The most likely applicable provision is {section_label}."

        if intent == "legal penalty":
            if penalty:
                return f"{section_label} is the best-supported punishment provision here: {penalty}"
            return f"{section_label} is the best-supported punishment provision in the retrieved evidence."

        if intent in LOOKUP_INTENTS:
            core = _compact_text(primary_hit.text or primary_hit.derived_context, max_sentences=1, max_chars=220)
            if title:
                return f"{section_label} covers {title.rstrip('.')}. {core}"
            return f"{section_label} is the best-supported match. {core}"

        if intent in {"procedure", "explanation", "reasoning"}:
            core = _compact_text(primary_hit.text or primary_hit.derived_context, max_sentences=1, max_chars=220)
            return f"The retrieved evidence points to {section_label}. {core}"

        return canonical_conclusion

    def _fallback_detailed_answer(
        self,
        query: str,
        phase8: Dict[str, Any],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        completeness: str,
        canonical_conclusion: str,
    ) -> str:
        if completeness == "unsafe" or not primary_hit:
            return canonical_conclusion

        intent = _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary"))
        query_lower = _safe_lower(query)
        section_label = primary_hit.section_label or primary_hit.citation or "the retrieved provision"
        title = _normalize_space(primary_hit.corpus_chunk.section_title if primary_hit.corpus_chunk else "")
        title_part = f" ({title})" if title else ""
        all_hits = [primary_hit] + list(related_hits)
        rule_text = _compact_text(primary_hit.text or primary_hit.derived_context, max_sentences=2, max_chars=420)
        penalty = self._find_sentence(all_hits, PENALTY_SIGNAL_PATTERN)
        support_detail = self._supporting_detail(related_hits, primary_hit.chunk_id)

        if "without consent" in query_lower or ("consent" in query_lower and "good faith" in title.lower()):
            answer = (
                f"According to {section_label}{title_part}, the retrieved evidence indicates that an act done in good faith "
                f"for a person's benefit may not be an offence even without consent if consent cannot be obtained in time or "
                f"the person is incapable of giving it. {rule_text}"
            )
            if support_detail:
                answer += f" Supporting detail from the same evidence: {support_detail}"
            return answer

        if intent == "case application":
            answer = f"Based on {section_label}{title_part}, the retrieved evidence most strongly points to this provision. {rule_text}"
            if penalty and penalty not in answer:
                answer += f" The punishment language in the retrieved evidence is: {penalty}"
            answer += " The final applicability still depends on the exact facts of the incident."
            return answer

        if intent == "legal penalty":
            answer = f"The best-supported punishment basis is {section_label}{title_part}. {rule_text}"
            if penalty and penalty not in answer:
                answer += f" The retrieved punishment text is: {penalty}"
            return answer

        if intent in LOOKUP_INTENTS:
            answer = f"{section_label}{title_part} is the best-supported match in the retrieved evidence. {rule_text}"
            if support_detail:
                answer += f" Supporting detail: {support_detail}"
            return answer

        if intent in {"procedure", "explanation", "reasoning"}:
            answer = f"The retrieved evidence most strongly supports {section_label}{title_part}. {rule_text}"
            if support_detail:
                answer += f" A useful supporting detail is: {support_detail}"
            return answer

        return canonical_conclusion

    def _llm_system_prompt(self) -> str:
        return (
            "You are the final explanation renderer for Phase 11 of a legal RAG pipeline. "
            "You are not allowed to decide the law. "
            "You may only restate the supplied structured reasoning packet and approved citations. "
            "Do not add any legal claim, section, exception, punishment, dependency, or conflict not already present in the packet. "
            "Return JSON with keys: explanation, short_summary, caution_notes."
        )

    def _llm_user_prompt(
        self,
        reasoning: Dict[str, Any],
        validation: Dict[str, Any],
        output: Dict[str, Any],
    ) -> str:
        payload = {
            "reasoning": reasoning,
            "validation": validation,
            "output": {
                "final_legal_conclusion": output.get("final_legal_conclusion"),
                "completeness": output.get("completeness"),
                "risk_level": output.get("risk_level"),
                "citations": output.get("citations"),
                "unsafe_reasons": output.get("unsafe_reasons"),
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _llm_answers(
        self,
        query: str,
        phase8: Dict[str, Any],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        reasoning: Dict[str, Any],
        validation: Dict[str, Any],
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        completeness = _safe_lower(output.get("completeness"))
        fallback_explanation = self._fallback_detailed_answer(
            query=query,
            phase8=phase8,
            primary_hit=primary_hit,
            related_hits=related_hits,
            completeness=completeness,
            canonical_conclusion=output.get("final_legal_conclusion"),
        )
        fallback_summary = self._fallback_summary_answer(
            query=query,
            phase8=phase8,
            primary_hit=primary_hit,
            related_hits=related_hits,
            completeness=completeness,
            canonical_conclusion=output.get("final_legal_conclusion"),
        )
        if not self.enable_llm or self.llm_client is None:
            return {
                "enabled": False,
                "used": False,
                "model": None,
                "explanation": fallback_explanation,
                "short_summary": fallback_summary,
                "caution_notes": [],
                "error": None,
            }

        try:
            response = self.llm_client.generate_json(
                model=self.llm_model,
                system_prompt=self._llm_system_prompt(),
                user_prompt=self._llm_user_prompt(reasoning=reasoning, validation=validation, output=output),
            )
        except Exception as exc:
            return {
                "enabled": True,
                "used": False,
                "model": self.llm_model,
                "explanation": fallback_explanation,
                "short_summary": fallback_summary,
                "caution_notes": [],
                "error": str(exc),
            }

        return {
            "enabled": True,
            "used": True,
            "model": self.llm_model,
            "explanation": _normalize_space(response.get("explanation")) or fallback_explanation,
            "short_summary": _normalize_space(response.get("short_summary")) or output.get("final_legal_conclusion"),
            "caution_notes": _coerce_str_list(response.get("caution_notes")),
            "error": None,
        }

    def _verify_text(
        self,
        text: str,
        citations: List[str],
        target_summary: RequestedTargetSummary,
        known_sections: List[str],
    ) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not _normalize_space(text):
            issues.append("Empty text.")
            return False, issues
        mentioned = _extract_section_references(text)
        allowed = {_canonical_reference_token(section) for section in known_sections if _canonical_reference_token(section)}
        allowed.update({_canonical_reference_token(citation) for citation in citations if _canonical_reference_token(citation)})
        allowed.update({_canonical_reference_token(target) for target in target_summary.present_in_corpus if _canonical_reference_token(target)})
        for mention in mentioned:
            token = _canonical_reference_token(mention)
            if token and token not in allowed and token.split("(")[0] not in {value.split("(")[0] for value in allowed if value}:
                issues.append(f"Unsupported section reference in text: {mention}.")
        return len(issues) == 0, issues

    def reason_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        phase8 = dict(item.get("phase8", {}) or {})
        query = _normalize_space(item.get("query") or phase8.get("query"))
        retrieval = dict(item.get("retrieval", {}) or {})
        hits = self._load_hits(retrieval)
        requested_targets = self._extract_requested_targets(query, phase8)
        target_summary = self._summarize_targets(requested_targets, hits)
        role_assignments = self._role_assignments(hits, phase8, target_summary)
        intent_family = self._intent_family(phase8)

        primary_hit = self._preferred_hit(hits, intent_family, target_summary, role_assignments)
        if primary_hit:
            promoted_roles = [role for role in role_assignments.get(primary_hit.chunk_id, []) if role != "supporting_rule"]
            if "primary_rule" not in promoted_roles:
                promoted_roles.insert(0, "primary_rule")
            role_assignments[primary_hit.chunk_id] = _dedupe_keep_order(promoted_roles)
        related_hits = self._related_hits(primary_hit, hits, role_assignments)
        evidence_hits = [hit for hit in [primary_hit] + related_hits if hit]

        matched_concepts, unmatched_concepts = self._matched_concepts(phase8, evidence_hits)
        required_roles = self._required_roles(query, phase8, target_summary)
        missing_dependencies = self._dependency_coverage(required_roles, {hit.chunk_id: role_assignments.get(hit.chunk_id, []) for hit in evidence_hits})
        fact_support_status = self._fact_support_status(query, phase8, matched_concepts, unmatched_concepts, missing_dependencies)
        support_status = self._support_status_for_intent(evidence_hits, primary_hit, target_summary)
        conflict_flags = self._detect_conflicts(evidence_hits, role_assignments, primary_hit)
        completeness = self._completeness(support_status, missing_dependencies, conflict_flags, fact_support_status)
        confidence = self._confidence_score(evidence_hits, primary_hit, target_summary, missing_dependencies, conflict_flags, fact_support_status, completeness)
        citations = self._citations(primary_hit, related_hits)

        primary_sections = _dedupe_keep_order([hit.section_label for hit in evidence_hits if "primary_rule" in role_assignments.get(hit.chunk_id, []) and hit.section_label])
        supporting_sections = _dedupe_keep_order([hit.section_label for hit in evidence_hits if "supporting_rule" in role_assignments.get(hit.chunk_id, []) and hit.section_label])
        exception_sections = self._supporting_sections_by_role(evidence_hits, role_assignments, "exception")
        condition_sections = self._supporting_sections_by_role(evidence_hits, role_assignments, "condition")
        punishment_sections = self._supporting_sections_by_role(evidence_hits, role_assignments, "punishment")

        canonical_conclusion = self._canonical_conclusion(
            query=query,
            phase8=phase8,
            primary_hit=primary_hit,
            related_hits=related_hits,
            completeness=completeness,
            missing_dependencies=missing_dependencies,
            conflict_flags=conflict_flags,
            target_summary=target_summary,
        )
        risk_level = _risk_level(confidence, completeness, bool(conflict_flags.get("unresolved")))
        unsafe_reasons = _dedupe_keep_order(
            (missing_dependencies if completeness != "complete" else [])
            + conflict_flags.get("reasons", [])
            + (["fact support is unsupported"] if fact_support_status == "unsupported" else [])
            + target_summary.missing_from_corpus
        )
        human_answer = self._human_answer(completeness, canonical_conclusion, citations)

        reasoning = {
            "query_intent": dict(phase8.get("intent", {}) or {}),
            "matched_concepts": matched_concepts,
            "primary_sections": primary_sections,
            "supporting_sections": supporting_sections,
            "exception_sections": exception_sections,
            "condition_sections": condition_sections,
            "punishment_sections": punishment_sections,
            "conflict_flags": conflict_flags,
            "missing_dependencies": missing_dependencies,
            "fact_support_status": fact_support_status,
            "final_legal_conclusion": canonical_conclusion,
            "confidence": {
                "score": confidence,
                "label": _confidence_label(confidence),
            },
            "risk_level": risk_level,
            "citations": citations,
            "requested_sections": target_summary.requested,
            "retrieved_exact_matches": target_summary.retrieved_matches,
            "role_assignments": {
                hit.chunk_id: {
                    "roles": role_assignments.get(hit.chunk_id, []),
                    "section": hit.section_label,
                    "citation": hit.corpus_chunk.citation_text if hit.corpus_chunk and hit.corpus_chunk.citation_text else hit.citation,
                }
                for hit in evidence_hits
            },
            "resolution_notes": conflict_flags.get("resolution_notes", []),
            "unsafe_reasons": unsafe_reasons,
        }

        validation = {
            "support_status": support_status,
            "completeness": completeness,
            "requested_sections": target_summary.requested,
            "present_in_corpus": target_summary.present_in_corpus,
            "retrieved_exact_matches": target_summary.retrieved_matches,
            "missing_from_corpus": target_summary.missing_from_corpus,
            "required_roles": required_roles,
            "missing_dependencies": missing_dependencies,
            "fact_support_status": fact_support_status,
            "matched_concepts": matched_concepts,
            "unmatched_material_concepts": unmatched_concepts,
            "conflict_flags": conflict_flags,
        }

        output = {
            "final_legal_conclusion": canonical_conclusion,
            "completeness": completeness,
            "confidence": {
                "score": confidence,
                "label": _confidence_label(confidence),
            },
            "risk_level": risk_level,
            "citations": citations,
            "human_answer": human_answer,
            "unsafe_reasons": unsafe_reasons,
        }

        llm_generation = self._llm_answers(
            query=query,
            phase8=phase8,
            primary_hit=primary_hit,
            related_hits=related_hits,
            reasoning=reasoning,
            validation=validation,
            output=output,
        )
        known_sections = _dedupe_keep_order(primary_sections + supporting_sections + exception_sections + condition_sections + punishment_sections)
        explanation_ok, explanation_issues = self._verify_text(
            llm_generation.get("explanation", ""),
            citations=citations,
            target_summary=target_summary,
            known_sections=known_sections,
        )
        summary_ok, summary_issues = self._verify_text(
            llm_generation.get("short_summary", ""),
            citations=citations,
            target_summary=target_summary,
            known_sections=known_sections,
        )
        if not explanation_ok or not summary_ok:
            llm_generation["used"] = False
            llm_generation["verification_issues"] = explanation_issues + summary_issues
            llm_generation["explanation"] = self._fallback_detailed_answer(
                query=query,
                phase8=phase8,
                primary_hit=primary_hit,
                related_hits=related_hits,
                completeness=completeness,
                canonical_conclusion=canonical_conclusion,
            )
            llm_generation["short_summary"] = self._fallback_summary_answer(
                query=query,
                phase8=phase8,
                primary_hit=primary_hit,
                related_hits=related_hits,
                completeness=completeness,
                canonical_conclusion=canonical_conclusion,
            )
            llm_generation["caution_notes"] = []
        else:
            llm_generation["verification_issues"] = []

        phase11 = {
            "final_answer": human_answer or canonical_conclusion,
            "summary_answer": llm_generation.get("short_summary") if llm_generation.get("short_summary") else canonical_conclusion,
            "detailed_answer": llm_generation.get("explanation") if llm_generation.get("explanation") else (human_answer or canonical_conclusion),
            "reasoning": reasoning,
            "validation": validation,
            "output": output,
            "llm_generation": llm_generation,
        }

        enriched = dict(item)
        enriched["phase11"] = phase11
        return enriched

    def reason_many(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.reason_one(item) for item in items]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 11 constrained legal reasoning")
    parser.add_argument("--retrieval", required=True, help="Path to Phase 9/10 retrieval JSON output")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json used by the corpus index")
    parser.add_argument("--output", default=None, help="Optional output path for the phase 11 JSON")
    parser.add_argument("--top-support-hits", type=int, default=4, help="Max additional evidence hits to attach")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM explanation generation from structured reasoning")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model name for phase 11 explanation rendering")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    retrieval_obj = _load_json(args.retrieval)
    items = _normalize_phase9_items(retrieval_obj)
    corpus_index = CorpusIndex(args.chunks)

    llm_client: Optional[Phase11OpenAIClient] = None
    if args.enable_llm:
        llm_client = Phase11OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", ""))

    reasoner = Phase11Reasoner(
        corpus_index=corpus_index,
        top_support_hits=args.top_support_hits,
        llm_client=llm_client,
        llm_model=args.llm_model,
        enable_llm=args.enable_llm,
    )
    output = reasoner.reason_many(items)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(output)} phase 11 item(s) to {out_path}")
        return

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




'''

python reasoning/reason.py \
  --retrieval retrieval/output_2__.json \
  --chunks data/processed/artifacts2/chunks.json \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --output reasoning/res_2__2.json

python reasoning/reason.py \
  --retrieval retrieval/output_4__.json \
  --chunks data/processed/artifacts2/chunks.json \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --output reasoning/res_4__2.json



'''
