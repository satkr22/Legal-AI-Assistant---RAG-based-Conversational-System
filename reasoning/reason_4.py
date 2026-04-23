from __future__ import annotations

"""Phase 11 - Grounded answer synthesis and validation for legal RAG.

This version keeps Phase 11 constrained:
- it consumes Phase 9/10 evidence
- it does not traverse the graph
- it selects the strongest supported legal provision
- it reasons over structured evidence only
- the LLM, if enabled, only rewrites a grounded summary
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


BASE_CHUNK_TYPES = {"section", "subsection", "clause", "content"}
EXPLANATORY_CHUNK_TYPES = {"explanation", "illustration"}

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

# Structural, not word-based, priorities.
SECTION_TYPE_PRIORITY = {
    "rule": 5,
    "definition": 5,
    "provision": 5,
    "section": 4,
    "clause": 4,
    "subsection": 4,
    "exception": 3,
    "illustration": 1,
    "explanation": 2,
}

CHUNK_TYPE_PRIORITY = {
    "content": 6,
    "subsection": 5,
    "clause": 4,
    "section": 4,
    "explanation": 2,
    "illustration": 1,
}

# ---------------------------------------------------------------------------
# Structured legal reasoning constants (generic — not query/law specific)
# ---------------------------------------------------------------------------

# Relationship types between sections that share a base concept
_REL_CONDITIONAL_VARIANT = "conditional_variant"
_REL_EXCEPTION = "exception"
_REL_EXPLANATION = "explanation"
_REL_INDEPENDENT = "independent"

# Uncertainty labels
_UNC_LOW = "low"
_UNC_MEDIUM = "medium"
_UNC_HIGH = "high"

SECTION_REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|subsection|sub-section|clause)\s*"
    r"(\d+[a-z]?)(?:\s*\(\s*([0-9a-z]+)\s*\))?",
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
    raw = _normalize_space(raw)
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
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


def _risk_level(score: float, completeness: str, weak_alignment: bool) -> str:
    if completeness == "unsafe" or weak_alignment:
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
        for chunk in self.by_section_token.get(section_token, []):
            if chunk.section_title:
                return chunk.section_title
        return ""

    def matching_chunks(self, reference: str) -> List[CorpusChunk]:
        token = _canonical_reference_token(reference)
        if not token:
            return []
        base = token.split("(")[0]
        return _dedupe_keep_order(list(self.citation_token_to_chunks.get(token, [])) + list(self.by_section_token.get(base, [])))


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


# ---------------------------------------------------------------------------
# Structured Legal Reasoning Engine (Phase 11-SR)
# Generic — works over any evidence set, any legal domain, no hardcoded rules
# ---------------------------------------------------------------------------

@dataclass
class SectionAnalysis:
    """Result of reasoning over a single retrieved section."""
    section: str
    role: str                   # conditional_variant | exception | explanation | independent
    condition_summary: str
    match_score: float          # 0.0–1.0, condition-based (not retrieval score)
    concept_overlap: List[str]
    raw_retrieval_score: float
    chunk_ids: List[str]
    hit_count: int
    is_base: bool
    note: str = ""


@dataclass
class StructuredReasoningResult:
    base_concept: str
    sections_analyzed: List[SectionAnalysis]
    selected_section: Optional[str]
    selected_hit: Optional["EvidenceHit"]
    selection_reason: str
    alternative_sections: List[str]
    uncertainty: str
    uncertainty_reason: str
    overrides_score_selection: bool     # True when condition logic differs from top-score pick


class StructuredLegalReasoner:
    """
    Performs generic structured legal reasoning over retrieved evidence.

    Steps:
      1. Identify base legal concept
      2. Group sections by shared concept
      3. Detect relationship type between sections in each group
      4. Extract conditions abstractly from section text
      5. Match query semantics against conditions
      6. Select correct section by condition satisfaction (not raw score)
      7. Produce structured output
    """

    # ------------------------------------------------------------------ #
    # STEP 1 — BASE CONCEPT                                                #
    # ------------------------------------------------------------------ #

    def _identify_base_concept(
        self,
        hits: List["EvidenceHit"],
        query_concepts: List[str],
    ) -> str:
        """
        The base concept is the legal concept that appears most frequently
        across all retrieved hits and overlaps with query concepts.
        Falls back to the most frequent concept in hits alone.
        """
        counter: Counter = Counter()
        for hit in hits:
            for concept in hit.legal_concepts:
                counter[_safe_lower(concept)] += 1

        if not counter:
            return _join_brief(query_concepts[:2]) or "legal provision"

        query_lower = {_safe_lower(c) for c in query_concepts}

        # Prefer concepts that appear in both query and hits
        overlapping = [(c, n) for c, n in counter.most_common() if c in query_lower]
        if overlapping:
            return overlapping[0][0]

        return counter.most_common(1)[0][0]

    # ------------------------------------------------------------------ #
    # STEP 2 — GROUP SECTIONS BY CONCEPT                                  #
    # ------------------------------------------------------------------ #

    def _group_by_concept(
        self,
        hits: List["EvidenceHit"],
    ) -> Dict[str, List["EvidenceHit"]]:
        """
        Group hits by section token. Sections that share the majority of
        their legal_concepts with the most-common concept are in the
        'primary' group; others are in their own groups.
        Returns dict: section_token → list of hits.
        """
        grouped: Dict[str, List["EvidenceHit"]] = defaultdict(list)
        for hit in hits:
            key = hit.section_token or hit.chunk_id
            grouped[key].append(hit)
        return dict(grouped)

    # ------------------------------------------------------------------ #
    # STEP 3 — DETECT RELATIONSHIP TYPE                                   #
    # ------------------------------------------------------------------ #

    def _detect_relationship(
        self,
        chunk_type: str,
        section_type: str,
        text: str,
        title: str,
        all_section_titles: List[str],
        all_section_texts: List[str],
    ) -> str:
        """
        Infer relationship type from structural metadata and text signals.
        No hardcoded domain terms — uses structural patterns.
        """
        ct = _safe_lower(chunk_type)
        st = _safe_lower(section_type)
        t  = _safe_lower(text)
        tl = _safe_lower(title)

        # Chunk type signals
        if ct in {"explanation", "illustration"}:
            return _REL_EXPLANATION

        # Section type signals
        if st == "exception":
            return _REL_EXCEPTION

        # Text-level structural signals — negation/conditionality patterns
        # These are syntactic, not domain-specific
        negation_markers = [
            "notwithstanding", "except", "unless", "shall not apply",
            "does not apply", "shall not mitigate", "will not mitigate",
            "provided that", "nothing in this section",
        ]
        conditional_contrast_markers = [
            "otherwise than", "other than", "except when", "not on",
            "without", "in the absence of",
        ]
        positive_condition_markers = [
            " on ", " upon ", " where ", " when ", " if ",
            " with intent", " in attempting", " in consequence",
            " being a ", " as such ",
        ]

        has_negation = any(m in t for m in negation_markers)
        has_contrast = any(m in t for m in conditional_contrast_markers)
        has_positive_cond = any(m in t for m in positive_condition_markers)

        # If the section text has a negation/contrast pattern AND there are other
        # sections with similar base content → conditional_variant
        if has_contrast or has_negation:
            return _REL_CONDITIONAL_VARIANT

        # If it has a positive condition modifier → also a variant
        if has_positive_cond and ct in {"content", "subsection", "clause"}:
            return _REL_CONDITIONAL_VARIANT

        # Default: treat as independent rule
        return _REL_INDEPENDENT

    # ------------------------------------------------------------------ #
    # STEP 4 — EXTRACT CONDITIONS                                         #
    # ------------------------------------------------------------------ #

    def _extract_condition_summary(self, text: str, title: str) -> str:
        """
        Extract a concise, abstract condition summary from section text.
        Identifies the principal conditional clause (positive or negative).
        No domain-specific hardcoding.
        """
        t = _normalize_space(text)
        if not t:
            return _normalize_space(title) or "No condition text available."

        sentences = _split_sentences(t)
        if not sentences:
            return _normalize_space(title)

        # Prefer the sentence most likely to contain a condition
        condition_indicators = [
            "otherwise than", "other than", "except", "unless", "notwithstanding",
            "if ", "when ", "where ", "provided", "on ", "upon ",
            "with intent", "in attempting", "in consequence", "being a",
            "shall not", "will not", "does not",
        ]
        scored: List[Tuple[int, str]] = []
        for sent in sentences:
            sl = sent.lower()
            score_val = sum(1 for ind in condition_indicators if ind in sl)
            scored.append((score_val, sent))

        # Pick the highest-scoring sentence; fall back to first
        best = max(scored, key=lambda x: x[0])
        chosen = best[1] if best[0] > 0 else sentences[0]
        return _compact_text(chosen, max_sentences=1, max_chars=360)

    # ------------------------------------------------------------------ #
    # STEP 5 — MATCH QUERY AGAINST CONDITIONS                             #
    # ------------------------------------------------------------------ #

    def _condition_match_score(
        self,
        hit: "EvidenceHit",
        query_concepts: List[str],
        query_text: str,
    ) -> float:
        """
        Compute a condition-match score (0.0–1.0) for a hit against the query.

        Combines:
          (a) Concept overlap between hit.legal_concepts and query_concepts
          (b) Keyword overlap between hit.keywords and query tokens
          (c) A positive/negative condition polarity bonus:
              If the query contains affirmative signals that match the
              positive-condition branch of the hit, score goes up.
              If the hit's condition is the negation branch and the query
              implies the condition IS present, score goes down for that hit.
        """
        q_concepts = {_safe_lower(c) for c in query_concepts}
        h_concepts = {_safe_lower(c) for c in hit.legal_concepts}
        concept_union = q_concepts | h_concepts
        concept_intersect = q_concepts & h_concepts
        concept_jaccard = len(concept_intersect) / max(len(concept_union), 1)

        q_tokens = set(re.findall(r"\b\w{3,}\b", _safe_lower(query_text)))
        h_keywords = {_safe_lower(k) for k in hit.keywords}
        keyword_union = q_tokens | h_keywords
        keyword_intersect = q_tokens & h_keywords
        keyword_jaccard = len(keyword_intersect) / max(len(keyword_union), 1)

        # Base score: weighted combination
        base = concept_jaccard * 0.60 + keyword_jaccard * 0.40

        # Polarity adjustment: detect whether the query affirms the
        # operative condition that is present in this section's text.
        # e.g. if query says "I provoked them" and section requires provocation
        # to BE present → positive match boost.
        # If section requires provocation to be ABSENT ("otherwise than") and
        # query says provocation WAS present → penalty.
        hit_text_lower = _safe_lower(hit.text)
        query_lower = _safe_lower(query_text)

        # Negation pattern in section text means the section applies when
        # the condition is ABSENT from the fact pattern.
        negation_in_section = any(m in hit_text_lower for m in [
            "otherwise than", "other than", "except", "unless",
            "shall not", "will not", "does not apply",
        ])

        # Detect affirmative presence of the concept in the query
        # by checking whether query concepts overlap with the section's
        # distinguishing keywords (those NOT in every section)
        affirmative_in_query = bool(concept_intersect)  # at least one concept matches

        if negation_in_section and affirmative_in_query:
            # Section applies when condition is absent, but query implies
            # the condition is present → reduce score
            base -= 0.18
        elif not negation_in_section and affirmative_in_query:
            # Section applies when condition is present, and query has it → boost
            base += 0.12

        return round(_clamp(base, 0.0, 1.0), 4)

    # ------------------------------------------------------------------ #
    # STEP 6 — SELECT CORRECT SECTION                                     #
    # ------------------------------------------------------------------ #

    def _select_section(
        self,
        analyses: List[SectionAnalysis],
        hits_by_section: Dict[str, List["EvidenceHit"]],
        score_based_primary: Optional["EvidenceHit"],
    ) -> Tuple[Optional[SectionAnalysis], bool]:
        """
        Select the best section by condition match score, not raw retrieval score.
        Returns (selected_analysis, overrides_score_selection).

        Only compares among conditional_variant sections within the primary
        concept group. Falls back to score-based selection if no variants found.
        """
        variants = [a for a in analyses if a.role == _REL_CONDITIONAL_VARIANT]

        if not variants:
            # No conditional structure detected — use score-based selection
            best = max(analyses, key=lambda a: (a.raw_retrieval_score, a.hit_count), default=None)
            return best, False

        # Among variants, pick highest condition-match score
        best_variant = max(variants, key=lambda a: (a.match_score, a.raw_retrieval_score))

        # Check if this differs from what the score-based selection chose
        score_based_section = score_based_primary.section_token if score_based_primary else None
        overrides = (
            score_based_section is not None
            and best_variant.section != f"Section {score_based_section}"
        )

        return best_variant, overrides

    # ------------------------------------------------------------------ #
    # STEP 7 — UNCERTAINTY ASSESSMENT                                     #
    # ------------------------------------------------------------------ #

    def _assess_uncertainty(
        self,
        selected: Optional[SectionAnalysis],
        analyses: List[SectionAnalysis],
        overrides_score: bool,
    ) -> Tuple[str, str]:
        """
        Returns (uncertainty_label, reason_string).
        Generic — based on score gaps and structural factors.
        """
        if not selected:
            return _UNC_HIGH, "No section could be selected from retrieved evidence."

        variants = [a for a in analyses if a.role == _REL_CONDITIONAL_VARIANT]
        explanations = [a for a in analyses if a.role == _REL_EXPLANATION]

        if len(variants) >= 2:
            sorted_v = sorted(variants, key=lambda a: a.match_score, reverse=True)
            gap = sorted_v[0].match_score - sorted_v[1].match_score
            if gap < 0.10:
                reason = (
                    f"Two conditional variants are closely scored "
                    f"({sorted_v[0].section}: {sorted_v[0].match_score:.2f}, "
                    f"{sorted_v[1].section}: {sorted_v[1].match_score:.2f}). "
                    "A factual determination is required to resolve the correct branch."
                )
                return _UNC_MEDIUM, reason

        if explanations:
            exp_texts = [e.condition_summary for e in explanations if "question of fact" in e.condition_summary.lower()]
            if exp_texts:
                return _UNC_MEDIUM, (
                    "An explanation clause in the retrieved evidence states that a key "
                    "condition is a question of fact, not a pure legal determination."
                )

        if overrides_score:
            return _UNC_MEDIUM, (
                "Condition-based reasoning selected a different section than the "
                "highest-scored retrieval hit. Medium uncertainty applies until "
                "factual conditions are confirmed."
            )

        if selected.match_score >= 0.65:
            return _UNC_LOW, "Selected section has strong condition match and no closely competing variants."

        return _UNC_MEDIUM, "Condition match score is moderate; factual confirmation recommended."

    # ------------------------------------------------------------------ #
    # MAIN ENTRY POINT                                                     #
    # ------------------------------------------------------------------ #

    def reason(
        self,
        hits: List["EvidenceHit"],
        query_text: str,
        query_concepts: List[str],
        score_based_primary: Optional["EvidenceHit"],
    ) -> StructuredReasoningResult:
        """
        Full 7-step structured reasoning pipeline.
        Returns StructuredReasoningResult.
        """

        # --- STEP 1 ---
        base_concept = self._identify_base_concept(hits, query_concepts)

        # --- STEP 2 ---
        hits_by_section = self._group_by_concept(hits)

        # Collect all titles and texts for relationship detection context
        all_titles = [
            (h.corpus_chunk.section_title if h.corpus_chunk else "") or h.citation
            for h in hits
        ]
        all_texts = [h.text for h in hits]

        # --- STEPS 3–5: analyse each section group ---
        analyses: List[SectionAnalysis] = []

        for section_token, section_hits in hits_by_section.items():
            best_hit = max(
                section_hits,
                key=lambda h: (
                    1 if h.is_base else 0,
                    h.concept_coverage_ratio,
                    h.score,
                ),
            )

            section_label = f"Section {best_hit.section_token}" if best_hit.section_token else best_hit.citation or section_token

            chunk_type = best_hit.chunk_type
            section_type = _safe_lower(best_hit.corpus_chunk.section_type if best_hit.corpus_chunk else "")
            text = best_hit.text
            title = (best_hit.corpus_chunk.section_title if best_hit.corpus_chunk else "") or ""

            # STEP 3
            role = self._detect_relationship(
                chunk_type=chunk_type,
                section_type=section_type,
                text=text,
                title=title,
                all_section_titles=all_titles,
                all_section_texts=all_texts,
            )

            # STEP 4
            condition_summary = self._extract_condition_summary(text, title)

            # STEP 5
            match_score = self._condition_match_score(best_hit, query_concepts, query_text)

            # Concept overlap for output
            q_concepts_lower = {_safe_lower(c) for c in query_concepts}
            h_concepts_lower = {_safe_lower(c) for c in best_hit.legal_concepts}
            overlap = sorted(q_concepts_lower & h_concepts_lower)

            raw_score = max(h.score for h in section_hits)
            is_base_section = any(h.is_base for h in section_hits)

            analyses.append(SectionAnalysis(
                section=section_label,
                role=role,
                condition_summary=condition_summary,
                match_score=match_score,
                concept_overlap=overlap,
                raw_retrieval_score=round(raw_score, 6),
                chunk_ids=[h.chunk_id for h in section_hits],
                hit_count=len(section_hits),
                is_base=is_base_section,
            ))

        # --- STEP 6 ---
        selected_analysis, overrides = self._select_section(
            analyses, hits_by_section, score_based_primary
        )

        # Resolve selected hit from the selected section
        selected_hit: Optional["EvidenceHit"] = None
        if selected_analysis:
            candidates_for_section = hits_by_section.get(
                # strip "Section " prefix to match section_token key
                (selected_analysis.section.replace("Section ", "").strip()
                 if selected_analysis.section.startswith("Section ")
                 else selected_analysis.section),
                []
            )
            if candidates_for_section:
                selected_hit = max(
                    candidates_for_section,
                    key=lambda h: (1 if h.is_base else 0, h.concept_coverage_ratio, h.score),
                )

        # Build selection reason
        if selected_analysis:
            reason_parts = [
                f"Selected {selected_analysis.section} by condition-match score "
                f"({selected_analysis.match_score:.2f}) over retrieval score "
                f"({selected_analysis.raw_retrieval_score:.4f}).",
                f"Role: {selected_analysis.role}.",
                f"Condition: {selected_analysis.condition_summary}",
            ]
            if selected_analysis.concept_overlap:
                reason_parts.append(
                    f"Concept overlap with query: {', '.join(selected_analysis.concept_overlap)}."
                )
            if overrides:
                score_section = (
                    f"Section {score_based_primary.section_token}"
                    if score_based_primary and score_based_primary.section_token
                    else "the highest-scored hit"
                )
                reason_parts.append(
                    f"This overrides the score-based selection of {score_section}."
                )
            selection_reason = " ".join(reason_parts)
        else:
            selection_reason = "No section could be selected from the retrieved evidence."

        # Alternatives: all conditional_variant sections that were NOT selected
        alternatives: List[str] = []
        for a in sorted(analyses, key=lambda x: x.match_score, reverse=True):
            if selected_analysis and a.section == selected_analysis.section:
                continue
            if a.role in {_REL_CONDITIONAL_VARIANT, _REL_EXCEPTION}:
                alternatives.append(
                    f"{a.section} (role={a.role}, condition_match={a.match_score:.2f})"
                )
        # Also include explanations that might affect the outcome
        for a in analyses:
            if a.role == _REL_EXPLANATION:
                alternatives.append(
                    f"{a.section} (role=explanation — may qualify or override selected section)"
                )

        # STEP 7 — uncertainty
        uncertainty, uncertainty_reason = self._assess_uncertainty(
            selected_analysis, analyses, overrides
        )

        return StructuredReasoningResult(
            base_concept=base_concept,
            sections_analyzed=analyses,
            selected_section=selected_analysis.section if selected_analysis else None,
            selected_hit=selected_hit,
            selection_reason=selection_reason,
            alternative_sections=_dedupe_keep_order(alternatives),
            uncertainty=uncertainty,
            uncertainty_reason=uncertainty_reason,
            overrides_score_selection=overrides,
        )


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
        self._sr = StructuredLegalReasoner()

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

    def _section_candidates(
        self,
        hits: List[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_family: str,
    ) -> List[Tuple[str, List[EvidenceHit], float]]:
        grouped: Dict[str, List[EvidenceHit]] = defaultdict(list)
        for hit in hits:
            token = hit.section_token or hit.chunk_id
            grouped[token].append(hit)

        target_tokens = {
            _canonical_reference_token(target).split("(")[0]
            for target in target_summary.present_in_corpus
            if _canonical_reference_token(target)
        }

        candidates: List[Tuple[str, List[EvidenceHit], float]] = []
        for token, section_hits in grouped.items():
            section_hits_sorted = sorted(section_hits, key=lambda h: (-h.score, -h.concept_coverage_ratio, -h.concept_match_count, h.rank))
            best = section_hits_sorted[0]
            base_hits = sum(1 for h in section_hits if h.is_base)
            explanatory_hits = sum(1 for h in section_hits if h.is_explanatory)
            avg_cov = sum(h.concept_coverage_ratio for h in section_hits) / max(len(section_hits), 1)
            avg_match = sum(h.concept_match_count for h in section_hits) / max(len(section_hits), 1)

            score = best.score * 0.55 + avg_cov * 0.20 + min(avg_match / 5.0, 0.15)
            score += min(base_hits * 0.03, 0.12)
            score += min(explanatory_hits * 0.01, 0.03)

            if token in target_tokens:
                score += 0.30

            if intent_family == "lookup":
                score += 0.07 if best.is_base else -0.03
            elif intent_family == "application":
                score += 0.09 if best.is_base else 0.0
            elif intent_family == "explanation":
                score += 0.03 if explanatory_hits else 0.0

            sec_type = _safe_lower(best.corpus_chunk.section_type if best.corpus_chunk else "")
            score += SECTION_TYPE_PRIORITY.get(sec_type, 0) * 0.01

            candidates.append((token, section_hits_sorted, round(score, 6)))

        candidates.sort(key=lambda item: (item[2], max(h.score for h in item[1]), len(item[1])), reverse=True)
        return candidates

    def _preferred_hit(
        self,
        hits: List[EvidenceHit],
        intent_family: str,
        target_summary: RequestedTargetSummary,
    ) -> Optional[EvidenceHit]:
        if not hits:
            return None

        candidates = self._section_candidates(hits, target_summary, intent_family)
        if not candidates:
            return max(hits, key=lambda h: (h.score, h.concept_coverage_ratio, h.concept_match_count))

        best_section_token, best_section_hits, _ = candidates[0]
        # Choose the strongest chunk within the best-scoring section.
        def within_section_key(hit: EvidenceHit) -> Tuple[int, int, float, float, int]:
            sec_type = _safe_lower(hit.corpus_chunk.section_type if hit.corpus_chunk else "")
            chunk_priority = CHUNK_TYPE_PRIORITY.get(hit.chunk_type, 0)
            type_priority = SECTION_TYPE_PRIORITY.get(sec_type, 0)
            return (
                1 if hit.is_base else 0,
                type_priority,
                chunk_priority,
                hit.concept_coverage_ratio,
                hit.score,
            )

        return max(best_section_hits, key=within_section_key)

    def _related_hits(
        self,
        primary_hit: Optional[EvidenceHit],
        all_hits: List[EvidenceHit],
        intent_family: str,
        top_k: Optional[int] = None,
    ) -> List[EvidenceHit]:
        if not primary_hit:
            return []
        top_k = self.top_support_hits if top_k is None else max(1, int(top_k))
        same_section = [hit for hit in all_hits if hit.section_token == primary_hit.section_token and hit.chunk_id != primary_hit.chunk_id]
        same_section = sorted(same_section, key=lambda h: (-h.score, -h.concept_coverage_ratio, -h.concept_match_count, h.rank))

        # Additional evidence from other sections only when it is highly relevant.
        other_hits = [hit for hit in all_hits if hit.section_token != primary_hit.section_token]
        other_hits = sorted(other_hits, key=lambda h: (-h.concept_coverage_ratio, -h.concept_match_count, -h.score, h.rank))

        ordered: List[EvidenceHit] = []
        seen = set()

        for hit in same_section + other_hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            ordered.append(hit)

        return ordered[:top_k]

    def _section_distribution(self, hits: List[EvidenceHit]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[EvidenceHit]] = defaultdict(list)
        for hit in hits:
            token = hit.section_token or hit.chunk_id
            grouped[token].append(hit)

        ranked = sorted(
            grouped.items(),
            key=lambda item: (max((h.score for h in item[1]), default=0.0), len(item[1])),
            reverse=True,
        )

        out: List[Dict[str, Any]] = []
        for section_token, section_hits in ranked[:5]:
            title = self.corpus_index.section_title(section_token)
            out.append(
                {
                    "section": f"Section {section_token}" if section_token else None,
                    "section_title": title,
                    "hit_count": len(section_hits),
                    "best_score": round(max((hit.score for hit in section_hits), default=0.0), 6),
                    "chunk_types": Counter(hit.chunk_type for hit in section_hits),
                    "concept_coverage": round(max((hit.concept_coverage_ratio for hit in section_hits), default=0.0), 4),
                }
            )
        return out

    def _intent_alignment(self, intent_family: str, hits: List[EvidenceHit], primary_hit: Optional[EvidenceHit]) -> Tuple[str, List[str]]:
        if not hits:
            return "weak", ["No retrieved evidence is available for Phase 11 synthesis."]

        notes: List[str] = []
        base_count = sum(1 for hit in hits if hit.is_base)
        explanatory_count = sum(1 for hit in hits if hit.is_explanatory)

        if intent_family == "lookup":
            if primary_hit and primary_hit.is_base:
                notes.append("Lookup intent is anchored by a direct base legal chunk.")
                return "strong", notes
            if base_count:
                notes.append("Lookup intent has base support but the primary hit is not the cleanest direct rule.")
                return "moderate", notes
            notes.append("Lookup intent is mostly supported by explanatory material.")
            return "weak", notes

        if intent_family == "application":
            if primary_hit and primary_hit.is_base and base_count >= 1:
                notes.append("Application intent has a direct rule chunk to anchor the answer.")
                return "strong", notes
            if base_count >= 1:
                notes.append("Application intent has some direct legal support, but the match is not ideal.")
                return "moderate", notes
            notes.append("Application intent lacks a direct rule anchor.")
            return "weak", notes

        if explanatory_count >= 1 and base_count >= 1:
            notes.append("Explanatory intent has both rule text and supporting material.")
            return "strong", notes
        if base_count >= 1:
            notes.append("Explanatory intent has rule support but limited companion explanation.")
            return "moderate", notes

        notes.append("Retrieved evidence does not align strongly with the requested intent.")
        return "weak", notes

    def _confidence_score(
        self,
        hits: List[EvidenceHit],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_alignment: str,
        intent_family: str,
    ) -> float:
        if not hits or not primary_hit:
            return 0.12

        score = 0.25
        if target_summary.support_status == "exact_section_supported":
            score += 0.28
        elif target_summary.support_status == "section_exists_but_not_retrieved_well":
            score += 0.08
        elif target_summary.support_status == "section_not_in_corpus":
            score -= 0.22

        if primary_hit.is_base:
            score += 0.18
        elif primary_hit.is_explanatory:
            score += 0.05

        score += min(primary_hit.concept_coverage_ratio, 0.20)
        score += min(primary_hit.score * 0.18, 0.18)
        score += min(len(related_hits) * 0.03, 0.12)

        if intent_alignment == "strong":
            score += 0.12
        elif intent_alignment == "moderate":
            score += 0.04
        else:
            score -= 0.10

        if target_summary.missing_from_corpus:
            score -= 0.12

        section_candidates = self._section_candidates(hits, target_summary, intent_family)
        if len(section_candidates) >= 2:
            gap = section_candidates[0][2] - section_candidates[1][2]
            if gap < 0.08:
                score -= 0.06

        return round(_clamp(score, 0.05, 0.98), 4)

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

    def _completion_state(
        self,
        target_summary: RequestedTargetSummary,
        primary_hit: Optional[EvidenceHit],
        intent_alignment: str,
        hits: List[EvidenceHit],
    ) -> str:
        if not primary_hit:
            return "unsafe"
        if target_summary.support_status == "section_not_in_corpus":
            return "unsafe"
        if target_summary.support_status == "section_exists_but_not_retrieved_well":
            return "partial"

        if intent_alignment == "weak":
            return "partial" if hits else "unsafe"

        # If the evidence is sparse, stay conservative.
        if len(hits) == 1 and primary_hit.concept_coverage_ratio < 0.35:
            return "partial"
        return "complete"

    def _warning_messages(
        self,
        hits: List[EvidenceHit],
        primary_hit: Optional[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_alignment: str,
        intent_family: str,
    ) -> List[str]:
        warnings: List[str] = []

        if not hits:
            warnings.append("Phase 9 did not return any evidence for this query.")
            return warnings

        if target_summary.missing_from_corpus:
            warnings.append(f"{_join_brief(target_summary.missing_from_corpus)} is not present in the indexed corpus.")

        if target_summary.support_status == "section_exists_but_not_retrieved_well":
            warnings.append("The requested section exists in the corpus, but the current retrieval set does not ground it strongly.")

        if intent_alignment == "weak":
            warnings.append("Evidence-to-intent alignment is weak, so the answer is intentionally cautious.")

        # Generic caution for closely competing sections; do not treat this as a hard conflict.
        candidates = self._section_candidates(hits, target_summary, intent_family)
        if len(candidates) >= 2:
            gap = candidates[0][2] - candidates[1][2]
            if gap < 0.08:
                warnings.append("Several sections are close in relevance; the answer uses the strongest supported section.")

        return warnings

    def _core_rule_text(self, hit: Optional[EvidenceHit]) -> str:
        if not hit:
            return ""
        text = hit.corpus_chunk.derived_context if hit.corpus_chunk and hit.corpus_chunk.derived_context else (hit.derived_context or hit.text)
        return _compact_text(text, max_sentences=2, max_chars=420)

    def _supporting_detail(self, hits: Sequence[EvidenceHit], primary_chunk_id: Optional[str]) -> str:
        for hit in hits:
            if hit.chunk_id == primary_chunk_id:
                continue
            detail = _compact_text(hit.text, max_sentences=1, max_chars=260)
            if detail:
                return detail
        return ""

    def _build_answer(
        self,
        query: str,
        phase8: Dict[str, Any],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_alignment: str,
        completeness: str,
    ) -> str:
        intent = dict(phase8.get("intent", {}) or {})
        primary_intent = _safe_lower(intent.get("primary", "explanation"))
        intent_family = self._intent_family(phase8)

        if target_summary.support_status == "section_not_in_corpus":
            if primary_hit and primary_hit.section_token:
                section_label = f"Section {primary_hit.section_token}"
                title = primary_hit.corpus_chunk.section_title if primary_hit.corpus_chunk else ""
                title_part = f" ({title})" if title else ""
                return (
                    f"{_join_brief(target_summary.missing_from_corpus)} is not present in the indexed corpus. "
                    f"The closest retrieved material for your intent points to {section_label}{title_part}, "
                    "but Phase 11 does not substitute that for the missing requested section."
                )
            return (
                f"{_join_brief(target_summary.missing_from_corpus)} is not present in the indexed corpus, "
                "so Phase 11 cannot produce a grounded section answer."
            )

        if target_summary.support_status == "section_exists_but_not_retrieved_well":
            return (
                f"{_join_brief(target_summary.present_in_corpus)} exists in the indexed corpus, "
                "but the current retrieval results do not support a reliable grounded answer yet."
            )

        if not primary_hit:
            return "The current retrieval results do not provide enough evidence to compose a grounded answer."

        section_label = f"Section {primary_hit.section_token}" if primary_hit.section_token else primary_hit.citation or "the retrieved provision"
        act = primary_hit.act or self.corpus_index.doc_meta.get("act") or "the indexed corpus"
        section_title = primary_hit.corpus_chunk.section_title if primary_hit.corpus_chunk and primary_hit.corpus_chunk.section_title else ""
        title_part = f" ({section_title})" if section_title else ""
        rule_text = self._core_rule_text(primary_hit)
        support_detail = self._supporting_detail(related_hits, primary_hit.chunk_id)

        if completeness == "unsafe":
            return (
                f"The evidence points toward {section_label}{title_part} in {act}, but the legal conclusion remains unsafe because "
                f"{_join_brief(target_summary.missing_from_corpus or ['the evidence is incomplete'])}."
            )

        if intent_family == "lookup":
            answer = f"{section_label}{title_part} in {act} is the best-supported match for your query. {rule_text}"
            if support_detail:
                answer += f" Supporting detail from nearby evidence: {support_detail}"
            return answer

        if primary_intent == "legal penalty":
            answer = f"Based on the retrieved corpus, {section_label}{title_part} is the best-supported provision for this penalty-focused query. {rule_text}"
            if intent_alignment != "strong":
                answer += " The answer remains cautious because the retrieved support is not perfectly aligned."
            if support_detail:
                answer += f" Nearby supporting evidence: {support_detail}"
            return answer

        if primary_intent == "case application":
            answer = f"Based on the retrieved corpus, the strongest directly supported provision is {section_label}{title_part}. {rule_text}"
            if support_detail:
                answer += f" Nearby supporting evidence: {support_detail}"
            if completeness == "partial":
                answer += " This answer is partial because the evidence set is limited or only loosely aligned."
            else:
                answer += " This is a grounded indication based on the retrieved evidence; exact applicability still depends on the facts."
            return answer

        if intent_family == "explanation":
            answer = f"The retrieved evidence most strongly supports {section_label}{title_part}. {rule_text}"
            if support_detail:
                answer += f" Supporting detail: {support_detail}"
            return answer

        answer = f"The best-supported answer from the indexed corpus points to {section_label}{title_part}. {rule_text}"
        if support_detail:
            answer += f" Supporting detail: {support_detail}"
        return answer

    def _build_answer_from_reasoning(
        self,
        query: str,
        phase8: Dict[str, Any],
        sr: StructuredReasoningResult,
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_alignment: str,
        completeness: str,
    ) -> str:
        """
        Build the deterministic answer using structured reasoning output.
        Uses sr.selected_hit when reasoning overrides the score-based primary;
        otherwise falls through to the standard _build_answer path.
        """
        # If structured reasoning produced no useful selection, fall back
        if not sr.selected_section:
            return self._build_answer(
                query, phase8, primary_hit, related_hits,
                target_summary, intent_alignment, completeness,
            )

        # Determine which hit to anchor the answer on
        reasoning_hit = sr.selected_hit if sr.selected_hit else primary_hit
        if not reasoning_hit:
            return self._build_answer(
                query, phase8, primary_hit, related_hits,
                target_summary, intent_alignment, completeness,
            )

        # Build answer components
        section_label = (
            f"Section {reasoning_hit.section_token}"
            if reasoning_hit.section_token
            else reasoning_hit.citation or sr.selected_section
        )
        act = reasoning_hit.act or self.corpus_index.doc_meta.get("act") or "the indexed corpus"
        section_title = (
            reasoning_hit.corpus_chunk.section_title
            if reasoning_hit.corpus_chunk and reasoning_hit.corpus_chunk.section_title
            else ""
        )
        title_part = f" ({section_title})" if section_title else ""
        rule_text = self._core_rule_text(reasoning_hit)

        # Gather supporting detail from related hits under the same section
        sr_related = [
            h for h in related_hits
            if h.chunk_id != reasoning_hit.chunk_id
            and h.section_token == reasoning_hit.section_token
        ]
        # If no same-section support, fall back to any related hit
        if not sr_related:
            sr_related = [h for h in related_hits if h.chunk_id != reasoning_hit.chunk_id]
        support_detail = self._supporting_detail(sr_related, reasoning_hit.chunk_id)

        intent = dict(phase8.get("intent", {}) or {})
        intent_family = self._intent_family(phase8)
        primary_intent = _safe_lower(intent.get("primary", "explanation"))

        # Uncertainty qualifier
        unc_qualifier = ""
        if sr.uncertainty == _UNC_HIGH:
            unc_qualifier = " The evidence base is insufficient for a definitive legal conclusion."
        elif sr.uncertainty == _UNC_MEDIUM:
            unc_qualifier = f" {sr.uncertainty_reason}"

        # Override notice
        override_notice = ""
        if sr.overrides_score_selection:
            override_notice = (
                f" (Condition-based reasoning selected {sr.selected_section} "
                "over the highest-scored retrieval hit.)"
            )

        if completeness == "unsafe":
            return (
                f"The evidence points toward {section_label}{title_part} in {act}, "
                f"but the legal conclusion is not safe to state definitively.{unc_qualifier}"
            )

        if intent_family == "lookup":
            answer = f"{section_label}{title_part} in {act} is the best-supported match. {rule_text}"
        elif primary_intent in {"legal penalty", "punishment"}:
            answer = (
                f"Based on retrieved evidence, {section_label}{title_part} is the "
                f"applicable provision for this penalty-related query. {rule_text}"
            )
            if intent_alignment != "strong":
                answer += " The answer remains cautious as retrieved support is not perfectly aligned."
        elif primary_intent == "case application":
            answer = (
                f"Based on retrieved evidence, the strongest condition-matched provision "
                f"is {section_label}{title_part}. {rule_text}"
            )
            if completeness == "partial":
                answer += " This answer is partial; the evidence is limited or loosely aligned."
            else:
                answer += (
                    " This is a grounded indication from retrieved evidence; "
                    "exact applicability still depends on the specific facts."
                )
        else:
            # consequence, explanation, reasoning, hypothetical, comparison, etc.
            answer = (
                f"Applying structured reasoning over the retrieved evidence, "
                f"{section_label}{title_part} is the condition-matched applicable provision. "
                f"{rule_text}"
            )

        if support_detail:
            answer += f" Supporting detail: {support_detail}"
        if unc_qualifier and primary_intent not in {"lookup"}:
            answer += unc_qualifier
        if override_notice:
            answer += override_notice

        return answer

    def _citations(self, primary_hit: Optional[EvidenceHit], related_hits: List[EvidenceHit]) -> List[str]:
        citations: List[str] = []
        for hit in [primary_hit] + list(related_hits):
            if not hit:
                continue
            citation = hit.corpus_chunk.citation_text if hit.corpus_chunk and hit.corpus_chunk.citation_text else hit.citation
            if citation:
                citations.append(citation)
        return _dedupe_keep_order(citations)

    def _llm_system_prompt(self) -> str:
        return (
            "You are Phase 11 of a legal RAG pipeline. "
            "You must answer only from the validated evidence provided. "
            "You will be given a 'structured_reasoning' block that contains the result of "
            "multi-step condition-based legal reasoning over the retrieved sections. "
            "You MUST use the 'selected_section' from structured_reasoning as your primary anchor "
            "unless the uncertainty is 'high'. "
            "If structured_reasoning overrides the score-based selection, respect that override. "
            "Do not invent sections, punishments, facts, or legal conclusions outside the evidence. "
            "If evidence is limited, be explicit about the limitation. "
            "Return JSON with keys: summary_answer, detailed_answer, reasoning_trace, caution_notes. "
            "summary_answer must be concise. detailed_answer may be fuller but still grounded. "
            "reasoning_trace must be a short list of evidence-based bullets that follow the "
            "structured_reasoning steps: base concept → condition match → section selected → why. "
            "caution_notes must be a short list."
        )

    def _llm_user_prompt(
        self,
        query: str,
        phase8: Dict[str, Any],
        target_summary: RequestedTargetSummary,
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        deterministic_answer: str,
        warnings: List[str],
        confidence: float,
        sr: Optional[StructuredReasoningResult] = None,
    ) -> str:
        intent = dict(phase8.get("intent", {}) or {})
        evidence_hits = [hit for hit in [primary_hit] + list(related_hits) if hit]
        evidence_rows: List[Dict[str, Any]] = []
        for hit in evidence_hits:
            evidence_rows.append(
                {
                    "chunk_id": hit.chunk_id,
                    "chunk_type": hit.chunk_type,
                    "section": f"Section {hit.section_token}" if hit.section_token else None,
                    "section_title": hit.corpus_chunk.section_title if hit.corpus_chunk else "",
                    "citation": hit.corpus_chunk.citation_text if hit.corpus_chunk and hit.corpus_chunk.citation_text else hit.citation,
                    "text": hit.text,
                    "derived_context": hit.derived_context,
                    "score": round(hit.score, 6),
                    "concept_coverage_ratio": round(hit.concept_coverage_ratio, 4),
                    "reasons": hit.reasons,
                }
            )

        payload = {
            "query": query,
            "intent": intent,
            "support_status": target_summary.support_status,
            "requested_sections": target_summary.requested,
            "present_in_corpus": target_summary.present_in_corpus,
            "retrieved_exact_matches": target_summary.retrieved_matches,
            "missing_from_corpus": target_summary.missing_from_corpus,
            "confidence_score": confidence,
            "deterministic_fallback_answer": deterministic_answer,
            "warnings": warnings,
            "evidence": evidence_rows,
        }
        if sr is not None:
            payload["structured_reasoning"] = {
                "base_concept": sr.base_concept,
                "selected_section": sr.selected_section,
                "selection_reason": sr.selection_reason,
                "overrides_score_selection": sr.overrides_score_selection,
                "uncertainty": sr.uncertainty,
                "uncertainty_reason": sr.uncertainty_reason,
                "alternative_sections": sr.alternative_sections,
                "sections_analyzed": [
                    {
                        "section": a.section,
                        "role": a.role,
                        "condition_summary": a.condition_summary,
                        "condition_match_score": a.match_score,
                        "concept_overlap": a.concept_overlap,
                        "raw_retrieval_score": a.raw_retrieval_score,
                    }
                    for a in sr.sections_analyzed
                ],
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _llm_answers(
        self,
        query: str,
        phase8: Dict[str, Any],
        target_summary: RequestedTargetSummary,
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        deterministic_answer: str,
        warnings: List[str],
        confidence: float,
        sr: Optional[StructuredReasoningResult] = None,
    ) -> Dict[str, Any]:
        if not self.enable_llm or self.llm_client is None:
            return {
                "enabled": False,
                "used": False,
                "model": None,
                "summary_answer": deterministic_answer,
                "detailed_answer": deterministic_answer,
                "reasoning_trace": [],
                "caution_notes": [],
                "error": None,
            }

        try:
            response = self.llm_client.generate_json(
                model=self.llm_model,
                system_prompt=self._llm_system_prompt(),
                user_prompt=self._llm_user_prompt(
                    query=query,
                    phase8=phase8,
                    target_summary=target_summary,
                    primary_hit=primary_hit,
                    related_hits=related_hits,
                    deterministic_answer=deterministic_answer,
                    warnings=warnings,
                    confidence=confidence,
                    sr=sr,
                ),
            )
        except Exception as exc:
            return {
                "enabled": True,
                "used": False,
                "model": self.llm_model,
                "summary_answer": deterministic_answer,
                "detailed_answer": deterministic_answer,
                "reasoning_trace": [],
                "caution_notes": [],
                "error": str(exc),
            }

        summary_answer = _normalize_space(response.get("summary_answer")) or deterministic_answer
        detailed_answer = _normalize_space(response.get("detailed_answer")) or deterministic_answer
        reasoning_trace = _coerce_str_list(response.get("reasoning_trace"))
        caution_notes = _coerce_str_list(response.get("caution_notes"))

        return {
            "enabled": True,
            "used": True,
            "model": self.llm_model,
            "summary_answer": summary_answer,
            "detailed_answer": detailed_answer,
            "reasoning_trace": reasoning_trace,
            "caution_notes": caution_notes,
            "error": None,
        }

    def _verify_text(
        self,
        text: str,
        citations: List[str],
        target_summary: RequestedTargetSummary,
        allowed_sections: List[str],
    ) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not _normalize_space(text):
            issues.append("Empty text.")
            return False, issues

        mentioned = _extract_section_references(text)
        allowed = {_canonical_reference_token(section) for section in allowed_sections if _canonical_reference_token(section)}
        allowed.update({_canonical_reference_token(citation) for citation in citations if _canonical_reference_token(citation)})
        allowed.update({_canonical_reference_token(target) for target in target_summary.present_in_corpus if _canonical_reference_token(target)})

        for mention in mentioned:
            token = _canonical_reference_token(mention)
            if token and token not in allowed and token.split("(")[0] not in {value.split("(")[0] for value in allowed if value}:
                issues.append(f"Unsupported section reference in text: {mention}.")
        return len(issues) == 0, issues

    def _verify_answer(
        self,
        answer: str,
        primary_hit: Optional[EvidenceHit],
        citations: List[str],
        target_summary: RequestedTargetSummary,
    ) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not _normalize_space(answer):
            issues.append("Empty final answer.")

        if primary_hit and primary_hit.section_token:
            mentioned = _extract_section_references(answer)
            normalized_mentioned = {_canonical_reference_token(item) for item in mentioned if _canonical_reference_token(item)}
            if target_summary.present_in_corpus:
                allowed = {
                    _canonical_reference_token(target).split("(")[0]
                    for target in target_summary.present_in_corpus
                    if _canonical_reference_token(target)
                }
                if normalized_mentioned and not any(token and token.split("(")[0] in allowed for token in normalized_mentioned):
                    issues.append("Final answer mentions a section that does not match the requested supported target.")

        if primary_hit and citations:
            primary_citation = primary_hit.corpus_chunk.citation_text if primary_hit.corpus_chunk else primary_hit.citation
            if primary_citation and primary_citation not in citations:
                issues.append("Primary supporting citation is missing from the citation list.")

        return len(issues) == 0, issues

    def reason_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        phase8 = dict(item.get("phase8", {}) or {})
        query = _normalize_space(item.get("query") or phase8.get("query"))
        retrieval = dict(item.get("retrieval", {}) or {})
        hits = self._load_hits(retrieval)

        requested_targets = self._extract_requested_targets(query, phase8)
        target_summary = self._summarize_targets(requested_targets, hits)
        intent_family = self._intent_family(phase8)

        primary_hit = self._preferred_hit(hits, intent_family, target_summary)
        related_hits = self._related_hits(primary_hit, hits, intent_family)

        intent_alignment, intent_notes = self._intent_alignment(intent_family, hits, primary_hit)
        confidence = self._confidence_score(hits, primary_hit, related_hits, target_summary, intent_alignment, intent_family)
        support_status = self._support_status_for_intent(hits, primary_hit, target_summary)
        completeness = self._completion_state(target_summary, primary_hit, intent_alignment, hits)
        warnings = self._warning_messages(hits, primary_hit, target_summary, intent_alignment, intent_family)

        # ----------------------------------------------------------------
        # Structured Legal Reasoning (condition-based selection)
        # ----------------------------------------------------------------
        query_concepts = list(
            dict.fromkeys(
                _coerce_str_list(phase8.get("concepts"))
                + [c for sq in (phase8.get("decomposition", {}) or {}).get("sub_queries", []) or []
                   for c in _coerce_str_list((sq or {}).get("concepts"))]
            )
        )
        sr: StructuredReasoningResult = self._sr.reason(
            hits=hits,
            query_text=query,
            query_concepts=query_concepts,
            score_based_primary=primary_hit,
        )

        # Use the SR-selected hit as the effective primary when it overrides
        effective_primary = sr.selected_hit if (sr.overrides_score_selection and sr.selected_hit) else primary_hit

        # Recompute related hits against the effective primary
        effective_related = self._related_hits(effective_primary, hits, intent_family)

        # Build answer using structured reasoning result
        answer = self._build_answer_from_reasoning(
            query=query,
            phase8=phase8,
            sr=sr,
            primary_hit=effective_primary,
            related_hits=effective_related,
            target_summary=target_summary,
            intent_alignment=intent_alignment,
            completeness=completeness,
        )

        citations = self._citations(effective_primary, effective_related)

        # Add a warning when SR overrides the score-based selection
        if sr.overrides_score_selection:
            warnings.append(
                f"Structured reasoning selected {sr.selected_section} over the "
                f"highest-scored retrieval hit. Reason: {sr.uncertainty_reason}"
            )
        if sr.uncertainty == _UNC_MEDIUM:
            warnings.append(f"Reasoning uncertainty: medium. {sr.uncertainty_reason}")
        elif sr.uncertainty == _UNC_HIGH:
            warnings.append(f"Reasoning uncertainty: high. {sr.uncertainty_reason}")

        verified, verification_issues = self._verify_answer(answer, effective_primary, citations, target_summary)
        if not verified:
            warnings.extend(verification_issues)
            if completeness != "unsafe":
                answer = "The retrieved evidence is not strong enough to produce a more specific grounded answer."

        llm_answers = self._llm_answers(
            query=query,
            phase8=phase8,
            target_summary=target_summary,
            primary_hit=effective_primary,
            related_hits=effective_related,
            deterministic_answer=answer,
            warnings=warnings,
            confidence=confidence,
            sr=sr,
        )

        allowed_sections = _dedupe_keep_order(
            [f"Section {hit.section_token}" for hit in [effective_primary] + effective_related if hit and hit.section_token]
        )
        llm_verified, llm_verification_issues = self._verify_text(
            llm_answers.get("summary_answer", ""),
            citations=citations,
            target_summary=target_summary,
            allowed_sections=allowed_sections,
        )
        if llm_verified:
            llm_verified_2, llm_verification_issues_2 = self._verify_text(
                llm_answers.get("detailed_answer", ""),
                citations=citations,
                target_summary=target_summary,
                allowed_sections=allowed_sections,
            )
            llm_verified = llm_verified and llm_verified_2
            llm_verification_issues.extend(llm_verification_issues_2)

        if not llm_verified:
            warnings.extend(llm_verification_issues)
            llm_answers["used"] = False
            llm_answers["summary_answer"] = answer
            llm_answers["detailed_answer"] = answer
            llm_answers["reasoning_trace"] = []
            llm_answers["caution_notes"] = []

        matched_sections = _dedupe_keep_order(
            [f"Section {hit.section_token}" for hit in [effective_primary] + effective_related if hit and hit.section_token]
        )
        selected_chunk_ids = [hit.chunk_id for hit in [effective_primary] + effective_related if hit]
        evidence_distribution = self._section_distribution(hits)

        # Serialise structured_reasoning for output
        sr_output = {
            "base_concept": sr.base_concept,
            "selected_section": sr.selected_section,
            "selection_reason": sr.selection_reason,
            "overrides_score_selection": sr.overrides_score_selection,
            "uncertainty": sr.uncertainty,
            "uncertainty_reason": sr.uncertainty_reason,
            "alternative_sections": sr.alternative_sections,
            "sections_analyzed": [
                {
                    "section": a.section,
                    "role": a.role,
                    "condition_summary": a.condition_summary,
                    "condition_match_score": a.match_score,
                    "concept_overlap": a.concept_overlap,
                    "raw_retrieval_score": a.raw_retrieval_score,
                    "hit_count": a.hit_count,
                    "is_base": a.is_base,
                    "note": a.note,
                }
                for a in sr.sections_analyzed
            ],
        }

        phase11 = {
            "answer_type": _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary", "explanation")) or "explanation",
            "support_status": support_status,
            "confidence": {
                "score": confidence,
                "label": _confidence_label(confidence),
            },
            "final_answer": answer,
            "summary_answer": llm_answers.get("summary_answer", answer),
            "detailed_answer": llm_answers.get("detailed_answer", answer),
            "citations": citations,
            "matched_sections": matched_sections,
            "missing_sections": target_summary.missing_from_corpus,
            "selected_chunk_ids": selected_chunk_ids,
            "warnings": _dedupe_keep_order(warnings),
            "structured_reasoning": sr_output,
            "validation": {
                "requested_sections": target_summary.requested,
                "present_in_corpus": target_summary.present_in_corpus,
                "retrieved_exact_matches": target_summary.retrieved_matches,
                "missing_from_corpus": target_summary.missing_from_corpus,
                "intent_alignment": intent_alignment,
                "intent_notes": intent_notes,
                "answer_verified": verified,
                "verification_issues": verification_issues,
                "llm_answer_verified": llm_verified,
                "llm_verification_issues": llm_verification_issues,
                "completeness": completeness,
            },
            "evidence": {
                "primary_chunk_id": effective_primary.chunk_id if effective_primary else None,
                "primary_chunk_type": effective_primary.chunk_type if effective_primary else None,
                "primary_section": f"Section {effective_primary.section_token}" if effective_primary and effective_primary.section_token else None,
                "primary_section_title": effective_primary.corpus_chunk.section_title if effective_primary and effective_primary.corpus_chunk else "",
                "related_chunk_ids": [hit.chunk_id for hit in effective_related],
                "top_sections": evidence_distribution,
            },
            "llm_generation": {
                "enabled": llm_answers.get("enabled", False),
                "used": llm_answers.get("used", False),
                "model": llm_answers.get("model"),
                "reasoning_trace": llm_answers.get("reasoning_trace", []),
                "caution_notes": llm_answers.get("caution_notes", []),
                "error": llm_answers.get("error"),
            },
        }

        enriched = dict(item)
        enriched["phase11"] = phase11
        return enriched

    def reason_many(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.reason_one(item) for item in items]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 11 grounded answer synthesis for legal RAG")
    parser.add_argument("--retrieval", required=True, help="Path to Phase 9 retrieval JSON output")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json used by the corpus index")
    parser.add_argument("--output", default=None, help="Optional output path for the phase 11 JSON")
    parser.add_argument("--top-support-hits", type=int, default=4, help="Max additional supporting hits to attach")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM answer synthesis on top of validated evidence")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model name for phase 11 synthesis")
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

python reasoning/reason_4.py \
  --retrieval retrieval/output_7__.json \
  --chunks data/processed/artifacts2/chunks.json \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --output reasoning/res_7__4.json
  
python reasoning/reason_4.py \
  --retrieval retrieval/output_8__.json \
  --chunks data/processed/artifacts2/chunks.json \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --output reasoning/res_8__4.json



'''