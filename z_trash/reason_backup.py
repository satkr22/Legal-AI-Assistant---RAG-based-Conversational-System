from __future__ import annotations

"""Phase 11 - Grounded answer synthesis and validation for legal RAG.

Phase 11 consumes Phase 9 retrieval output and turns it into a safe,
corpus-grounded answer package. It does not change retrieval behaviour.

Primary responsibilities:
- validate requested sections against the indexed corpus
- check whether retrieved chunks support the user's intent
- prefer direct legal rule chunks over illustrations for lookup queries
- assemble evidence around the best-supported section
- produce a grounded answer with citations, warnings, and confidence

Expected input:
- Phase 9 JSON output from retrieval/rq.py
- chunks.json from the indexed corpus
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
APPLICATION_INTENTS = {"case application", "legal penalty"}

SECTION_REFERENCE_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.|subsection|sub-section|clause)\s*"
    r"(\d+[a-z]?)(?:\s*\(\s*([0-9a-z]+)\s*\))?",
    flags=re.IGNORECASE,
)

PENALTY_SIGNAL_PATTERN = re.compile(
    r"\b(?:shall be punished|punished with|imprisonment|fine|extend to)\b",
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

    def matching_chunks(self, reference: str) -> List[CorpusChunk]:
        token = _canonical_reference_token(reference)
        if not token:
            return []
        base = token.split("(")[0]
        return _dedupe_keep_order(
            list(self.citation_token_to_chunks.get(token, [])) + list(self.by_section_token.get(base, []))
        )

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
        retrieved = [
            target for target in present
            if any(self._target_matches_hit(target, hit) for hit in hits)
        ]

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

    def _preferred_hit(self, hits: List[EvidenceHit], intent_family: str, target_summary: RequestedTargetSummary) -> Optional[EvidenceHit]:
        if not hits:
            return None

        target_tokens = {
            _canonical_reference_token(target).split("(")[0]
            for target in target_summary.present_in_corpus
            if _canonical_reference_token(target)
        }

        type_priority_lookup = {
            "content": 6,
            "subsection": 5,
            "clause": 4,
            "section": 3,
            "explanation": 2,
            "illustration": 1,
        }
        type_priority_general = {
            "content": 6,
            "subsection": 5,
            "section": 4,
            "clause": 3,
            "explanation": 2,
            "illustration": 1,
        }
        type_priority = type_priority_lookup if intent_family == "lookup" else type_priority_general

        def sort_key(hit: EvidenceHit) -> Tuple[int, int, float, float, float]:
            target_match = 1 if hit.section_token in target_tokens else 0
            return (
                target_match,
                type_priority.get(hit.chunk_type, 0),
                hit.concept_match_count,
                hit.concept_coverage_ratio,
                hit.score,
            )

        return max(hits, key=sort_key)

    def _related_hits(
        self,
        primary_hit: Optional[EvidenceHit],
        all_hits: List[EvidenceHit],
        intent_family: str,
    ) -> List[EvidenceHit]:
        if not primary_hit:
            return []

        same_section = [hit for hit in all_hits if hit.section_token == primary_hit.section_token and hit.chunk_id != primary_hit.chunk_id]
        base_hits = [hit for hit in same_section if hit.is_base]
        expl_hits = [hit for hit in same_section if hit.is_explanatory]

        if intent_family == "lookup":
            expl_priority = {"explanation": 2, "illustration": 1}
            ordered = (
                sorted(base_hits, key=lambda hit: (-hit.score, hit.rank))
                + sorted(expl_hits, key=lambda hit: (-expl_priority.get(hit.chunk_type, 0), -hit.score, hit.rank))
            )
        else:
            ordered = sorted(base_hits + expl_hits, key=lambda hit: (-hit.score, hit.rank))
        return ordered[: self.top_support_hits]

    def _section_distribution(self, hits: List[EvidenceHit]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[EvidenceHit]] = defaultdict(list)
        for hit in hits:
            if hit.section_token:
                grouped[hit.section_token].append(hit)

        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                len(item[1]),
                max((hit.score for hit in item[1]), default=0.0),
            ),
            reverse=True,
        )
        out: List[Dict[str, Any]] = []
        for section_token, section_hits in ranked[:5]:
            title = self.corpus_index.section_title(section_token)
            out.append(
                {
                    "section": f"Section {section_token}",
                    "section_title": title,
                    "hit_count": len(section_hits),
                    "best_score": round(max((hit.score for hit in section_hits), default=0.0), 6),
                    "chunk_types": Counter(hit.chunk_type for hit in section_hits),
                }
            )
        return out

    def _intent_alignment(self, intent_family: str, hits: List[EvidenceHit], primary_hit: Optional[EvidenceHit]) -> Tuple[str, List[str]]:
        if not hits:
            return "weak", ["No retrieved evidence is available for phase 11 synthesis."]

        notes: List[str] = []
        base_count = sum(1 for hit in hits if hit.is_base)
        explanatory_count = sum(1 for hit in hits if hit.is_explanatory)

        if intent_family == "lookup":
            if primary_hit and primary_hit.is_base:
                notes.append("Lookup intent is aligned with a direct base legal chunk.")
                return "strong", notes
            notes.append("Lookup intent is currently led by explanatory material rather than direct rule text.")
            return "moderate", notes

        if intent_family == "application":
            if primary_hit and primary_hit.is_base and base_count >= 1:
                notes.append("Application intent has at least one direct legal rule chunk to anchor the answer.")
                return "strong", notes
            notes.append("Application intent lacks a strong base rule chunk and should be answered cautiously.")
            return "weak", notes

        if explanatory_count >= 1 and base_count >= 1:
            notes.append("Explanatory intent has both rule text and supporting explanatory material.")
            return "strong", notes
        if base_count >= 1:
            notes.append("Explanatory intent has direct rule support but limited companion explanation.")
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

        score += min(primary_hit.concept_coverage_ratio, 0.2)
        score += min(primary_hit.score * 0.18, 0.18)
        score += min(len(related_hits) * 0.04, 0.12)

        if intent_alignment == "strong":
            score += 0.12
        elif intent_alignment == "moderate":
            score += 0.04
        else:
            score -= 0.1

        if target_summary.missing_from_corpus:
            score -= 0.12

        competing_sections: List[Tuple[str, float]] = []
        seen_sections = set()
        for hit in hits:
            if not hit.section_token or hit.section_token in seen_sections:
                continue
            seen_sections.add(hit.section_token)
            competing_sections.append((hit.section_token, hit.score))
            if len(competing_sections) >= 2:
                break
        if len(competing_sections) >= 2:
            gap = competing_sections[0][1] - competing_sections[1][1]
            if gap < 0.15:
                score -= 0.08

        return round(_clamp(score, 0.05, 0.98), 4)

    def _warning_messages(
        self,
        hits: List[EvidenceHit],
        primary_hit: Optional[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_alignment: str,
    ) -> List[str]:
        warnings: List[str] = []

        if not hits:
            warnings.append("Phase 9 did not return any evidence for this query.")
            return warnings

        if target_summary.missing_from_corpus:
            warnings.append(
                f"{_join_brief(target_summary.missing_from_corpus)} is not present in the indexed corpus."
            )

        if target_summary.support_status == "section_exists_but_not_retrieved_well":
            warnings.append("The requested section exists in the corpus, but the current retrieval set does not ground it strongly.")

        top_hit = hits[0]
        if primary_hit and top_hit.chunk_id != primary_hit.chunk_id and top_hit.is_explanatory and primary_hit.is_base:
            warnings.append("Phase 11 preferred a direct rule chunk over a higher-ranked explanatory or illustrative hit.")

        if intent_alignment == "weak":
            warnings.append("Evidence-to-intent alignment is weak, so the answer is intentionally cautious.")

        distinct_sections = _dedupe_keep_order([hit.section_token for hit in hits if hit.section_token])
        if len(distinct_sections) > 1 and target_summary.support_status == "no_specific_section_requested":
            warnings.append("Multiple nearby sections were retrieved, so exact applicability depends on the facts.")

        return warnings

    def _core_rule_text(self, hit: Optional[EvidenceHit]) -> str:
        if not hit:
            return ""
        if hit.corpus_chunk and hit.corpus_chunk.derived_context:
            text = hit.corpus_chunk.derived_context
        else:
            text = hit.derived_context or hit.text
        sentences = _split_sentences(text)
        if not sentences:
            return ""

        citation_sentence = []
        content_sentence = []
        for sentence in sentences:
            lowered = sentence.lower()
            if lowered.startswith("act:") or lowered.startswith("chapter "):
                citation_sentence.append(sentence)
                continue
            if lowered.startswith("section "):
                citation_sentence.append(sentence)
                continue
            content_sentence.append(sentence)

        if content_sentence:
            return _compact_text(" ".join(content_sentence), max_sentences=2, max_chars=420)
        return _compact_text(" ".join(citation_sentence), max_sentences=2, max_chars=420)

    def _penalty_text(self, hits: Sequence[EvidenceHit]) -> str:
        for hit in hits:
            for sentence in _split_sentences(hit.text):
                if PENALTY_SIGNAL_PATTERN.search(sentence):
                    return _compact_text(sentence, max_sentences=1, max_chars=280)
        return ""

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
        hits: List[EvidenceHit],
        primary_hit: Optional[EvidenceHit],
        related_hits: List[EvidenceHit],
        target_summary: RequestedTargetSummary,
        intent_alignment: str,
    ) -> str:
        intent = dict(phase8.get("intent", {}) or {})
        primary_intent = _safe_lower(intent.get("primary", "explanation"))
        intent_family = self._intent_family(phase8)

        if target_summary.support_status == "section_not_in_corpus":
            closest = primary_hit
            if closest and closest.section_token:
                section_label = f"Section {closest.section_token}"
                title = closest.corpus_chunk.section_title if closest.corpus_chunk else ""
                title_part = f" ({title})" if title else ""
                return (
                    f"{_join_brief(target_summary.missing_from_corpus)} is not present in the indexed corpus. "
                    f"The closest retrieved material for your intent points to {section_label}{title_part}, "
                    f"but Phase 11 does not treat that as a substitute for the missing requested section."
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
        section_title = ""
        if primary_hit.corpus_chunk and primary_hit.corpus_chunk.section_title:
            section_title = primary_hit.corpus_chunk.section_title
        title_part = f" ({section_title})" if section_title else ""
        rule_text = self._core_rule_text(primary_hit)
        support_detail = self._supporting_detail(related_hits, primary_hit.chunk_id)
        penalty = self._penalty_text([primary_hit] + list(related_hits))
        primary_penalty_text = _compact_text(primary_hit.text, max_sentences=1, max_chars=280)

        if intent_family == "lookup":
            answer = f"{section_label}{title_part} in {act} is the best-supported match for your query. {rule_text}"
            if support_detail:
                answer += f" Supporting detail from the same section: {support_detail}"
            return answer

        if primary_intent == "legal penalty":
            answer = f"Based on the retrieved corpus, {section_label}{title_part} is the best-supported provision for this penalty-focused query. {rule_text}"
            if penalty and penalty != primary_penalty_text and penalty not in answer:
                answer += f" Penalty text in the retrieved evidence: {penalty}"
            if intent_alignment != "strong":
                answer += " The answer remains cautious because the retrieved support is not perfectly aligned."
            return answer

        if primary_intent == "case application":
            answer = (
                f"Based on the retrieved corpus, the closest directly supported provision is {section_label}{title_part}. "
                f"{rule_text}"
            )
            if penalty and penalty != primary_penalty_text and penalty not in answer:
                answer += f" The retrieved punishment language says: {penalty}"
            answer += " This is an evidence-grounded indication, not a final legal conclusion, because real applicability depends on the exact facts."
            return answer

        if intent_family == "explanation":
            answer = f"The retrieved evidence most strongly supports {section_label}{title_part}. {rule_text}"
            if support_detail:
                answer += f" A useful supporting detail is: {support_detail}"
            return answer

        answer = f"The best-supported answer from the indexed corpus points to {section_label}{title_part}. {rule_text}"
        if support_detail:
            answer += f" Supporting detail: {support_detail}"
        return answer

    def _citations(self, primary_hit: Optional[EvidenceHit], related_hits: List[EvidenceHit]) -> List[str]:
        citations: List[str] = []
        for hit in [primary_hit] + list(related_hits):
            if not hit:
                continue
            citation = ""
            if hit.corpus_chunk and hit.corpus_chunk.citation_text:
                citation = hit.corpus_chunk.citation_text
            elif hit.citation:
                citation = hit.citation
            if citation:
                citations.append(citation)
        return _dedupe_keep_order(citations)

    def _llm_system_prompt(self) -> str:
        return (
            "You are Phase 11 of a legal RAG pipeline. "
            "You must answer only from the validated evidence provided. "
            "Do not invent sections, punishments, facts, or legal conclusions outside the evidence. "
            "If evidence is limited, be explicit about the limitation. "
            "Return JSON with keys: summary_answer, detailed_answer, reasoning_trace, caution_notes. "
            "summary_answer must be concise. detailed_answer may be fuller but still grounded. "
            "reasoning_trace must be a short list of evidence-based bullets. "
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

    def _verify_answer(
        self,
        answer: str,
        primary_hit: Optional[EvidenceHit],
        citations: List[str],
        target_summary: RequestedTargetSummary,
    ) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        if not answer:
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

        return (len(issues) == 0), issues

    def _verify_llm_answers(
        self,
        llm_answers: Dict[str, Any],
        primary_hit: Optional[EvidenceHit],
        citations: List[str],
        target_summary: RequestedTargetSummary,
    ) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        for key in ("summary_answer", "detailed_answer"):
            ok, sub_issues = self._verify_answer(
                _normalize_space(llm_answers.get(key)),
                primary_hit=primary_hit,
                citations=citations,
                target_summary=target_summary,
            )
            if not ok:
                issues.extend([f"{key}: {issue}" for issue in sub_issues])
        return (len(issues) == 0), issues

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
        confidence = self._confidence_score(hits, primary_hit, related_hits, target_summary, intent_alignment)
        warnings = self._warning_messages(hits, primary_hit, target_summary, intent_alignment)
        answer = self._build_answer(query, phase8, hits, primary_hit, related_hits, target_summary, intent_alignment)
        citations = self._citations(primary_hit, related_hits)
        verified, verification_issues = self._verify_answer(answer, primary_hit, citations, target_summary)

        if not verified:
            warnings.extend(verification_issues)
            if target_summary.support_status != "exact_section_supported":
                answer = "The retrieved evidence is not strong enough to produce a more specific grounded answer."

        llm_answers = self._llm_answers(
            query=query,
            phase8=phase8,
            target_summary=target_summary,
            primary_hit=primary_hit,
            related_hits=related_hits,
            deterministic_answer=answer,
            warnings=warnings,
            confidence=confidence,
        )
        llm_verified, llm_verification_issues = self._verify_llm_answers(
            llm_answers,
            primary_hit=primary_hit,
            citations=citations,
            target_summary=target_summary,
        )
        if not llm_verified:
            warnings.extend(llm_verification_issues)
            llm_answers["used"] = False
            llm_answers["summary_answer"] = answer
            llm_answers["detailed_answer"] = answer
            llm_answers["reasoning_trace"] = []
            llm_answers["caution_notes"] = []

        selected_chunk_ids = [hit.chunk_id for hit in [primary_hit] + related_hits if hit]
        matched_sections = _dedupe_keep_order(
            [f"Section {hit.section_token}" for hit in [primary_hit] + related_hits if hit and hit.section_token]
        )
        evidence_distribution = self._section_distribution(hits)

        phase11 = {
            "answer_type": _safe_lower(dict(phase8.get("intent", {}) or {}).get("primary", "explanation")) or "explanation",
            "support_status": target_summary.support_status,
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
            },
            "evidence": {
                "primary_chunk_id": primary_hit.chunk_id if primary_hit else None,
                "primary_chunk_type": primary_hit.chunk_type if primary_hit else None,
                "primary_section": f"Section {primary_hit.section_token}" if primary_hit and primary_hit.section_token else None,
                "primary_section_title": primary_hit.corpus_chunk.section_title if primary_hit and primary_hit.corpus_chunk else "",
                "related_chunk_ids": [hit.chunk_id for hit in related_hits],
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

python reasoning/reason.py \
  --retrieval retrieval/output_2__.json \
  --chunks data/processed/artifacts2/chunks.json \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --output reasoning/res_2__.json

python reasoning/reason.py \
  --retrieval retrieval/output_4__.json \
  --chunks data/processed/artifacts2/chunks.json \
  --enable-llm \
  --llm-model gpt-4o-mini \
  --output reasoning/res_4__.json



'''
