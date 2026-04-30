from __future__ import annotations

"""Phase 8 — Query Analysis for legal RAG.

This version does one-shot LLM analysis:
- One LLM call for the main query, including decomposition when needed.
- No recursive re-analysis of sub-queries.
- Deterministic validation/cleanup after the LLM call.
- Chunk-grounded hints are used to guide the LLM.

Expected input artifact:
- chunks.json

Typical usage:
    export OPENAI_API_KEY=...
    python phase8_query_analyzer.py \
        --chunks /path/to/chunks.json \
        --query "If five persons conjointly commit robbery and one of them commits murder, how will they be punished?"

Output is JSON printed to stdout and optionally written to a file.
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


FINAL_INTENTS = [
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
    "legal_scope",
    "legal_exception",
    "legal_condition",
    "legal_penalty",
    "case_application",
]

TARGET_TYPES = [
    "section",
    "subsection",
    "clause",
    "explanation",
    "illustration",
    "definition",
    "concept",
    "act",
]

SPECIFICITY_ENUM = ["broad", "narrow"]

GENERIC_CONCEPT_STOPWORDS = {
    "act",
    "law",
    "legal issue",
    "issue",
    "case",
    "matter",
    "thing",
    "does",
    "do",
    "mean",
    "means",
    "between",
    "section",
    "subsection",
    "clause",
    "explanation",
    "illustration",
    "code",
    "rule",
    "provision",
    "provisions",
    "shall",
    "may",
    "would",
    "could",
    "must",
    "whoever",
    "any person",
    "person",
    "persons",
    "someone",
    "something",
    "whenever",
    "wherever",
    "public",
    "general",
    "nothing",
    "any",
    "one",
    "two",
    "three",
    "four",
    "five",
}


@dataclass
class RuleSignals:
    strong_signals: int = 0
    medium_signals: int = 0
    ambiguity_hits: int = 0
    weak_signals: int = 0
    multi_part_hits: int = 0
    short_query: bool = False
    exact_reference: bool = False




@dataclass
class ChunkCandidate:
    label: str
    score: float
    evidence: str
    source: str


@dataclass
class ChunkBankItem:
    chunk_id: str
    chunk_type: str
    act: str
    chapter_title: str
    section_number: Optional[int]
    section_title: str
    keywords: List[str]
    legal_concepts: List[str]
    semantic_summary: str
    plain_english_paraphrase: str
    citation_text: str
    embedding_text: str
    derived_context: str

    @classmethod
    def from_chunk(cls, chunk: Dict[str, Any]) -> "ChunkBankItem":
        section = chunk.get("section", {}) or {}
        chapter = chunk.get("chapter", {}) or {}
        section_number = section.get("section_number")
        if isinstance(section_number, str) and section_number.isdigit():
            section_number = int(section_number)
        if not isinstance(section_number, int):
            section_number = None

        return cls(
            chunk_id=str(chunk.get("chunk_id", "")),
            chunk_type=str(chunk.get("chunk_type", "")),
            act=str(chunk.get("act", "") or ""),
            chapter_title=str(chapter.get("chapter_title", "") or ""),
            section_number=section_number,
            section_title=str(section.get("section_title", "") or ""),
            keywords=[str(x).strip().lower() for x in (chunk.get("keywords", []) or []) if str(x).strip()],
            legal_concepts=[str(x).strip().lower() for x in (chunk.get("legal_concepts", []) or []) if str(x).strip()],
            semantic_summary=str(chunk.get("semantic_summary", "") or ""),
            plain_english_paraphrase=str(chunk.get("plain_english_paraphrase", "") or ""),
            citation_text=str((chunk.get("citation") or {}).get("citation_text", "") or ""),
            embedding_text=str(chunk.get("embedding_text", "") or ""),
            derived_context=str(chunk.get("derived_context", "") or ""),
        )

    def candidate_phrases(self) -> List[str]:
        phrases: List[str] = []
        phrases.extend(self.legal_concepts)
        phrases.extend(self.keywords)
        if self.section_title:
            phrases.append(self.section_title)
        if self.chapter_title:
            phrases.append(self.chapter_title)
        if self.semantic_summary:
            phrases.extend(_extract_salient_phrases(self.semantic_summary))
        if self.plain_english_paraphrase:
            phrases.extend(_extract_salient_phrases(self.plain_english_paraphrase))
        return _dedupe_keep_order([_normalize_phrase(p) for p in phrases if _normalize_phrase(p)])


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_phrase(text: str) -> str:
    text = _normalize_space(text).lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_good_concept_phrase(text: str) -> bool:
    phrase = _normalize_phrase(text)
    if not phrase or phrase in GENERIC_CONCEPT_STOPWORDS:
        return False
    tokens = phrase.split()
    if not tokens:
        return False
    if len(tokens) > 5:
        return False
    bad_tokens = {
        "be", "is", "are", "was", "were", "will", "would", "could", "should", "do", "does", "did",
        "may", "shall", "can", "must", "have", "has", "had", "and", "or", "if", "then", "one",
        "two", "three", "four", "five", "them", "their", "his", "her", "its", "that", "this",
        "those", "these", "to", "of", "for", "with", "from", "by", "as", "at", "in", "on",
        "commit", "commits", "punish", "punished", "jointly", "conjointly", "joint", "they", "them",
        "their", "person", "persons",
    }
    if tokens[0] in bad_tokens or tokens[-1] in bad_tokens:
        return False
    if any(t in {"may", "shall", "will", "would", "could", "should", "can", "must", "be", "been", "being", "do", "does", "did", "have", "has", "had"} for t in tokens):
        return False
    content = [t for t in tokens if t not in bad_tokens]
    if not content:
        return False
    if not any(len(t) >= 4 for t in content):
        return False
    return True


def _dedupe_keep_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        key = item if isinstance(item, (str, int, float, tuple)) else json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_candidates(items: List[ChunkCandidate]) -> List[ChunkCandidate]:
    seen = set()
    out: List[ChunkCandidate] = []
    for item in items:
        key = item.label.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _tokenize(text: str) -> List[str]:
    text = _normalize_space(text).lower()
    return re.findall(r"[a-z0-9]+", text)


def _token_bigrams(tokens: Sequence[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def _token_trigrams(tokens: Sequence[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]


def _extract_salient_phrases(text: str) -> List[str]:
    text = _normalize_space(text)
    if not text:
        return []

    phrases: List[str] = []
    parts = re.split(r"[;:\n\.\(\)]", text)
    for part in parts:
        part = _normalize_space(part)
        if not part:
            continue
        if 3 <= len(part.split()) <= 7:
            phrases.append(part)

    for m in re.finditer(r"\b(?:[A-Za-z][A-Za-z\-]+\s+){1,4}[A-Za-z][A-Za-z\-]+\b", text):
        candidate = _normalize_space(m.group(0))
        if 2 <= len(candidate.split()) <= 6:
            phrases.append(candidate)

    return _dedupe_keep_order(phrases)


# ---------------------------------------------------------------------------
# OpenAI client wrapper
# ---------------------------------------------------------------------------


class OpenAIClient:
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, model: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a legal query analysis engine. "
                        "Return only valid JSON. No markdown. No commentary."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"


# ---------------------------------------------------------------------------
# Chunk bank: build grounding context from chunks.json
# ---------------------------------------------------------------------------


class ChunkBank:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.items: List[ChunkBankItem] = [ChunkBankItem.from_chunk(c) for c in chunks]
        self._phrase_frequency: Dict[str, int] = {}
        self._phrase_examples: Dict[str, List[str]] = {}
        self._build_phrase_index()

    @classmethod
    def from_path(cls, path: str | Path) -> "ChunkBank":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            chunks = data.get("chunks", []) or []
        elif isinstance(data, list):
            chunks = data
        else:
            chunks = []
        return cls(chunks)

    def _add_phrase(self, phrase: str, source: str) -> None:
        phrase = _normalize_phrase(phrase)
        if not _is_good_concept_phrase(phrase):
            return
        self._phrase_frequency[phrase] = self._phrase_frequency.get(phrase, 0) + 1
        self._phrase_examples.setdefault(phrase, [])
        if len(self._phrase_examples[phrase]) < 3 and source not in self._phrase_examples[phrase]:
            self._phrase_examples[phrase].append(source)

    def _build_phrase_index(self) -> None:
        for item in self.items:
            source = item.citation_text or item.section_title or item.chunk_id
            for phrase in item.candidate_phrases():
                tokens = phrase.split()
                if len(tokens) == 1 and tokens[0] in GENERIC_CONCEPT_STOPWORDS:
                    continue
                self._add_phrase(phrase, source)

    def _score_phrase(
        self,
        query_tokens: Sequence[str],
        query_bigrams: Sequence[str],
        query_trigrams: Sequence[str],
        phrase: str,
    ) -> float:
        phrase_tokens = _tokenize(phrase)
        if not phrase_tokens:
            return 0.0

        score = 0.0
        overlap = len(set(query_tokens) & set(phrase_tokens))
        score += 0.35 * overlap
        phrase_norm = " ".join(phrase_tokens)
        if phrase_norm in query_bigrams:
            score += 1.0
        if phrase_norm in query_trigrams:
            score += 1.2
        if any(phrase_norm in bg for bg in query_bigrams):
            score += 0.4
        if any(phrase_norm in tg for tg in query_trigrams):
            score += 0.6
        if len(phrase_tokens) <= 3:
            score += 0.15
        if any(tok in {"section", "clause", "explanation", "illustration"} for tok in phrase_tokens):
            score += 0.25
        return score

    def candidate_concepts(self, query: str, limit: int = 25) -> List[ChunkCandidate]:
        q_tokens = _tokenize(query)
        q_bigrams = _token_bigrams(q_tokens)
        q_trigrams = _token_trigrams(q_tokens)

        scored: List[ChunkCandidate] = []
        for phrase, freq in self._phrase_frequency.items():
            score = self._score_phrase(q_tokens, q_bigrams, q_trigrams, phrase)
            if score <= 0:
                continue
            score += min(0.35, math.log1p(freq) * 0.08)
            examples = self._phrase_examples.get(phrase, [])
            evidence = examples[0] if examples else phrase
            src = ", ".join(examples[:2]) if examples else phrase
            scored.append(ChunkCandidate(label=phrase, score=score, evidence=evidence, source=src))

        scored.sort(key=lambda x: (-x.score, len(x.label), x.label))
        return scored[:limit]

    def candidate_sections(self, query: str, limit: int = 10) -> List[ChunkCandidate]:
        q_tokens = _tokenize(query)
        q_bigrams = _token_bigrams(q_tokens)
        q_trigrams = _token_trigrams(q_tokens)

        scored: List[ChunkCandidate] = []
        for item in self.items:
            title = item.section_title.strip()
            if not title:
                continue
            phrase = _normalize_phrase(title)
            score = self._score_phrase(q_tokens, q_bigrams, q_trigrams, phrase)
            if score <= 0:
                continue
            evidence = item.citation_text or f"Section {item.section_number}: {item.section_title}"
            scored.append(ChunkCandidate(label=title, score=score, evidence=evidence, source=item.chunk_id))

        scored.sort(key=lambda x: (-x.score, len(x.label), x.label))
        return _dedupe_candidates(scored[:limit])

    def grounding_block(self, query: str, limit_concepts: int = 12, limit_sections: int = 5) -> str:
        concepts = self.candidate_concepts(query, limit=limit_concepts)
        sections = self.candidate_sections(query, limit=limit_sections)

        lines: List[str] = []
        if concepts:
            lines.append("Candidate concepts from chunks.json:")
            for c in concepts:
                lines.append(f"- {c.label}  | evidence: {c.evidence}")
        if sections:
            lines.append("Relevant section titles from chunks.json:")
            for s in sections:
                lines.append(f"- {s.label}  | citation: {s.evidence}")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Query analysis
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        llm_model: str = "gpt-4o-mini",
        chunk_bank: Optional[ChunkBank] = None,
        rule_high_threshold: float = 0.80,
        rule_use_llm_threshold: float = 0.68,
        enable_query_decomposition: bool = True,
        max_sub_queries: int = 4
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.chunk_bank = chunk_bank
        self.rule_high_threshold = rule_high_threshold
        self.rule_use_llm_threshold = rule_use_llm_threshold
        self.enable_query_decomposition = enable_query_decomposition
        self.max_sub_queries = max(2, int(max_sub_queries))


    # -------------------- public API --------------------

    def analyze(self, query: str) -> Dict[str, Any]:
        query = self._normalize_query(query)
        result = self._analyze_single_query(query)
        return self._validate_and_fix(result)

    # -------------------- core flow --------------------

    def _analyze_single_query(self, query: str) -> Dict[str, Any]:
        rule_result = self._rule_based_analysis(query)

        # High-confidence rules or no LLM available: stay with rules.
        if rule_result["confidence"]["rule_based"] >= self.rule_high_threshold or self.llm_client is None:
            rule_result["method"] = "rules"
            return rule_result

        # Medium/low confidence: let LLM take over semantic interpretation.
        if rule_result["confidence"]["rule_based"] <= self.rule_use_llm_threshold:
            llm_result = self._llm_analysis(query, rule_result)
            if llm_result is not None:
                merged = self._merge_rule_and_llm(rule_result, llm_result)
                merged["method"] = "llm_guided"
                return merged

        rule_result["method"] = "rules_fallback"
        return rule_result

    # -------------------- decomposition logic --------------------

    

    

    def _rule_based_analysis(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        signals = RuleSignals()
        notes: List[str] = []

        intent = "explanation"
        secondary: List[str] = []
        targets: List[Dict[str, str]] = []
        concepts: List[str] = []

        constraints: Dict[str, Any] = {
            "jurisdiction": None,
            "time": None,
            "specificity": "broad",
        }

        features: Dict[str, Any] = {
            "is_multi_hop": False,
            "requires_reasoning": False,
            "requires_exact_match": False,
        }

        if re.search(r"\b(section|article|rule)\s+\d+[a-z]?(?:\(\d+\))*\b", q):
            intent = "lookup"
            signals.strong_signals += 1
            signals.exact_reference = True
            features["requires_exact_match"] = True
            constraints["specificity"] = "narrow"
            ref = self._extract_section_like(query)
            if ref:
                num_match = re.search(r"\d+", ref)
                if num_match:
                    targets.append({"type": "section", "value": num_match.group(0)})
            notes.append("Detected exact statutory reference.")

        if re.search(r"\bdifference between\b|\bcompare\b|\bvs\b|\bversus\b", q):
            intent = "comparison"
            signals.strong_signals += 1
            features["is_multi_hop"] = True
            features["requires_reasoning"] = True
            secondary.append("reasoning")

        if re.search(r"\bhow to\b|\bsteps to\b|\bprocedure\b|\bprocess\b|\bfile\b|\bapply\b|\bcomplain\b", q):
            if intent != "lookup":
                intent = "procedure"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        if re.search(r"\bwho can\b|\beligible\b|\beligibility\b|\bcan i\b|\bam i allowed\b", q):
            intent = "eligibility"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        if re.search(r"\bwhat is\b|\bdefine\b|\bmeaning of\b|\bwhat does .* mean\b", q):
            if intent == "explanation":
                intent = "definition"
            signals.strong_signals += 1
            features["requires_exact_match"] = True

        if re.search(r"\bwhat will happen\b|\bwhat could happen\b|\bwhat would happen\b|\bwhat happens\b|\bconsequence\b|\bpunishment\b|\bpenalty\b|\bbe punished\b|\bpunished\b", q):
            if intent not in {"lookup", "comparison", "definition"}:
                intent = "legal_penalty"
            signals.medium_signals += 1
            features["requires_reasoning"] = True
            secondary.append("reasoning")

        if re.search(r"\bif\b.*\bthen\b|\bif\b.*\bwhat\b|\bwhen\b.*\bwhat\b", q):
            features["is_multi_hop"] = True
            features["requires_reasoning"] = True
            signals.multi_part_hits += 1
            if intent == "explanation":
                intent = "hypothetical"

        if re.search(r"\bexception\b|\bexceptions\b", q):
            if intent == "explanation":
                intent = "legal_exception"
            features["requires_reasoning"] = True
            signals.medium_signals += 1
            secondary.append("legal_exception")

        if re.search(r"\bcondition\b|\bconditions\b", q):
            if intent == "explanation":
                intent = "legal_condition"
            features["requires_reasoning"] = True
            signals.medium_signals += 1
            secondary.append("legal_condition")

        if re.search(r"\bpunishment\b|\bpenalty\b|\bimprisonment\b|\bfine\b|\bsentence\b", q):
            if intent in {"explanation", "hypothetical", "consequence", "case_application"}:
                intent = "legal_penalty"
            features["requires_reasoning"] = True
            signals.medium_signals += 1
            secondary.append("legal_penalty")

        if re.search(r"\bmy\b|\bme\b|\bmine\b|\bwe\b|\bus\b|\bsomeone\b|\bhe\b|\bshe\b|\bthey\b|\bminor\b|\bfriend\b", q):
            if features["requires_reasoning"] or features["is_multi_hop"] or intent in {"hypothetical", "consequence", "procedure"}:
                secondary.append("case_application")

        concepts.extend(self._extract_semantic_concepts(query))
        targets.extend(self._extract_targets(query))

        if signals.multi_part_hits > 0 and "case_application" not in secondary:
            secondary.append("case_application")

        rule_conf = self._compute_rule_confidence(
            signals=signals,
            primary_intent=intent,
            target_count=len(targets),
            concept_count=len(concepts),
        )

        if intent in {"consequence", "hypothetical", "case_application", "procedure", "legal_penalty"}:
            features["requires_reasoning"] = True

        return {
            "query": query,
            "intent": {
                "primary": intent,
                "secondary": self._dedupe_list([s for s in secondary if s != intent]),
            },
            "targets": self._dedupe_targets(targets),
            "concepts": self._dedupe_list(concepts),
            "constraints": constraints,
            "query_features": features,
            "confidence": {
                "rule_based": round(rule_conf, 2),
                "llm": 0.0,
                "agreement": 0.0,
                "overall": round(rule_conf, 2),
            },
            "method": "rules",
            "notes": notes,
        }

    def _compute_rule_confidence(
        self,
        signals: RuleSignals,
        primary_intent: str,
        target_count: int,
        concept_count: int,
    ) -> float:
        score = 0.34
        score += 0.20 * signals.strong_signals
        score += 0.10 * signals.medium_signals
        score += 0.04 * signals.weak_signals

        if signals.exact_reference:
            score += 0.12
        if target_count > 0:
            score += min(0.08, 0.03 * target_count)
        if concept_count > 0:
            score += min(0.06, 0.02 * concept_count)

        score -= 0.10 * signals.ambiguity_hits
        if signals.short_query:
            score -= 0.08
        if signals.multi_part_hits > 0:
            score -= 0.03 * signals.multi_part_hits
        if primary_intent in {"reasoning", "hypothetical", "case_application"}:
            score -= 0.04

        return max(0.05, min(0.95, score))

    def _extract_targets(self, query: str) -> List[Dict[str, str]]:
        targets: List[Dict[str, str]] = []
        q = query.lower()

        sec_match = re.search(r"\bsection\s+(\d+)", q)
        if sec_match:
            targets.append({"type": "section", "value": sec_match.group(1)})

        for s in re.findall(r"\(\s*(\d+)\s*\)", query):
            targets.append({"type": "subsection", "value": s})
        for c in re.findall(r"\(\s*([a-zA-Z])\s*\)", query):
            targets.append({"type": "clause", "value": c})

        for hit in re.findall(r"\b(ipc|bns|crpc|cpc|constitution)\b", q):
            targets.append({"type": "act", "value": hit.upper()})

        if re.search(r"\bdefinition\b|\bmeaning\b|\bmeans\b", q):
            targets.append({"type": "definition", "value": "definition"})
        if re.search(r"\bexplain\b|\bexplanation\b|\btell me about\b", q):
            targets.append({"type": "explanation", "value": "explanation"})
        if re.search(r"\billustration\b", q):
            targets.append({"type": "illustration", "value": "illustration"})

        return self._dedupe_targets(targets)

    def _extract_semantic_concepts(self, query: str) -> List[str]:
        q = query.lower()
        concepts: List[str] = []

        soft_patterns = [
            (r"\bbully\b|\bbullying\b|\bharass\b|\bharassment\b|\bthreat\b|\bthreatening\b", "harassment"),
            (r"\bhit\b|\baccident\b|\breversing\b|\bcrash\b|\bcollision\b", "accident"),
            (r"\bsteal\b|\bstole\b|\btaken\b|\btook\b|\btheft\b|\brobbery\b", "theft"),
            (r"\bforce\b|\bforced\b|\bassault\b|\bviolence\b|\bpush\b|\bpushed\b", "assault"),
            (r"\bcheat\b|\bcheated\b|\bscam\b|\bfraud\b|\bblackmail\b", "fraud"),
            (r"\bconsent\b|\bagree\b|\bagreed\b|\bpermission\b", "consent"),
            (r"\bnegligent\b|\bcareless\b|\black of care\b", "negligence"),
            (r"\bpunish\b|\bpunishment\b|\bpenalty\b|\bfine\b|\bimprisonment\b", "punishment"),
            (r"\bbroke\b|\bdamage\b|\bdestroy\b", "property damage"),
            (r"\bdrinking\b.*\bdriving\b|\bdrunk\b.*\bdriving\b", "drunk driving"),
            (r"\bminor\b|\bchild\b|\bjuvenile\b", "child"),
            (r"\bfalse case\b|\bfalse complaint\b", "false complaint"),
            (r"\bphone\b|\bmobile\b|\bdevice\b", "property"),
            (r"\bmurder\b", "murder"),
            (r"\brobbery\b", "robbery"),
        ]

        for pattern, concept in soft_patterns:
            if re.search(pattern, q):
                concepts.append(concept)

        if re.search(r"\bwhat could happen\b|\bwhat will happen\b|\bwhat happens\b|\bwhat would happen\b", q):
            concepts.append("legal consequences")

        if re.search(r"\bwhat can i do\b|\bwhat should i do\b|\bwhat do i do\b", q):
            concepts.append("legal remedy")

        if re.search(r"\bif\b.*\band\b.*\bthen\b|\bif\b.*\bwhat\b", q):
            concepts.append("conditional liability")

        if self.chunk_bank is not None:
            candidates = self.chunk_bank.candidate_concepts(q, limit=8)
            for c in candidates:
                if not _is_good_concept_phrase(c.label):
                    continue
                if c.score >= 1.2:
                    concepts.append(c.label)

        if not concepts:
            concepts.append("legal issue")

        return self._dedupe_list([_normalize_phrase(c) for c in concepts if _normalize_phrase(c)])

    # -------------------- LLM analysis and validation --------------------

    def _llm_analysis(self, query: str, rule_hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = self._call_llm(query, rule_hint)
        parsed = self._safe_parse_json(raw)
        if parsed is None:
            return None

        parsed = self._normalize_llm_output(parsed, query, rule_hint)
        if not self._looks_valid(parsed):
            return None

        parsed = self._normalize_llm_subqueries(parsed)
        return parsed

    def _call_llm(self, query: str, rule_hint: Dict[str, Any]) -> Any:
        grounding_block = self._grounding_block(query)
        prompt = f"""
You are Phase 8: Query Analysis for a legal retrieval system.

The user writes in natural language, not legal vocabulary.
Your job is to infer the real retrieval need, not just mirror the words.

Use the chunk-grounded hints below. Prefer these concepts if they match the query.
Do not invent section numbers unless they appear explicitly in the user query.
Do not output generic concepts like "thing", "case", "law", "does", "mean", "between".
If the query contains more than one legal issue, set decomposition needed=true and identify the issues clearly.
If the query can be broken into a condition and outcome, do that.

Return exactly this JSON shape:
{{
  "query": "string",
  "intent": {{
    "primary": "one allowed intent",
    "secondary": ["optional additional intents"]
  }},
  "targets": [
    {{"type": "one of section, subsection, clause, explanation, illustration, definition, concept, act", "value": "string"}}
  ],
  "concepts": ["normalized semantic concepts"],
  "constraints": {{
    "jurisdiction": "string or null",
    "time": "string or null",
    "specificity": "broad or narrow"
  }},
  "query_features": {{
    "is_multi_hop": true/false,
    "requires_reasoning": true/false,
    "requires_exact_match": true/false
  }},
  "confidence": 0.0,
  "decomposition": {{
    "needed": true/false,
    "strategy": "none|rules|llm",
    "reason": "string",
    "count": 0
  }},
  "method": "llm",
  "sub_queries": [
    {{
      "query": "string",
      "intent": {{
        "primary": "one allowed intent",
        "secondary": ["optional additional intents"]
      }},
      "targets": [{{"type": "one of section, subsection, clause, explanation, illustration, definition, concept, act", "value": "string"}}],
      "concepts": ["normalized semantic concepts"],
      "constraints": {{
        "jurisdiction": "string or null",
        "time": "string or null",
        "specificity": "broad or narrow"
      }},
      "query_features": {{
        "is_multi_hop": false,
        "requires_reasoning": true/false,
        "requires_exact_match": true/false
      }},
      "confidence": it can be between 0-1 as per ur response confidence,
      "decomposition": {{
        "needed": false,
        "strategy": "none",
        "reason": "string",
        "count": 0
      }},
      "method": "llm_subquery"
    }}
  ],
  "notes": ["optional note"]
}}

Allowed intents:
{json.dumps(FINAL_INTENTS, ensure_ascii=False)}

Important guidance:
- Choose the MOST ACTIONABLE intent.
- If the user asks "what can I do" or "what should I do", prefer procedure or case_application.
- If the user asks "what will happen" or "what could happen", prefer consequence.
- If the user asks for a law's meaning, use definition.
- If the user asks how a law applies to their situation, use case_application.
- DO NOT ADD VAGUE OR TOO GENERIC WORDS IN "Concepts" field.
- If the user asks for steps to take, use procedure.
- If the query involves two different legal consequences or two different acts, mark is_multi_hop=true and decompose.
- Concepts should be short, normalized, and retrieval-friendly.
- Prefer chunk-grounded concepts like robbery, murder, theft, consent, negligence, assault, fraud, punishment, child, intimidation.
- If you use a new concept that is not in the grounding block, it must still be a standard legal concept.
- Sub-queries must be self-contained and focused on one legal issue each.
- Do not repeat the original query unchanged unless decomposition is not useful.
- Do not output commentary.

CRITICAL RULE:
- NEVER generate section numbers unless they appear in the user query.
- If unsure, output concept instead of section.
- Do not guess section numbers.

{grounding_block}

User query:
{query}

Rule-based hint:
{json.dumps(rule_hint, ensure_ascii=False)}
""".strip()

        return self.llm_client.generate(model=self.llm_model, prompt=prompt)

    def _merge_rule_and_llm(self, rule_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(llm_result)
        merged["query"] = rule_result["query"]
        merged["targets"] = merged.get("targets") or rule_result["targets"]
        merged["concepts"] = merged.get("concepts") or rule_result.get("concepts", [])
        merged["constraints"] = merged.get("constraints") or rule_result["constraints"]
        merged["query_features"] = merged.get("query_features") or rule_result["query_features"]

        llm_conf = self._safe_float(merged.get("confidence", 0.0), 0.0)
        rule_conf = self._safe_float(rule_result["confidence"]["rule_based"], 0.0)
        agreement = self._estimate_agreement(rule_result, merged)
        overall = self._combine_confidence(rule_conf, llm_conf, agreement)

        merged["confidence"] = {
            "rule_based": round(self._clamp(rule_conf), 2),
            "llm": round(self._clamp(llm_conf), 2),
            "agreement": round(self._clamp(agreement), 2),
            "overall": round(self._clamp(overall), 2),
        }
        merged["notes"] = rule_result.get("notes", []) + merged.get("notes", [])
        return merged

    def _estimate_agreement(self, rule_result: Dict[str, Any], llm_result: Dict[str, Any]) -> float:
        score = 0.0
        rule_primary = (rule_result.get("intent", {}) or {}).get("primary")
        llm_primary = (llm_result.get("intent", {}) or {}).get("primary")
        if rule_primary == llm_primary:
            score += 0.5

        rule_concepts = set(rule_result.get("concepts", []) or [])
        llm_concepts = set(llm_result.get("concepts", []) or [])
        if rule_concepts and llm_concepts:
            overlap = len(rule_concepts & llm_concepts) / max(1, len(rule_concepts | llm_concepts))
            score += 0.5 * overlap
        return _clamp(score)

    def _normalize_llm_output(self, obj: Dict[str, Any], query: str, rule_hint: Dict[str, Any]) -> Dict[str, Any]:
        if "query" not in obj:
            obj["query"] = query
        if "intent" not in obj or not isinstance(obj["intent"], dict):
            obj["intent"] = rule_hint["intent"]
        if "targets" not in obj or not isinstance(obj["targets"], list):
            obj["targets"] = rule_hint["targets"]
        if "concepts" not in obj or not isinstance(obj["concepts"], list):
            obj["concepts"] = rule_hint.get("concepts", [])
        if "constraints" not in obj or not isinstance(obj["constraints"], dict):
            obj["constraints"] = rule_hint["constraints"]
        if "query_features" not in obj or not isinstance(obj["query_features"], dict):
            obj["query_features"] = rule_hint["query_features"]
        if "decomposition" not in obj or not isinstance(obj["decomposition"], dict):
            obj["decomposition"] = {"needed": False, "strategy": "none", "reason": "", "count": 0}
        if "sub_queries" not in obj or not isinstance(obj["sub_queries"], list):
            obj["sub_queries"] = []
        if "notes" not in obj or not isinstance(obj["notes"], list):
            obj["notes"] = []

        try:
            conf = float(obj.get("confidence", 0.0))
            if not (0.0 <= conf <= 1.0):
                conf = 0.0
        except Exception:
            conf = 0.0
        obj["confidence"] = conf
        return obj

    def _normalize_llm_subqueries(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        sub_queries = obj.get("sub_queries", [])
        if not isinstance(sub_queries, list):
            obj["sub_queries"] = []
            return obj

        cleaned: List[Dict[str, Any]] = []
        seen_queries = set()
        for sq in sub_queries:
            if not isinstance(sq, dict):
                continue
            sq = self._normalize_llm_output(sq, str(obj.get("query", "")), obj)
            sq = self._validate_subquery(sq)
            query_text = str(sq.get("query", "")).strip()
            if not query_text:
                continue
            key = query_text.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            cleaned.append(sq)

        obj["sub_queries"] = cleaned[: self.max_sub_queries]

        decomp = obj.get("decomposition", {})
        if not isinstance(decomp, dict):
            decomp = {}
        if obj["sub_queries"]:
            decomp["needed"] = len(obj["sub_queries"]) >= 2
            decomp["strategy"] = "llm"
            decomp["count"] = len(obj["sub_queries"])
        else:
            decomp["needed"] = False
            decomp["strategy"] = "none"
            decomp["count"] = 0
        obj["decomposition"] = decomp
        return obj

    def _validate_subquery(self, sq: Dict[str, Any]) -> Dict[str, Any]:
        query = self._normalize_query(str(sq.get("query", "")))
        sq["query"] = query

        intent = sq.get("intent", {})
        if not isinstance(intent, dict):
            intent = {}
        primary = intent.get("primary", "explanation")
        if primary not in FINAL_INTENTS:
            primary = "explanation"
        secondary = intent.get("secondary", [])
        if not isinstance(secondary, list):
            secondary = []
        secondary = [s for s in secondary if isinstance(s, str) and s in FINAL_INTENTS and s != primary]
        sq["intent"] = {"primary": primary, "secondary": self._dedupe_list(secondary)}

        targets = sq.get("targets", [])
        clean_targets: List[Dict[str, str]] = []
        if isinstance(targets, list):
            for t in targets:
                if not isinstance(t, dict):
                    continue
                t_type = t.get("type")
                t_val = t.get("value")
                if t_type in TARGET_TYPES and isinstance(t_val, str) and t_val.strip():
                    clean_targets.append({"type": t_type, "value": t_val.strip()})
        sq["targets"] = self._dedupe_targets(clean_targets)

        concepts = sq.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []
        clean_concepts = [_normalize_phrase(x) for x in concepts if isinstance(x, str) and _normalize_phrase(x)]
        clean_concepts = [c for c in clean_concepts if c not in GENERIC_CONCEPT_STOPWORDS]
        sq["concepts"] = self._dedupe_list(clean_concepts)

        constraints = sq.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        specificity = constraints.get("specificity", "broad")
        if specificity not in SPECIFICITY_ENUM:
            specificity = "broad"
        sq["constraints"] = {
            "jurisdiction": constraints.get("jurisdiction", None),
            "time": constraints.get("time", None),
            "specificity": specificity,
        }

        features = sq.get("query_features", {})
        if not isinstance(features, dict):
            features = {}
        sq["query_features"] = {
            "is_multi_hop": bool(features.get("is_multi_hop", False)),
            "requires_reasoning": bool(features.get("requires_reasoning", False)),
            "requires_exact_match": bool(features.get("requires_exact_match", False)),
        }

        confidence = sq.get("confidence", 0.0)
        sq["confidence"] = round(self._clamp(self._safe_float(confidence, 0.0)), 2)

        decomp = sq.get("decomposition", {})
        if not isinstance(decomp, dict):
            decomp = {}
        sq["decomposition"] = {
            "needed": bool(decomp.get("needed", False)),
            "strategy": "none",
            "reason": str(decomp.get("reason", "")).strip(),
            "count": 0,
        }
        sq["method"] = "llm_subquery"
        sq["notes"] = [n for n in sq.get("notes", []) if isinstance(n, str)]
        return sq

    def _validate_and_fix(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result = self._validate_result_payload(result)
        result = self._enforce_no_hallucinated_sections(result)
        return result

    def _validate_result_payload(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result["query"] = str(result.get("query", "")).strip()

        intent = result.get("intent", {})
        if not isinstance(intent, dict):
            intent = {}
        primary = intent.get("primary", "explanation")
        if primary not in FINAL_INTENTS:
            primary = "explanation"
        secondary = intent.get("secondary", [])
        if not isinstance(secondary, list):
            secondary = []
        secondary = [s for s in secondary if isinstance(s, str) and s in FINAL_INTENTS and s != primary]
        result["intent"] = {"primary": primary, "secondary": self._dedupe_list(secondary)}

        targets = result.get("targets", [])
        clean_targets: List[Dict[str, str]] = []
        if isinstance(targets, list):
            for t in targets:
                if not isinstance(t, dict):
                    continue
                t_type = t.get("type")
                t_val = t.get("value")
                if t_type in TARGET_TYPES and isinstance(t_val, str) and t_val.strip():
                    clean_targets.append({"type": t_type, "value": t_val.strip()})
        result["targets"] = self._dedupe_targets(clean_targets)

        concepts = result.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []
        clean_concepts = [_normalize_phrase(x) for x in concepts if isinstance(x, str) and _normalize_phrase(x)]
        clean_concepts = [c for c in clean_concepts if c not in GENERIC_CONCEPT_STOPWORDS]
        result["concepts"] = self._dedupe_list(clean_concepts)

        constraints = result.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        specificity = constraints.get("specificity", "broad")
        if specificity not in SPECIFICITY_ENUM:
            specificity = "broad"
        result["constraints"] = {
            "jurisdiction": constraints.get("jurisdiction", None),
            "time": constraints.get("time", None),
            "specificity": specificity,
        }

        features = result.get("query_features", {})
        if not isinstance(features, dict):
            features = {}
        result["query_features"] = {
            "is_multi_hop": bool(features.get("is_multi_hop", False)),
            "requires_reasoning": bool(features.get("requires_reasoning", False)),
            "requires_exact_match": bool(features.get("requires_exact_match", False)),
        }

        confidence = result.get("confidence", {})
        if not isinstance(confidence, dict):
            confidence = {}
        rule_conf = self._safe_float(confidence.get("rule_based", 0.0), default=0.0)
        llm_conf = self._safe_float(confidence.get("llm", 0.0), default=0.0)
        agreement = self._safe_float(confidence.get("agreement", 0.0), default=0.0)
        overall = self._safe_float(confidence.get("overall", 0.0), default=0.0)
        if overall <= 0.0:
            overall = self._combine_confidence(rule_conf, llm_conf, agreement)
        result["confidence"] = {
            "rule_based": round(self._clamp(rule_conf), 2),
            "llm": round(self._clamp(llm_conf), 2),
            "agreement": round(self._clamp(agreement), 2),
            "overall": round(self._clamp(overall), 2),
        }

        notes = result.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        result["notes"] = [n for n in notes if isinstance(n, str)]

        decomposition = result.get("decomposition", {})
        if not isinstance(decomposition, dict):
            decomposition = {}
        strategy = str(decomposition.get("strategy", "none")).strip() or "none"
        if strategy not in {"none", "rules", "llm"}:
            strategy = "none"
        reason = str(decomposition.get("reason", "")).strip()
        count = int(self._safe_float(decomposition.get("count", 0), 0))
        result["decomposition"] = {
            "needed": bool(decomposition.get("needed", False)),
            "strategy": strategy,
            "reason": reason,
            "count": max(0, count),
        }

        sub_queries = result.get("sub_queries", [])
        clean_sub_queries: List[Dict[str, Any]] = []
        if isinstance(sub_queries, list):
            for sq in sub_queries:
                if not isinstance(sq, dict):
                    continue
                validated = self._validate_subquery(dict(sq))
                clean_sub_queries.append(validated)
        result["sub_queries"] = clean_sub_queries

        if not result["decomposition"]["needed"]:
            result["sub_queries"] = []

        if result["decomposition"]["needed"] and result["confidence"]["overall"] > 0.0:
            result["confidence"]["overall"] = round(_clamp(result["confidence"]["overall"] * 0.98), 2)

        return result

    def _enforce_no_hallucinated_sections(self, result: Dict[str, Any]) -> Dict[str, Any]:
        query = result.get("query", "").lower()
        has_explicit_section = bool(re.search(r"\b(section|article|rule)\s+\d+", query))

        cleaned_targets: List[Dict[str, str]] = []
        converted_concepts = list(result.get("concepts", []))

        for t in result.get("targets", []):
            if not isinstance(t, dict):
                continue
            t_type = t.get("type")
            t_value = str(t.get("value", "")).strip()

            if t_type == "section" and not has_explicit_section:
                concept = self.extract_concept_from_section_label(t_value)
                if concept:
                    converted_concepts.append(concept)
                continue

            if t_type == "section" and has_explicit_section:
                if t_value.lower() not in query:
                    continue

            cleaned_targets.append(t)

        result["targets"] = cleaned_targets
        result["concepts"] = self.dedupe_and_clean_concepts(converted_concepts)
        return result

    def extract_concept_from_section_label(self, text: str) -> Optional[str]:
        text = text.lower()
        text = re.sub(r"section\s*\d+\s*:?", "", text).strip()
        text = re.sub(r"[^a-z\s]", "", text)
        words = text.split()
        if not words:
            return None
        bad = {"punishment", "section", "law", "act"}
        words = [w for w in words if w not in bad]
        if not words:
            return None
        return " ".join(words)

    def dedupe_and_clean_concepts(self, concepts: List[str]) -> List[str]:
        seen = set()
        cleaned = []
        for c in concepts:
            c = c.strip().lower()
            if not c or c in seen:
                continue
            seen.add(c)
            cleaned.append(c)
        return cleaned

    # -------------------- parsing and helpers --------------------

    def _safe_parse_json(self, raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None

        text = raw.strip()
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _looks_valid(self, obj: Dict[str, Any]) -> bool:
        try:
            if not isinstance(obj, dict):
                return False
            intent = obj.get("intent", {})
            if not isinstance(intent, dict):
                return False
            primary = intent.get("primary")
            if primary not in FINAL_INTENTS:
                return False
            confidence = obj.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                return False
            if not (0.0 <= float(confidence) <= 1.0):
                return False
            return True
        except Exception:
            return False

    def _extract_section_like(self, query: str) -> Optional[str]:
        m = re.search(r"\b(section|article|rule)\s+\d+[a-z]?(?:\(\d+\))*\b", query, re.IGNORECASE)
        return m.group(0) if m else None

    def _normalize_query(self, query: str) -> str:
        return _normalize_space(query)

    

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _combine_confidence(self, rule_conf: float, llm_conf: float, agreement: float) -> float:
        return 0.30 * rule_conf + 0.45 * llm_conf + 0.25 * agreement

    def _clamp(self, x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _dedupe_targets(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        out = []
        for item in items:
            key = (item["type"], item["value"].lower())
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    def _dedupe_list(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            x = _normalize_phrase(x)
            if not _is_good_concept_phrase(x):
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _grounding_block(self, query: str) -> str:
        if self.chunk_bank is None:
            return "No chunk bank is loaded."
        return self.chunk_bank.grounding_block(query)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_analyzer(chunks_path: str, model: str, enable_llm: bool = True) -> QueryAnalyzer:
    chunk_bank = ChunkBank.from_path(chunks_path)
    llm_client: Optional[OpenAIClient] = None
    if enable_llm and os.getenv("OPENAI_API_KEY"):
        llm_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", ""))
    return QueryAnalyzer(
        llm_client=llm_client,
        llm_model=model,
        chunk_bank=chunk_bank,
        max_sub_queries=4,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 query analyzer")
    parser.add_argument("--chunks", required=True, help="Path to chunks.json")
    parser.add_argument("--query", help="Query to analyze")
    parser.add_argument("--input-json", help="Path to a JSON file containing a query or a list of queries")
    parser.add_argument("--output", help="Optional path to write JSON output")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use rules only")
    parser.add_argument("--stats", action="store_true", help="Print chunk-bank stats and exit")
    args = parser.parse_args()

    analyzer = build_analyzer(args.chunks, model=args.model, enable_llm=not args.no_llm)

    if args.stats:
        bank = analyzer.chunk_bank
        assert bank is not None
        stats = summarize_chunks(bank)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    queries: List[str] = []
    if args.query:
        queries = [args.query]
    elif args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, str):
            queries = [payload]
        elif isinstance(payload, list):
            queries = [str(item) for item in payload]
        elif isinstance(payload, dict) and "query" in payload:
            queries = [str(payload["query"])]
        else:
            raise ValueError("input JSON must be a string, a list of strings, or an object with a 'query' field")
    else:
        raise SystemExit("Provide --query or --input-json")

    outputs = [analyzer.analyze(q) for q in queries]
    payload: Any = outputs[0] if len(outputs) == 1 else outputs
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.write("\n")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def summarize_chunks(bank: ChunkBank) -> Dict[str, Any]:
    items = bank.items
    type_counts: Dict[str, int] = {}
    for item in items:
        type_counts[item.chunk_type] = type_counts.get(item.chunk_type, 0) + 1

    def missing(field: str) -> int:
        count = 0
        for item in items:
            value = getattr(item, field)
            if isinstance(value, list):
                if not value:
                    count += 1
            else:
                if not str(value).strip():
                    count += 1
        return count

    unique_phrases = len(bank._phrase_frequency)

    return {
        "chunk_count": len(items),
        "type_counts": type_counts,
        "missing": {
            "semantic_summary": missing("semantic_summary"),
            "plain_english_paraphrase": missing("plain_english_paraphrase"),
            "keywords": missing("keywords"),
            "legal_concepts": missing("legal_concepts"),
            "section_title": missing("section_title"),
        },
        "unique_grounded_phrases": unique_phrases,
    }


if __name__ == "__main__":
    main()




# python query_analysis/aq.py --chunks data/processed/artifacts2/chunks.json     --query "My phone was stolen in a bus and I also got injured while trying to stop the thief, what can I do?"     --output query_analysis/result.json