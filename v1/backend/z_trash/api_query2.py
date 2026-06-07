
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


class OpenAIClient:
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, model: str, prompt: str, response_format: str = "json") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query understanding engine. "
                            "Output ONLY valid JSON. No explanation. No markdown. No extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.05,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            match = re.search(r"\{.*\}", content, re.DOTALL)
            return match.group(0) if match else content
        except Exception as e:
            print(f"[OpenAI ERROR]: {e}")
            return "{}"


@dataclass
class RuleSignals:
    strong_signals: int = 0
    medium_signals: int = 0
    weak_signals: int = 0
    ambiguity_hits: int = 0
    multi_part_hits: int = 0
    short_query: bool = False
    exact_reference: bool = False


@dataclass
class QueryAnalysisResult:
    query: str
    intent: Dict[str, Any]
    targets: List[Dict[str, str]]
    concepts: List[str]
    constraints: Dict[str, Any]
    query_features: Dict[str, Any]
    confidence: Dict[str, float]
    method: str
    decomposition: Dict[str, Any] = field(default_factory=dict)
    sub_queries: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """
    Phase 8: natural-language query understanding for retrieval.

    Architecture:
    - Rules handle only obvious, deterministic patterns and cheap hints.
    - LLM handles semantic intent, concepts, and decomposition when the query is
      not obviously single-hop.
    - Decomposition is enabled more aggressively for conditional / consequence /
      case-application style queries, because those are usually implicit multi-hop.
    """

    def __init__(
        self,
        llm_client: Optional[OpenAIClient] = None,
        llm_model: str = "gpt-4o-mini",
        rule_high_threshold: float = 0.78,
        rule_use_llm_threshold: float = 0.70,
        enable_query_decomposition: bool = True,
        max_sub_queries: int = 4,
        llm_decompose_on_uncertain: bool = True,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.rule_high_threshold = rule_high_threshold
        self.rule_use_llm_threshold = rule_use_llm_threshold
        self.enable_query_decomposition = enable_query_decomposition
        self.max_sub_queries = max(2, int(max_sub_queries))
        self.llm_decompose_on_uncertain = llm_decompose_on_uncertain

    def analyze(self, query: str, allow_decomposition: bool = True) -> Dict[str, Any]:
        query = self._normalize_query(query)
        result = self._analyze_single_query(query)

        if allow_decomposition and self.enable_query_decomposition:
            result = self._attach_query_decomposition(result)

        return self._validate_and_fix(result)

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

    def _attach_query_decomposition(self, result: Dict[str, Any]) -> Dict[str, Any]:
        query = str(result.get("query", "")).strip()

        should_split, reason, force_llm = self._should_decompose_query(result)

        result["decomposition"] = {
            "needed": False,
            "strategy": "none",
            "reason": reason if should_split else "",
            "count": 0,
        }
        result["sub_queries"] = []

        if not should_split:
            return result

        raw_sub_queries: List[str] = []
        strategy = "rules"

        # LLM-first decomposition for uncertain or semantically implicit queries.
        if self.llm_client is not None and (force_llm or self.llm_decompose_on_uncertain):
            llm_sub_queries = self._llm_decompose_query(query, result)
            if llm_sub_queries:
                raw_sub_queries = llm_sub_queries
                strategy = "llm"

        # Deterministic fallback.
        if not raw_sub_queries:
            raw_sub_queries = self._rule_decompose_query(query, result)

        clean_sub_queries = self._clean_sub_queries(query, raw_sub_queries)
        if len(clean_sub_queries) < 2:
            return result

        analyzed_sub_queries = [
            self.analyze(sub_query, allow_decomposition=False)
            for sub_query in clean_sub_queries[: self.max_sub_queries]
        ]

        result["decomposition"] = {
            "needed": True,
            "strategy": strategy,
            "reason": reason,
            "count": len(analyzed_sub_queries),
        }
        result["sub_queries"] = analyzed_sub_queries
        result.setdefault("notes", []).append(
            f"Query decomposed into {len(analyzed_sub_queries)} retrieval sub-queries."
        )
        return result

    def _should_decompose_query(self, result: Dict[str, Any]) -> Tuple[bool, str, bool]:
        query = str(result.get("query", "")).strip()
        q = query.lower()
        features = result.get("query_features", {}) or {}
        secondary = (result.get("intent", {}) or {}).get("secondary", []) or []
        primary = (result.get("intent", {}) or {}).get("primary", "")

        reasons: List[str] = []
        force_llm = False

        connector_hits = len(re.findall(r"\b(and|also|plus|or|then|but)\b", q))

        if bool(features.get("is_multi_hop", False)):
            reasons.append("multi-hop query")
            force_llm = True

        if len(secondary) >= 1:
            reasons.append("multiple intents")
            force_llm = True

        if connector_hits >= 1 and len(q.split()) >= 8:
            reasons.append("joined clauses")

        if primary in {"comparison", "hypothetical", "case_application"}:
            reasons.append("composite reasoning intent")
            force_llm = True

        if primary == "consequence" and re.search(r"\bif\b", q):
            reasons.append("conditional consequence query")
            force_llm = True

        if re.search(r"\bsection\b.*\b(exception|exceptions|punishment|penalty|condition|conditions)\b", q):
            reasons.append("section query with attached legal sub-topic")
            force_llm = True

        if re.search(r"\bwhat (will|would|could|can) happen\b", q) and re.search(r"\bif\b|\bwhen\b", q):
            reasons.append("implicit condition-to-consequence query")
            force_llm = True

        if re.search(r"\bwhat can i do\b|\bwhat should i do\b|\bwhat do i do\b", q):
            reasons.append("action-seeking legal query")
            force_llm = True

        # If rules are uncertain, still try LLM decomposition because semantics matter more than surface form.
        if result.get("confidence", {}).get("rule_based", 0.0) < self.rule_use_llm_threshold:
            reasons.append("low rule confidence")
            force_llm = True

        return (len(reasons) > 0, ", ".join(self._dedupe_list(reasons)), force_llm)

    def _normalize_query(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip())

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

        # Exact statutory references.
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

        # Comparison.
        if re.search(r"\bdifference between\b|\bcompare\b|\bvs\b|\bversus\b", q):
            intent = "comparison"
            signals.strong_signals += 1
            features["is_multi_hop"] = True
            features["requires_reasoning"] = True
            secondary.append("reasoning")

        # Procedure.
        if re.search(r"\bhow to\b|\bsteps to\b|\bprocedure\b|\bprocess\b|\bfile\b|\bapply\b|\bcomplain\b", q):
            if intent != "lookup":
                intent = "procedure"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        # Eligibility.
        if re.search(r"\bwho can\b|\beligible\b|\beligibility\b|\bcan i\b|\bam i allowed\b", q):
            intent = "eligibility"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        # Definition.
        if re.search(r"\bwhat is\b|\bdefine\b|\bmeaning of\b|\bwhat does .* mean\b", q):
            if intent == "explanation":
                intent = "definition"
            signals.strong_signals += 1
            features["requires_exact_match"] = True

        # Reasoning.
        if re.search(r"\bwhy\b|\breason\b|\bcause\b|\beffect\b|\bimpact\b", q):
            if intent in {"explanation", "definition"}:
                intent = "reasoning"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        # Hypothetical / consequence-like conditionals.
        if re.search(r"\bwhat if\b|\bsuppose\b|\bin case\b|\bif\b", q):
            # Not every if-query is hypothetical, but it is a strong signal.
            if intent in {"explanation", "definition"}:
                intent = "hypothetical"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

            # Explicit consequence wording after a condition is multi-hop.
            if re.search(r"\bwhat (will|would|could|can) happen\b|\bwhat happens\b", q):
                features["is_multi_hop"] = True

        if re.search(r"\bsummary\b|\bsummarize\b|\bbrief\b|\boverview\b|\bgist\b", q):
            intent = "summary"
            signals.medium_signals += 1

        # Indian legal context.
        if re.search(r"\bipc\b|\bbns\b|\bcrpc\b|\bcc\b|\bconstitution\b|\bindian\b", q):
            constraints["jurisdiction"] = "india"
            signals.medium_signals += 1
            notes.append("Detected Indian legal context.")
            if intent in {"definition", "explanation"}:
                intent = "legal_scope"

        # Exceptions and conditions.
        if re.search(r"\bexception\b|\bexceptions\b|\bexcept\b|\bunless\b|\bnot apply\b", q):
            if intent in {"explanation", "definition"}:
                intent = "legal_exception"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        if re.search(r"\bcondition\b|\bconditions\b|\bapplicable\b|\bapply to\b|\bscope\b|\bwhen does\b", q):
            if intent in {"explanation", "definition"}:
                intent = "legal_condition"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        # Consequence.
        if re.search(r"\bwhat could happen\b|\bwhat will happen\b|\bwhat happens\b|\bwhat would happen\b|\bcan happen\b", q):
            if intent in {"explanation", "definition"}:
                intent = "consequence"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        # Penalty.
        if re.search(r"\bpunishment\b|\bpenalty\b|\bsentence\b|\bfine\b|\bimprisonment\b|\bliable to punishment\b", q):
            intent = "legal_penalty"
            signals.strong_signals += 1
            constraints["specificity"] = "narrow"
            features["requires_exact_match"] = True

        # Multi-part cues.
        if any(token in q for token in [" and ", " also ", " plus ", " or ", " but "]):
            signals.multi_part_hits += 1
            if len(re.findall(r"\b(and|also|plus|or|but)\b", q)) >= 2:
                features["is_multi_hop"] = True

        # Short queries.
        if len(q.split()) <= 3:
            signals.short_query = True
            signals.weak_signals += 1

        # Ambiguity.
        for marker in ["something", "thing", "that", "this", "it", "stuff", "random", "etc"]:
            if re.search(rf"\b{re.escape(marker)}\b", q):
                signals.ambiguity_hits += 1

        if "?" not in query and len(q.split()) <= 6:
            signals.ambiguity_hits += 1

        # Deterministic concept extraction.
        targets.extend(self._extract_targets(query))
        concepts.extend(self._extract_semantic_concepts(query))

        # Case-application cue for real-life situations.
        if re.search(r"\bmy\b|\bme\b|\bsomeone\b|\bhe\b|\bshe\b|\bthey\b|\bminor\b|\bfriend\b", q):
            if features["requires_reasoning"] or features["is_multi_hop"] or intent in {"hypothetical", "consequence", "procedure"}:
                secondary.append("case_application")

        if signals.multi_part_hits > 0 and "case_application" not in secondary:
            secondary.append("case_application")

        rule_conf = self._compute_rule_confidence(
            signals=signals,
            primary_intent=intent,
            target_count=len(targets),
            concept_count=len(concepts),
        )

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

    def _llm_analysis(self, query: str, rule_hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = self._call_llm(query, rule_hint)
        parsed = self._safe_parse_json(raw)
        if parsed is None:
            return None

        parsed = self._normalize_llm_output(parsed, query, rule_hint)
        if not self._looks_valid(parsed):
            return None

        return parsed

    def _call_llm(self, query: str, rule_hint: Dict[str, Any]) -> Any:
        prompt = self._build_prompt(query, rule_hint)
        return self.llm_client.generate(model=self.llm_model, prompt=prompt, response_format="json")

    def _build_prompt(self, query: str, rule_hint: Dict[str, Any]) -> str:
        return f"""
You are Phase 8: Query Analysis for a retrieval system.

The user writes in natural language, not legal vocabulary.
Your job is to infer the real user need, not just legal words.

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
  "confidence": 0.0
}}

Allowed intents:
{json.dumps(FINAL_INTENTS, ensure_ascii=False)}

Important guidance:
- Choose the MOST ACTIONABLE intent.
- Do NOT default to legal_scope just because the situation is legal.
- If the user asks "what can I do" or "what should I do", prefer procedure or case_application.
- If the user asks "what will happen" or "what could happen", prefer consequence.
- If the user asks about a real-life situation like bullying, cheating, threats, assault, theft, or harassment, map the language to legal concepts such as harassment, intimidation, assault, fraud, theft, negligence, or consent.
- If the user asks for a law's meaning, use definition.
- If the user asks how a law applies to their situation, use case_application.
- If the user asks for steps to take, use procedure.
- For concepts retrieval analyze the semantic meaning of the user's query, not a blind keyword copy.
- Keep concepts short and retrieval-friendly.
- Do not invent section numbers unless the user explicitly mentions them.
- NEVER output generic words like "thing", "case", "law", "does", "mean", "between".
- Convert phrases into proper legal concepts:
  "false case" -> "false complaint"
  "drinking and driving" -> "drunk driving"
  "took money not returning" -> "criminal breach of trust"
- If query describes a real-life situation, prefer intent = "case_application" unless it is clearly lookup or definition.
- Convert layman entity words to legal entities:
  "minor" -> "child"
  "police" -> "officer in charge"
- Output ONLY JSON.

Examples:
User: "A guy in my college bullying me what can i do for this?"
Intent: procedure or case_application
Concepts: ["bullying", "harassment", "intimidation"]

User: "I accidentally hit a car while reversing what could happen?"
Intent: consequence
Concepts: ["accident", "negligence"]

User: "Someone took my phone without asking"
Intent: case_application
Concepts: ["theft", "property"]

User: "What is theft?"
Intent: definition
Concepts: ["theft"]

User query:
{query}

Rule-based hint:
{json.dumps(rule_hint, ensure_ascii=False)}
"""

    def _llm_decompose_query(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        prompt = f"""
You split a legal search query into smaller retrieval-friendly queries.

Return exactly this JSON:
{{
  "sub_queries": [
    "query 1",
    "query 2"
  ]
}}

Rules:
- Output 2 to {self.max_sub_queries} sub-queries only when decomposition is useful.
- Preserve the user's legal meaning.
- Make each sub-query short, concrete, and retrieval-friendly.
- Keep statutory references if present.
- Do not output duplicates.
- Do not output commentary.
- If decomposition is not useful, return the original query as a single item.
- For real-life situations, split into:
  1) the underlying legal concept
  2) the likely legal consequence / procedure / exception
- For conditional queries, split condition and outcome when that helps retrieval.

Original query:
{query}

Current analysis:
{json.dumps(analysis, ensure_ascii=False)}
"""
        raw = self.llm_client.generate(model=self.llm_model, prompt=prompt, response_format="json")
        parsed = self._safe_parse_json(raw) or {}
        sub_queries = parsed.get("sub_queries", [])
        if not isinstance(sub_queries, list):
            return []
        return [str(item).strip() for item in sub_queries if str(item).strip()]

    def _rule_decompose_query(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        q = query.strip()
        q_lower = q.lower()
        sub_queries: List[str] = []

        compare_match = re.search(
            r"\b(?:difference between|compare)\s+(.+?)\s+\b(?:and|vs|versus)\b\s+(.+)",
            q,
            re.IGNORECASE,
        )
        if compare_match:
            left = self._normalize_sub_query_fragment(compare_match.group(1))
            right = self._normalize_sub_query_fragment(compare_match.group(2))
            if left:
                sub_queries.append(f"Explain {left}")
            if right:
                sub_queries.append(f"Explain {right}")
            return sub_queries

        section_match = self._extract_section_like(q)
        if section_match and re.search(r"\b(exception|exceptions|punishment|penalty|condition|conditions)\b", q_lower):
            sub_queries.append(f"Explain {section_match}")
            if re.search(r"\bexception|exceptions\b", q_lower):
                sub_queries.append(f"What are the exceptions to {section_match}?")
            if re.search(r"\bpunishment|penalty\b", q_lower):
                sub_queries.append(f"What is the punishment under {section_match}?")
            if re.search(r"\bcondition|conditions\b", q_lower):
                sub_queries.append(f"What conditions apply under {section_match}?")
            return sub_queries

        # Explicit conditional-to-consequence split.
        if re.search(r"\bif\b", q_lower) and re.search(r"\bwhat (will|would|could|can) happen\b|\bwhat happens\b", q_lower):
            cond = re.split(r"\bwhat (will|would|could|can) happen\b|\bwhat happens\b", q, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,?")
            if cond:
                sub_queries.append(self._normalize_sub_query_fragment(cond))
            sub_queries.append(self._normalize_sub_query_fragment(q))
            return sub_queries

        chunks = re.split(r"\b(?:and also|also|plus|and|but)\b", q, flags=re.IGNORECASE)
        if len(chunks) > 1:
            anchor = ""
            concepts = analysis.get("concepts", []) or []
            targets = analysis.get("targets", []) or []
            if concepts:
                anchor = str(concepts[0]).strip()
            elif targets:
                anchor = str(targets[0].get("value", "")).strip()

            for chunk in chunks:
                cleaned = self._normalize_sub_query_fragment(chunk)
                if cleaned:
                    if anchor and re.search(r"\b(caught|happen|happens|happened|apply|applies|punishment|penalty)\b", cleaned, re.IGNORECASE):
                        if anchor.lower() not in cleaned.lower():
                            cleaned = f"{cleaned} for {anchor}"
                    if "?" not in cleaned and not re.search(
                        r"^(what|how|why|when|who|which|can|is|are|explain|define|summarize)\b",
                        cleaned,
                        re.IGNORECASE,
                    ):
                        cleaned = f"Explain {cleaned}"
                    sub_queries.append(cleaned)
            return sub_queries

        if bool((analysis.get("query_features", {}) or {}).get("is_multi_hop", False)):
            query_without_tail = re.split(
                r"\b(?:then|what happens|what will happen|what could happen)\b",
                q,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" ,?")
            if query_without_tail and query_without_tail != q:
                sub_queries.append(query_without_tail)
                sub_queries.append(q)

        return sub_queries

    def _normalize_sub_query_fragment(self, text: str) -> str:
        cleaned = self._normalize_query(text.strip(" ,?"))
        cleaned = re.sub(r"^(then|also)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _clean_sub_queries(self, original_query: str, sub_queries: List[str]) -> List[str]:
        original_norm = self._normalize_query(original_query).lower()
        cleaned: List[str] = []
        seen = {original_norm}

        for sub_query in sub_queries:
            candidate = self._normalize_query(sub_query)
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            if len(candidate.split()) < 2:
                continue
            seen.add(lowered)
            cleaned.append(candidate)

        return cleaned[: self.max_sub_queries]

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

        try:
            conf = float(obj.get("confidence", 0.0))
            if not (0.0 <= conf <= 1.0):
                conf = 0.0
        except Exception:
            conf = 0.0
        obj["confidence"] = conf
        return obj

    def _merge_rule_and_llm(self, rule_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(llm_result)
        merged["query"] = rule_result["query"]
        merged["targets"] = merged.get("targets") or rule_result["targets"]
        merged["concepts"] = merged.get("concepts") or rule_result.get("concepts", [])
        merged["constraints"] = merged.get("constraints") or rule_result["constraints"]
        merged["query_features"] = merged.get("query_features") or rule_result["query_features"]

        llm_conf = self._safe_float(merged.get("confidence", 0.0), 0.0)
        rule_conf = self._safe_float(rule_result["confidence"]["rule_based"], 0.0)
        overall = self._combine_confidence(rule_conf, llm_conf, 0.5 if llm_conf > 0 else 0.0)

        merged["confidence"] = {
            "rule_based": round(self._clamp(rule_conf), 2),
            "llm": round(self._clamp(llm_conf), 2),
            "agreement": 0.5 if llm_conf > 0 else 0.0,
            "overall": round(self._clamp(overall), 2),
        }
        merged["notes"] = rule_result.get("notes", [])
        return merged

    def _validate_and_fix(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return self._validate_result_payload(result, include_sub_queries=True)

    def _validate_result_payload(self, result: Dict[str, Any], include_sub_queries: bool) -> Dict[str, Any]:
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
        clean_concepts = [str(x).strip().lower() for x in concepts if isinstance(x, str) and x.strip()]
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
        if include_sub_queries and isinstance(sub_queries, list):
            for item in sub_queries[: self.max_sub_queries]:
                if not isinstance(item, dict):
                    continue
                fixed = self._validate_result_payload(dict(item), include_sub_queries=False)
                fixed["decomposition"] = {"needed": False, "strategy": "none", "reason": "", "count": 0}
                fixed["sub_queries"] = []
                clean_sub_queries.append(fixed)
        result["sub_queries"] = clean_sub_queries
        result["decomposition"]["count"] = len(clean_sub_queries)
        result["decomposition"]["needed"] = len(clean_sub_queries) >= 2
        if not result["decomposition"]["needed"]:
            result["decomposition"]["strategy"] = "none"

        result["method"] = str(result.get("method", "rules"))
        return result

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

    def _extract_section_like(self, query: str) -> Optional[str]:
        m = re.search(r"\b(section|article|rule)\s+\d+[a-z]?(?:\(\d+\))*\b", query, re.IGNORECASE)
        return m.group(0) if m else None

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
            (r"\bminor\b|\bchild\b|\bjuvenile\b", "juvenile"),
            (r"\bfalse case\b|\bfalse complaint\b", "false complaint"),
            (r"\bphone\b|\bmobile\b|\bdevice\b", "property"),
        ]

        for pattern, concept in soft_patterns:
            if re.search(pattern, q):
                concepts.append(concept)

        if re.search(r"\bwhat could happen\b|\bwhat will happen\b|\bwhat happens\b", q):
            concepts.append("legal consequences")

        if re.search(r"\bwhat can i do\b|\bwhat should i do\b|\bwhat do i do\b", q):
            concepts.append("legal remedy")

        if not concepts:
            concepts.append("legal issue")

        return self._dedupe_list(concepts)

    def _combine_confidence(self, rule_conf: float, llm_conf: float, agreement: float) -> float:
        return 0.30 * rule_conf + 0.45 * llm_conf + 0.25 * agreement

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

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
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    llm_client = OpenAIClient(api_key=api_key) if api_key else None

    analyzer = QueryAnalyzer(
        llm_client=llm_client,
        llm_model="gpt-4o-mini",
        enable_query_decomposition=True,
        max_sub_queries=4,
    )

    test_queries = [
        "If five persons conjointly commit robbery and one of them commits murder, how doe they will be punished?"
    ]
    
    # test_queries = [
    #     "A guy in my college bullying me what can i do for this?",
    #     "I accidentally hit a car while reversing what could happen?",
    #     "What is section 375 and its exceptions?",
    #     "Difference between murder and culpable homicide",
    #     "If someone forces consent later what law applies",
    #     "If a minor steals something what will happen?",
    # ]

    results = []
    for q in test_queries:
        result = analyzer.analyze(q)
        results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("=" * 80)

    result_path = os.path.join("query_analysis", "result3.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {result_path}")
