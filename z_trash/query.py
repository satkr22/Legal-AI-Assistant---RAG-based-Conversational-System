from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llama_cpp import Llama
from Llama.generation.generator import LLM


MODEL_PATH = "/home/usatkr/llm/models/llama-3.1-8b/llama-3.1-8b-q4_k_m.gguf"

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

TARGET_TYPES = ["section", "subsection", "clause", "explanation", "illustration", "definition", "concept", "act"]
SPECIFICITY_ENUM = ["broad", "narrow"]


class LlamaCppClient:
    def __init__(self, model_path: str):
        self.llama = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False,
        )
        self.generator = LLM(self.llama)

    def generate(self, model: str, prompt: str, response_format: str = "json") -> str:
        formatted_prompt = self.generator.create_evaluation_prompt(
            question=prompt,
            system_msg=(
                "You are a query understanding engine. "
                "Output ONLY valid JSON. No explanation. No markdown. No extra text."
            ),
        )

        output = self.generator.generate(
            formatted_prompt,
            max_tokens=512,
            temperature=0.05,
        )

        if isinstance(output, str):
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if match:
                return match.group(0)
            return output
        return json.dumps(output, ensure_ascii=False)


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
    notes: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """
    Phase 8: natural-language query understanding for retrieval.

    This version keeps rules narrow and lets the LLM do the semantic work.
    The main output for Phase 9 is:
      - intent
      - concepts
      - targets
      - a few boolean query features
    """

    def __init__(
        self,
        llm_client: Optional[LlamaCppClient] = None,
        llm_model: str = "llama-3.1",
        rule_high_threshold: float = 0.78,
        rule_use_llm_threshold: float = 0.50,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.rule_high_threshold = rule_high_threshold
        self.rule_use_llm_threshold = rule_use_llm_threshold

    def analyze(self, query: str) -> Dict[str, Any]:
        query = self._normalize_query(query)
        rule_result = self._rule_based_analysis(query)

        if rule_result["confidence"]["rule_based"] >= self.rule_high_threshold or self.llm_client is None:
            rule_result["method"] = "rules"
            return self._validate_and_fix(rule_result)

        if rule_result["confidence"]["rule_based"] <= self.rule_use_llm_threshold:
            llm_result = self._llm_analysis(query, rule_result)
            if llm_result is not None:
                merged = self._merge_rule_and_llm(rule_result, llm_result)
                merged["method"] = "llm_guided"
                return self._validate_and_fix(merged)

        rule_result["method"] = "rules_fallback"
        return self._validate_and_fix(rule_result)

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

        # Exact statutory references are safe to handle deterministically.
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

        if re.search(r"\bhow to\b|\bsteps to\b|\bprocedure\b|\bprocess\b|\bfile\b|\bapply\b", q):
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

        if re.search(r"\bwhy\b|\breason\b|\bcause\b|\beffect\b|\bimpact\b|\bconsequence\b", q):
            if intent in {"explanation", "definition"}:
                intent = "reasoning"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        if re.search(r"\bwhat if\b|\bsuppose\b|\bin case\b|\bif .* then\b", q):
            intent = "hypothetical"
            signals.medium_signals += 1
            features["requires_reasoning"] = True
            features["is_multi_hop"] = True

        if re.search(r"\bsummary\b|\bsummarize\b|\bbrief\b|\boverview\b|\bgist\b", q):
            intent = "summary"
            signals.medium_signals += 1

        if re.search(r"\bipc\b|\bbns\b|\bcrpc\b|\bcc\b|\bconstitution\b|\bindian\b", q):
            constraints["jurisdiction"] = "india"
            signals.medium_signals += 1
            notes.append("Detected Indian legal context.")
            if intent in {"definition", "explanation"}:
                intent = "legal_scope"

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

        if re.search(r"\bwhat could happen\b|\bwhat will happen\b|\bwhat happens\b|\bwhat would happen\b|\bcan happen\b", q):
            if intent in {"explanation", "definition"}:
                intent = "consequence"
            signals.medium_signals += 1
            features["requires_reasoning"] = True

        if re.search(r"\bpunishment\b|\bpenalty\b|\bsentence\b|\bfine\b|\bimprisonment\b|\bliable to punishment\b", q):
            intent = "legal_penalty"
            signals.strong_signals += 1
            constraints["specificity"] = "narrow"
            features["requires_exact_match"] = True

        if any(token in q for token in [" and ", " also ", " plus ", " or "]):
            signals.multi_part_hits += 1
            if len(re.findall(r"\b(and|also|plus|or)\b", q)) >= 2:
                features["is_multi_hop"] = True

        if len(q.split()) <= 3:
            signals.short_query = True
            signals.weak_signals += 1

        for marker in ["something", "thing", "that", "this", "it", "stuff", "random", "etc"]:
            if re.search(rf"\b{re.escape(marker)}\b", q):
                signals.ambiguity_hits += 1

        if "?" not in query and len(q.split()) <= 6:
            signals.ambiguity_hits += 1

        targets.extend(self._extract_targets(query))
        concepts.extend(self._extract_semantic_concepts(query))

        if signals.multi_part_hits > 0:
            secondary.append("case_application")

        rule_conf = self._compute_rule_confidence(
            signals=signals,
            primary_intent=intent,
            target_count=len(targets),
            concept_count=len(concepts),
        )

        return {
            "query": query,
            "intent": {"primary": intent, "secondary": self._dedupe_list([s for s in secondary if s != intent])},
            "targets": self._dedupe_targets(targets),
            "concepts": self._dedupe_list(concepts),
            "constraints": constraints,
            "query_features": features,
            "confidence": {"rule_based": round(rule_conf, 2), "llm": 0.0, "agreement": 0.0, "overall": round(rule_conf, 2)},
            "method": "rules",
            "notes": notes,
        }

    def _compute_rule_confidence(self, signals: RuleSignals, primary_intent: str, target_count: int, concept_count: int) -> float:
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
- For concepts retrieval analyse the semantic meaning of the user's query not blindly extract words from user's query.
- Keep concepts short and retrieval-friendly.
- Do not invent section numbers unless the user explicitly mentions them.
- NEVER output generic words like "thing", "case", "law", "does", "mean", "between".
- Convert phrases into proper legal concepts:
  "false case" → "false complaint"
  "drinking and driving" → "drunk driving"
  "took money not returning" → "criminal breach of trust"
  - If query describes a real-life situation → intent MUST be "case_application"
  - Convert layman entity words to legal entities:
  "minor" -> "child" or "juvenile"
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
    # harassment / bullying
    (r"\bbully\b|\bbullying\b|\bharass\b|\bharassment\b|\bthreat\b|\bthreatening\b", "harassment"),

    # accident / negligence
    (r"\bhit\b|\baccident\b|\breversing\b|\bcrash\b|\bcollision\b", "accident"),

    # theft / property
    (r"\bsteal\b|\bstole\b|\btaken\b|\btook\b|\btheft\b|\brobbery\b", "theft"),

    # assault / violence
    (r"\bforce\b|\bforced\b|\bassault\b|\bviolence\b|\bpush\b|\bpushed\b", "assault"),

    # fraud / cheating
    (r"\bcheat\b|\bcheated\b|\bscam\b|\bfraud\b|\bblackmail\b", "fraud"),

    # consent
    (r"\bconsent\b|\bagree\b|\bagreed\b|\bpermission\b", "consent"),

    # negligence
    (r"\bnegligent\b|\bcareless\b|\black of care\b", "negligence"),

    # punishment
    (r"\bpunish\b|\bpunishment\b|\bpenalty\b|\bfine\b|\bimprisonment\b", "punishment"),

    # property damage
    (r"\bbroke\b|\bdamage\b|\bdestroy\b", "property damage"),

    # drunk driving
    (r"\bdrinking\b.*\bdriving\b|\bdrunk\b.*\bdriving\b", "drunk driving"),
]
        for pattern, concept in soft_patterns:
            if re.search(pattern, q):
                concepts.append(concept)

        stopwords = {
            "what", "when", "where", "why", "how", "which", "that", "this", "there",
            "have", "been", "will", "would", "could", "should", "about", "from", "into",
            "with", "without", "then", "them", "they", "their", "your", "you", "i", "me",
            "him", "her", "our", "was", "were", "are", "am", "can", "may", "might",
            "happen", "happens", "happening", "now", "please", "tell", "explain", "do",
            "for", "of", "a", "an", "the", "and", "or", "to", "in", "on", "at", "my",
        }
        
        if not concepts:
            concepts.append("legal issue")
        # words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", q)
        # for w in words:
        #     if w not in stopwords and w not in concepts:
        #         if w not in {"college", "guy", "bullying", "accidentally", "reversing", "someone"}:
        #             concepts.append(w)

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
    llm_client = LlamaCppClient(MODEL_PATH)
    analyzer = QueryAnalyzer(llm_client=llm_client, llm_model="llama-3.1")

    tests = [
        "A guy in my college bullying me what can i do for this?",
        "I accidentally hit a car while reversing what could happen?",
        "What is section 375 and its exceptions?",
        "Difference between murder and culpable homicide",
        "If someone forces consent later what law applies",
    ]
    
#     test_queries = [

#     # 1. Definition
#     "What is theft?",
#     "Define murder",
#     "Meaning of consent in law",
#     "What does negligence mean?",
#     "What is cheating under IPC?",

#     # # 2. Section / Lookup
#     "What is section 375?",
#     "Explain section 420 IPC",
#     "Section 300 definition",
#     "What does section 34 say?",
#     "Section 302 punishment",

#     # # 3. Legal Exception / Condition
#     "What are the exceptions to section 375?",
#     "When does section 300 not apply?",
#     "Conditions for self defence",
#     "When is consent not valid?",
#     "Where does this law not apply?",

#     # 4. Consequence
#     "I hit a bike accidentally what will happen?",
#     "If I steal something what can happen?",
#     "What happens if someone files a false case?",
#     "I broke someone's phone what could happen?",
#     "If police catch me drinking and driving what happens?",

#     # 5. Procedure
#     "What should I do if someone is threatening me?",
#     "How to file an FIR?",
#     "Steps to report cyber fraud",
#     "What can I do if someone stole my phone?",
#     "How to complain against harassment?",

#     # # 6. Case Application
#     "My friend took my money and is not returning it",
#     "Someone is blackmailing me online",
#     "A guy is spreading rumors about me in college",
#     "My landlord is forcing me to leave",
#     "Someone used my photo without permission",

#     # # 7. Comparison
#     "Difference between theft and robbery",
#     "Compare murder and culpable homicide",
#     "What is the difference between IPC and BNS?",
#     "Fraud vs cheating",
#     "Assault vs battery",

#     # 8. Hypothetical
#     "What if someone kills in self defence?",
#     "If I hit someone by mistake is it a crime?",
#     "Suppose a minor commits theft what happens?",
#     "If consent is taken later is it valid?",
#     "What if both people agree but law disagrees?",

#     # # 9. Eligibility
#     "Who can file an FIR?",
#     "Am I allowed to record a phone call?",
#     "Can a minor be punished?",
#     "Who is eligible for bail?",
#     "Can police arrest without warrant?",

#     # # 10. Reasoning
#     "Why is murder punished severely?",
#     "Why is consent important in law?",
#     "Reason behind strict cyber laws",
#     "Why is bail granted?",
#     "Why is negligence considered crime?",

#     # 11. Multi-intent
#     "What is section 420 and what is its punishment?",
#     "Explain theft and also what happens if caught",
#     # "What is IPC and how does it apply?",
#     "Difference between fraud and what punishment applies",
#     "What is consent and when is it invalid?",

#     # # 12. Ambiguous / Noisy
#     "Something happened with my phone what can I do",
#     "That guy did something illegal to me",
#     "I think I got cheated maybe not sure",
#     "Problem with police what to do",
#     "Random issue happened need help",

#     # # # 13. Short queries
#     "theft law",
#     "section 420",
#     "punishment murder",
#     "consent valid?",
#     "fraud meaning",

#     # # 14. Adversarial
#     "He didn't exactly steal but took it without asking",
#     "It wasn't forced but I felt pressured",
#     "He didn't hit me but pushed me",
#     "Money was taken but returned later",
#     "She agreed but later said no",
# ]

    test_queries = ["what if i kill someone in my defence then what will happen?"]
    
    results = []

    for q in test_queries:
        result = analyzer.analyze(q)
        results.append(result)
        print("done")
        
    result_path = "query_analysis/result.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # for q in test_queries:
    #     print("\n" + "=" * 80)
    #     print("QUERY:", q)
    #     # with open(result_path, "w") as f:
    #     #     result = analyzer.analyze(q)
    #     #     json.dump(result, f, indent=2, ensure_ascii=False)
    #     print(json.dumps(analyzer.analyze(q), indent=2, ensure_ascii=False))
    #     print("done")









# API_KEY = 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 

# export OPENAI_API_KEY="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"