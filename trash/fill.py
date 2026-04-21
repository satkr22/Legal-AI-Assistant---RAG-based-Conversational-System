#!/usr/bin/env python3
"""Optimized legal-document enrichment for BNS-style structured JSON.

Goals:
- preserve every existing non-empty field
- drastically reduce LLM calls via batching
- fill missing section/node summaries with retries
- infer node_type deterministically so it does not stay null
- stay safe on 6 GB VRAM with a 4k context window
"""

# python ingestion/enrichment/fill.py --input data/processed/jsons/enriched_jsons/context.json --enums data/processed/jsons/json_blueprints/enr_enums.json --output data/processed/jsons/enriched_jsons/context_filled2.json

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from llama_cpp import Llama
except Exception as exc:  # pragma: no cover - import-time fail in the local test env is expected
    Llama = None  # type: ignore
    _LLAMA_IMPORT_ERROR = exc
else:
    _LLAMA_IMPORT_ERROR = None


MODEL_PATH_DEFAULT = "/home/usatkr/llm/models/llama-3.1-8b/llama-3.1-8b-q4_k_m.gguf"

SYSTEM_MSG = (
    "You are a strict legal-document enrichment assistant. "
    "Use only the provided context and metadata. "
    "Do not use raw source text outside the provided context, and do not invent facts. "
    "Return valid JSON only."
)

DEFAULT_NODE_TYPES = {
    "definition",
    "rule",
    "exception",
    "explanation",
    "illustration",
    "punishment",
    "condition",
    "provison",
}

DEFAULT_SECTION_TYPES = {
    "definition",
    "offence",
    "punishment",
    "procedure",
    "general_rule",
}

ALIAS_MAP = {
    "provision": "provison",
    "proviso": "provison",
}

GENERIC_WORDS = {
    "thing", "things", "person", "persons", "section", "subsection", "clause",
    "act", "law", "legal", "case", "matter", "provision", "provisions",
    "document", "documents", "rule", "rules", "content", "body",
}

SUMMARY_MAX_CHARS = 220
PARAPHRASE_MAX_CHARS = 240
REPRESENTATIVE_CONTEXT_CHARS = 260
NODE_CONTEXT_CHARS = 360
SECTION_BATCH_MAX_ITEMS = 6
NODE_BATCH_MAX_ITEMS = 5
SECTION_BATCH_MAX_CHARS = 3200
NODE_BATCH_MAX_CHARS = 2800


# ----------------------------- IO / utilities -----------------------------

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def normalize_phrase(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r\"'`.,;:()[]{}<>")
    return text


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        norm = normalize_phrase(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def clean_keyword_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, list):
        raw = values
    elif isinstance(values, str):
        raw = re.split(r"[\n,;]+", values)
    else:
        raw = [values]

    out: List[str] = []
    for v in raw:
        kw = normalize_phrase(v)
        if not kw:
            continue
        if len(kw) < 3:
            continue
        if kw in GENERIC_WORDS:
            continue
        out.append(kw)
    return dedupe_preserve_order(out)


def coerce_enum(value: Any, allowed: Set[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = normalize_phrase(value)
    candidate = ALIAS_MAP.get(candidate, candidate)
    return candidate if candidate in allowed else None


def safe_extract_json_value(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    # First try full string.
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass

    # Then try to extract the first full JSON object or array.
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, flags=re.S)
        if not match:
            continue
        chunk = match.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(chunk)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    raise ValueError(f"Could not parse JSON from model output:\n{text[:1200]}")


def normalize_result_payload(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list):
        return {"items": obj}
    if isinstance(obj, dict):
        for key in ("items", "sections", "nodes", "results", "output"):
            val = obj.get(key)
            if isinstance(val, list):
                return {"items": val}
    raise ValueError("Model output did not contain a JSON list of items.")


def truncate(text: Any, max_chars: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ----------------------------- heuristics -----------------------------

def load_allowed_enums(enum_path: str | Path) -> Tuple[Set[str], Set[str]]:
    data = load_json(enum_path)
    node_types = set(DEFAULT_NODE_TYPES)
    section_types = set(DEFAULT_SECTION_TYPES)

    if isinstance(data, dict):
        if isinstance(data.get("node_type"), list):
            node_types = {
                ALIAS_MAP.get(normalize_phrase(x), normalize_phrase(x))
                for x in data["node_type"]
                if str(x).strip()
            }
        if isinstance(data.get("section_type"), list):
            section_types = {
                ALIAS_MAP.get(normalize_phrase(x), normalize_phrase(x))
                for x in data["section_type"]
                if str(x).strip()
            }

    return node_types, section_types


def infer_section_type(section: Dict[str, Any]) -> Optional[str]:
    title = normalize_phrase(section.get("section_title", ""))
    if not title:
        return None
    if "definition" in title:
        return "definition"
    if "punishment" in title or "punishments" in title:
        return "punishment"
    if "procedure" in title:
        return "procedure"
    if "offence" in title or "offences" in title or "offense" in title:
        return "offence"
    if any(tok in title for tok in ("general", "application", "preliminary", "short title")):
        return "general_rule"
    return None


def looks_like_definition(text: str) -> bool:
    t = normalize_phrase(text)
    if not t:
        return False
    if re.match(r'^["“].+?["”]\s*(means|denotes|includes|shall mean|is said to|is said to be)\b', t):
        return True
    if re.match(r'^["“][^"”]+["”].{0,12}\b(means|denotes|includes)\b', t):
        return True
    if " means " in f" {t} " or " denotes " in f" {t} " or " includes " in f" {t} ":
        # Avoid classifying a general sentence as a definition unless it looks term-led.
        if t.startswith(("“", '"')) or re.match(r'^[a-z][a-z\s\-]{0,40}\b(means|denotes|includes)\b', t):
            return True
    return False


def looks_like_condition(text: str) -> bool:
    t = normalize_phrase(text)
    if not t:
        return False
    return bool(
        re.match(r"^(if|when|whenever|where|wherever|in case|provided that|provided further that)\b", t)
        or t.startswith("subject to")
    )


def looks_like_exception(text: str) -> bool:
    t = normalize_phrase(text)
    if not t:
        return False
    return bool(
        re.match(r"^(except|except that|except as|notwithstanding|nothing in this|nothing in the|no\b)\b", t)
        or "shall not" in t
    )


def looks_like_proviso(text: str) -> bool:
    t = normalize_phrase(text)
    if not t:
        return False
    return bool(re.match(r"^(provided that|provided further that)\b", t))


def looks_like_punishment(text: str) -> bool:
    t = normalize_phrase(text)
    if not t:
        return False
    return bool(
        "liable to punishment" in t
        or "punishable" in t
        or "imprisonment" in t
        or "fine" in t and ("shall be" in t or "liable" in t)
    )


def infer_node_type(node: Dict[str, Any], allowed_node_types: Set[str], fallback_rule: bool = True) -> Optional[str]:
    current = normalize_phrase(node.get("node_type"))
    if current:
        current = ALIAS_MAP.get(current, current)
        if current in allowed_node_types:
            return current

    category = normalize_phrase(node.get("node_category"))
    label = normalize_phrase(node.get("node_label"))
    text = str(node.get("text") or node.get("derived_context") or "")

    if category == "explanation" or "explanation" in label:
        return "explanation" if "explanation" in allowed_node_types else None
    if category == "illustration" or "illustration" in label:
        return "illustration" if "illustration" in allowed_node_types else None
    if category == "exception" or label.startswith("except") or looks_like_exception(text):
        return "exception" if "exception" in allowed_node_types else None
    if looks_like_proviso(text):
        return "provison" if "provison" in allowed_node_types else None
    if looks_like_definition(text):
        return "definition" if "definition" in allowed_node_types else None
    if looks_like_punishment(text):
        return "punishment" if "punishment" in allowed_node_types else None
    if looks_like_condition(text):
        return "condition" if "condition" in allowed_node_types else None

    if fallback_rule and category in {"subsection", "clause", "content", "body"}:
        return "rule" if "rule" in allowed_node_types else None

    return None


# ----------------------------- context building -----------------------------

def iter_sections(data: Dict[str, Any]) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for chapter in data.get("chapters", []) or []:
        for section in chapter.get("sections", []) or []:
            yield chapter, section


def iter_nodes(nodes: Sequence[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for node in nodes or []:
        yield node
        for child in node.get("children", []) or []:
            yield from iter_nodes([child])


def count_nodes(nodes: Sequence[Dict[str, Any]]) -> int:
    total = 0
    for node in nodes or []:
        total += 1
        total += count_nodes(node.get("children", []) or [])
    return total


def representative_nodes_for_section(section: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    reps: List[Dict[str, Any]] = []
    for node in iter_nodes(section.get("nodes", []) or []):
        text = str(node.get("derived_context") or node.get("text") or "").strip()
        if not text:
            continue
        reps.append(node)
        if len(reps) >= limit:
            break
    return reps


def build_section_context(act: str, chapter: Dict[str, Any], section: Dict[str, Any]) -> str:
    parts = [
        f"Act: {act}",
        f"Chapter {chapter.get('chapter_number')}: {chapter.get('chapter_title')}",
        f"Section {section.get('section_number')}: {section.get('section_title')}",
    ]

    if is_nonempty(section.get("section_type")):
        parts.append(f"Existing section_type: {section.get('section_type')}")
    if is_nonempty(section.get("semantic_summary")):
        parts.append(f"Existing semantic_summary: {section.get('semantic_summary')}")

    reps = representative_nodes_for_section(section, limit=3)
    if reps:
        parts.append("Representative node contexts:")
        for idx, node in enumerate(reps, 1):
            node_text = truncate(node.get("derived_context") or node.get("text") or "", REPRESENTATIVE_CONTEXT_CHARS)
            parts.append(f"{idx}. [{node.get('node_id')}] {node_text}")

    return "\n".join(parts)


def build_node_context(act: str, chapter: Dict[str, Any], section: Dict[str, Any], node: Dict[str, Any]) -> str:
    parts = [
        f"Act: {act}",
        f"Chapter {chapter.get('chapter_number')}: {chapter.get('chapter_title')}",
        f"Section {section.get('section_number')}: {section.get('section_title')}",
    ]
    if is_nonempty(section.get("section_type")):
        parts.append(f"Section type: {section.get('section_type')}")
    if is_nonempty(section.get("semantic_summary")):
        parts.append(f"Section summary: {section.get('semantic_summary')}")

    node_text = node.get("derived_context")
    if not is_nonempty(node_text):
        # Fallback only when the derived_context is absent.
        node_text = node.get("text")
    parts.append(f"Node: [{node.get('node_id')}] {node.get('node_label')} | {node.get('node_category')} | {node.get('level')}")
    parts.append(f"Context: {truncate(node_text or '', NODE_CONTEXT_CHARS)}")
    return "\n".join(parts)


# ----------------------------- LLM wrapper -----------------------------

class LocalLlamaJSONClient:
    def __init__(
        self,
        model_path: str,
        n_ctx: int,
        n_threads: int,
        n_gpu_layers: int,
        n_batch: int,
        verbose: bool = False,
    ) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama_cpp is not available in this environment. "
                f"Original import error: {_LLAMA_IMPORT_ERROR}"
            )

        init_attempts = [
            dict(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers, n_batch=n_batch, verbose=verbose),
            dict(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers, verbose=verbose),
            dict(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=verbose),
        ]
        last_exc: Optional[Exception] = None
        self.model = None
        for kwargs in init_attempts:
            try:
                self.model = Llama(**kwargs)  # type: ignore[arg-type]
                break
            except TypeError as exc:
                last_exc = exc
                continue
        if self.model is None:
            raise RuntimeError(f"Could not initialize Llama. Last error: {last_exc}")

    def _chat_once(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        assert self.model is not None
        if hasattr(self.model, "create_chat_completion"):
            try:
                response = self.model.create_chat_completion(
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                )
                return response["choices"][0]["message"]["content"]
            except TypeError:
                pass
            except Exception:
                # fall back to completion below
                pass

        prompt = self._messages_to_prompt(messages)
        response = self.model.create_completion(  # type: ignore[attr-defined]
            prompt=prompt,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            stop=["```"],
        )
        return response["choices"][0]["text"]

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        # Minimal Llama-3-style fallback prompt.
        chunks = ["<|begin_of_text|>"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            chunks.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}\n<|eot_id|>")
        chunks.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(chunks)

    def chat_json(self, system_msg: str, user_msg: str, max_tokens: int, retries: int = 2) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        for attempt in range(retries):
            try:
                text = self._chat_once(messages, max_tokens=max_tokens)
                payload = normalize_result_payload(safe_extract_json_value(text))
                return payload
            except Exception as exc:
                last_error = exc
                if attempt == retries - 1:
                    break
                # Strengthen the prompt on retry without changing the task.
                messages = [
                    {"role": "system", "content": system_msg + " Be stricter: output only valid JSON and nothing else."},
                    {"role": "user", "content": user_msg + "\n\nReturn only a single valid JSON value. No markdown, no commentary."},
                ]
        raise RuntimeError(f"LLM JSON generation failed after retries: {last_error}")


# ----------------------------- prompt builders -----------------------------

def build_section_batch_prompt(
    sections: List[Dict[str, Any]],
    allowed_section_types: Sequence[str],
) -> str:
    schema = {
        "items": [
            {
                "section_id": "...",
                "semantic_summary": "...",
                "plain_english_paraphrase": "...",
                "additional_keywords": [],
                "legal_concepts": [],
                "section_type": "...",
            }
        ]
    }

    payload_items = []
    for sec in sections:
        payload_items.append(
            {
                "section_id": sec["section_id"],
                "section_number": sec.get("section_number"),
                "section_title": sec.get("section_title"),
                "current_section_type": sec.get("section_type"),
                "missing_fields": sec.get("missing_fields", []),
                "context": sec["context"],
            }
        )

    user_msg = f"""
Task: enrich multiple structured legal sections.

Rules:
1. Use only the provided context and metadata.
2. Do not invent facts.
3. Do not modify any existing non-empty field.
4. If a field is already present, leave it untouched in the output object by returning it as null/empty only when you are filling missing fields.
5. semantic_summary: 1 concise sentence, at most 2.
6. plain_english_paraphrase: 1 concise sentence, at most 2.
7. additional_keywords: short phrases only.
8. legal_concepts: concise legal phrases only.
9. section_type must be one of: {compact_json(list(allowed_section_types))}; if uncertain, null.
10. Return JSON only.

Return exactly this shape:
{compact_json(schema)}

Items to enrich:
{compact_json(payload_items)}
""".strip()
    return user_msg


def build_node_batch_prompt(nodes: List[Dict[str, Any]]) -> str:
    schema = {
        "items": [
            {
                "node_id": "...",
                "semantic_summary": "...",
                "plain_english_paraphrase": "...",
                "additional_keywords": [],
                "legal_concepts": [],
            }
        ]
    }

    payload_items = []
    for item in nodes:
        node = item.get("node")
        if not isinstance(node, dict):
            continue   # <-- FIX

        payload_items.append(
            {
                "node_id": node.get("node_id"),
                "node_label": node.get("node_label"),
                "node_category": node.get("node_category"),
                "level": node.get("level"),
                "missing_fields": item.get("missing_fields", []),
                "context": item.get("context"),
            }
        )

    user_msg = f"""
Task: enrich multiple legal nodes.

Rules:
1. Use only the provided context and metadata.
2. Do not invent facts.
3. Do not modify any existing non-empty field.
4. semantic_summary: 1 concise sentence, at most 2.
5. plain_english_paraphrase: 1 concise sentence, at most 2.
6. additional_keywords: short phrases only.
7. legal_concepts: concise legal phrases only.
8. Return JSON only.

Return exactly this shape:
{compact_json(schema)}

Items to enrich:
{compact_json(payload_items)}
""".strip()
    return user_msg


# ----------------------------- merging / updating -----------------------------

def merge_list_fields(existing: Any, new_values: Any) -> List[str]:
    merged = clean_keyword_list(existing) + clean_keyword_list(new_values)
    return dedupe_preserve_order(merged)


def clamp_text(value: Any, max_chars: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def fallback_section_summary(section: Dict[str, Any]) -> str:
    title = str(section.get("section_title") or "").strip().rstrip(".")
    sec_type = normalize_phrase(section.get("section_type"))
    if sec_type == "definition":
        return "This section defines key terms used in the Act."
    if sec_type == "punishment":
        return "This section sets out punishment-related provisions."
    if sec_type == "procedure":
        return "This section lays down procedural provisions."
    if sec_type == "offence":
        return "This section deals with offence-related rules."
    if title:
        return f"This section concerns {title.lower()}."
    return "This section sets out a legal provision."


def fallback_node_summary(node: Dict[str, Any], section: Dict[str, Any]) -> str:
    text = str(node.get("text") or node.get("derived_context") or "").strip()
    text = re.sub(r"\s+", " ", text)
    if looks_like_definition(text):
        m = re.match(r'^["“](.+?)["”]\s*(?:means|denotes|includes|shall mean|is said to|is said to be)\b', text, flags=re.I)
        term = m.group(1) if m else None
        if term:
            return f'Defines the term "{term}".'
        return "Defines a legal term."
    if looks_like_proviso(text):
        return "States a proviso or qualifying condition."
    if looks_like_condition(text):
        return "States a condition for the rule to apply."
    if looks_like_exception(text):
        return "States an exception or saving clause."
    if looks_like_punishment(text):
        return "States a punishment-related rule."
    if text:
        return truncate(text, SUMMARY_MAX_CHARS)
    return fallback_section_summary(section)


def fallback_paraphrase_from_text(node: Dict[str, Any], section: Dict[str, Any]) -> str:
    text = str(node.get("text") or node.get("derived_context") or "").strip()
    text = re.sub(r"\s+", " ", text)
    if looks_like_definition(text):
        m = re.match(r'^["“](.+?)["”]\s*(?:means|denotes|includes|shall mean|is said to|is said to be)\b', text, flags=re.I)
        term = m.group(1) if m else None
        if term:
            return f'This defines "{term}" for the Act.'
        return "This defines a term used in the Act."
    if looks_like_proviso(text):
        return "This adds a proviso or qualification to the rule."
    if looks_like_condition(text):
        return "This says the rule applies only when the stated condition is met."
    if looks_like_exception(text):
        return "This creates an exception or saves other laws from being affected."
    if looks_like_punishment(text):
        return "This deals with punishment or liability under the Act."
    if text:
        return truncate(text, PARAPHRASE_MAX_CHARS)
    return fallback_section_summary(section)


def update_section(section: Dict[str, Any], result: Dict[str, Any], allowed_section_types: Set[str]) -> None:
    if not is_nonempty(section.get("semantic_summary")):
        val = clamp_text(result.get("semantic_summary"), SUMMARY_MAX_CHARS)
        if val:
            section["semantic_summary"] = val

    if not is_nonempty(section.get("plain_english_paraphrase")):
        val = clamp_text(result.get("plain_english_paraphrase"), PARAPHRASE_MAX_CHARS)
        if val:
            section["plain_english_paraphrase"] = val

    section["keywords"] = merge_list_fields(section.get("keywords", []), result.get("additional_keywords", []))
    section["legal_concepts"] = merge_list_fields(section.get("legal_concepts", []), result.get("legal_concepts", []))

    if not is_nonempty(section.get("section_type")):
        section_type = coerce_enum(result.get("section_type"), allowed_section_types)
        if section_type:
            section["section_type"] = section_type


def update_node(node: Dict[str, Any], result: Dict[str, Any]) -> None:
    if not is_nonempty(node.get("semantic_summary")):
        val = clamp_text(result.get("semantic_summary"), SUMMARY_MAX_CHARS)
        if val:
            node["semantic_summary"] = val

    if not is_nonempty(node.get("plain_english_paraphrase")):
        val = clamp_text(result.get("plain_english_paraphrase"), PARAPHRASE_MAX_CHARS)
        if val:
            node["plain_english_paraphrase"] = val

    node["keywords"] = merge_list_fields(node.get("keywords", []), result.get("additional_keywords", []))
    node["legal_concepts"] = merge_list_fields(node.get("legal_concepts", []), result.get("legal_concepts", []))


def apply_final_fallbacks(data: Dict[str, Any], allowed_node_types: Set[str], allowed_section_types: Set[str]) -> None:
    for chapter, section in iter_sections(data):
        if not is_nonempty(section.get("section_type")):
            inferred = infer_section_type(section) or "general_rule"
            if inferred in allowed_section_types:
                section["section_type"] = inferred
            else:
                section["section_type"] = "general_rule"

        if not is_nonempty(section.get("semantic_summary")):
            section["semantic_summary"] = fallback_section_summary(section)
        if not is_nonempty(section.get("plain_english_paraphrase")):
            title = str(section.get("section_title") or "this section").strip()
            section["plain_english_paraphrase"] = f"This section covers {title.lower().rstrip('.')}."

        for node in iter_nodes(section.get("nodes", []) or []):
            inferred = infer_node_type(node, allowed_node_types, fallback_rule=True)
            if inferred:
                node["node_type"] = inferred

            if not is_nonempty(node.get("semantic_summary")):
                node["semantic_summary"] = fallback_node_summary(node, section)
            if not is_nonempty(node.get("plain_english_paraphrase")):
                node["plain_english_paraphrase"] = fallback_paraphrase_from_text(node, section)


# ----------------------------- batching -----------------------------

def batch_by_size(items: List[Dict[str, Any]], max_items: int, max_chars: int, text_key: str = "context") -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0

    for item in items:
        item_chars = len(str(item.get(text_key, ""))) + len(compact_json({k: v for k, v in item.items() if k != text_key}))
        if current and (len(current) >= max_items or current_chars + item_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches


def prepare_section_jobs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for chapter, section in iter_sections(data):
        if is_nonempty(section.get("semantic_summary")) and is_nonempty(section.get("plain_english_paraphrase")) and is_nonempty(section.get("section_type")):
            continue

        missing = []
        if not is_nonempty(section.get("semantic_summary")):
            missing.append("semantic_summary")
        if not is_nonempty(section.get("plain_english_paraphrase")):
            missing.append("plain_english_paraphrase")
        if not is_nonempty(section.get("section_type")):
            missing.append("section_type")

        jobs.append(
            {
                "chapter": chapter,
                "section": section,
                "missing_fields": missing,
                "context": build_section_context(data.get("act", ""), chapter, section),
                "section_id": section.get("section_id"),
                "section_number": section.get("section_number"),
                "section_title": section.get("section_title"),
                "section_type": section.get("section_type"),
            }
        )
    return jobs


def prepare_node_jobs(data: Dict[str, Any], allowed_node_types: Set[str]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for chapter, section in iter_sections(data):
        section_summary = section.get("semantic_summary")
        section_type = section.get("section_type")
        for node in iter_nodes(section.get("nodes", []) or []):
            inferred = infer_node_type(node, allowed_node_types, fallback_rule=True)
            if inferred and not is_nonempty(node.get("node_type")):
                node["node_type"] = inferred

            missing = []
            if not is_nonempty(node.get("semantic_summary")):
                missing.append("semantic_summary")
            if not is_nonempty(node.get("plain_english_paraphrase")):
                missing.append("plain_english_paraphrase")

            if not missing:
                continue

            jobs.append(
                {
                    "chapter": chapter,
                    "section": section,
                    "node": node,
                    "missing_fields": missing,
                    "context": build_node_context(data.get("act", ""), chapter, section, node),
                    "section_summary": section_summary,
                    "section_type": section_type,
                }
            )
    return jobs


# ----------------------------- LLM application -----------------------------

def enrich_sections_batched(
    client: LocalLlamaJSONClient,
    data: Dict[str, Any],
    allowed_section_types: Set[str],
    batch_max_items: int,
    batch_max_chars: int,
    max_tokens: int,
) -> None:
    # Group by chapter to keep prompts coherent.
    for chapter in data.get("chapters", []) or []:
        section_jobs: List[Dict[str, Any]] = []
        for section in chapter.get("sections", []) or []:
            if is_nonempty(section.get("semantic_summary")) and is_nonempty(section.get("plain_english_paraphrase")) and is_nonempty(section.get("section_type")):
                continue
            missing = []
            if not is_nonempty(section.get("semantic_summary")):
                missing.append("semantic_summary")
            if not is_nonempty(section.get("plain_english_paraphrase")):
                missing.append("plain_english_paraphrase")
            if not is_nonempty(section.get("section_type")):
                missing.append("section_type")
            section_jobs.append(
                {
                    "section": section,
                    "missing_fields": missing,
                    "context": build_section_context(data.get("act", ""), chapter, section),
                    "section_id": section.get("section_id"),
                    "section_number": section.get("section_number"),
                    "section_title": section.get("section_title"),
                    "section_type": section.get("section_type"),
                }
            )

        batches = batch_by_size(section_jobs, max_items=batch_max_items, max_chars=batch_max_chars, text_key="context")
        for idx, batch in enumerate(batches, 1):
            user_msg = build_section_batch_prompt(batch, sorted(allowed_section_types))
            try:
                payload = client.chat_json(SYSTEM_MSG, user_msg, max_tokens=max_tokens, retries=2)
            except Exception as e:
                print(f"[ERROR] batch failed: {e}")
                continue
            
            # payload = client.chat_json(SYSTEM_MSG, user_msg, max_tokens=max_tokens, retries=2)
            items = payload.get("items", [])
            by_id = {item.get("section_id"): item for item in items if isinstance(item, dict)}
            for job in batch:
                sec = job["section"]
                result = by_id.get(sec.get("section_id"), {})
                if result:
                    update_section(sec, result, allowed_section_types)
                # Always run final fallback on missing fields for safety.
                if not is_nonempty(sec.get("semantic_summary")):
                    sec["semantic_summary"] = fallback_section_summary(sec)
                if not is_nonempty(sec.get("plain_english_paraphrase")):
                    title = str(sec.get("section_title") or "this section").strip()
                    sec["plain_english_paraphrase"] = f"This section covers {title.lower().rstrip('.')}."
                if not is_nonempty(sec.get("section_type")):
                    sec["section_type"] = infer_section_type(sec) or "general_rule"

            print(f"[sections] Chapter {chapter.get('chapter_number')} batch {idx}/{len(batches)} done ({len(batch)} sections)")


def enrich_nodes_batched(
    client: LocalLlamaJSONClient,
    data: Dict[str, Any],
    allowed_node_types: Set[str],
    batch_max_items: int,
    batch_max_chars: int,
    max_tokens: int,
) -> None:
    for chapter in data.get("chapters", []) or []:
        for section in chapter.get("sections", []) or []:
            node_jobs: List[Dict[str, Any]] = []
            for node in iter_nodes(section.get("nodes", []) or []):
                inferred = infer_node_type(node, allowed_node_types, fallback_rule=True)
                if inferred and not is_nonempty(node.get("node_type")):
                    node["node_type"] = inferred

                missing = []
                if not is_nonempty(node.get("semantic_summary")):
                    missing.append("semantic_summary")
                if not is_nonempty(node.get("plain_english_paraphrase")):
                    missing.append("plain_english_paraphrase")
                if not missing:
                    continue

                node_jobs.append(
                    {
                        "node": node,
                        "missing_fields": missing,
                        "context": build_node_context(data.get("act", ""), chapter, section, node),
                        "section_summary": section.get("semantic_summary"),
                        "section_type": section.get("section_type"),
                    }
                )

            if not node_jobs:
                continue

            batches = batch_by_size(node_jobs, max_items=batch_max_items, max_chars=batch_max_chars, text_key="context")
            for idx, batch in enumerate(batches, 1):
                user_msg = build_node_batch_prompt(batch) 
                try:
                    payload = client.chat_json(SYSTEM_MSG, user_msg, max_tokens=max_tokens, retries=2)
                except Exception as e:
                    print(f"[ERROR] batch failed: {e}")
                    continue
                # payload = client.chat_json(SYSTEM_MSG, user_msg, max_tokens=max_tokens, retries=2)
                items = payload.get("items", [])
                by_id = {
                    item.get("node_id"): item
                    for item in items
                    if isinstance(item, dict) and item.get("node_id")
                }
                for job in batch:
                    node = job["node"]
                    result = by_id.get(node.get("node_id"), {})
                    if result:
                        update_node(node, result)
                    # Final safety pass for any remaining empties.
                    inferred = infer_node_type(node, allowed_node_types, fallback_rule=True)
                    if inferred:
                        node["node_type"] = inferred
                    if not is_nonempty(node.get("semantic_summary")):
                        node["semantic_summary"] = fallback_node_summary(node, section)
                    if not is_nonempty(node.get("plain_english_paraphrase")):
                        node["plain_english_paraphrase"] = fallback_paraphrase_from_text(node, section)

                print(
                    f"[nodes] Chapter {chapter.get('chapter_number')} / Section {section.get('section_number')} batch {idx}/{len(batches)} done ({len(batch)} nodes)"
                )


# ----------------------------- verification -----------------------------

def collect_stats(data: Dict[str, Any]) -> Dict[str, int]:
    stats = {
        "sections": 0,
        "nodes": 0,
        "section_semantic_missing": 0,
        "section_paraphrase_missing": 0,
        "section_type_missing": 0,
        "node_semantic_missing": 0,
        "node_paraphrase_missing": 0,
        "node_type_missing": 0,
    }
    for _, section in iter_sections(data):
        stats["sections"] += 1
        if not is_nonempty(section.get("semantic_summary")):
            stats["section_semantic_missing"] += 1
        if not is_nonempty(section.get("plain_english_paraphrase")):
            stats["section_paraphrase_missing"] += 1
        if not is_nonempty(section.get("section_type")):
            stats["section_type_missing"] += 1
        for node in iter_nodes(section.get("nodes", []) or []):
            stats["nodes"] += 1
            if not is_nonempty(node.get("semantic_summary")):
                stats["node_semantic_missing"] += 1
            if not is_nonempty(node.get("plain_english_paraphrase")):
                stats["node_paraphrase_missing"] += 1
            if not is_nonempty(node.get("node_type")):
                stats["node_type_missing"] += 1
    return stats


def print_stats(prefix: str, stats: Dict[str, int]) -> None:
    print(
        f"{prefix} sections={stats['sections']} nodes={stats['nodes']} | "
        f"section_missing(summary={stats['section_semantic_missing']}, paraphrase={stats['section_paraphrase_missing']}, type={stats['section_type_missing']}) | "
        f"node_missing(summary={stats['node_semantic_missing']}, paraphrase={stats['node_paraphrase_missing']}, type={stats['node_type_missing']})"
    )


# ----------------------------- main -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized legal JSON enrichment with batched llama calls.")
    parser.add_argument("--input", default="context.json", help="Input JSON file")
    parser.add_argument("--enums", default="enr_enums.json", help="Enum JSON file")
    parser.add_argument("--output", default="context_filled.json", help="Output JSON file")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, help="Path to GGUF model")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window size")
    parser.add_argument("--n-threads", type=int, default=max(4, (os.cpu_count() or 8) - 2), help="CPU threads")
    parser.add_argument("--n-gpu-layers", type=int, default=24, help="GPU layers to offload")
    parser.add_argument("--n-batch", type=int, default=256, help="Batch size for prompt processing")
    parser.add_argument("--section-batch-max-items", type=int, default=SECTION_BATCH_MAX_ITEMS, help="Max sections per LLM batch")
    parser.add_argument("--node-batch-max-items", type=int, default=NODE_BATCH_MAX_ITEMS, help="Max nodes per LLM batch")
    parser.add_argument("--section-batch-max-chars", type=int, default=SECTION_BATCH_MAX_CHARS, help="Max prompt characters for a section batch")
    parser.add_argument("--node-batch-max-chars", type=int, default=NODE_BATCH_MAX_CHARS, help="Max prompt characters for a node batch")
    parser.add_argument("--section-max-tokens", type=int, default=320, help="Max output tokens for section batches")
    parser.add_argument("--node-max-tokens", type=int, default=512, help="Max output tokens for node batches")
    parser.add_argument("--save-every-chapter", action="store_true", help="Save a checkpoint after each chapter")
    args = parser.parse_args()

    data = load_json(args.input)
    allowed_node_types, allowed_section_types = load_allowed_enums(args.enums)

    stats_before = collect_stats(data)
    print_stats("Before", stats_before)

    client = LocalLlamaJSONClient(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        verbose=False,
    )

    # First deterministic pass: remove easy nulls without spending model calls.
    for chapter, section in iter_sections(data):
        inferred = infer_section_type(section)
        if inferred and not is_nonempty(section.get("section_type")):
            section["section_type"] = inferred
        for node in iter_nodes(section.get("nodes", []) or []):
            inferred_node_type = infer_node_type(node, allowed_node_types, fallback_rule=True)
            if inferred_node_type and not is_nonempty(node.get("node_type")):
                node["node_type"] = inferred_node_type

    # LLM pass for sections and nodes.
    enrich_sections_batched(
        client=client,
        data=data,
        allowed_section_types=allowed_section_types,
        batch_max_items=args.section_batch_max_items,
        batch_max_chars=args.section_batch_max_chars,
        max_tokens=args.section_max_tokens,
    )

    if args.save_every_chapter:
        save_json(args.output, data)
        print(f"Checkpoint saved to {args.output} after section pass")

    enrich_nodes_batched(
        client=client,
        data=data,
        allowed_node_types=allowed_node_types,
        batch_max_items=args.node_batch_max_items,
        batch_max_chars=args.node_batch_max_chars,
        max_tokens=args.node_max_tokens,
    )

    # Final safety pass: fill anything the model missed.
    apply_final_fallbacks(data, allowed_node_types, allowed_section_types)

    save_json(args.output, data)
    stats_after = collect_stats(data)
    print_stats("After", stats_after)
    print(f"Done. Wrote output to {args.output}")


if __name__ == "__main__":
    main()
