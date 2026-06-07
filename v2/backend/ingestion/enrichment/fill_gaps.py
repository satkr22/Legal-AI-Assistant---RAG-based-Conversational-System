#!/usr/bin/env python3
# fill_context_fields.py

# To run:
# python ingestion/enrichment/fill_gaps.py --input data/processed/jsons/enriched_jsons/context.json --enums data/processed/jsons/json_blueprints/enr_enums.json --output data/processed/jsons/enriched_jsons/context_filled.json

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from llama_cpp import Llama
from Llama.generation.generator import LLM  # generator.py must be in the same folder, or on PYTHONPATH

MODEL_PATH = "/home/usatkr/llm/models/llama-3.1-8b/llama-3.1-8b-q4_k_m.gguf"

SYSTEM_MSG = (
    "You are a strict legal-document enrichment assistant. "
    "Use only the provided derived context and metadata. "
    "Do not use raw text or embedding text. "
    "Return JSON only."
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
}

GENERIC_WORDS = {
    "thing", "things", "person", "persons", "section", "subsection", "clause",
    "act", "law", "legal", "case", "matter", "provision", "provisions",
    "document", "documents", "rule", "rules"
}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
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


def normalize_phrase(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r\"'`.,;:()[]{}<>")
    return text


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        norm = normalize_phrase(item)
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def clean_keyword_list(values: Any) -> List[str]:
    items: List[str] = []
    if values is None:
        return items
    if isinstance(values, list):
        raw = values
    elif isinstance(values, str):
        raw = re.split(r"[\n,;]+", values)
    else:
        raw = [values]

    for v in raw:
        kw = normalize_phrase(str(v))
        if not kw:
            continue
        if len(kw) < 3:
            continue
        if kw in GENERIC_WORDS:
            continue
        items.append(kw)

    return dedupe_preserve_order(items)


def coerce_enum(value: Any, allowed: Set[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = normalize_phrase(str(value))
    candidate = ALIAS_MAP.get(candidate, candidate)
    return candidate if candidate in allowed else None


def safe_extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        chunk = match.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{text[:1200]}")


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


def collect_node_contexts(nodes: Sequence[Dict[str, Any]], limit: int = 4, max_chars: int = 1800) -> List[str]:
    snippets: List[str] = []
    used_chars = 0

    def walk(node: Dict[str, Any]) -> None:
        nonlocal used_chars
        if len(snippets) >= limit or used_chars >= max_chars:
            return

        dc = node.get("derived_context")
        if isinstance(dc, str):
            dc = dc.strip()
            if dc:
                snippets.append(dc)
                used_chars += len(dc)

        for child in node.get("children", []) or []:
            if len(snippets) >= limit or used_chars >= max_chars:
                return
            walk(child)

    for node in nodes or []:
        if len(snippets) >= limit or used_chars >= max_chars:
            break
        walk(node)

    return snippets


def build_node_context(act: str, chapter: Dict[str, Any], section: Dict[str, Any], node: Dict[str, Any]) -> str:
    parts = [
        f"Act: {act}",
        f"Chapter {chapter.get('chapter_number')}: {chapter.get('chapter_title')}",
        f"Section {section.get('section_number')}: {section.get('section_title')}",
    ]

    dc = node.get("derived_context")
    if isinstance(dc, str) and dc.strip():
        parts.append("Derived context:")
        parts.append(dc.strip())

    return "\n".join(parts)


def build_section_context(act: str, chapter: Dict[str, Any], section: Dict[str, Any]) -> str:
    snippets = collect_node_contexts(section.get("nodes", []), limit=4, max_chars=1800)

    parts = [
        f"Act: {act}",
        f"Chapter {chapter.get('chapter_number')}: {chapter.get('chapter_title')}",
        f"Section {section.get('section_number')}: {section.get('section_title')}",
    ]

    if is_nonempty(section.get("section_type")):
        parts.append(f"Existing section_type: {section.get('section_type')}")

    if is_nonempty(section.get("semantic_summary")):
        parts.append(f"Existing semantic_summary: {section.get('semantic_summary')}")

    if snippets:
        parts.append("Representative node contexts:")
        for i, snippet in enumerate(snippets, 1):
            parts.append(f"{i}. {snippet}")

    return "\n".join(parts)


def heuristic_section_type(section: Dict[str, Any]) -> Optional[str]:
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
    if "general" in title or "application" in title or "preliminary" in title:
        return "general_rule"
    return None


def heuristic_node_type(node: Dict[str, Any]) -> Optional[str]:
    current = normalize_phrase(str(node.get("node_type") or ""))
    if current:
        return current

    category = normalize_phrase(str(node.get("node_category") or ""))
    label = normalize_phrase(str(node.get("node_label") or ""))

    if category == "explanation" or "explanation" in label:
        return "explanation"
    if category == "illustration" or "illustration" in label:
        return "illustration"
    return None


def create_prompt(llm: LLM, item_kind: str, context_text: str, metadata: Dict[str, Any], missing_fields: List[str], allowed_enum_values: Sequence[str]) -> str:
    if item_kind == "node":
        output_schema = """{
  "semantic_summary": null,
  "plain_english_paraphrase": null,
  "additional_keywords": [],
  "legal_concepts": [],
  "node_type": null
}"""
        instruction = (
            "For node_type, choose exactly one allowed value from the allowed enum values, or null if uncertain. "
            "Do not change existing non-empty fields."
        )
    else:
        output_schema = """{
  "semantic_summary": null,
  "plain_english_paraphrase": null,
  "additional_keywords": [],
  "legal_concepts": [],
  "section_type": null
}"""
        instruction = (
            "For section_type, choose exactly one allowed value from the allowed enum values, or null if uncertain. "
            "Do not change existing non-empty fields."
        )

    user_msg = f"""
Task: fill missing legal enrichment fields for one structured legal item.

Rules:
1. Use only the provided derived context and metadata.
2. Do not use raw text or embedding text.
3. Do not invent facts.
4. Do not modify any existing non-empty field.
5. Keep semantic_summary to 1–2 concise sentences.
6. Keep plain_english_paraphrase to 1–2 concise sentences.
7. additional_keywords must be short phrases only.
8. legal_concepts must be concise legal phrases, not sentences.
9. {instruction}
10. Return JSON only.

Allowed enum values:
{json.dumps(list(allowed_enum_values), ensure_ascii=False)}

Missing fields:
{json.dumps(missing_fields, ensure_ascii=False)}

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)}

Context:
{context_text}

Return exactly this schema shape:
{output_schema}
""".strip()

    return llm.create_evaluation_prompt(question=user_msg, system_msg=SYSTEM_MSG)


def call_llm_json(llm: LLM, prompt: str, max_tokens: int) -> Dict[str, Any]:
    output = llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
    return safe_extract_json(output)


def merge_list_fields(existing: Any, new_values: Any) -> List[str]:
    merged = clean_keyword_list(existing) + clean_keyword_list(new_values)
    return dedupe_preserve_order(merged)


def update_node(node: Dict[str, Any], result: Dict[str, Any], allowed_node_types: Set[str]) -> None:
    if not is_nonempty(node.get("semantic_summary")):
        val = result.get("semantic_summary")
        if isinstance(val, str) and val.strip():
            node["semantic_summary"] = val.strip()

    if not is_nonempty(node.get("plain_english_paraphrase")):
        val = result.get("plain_english_paraphrase")
        if isinstance(val, str) and val.strip():
            node["plain_english_paraphrase"] = val.strip()

    node["keywords"] = merge_list_fields(node.get("keywords", []), result.get("additional_keywords", []))
    node["legal_concepts"] = merge_list_fields(node.get("legal_concepts", []), result.get("legal_concepts", []))

    if not is_nonempty(node.get("node_type")):
        node_type = coerce_enum(result.get("node_type"), allowed_node_types)
        if node_type:
            node["node_type"] = node_type


def update_section(section: Dict[str, Any], result: Dict[str, Any], allowed_section_types: Set[str]) -> None:
    if not is_nonempty(section.get("semantic_summary")):
        val = result.get("semantic_summary")
        if isinstance(val, str) and val.strip():
            section["semantic_summary"] = val.strip()

    if not is_nonempty(section.get("plain_english_paraphrase")):
        val = result.get("plain_english_paraphrase")
        if isinstance(val, str) and val.strip():
            section["plain_english_paraphrase"] = val.strip()

    section["keywords"] = merge_list_fields(section.get("keywords", []), result.get("additional_keywords", []))
    section["legal_concepts"] = merge_list_fields(section.get("legal_concepts", []), result.get("legal_concepts", []))

    if not is_nonempty(section.get("section_type")):
        section_type = coerce_enum(result.get("section_type"), allowed_section_types)
        if section_type:
            section["section_type"] = section_type


def fill_node_tree(
    llm: LLM,
    data: Dict[str, Any],
    chapter: Dict[str, Any],
    section: Dict[str, Any],
    node: Dict[str, Any],
    allowed_node_types: Set[str],
    allowed_section_types: Set[str],
) -> None:
    if is_nonempty(node.get("derived_context")):
        if not is_nonempty(node.get("node_type")):
            inferred = heuristic_node_type(node)
            if inferred and inferred in allowed_node_types:
                node["node_type"] = inferred

        missing_fields = []
        if not is_nonempty(node.get("semantic_summary")):
            missing_fields.append("semantic_summary")
        if not is_nonempty(node.get("plain_english_paraphrase")):
            missing_fields.append("plain_english_paraphrase")
        if not is_nonempty(node.get("node_type")):
            missing_fields.append("node_type")

        metadata = {
            "doc_id": data.get("doc_id"),
            "act": data.get("act"),
            "chapter_id": chapter.get("chapter_id"),
            "chapter_number": chapter.get("chapter_number"),
            "chapter_title": chapter.get("chapter_title"),
            "section_id": section.get("section_id"),
            "section_number": section.get("section_number"),
            "section_title": section.get("section_title"),
            "section_type": section.get("section_type"),
            "node_id": node.get("node_id"),
            "node_label": node.get("node_label"),
            "node_category": node.get("node_category"),
            "level": node.get("level"),
            "parent_node_id": node.get("parent_node_id"),
            "explains_node_id": node.get("explains_node_id"),
            "existing_keywords": node.get("keywords", []),
            "existing_legal_concepts": node.get("legal_concepts", []),
            "existing_node_type": node.get("node_type"),
        }

        context_text = build_node_context(data.get("act", ""), chapter, section, node)
        prompt = create_prompt(
            llm=llm,
            item_kind="node",
            context_text=context_text,
            metadata=metadata,
            missing_fields=missing_fields,
            allowed_enum_values=sorted(allowed_node_types),
        )

        result = call_llm_json(llm, prompt, max_tokens=384)
        update_node(node, result, allowed_node_types)

    for child in node.get("children", []) or []:
        fill_node_tree(
            llm=llm,
            data=data,
            chapter=chapter,
            section=section,
            node=child,
            allowed_node_types=allowed_node_types,
            allowed_section_types=allowed_section_types,
        )


def fill_section_fields(
    llm: LLM,
    data: Dict[str, Any],
    chapter: Dict[str, Any],
    section: Dict[str, Any],
    allowed_node_types: Set[str],
    allowed_section_types: Set[str],
) -> None:
    context_text = build_section_context(data.get("act", ""), chapter, section)

    if not is_nonempty(section.get("section_type")):
        inferred = heuristic_section_type(section)
        if inferred and inferred in allowed_section_types:
            section["section_type"] = inferred

    missing_fields = []
    if not is_nonempty(section.get("semantic_summary")):
        missing_fields.append("semantic_summary")
    if not is_nonempty(section.get("plain_english_paraphrase")):
        missing_fields.append("plain_english_paraphrase")
    if not is_nonempty(section.get("section_type")):
        missing_fields.append("section_type")

    metadata = {
        "doc_id": data.get("doc_id"),
        "act": data.get("act"),
        "chapter_id": chapter.get("chapter_id"),
        "chapter_number": chapter.get("chapter_number"),
        "chapter_title": chapter.get("chapter_title"),
        "section_id": section.get("section_id"),
        "section_number": section.get("section_number"),
        "section_title": section.get("section_title"),
        "section_order": section.get("section_order"),
        "current_section_type": section.get("section_type"),
        "existing_keywords": section.get("keywords", []),
        "existing_legal_concepts": section.get("legal_concepts", []),
    }

    prompt = create_prompt(
        llm=llm,
        item_kind="section",
        context_text=context_text,
        metadata=metadata,
        missing_fields=missing_fields,
        allowed_enum_values=sorted(allowed_section_types),
    )

    result = call_llm_json(llm, prompt, max_tokens=512)
    update_section(section, result, allowed_section_types)


def process_document(data: Dict[str, Any], llm: LLM, allowed_node_types: Set[str], allowed_section_types: Set[str]) -> Dict[str, Any]:
    for chapter in data.get("chapters", []) or []:
        for section in chapter.get("sections", []) or []:
            fill_section_fields(
                llm=llm,
                data=data,
                chapter=chapter,
                section=section,
                allowed_node_types=allowed_node_types,
                allowed_section_types=allowed_section_types,
            )

            for node in section.get("nodes", []) or []:
                fill_node_tree(
                    llm=llm,
                    data=data,
                    chapter=chapter,
                    section=section,
                    node=node,
                    allowed_node_types=allowed_node_types,
                    allowed_section_types=allowed_section_types,
                )

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="context.json", help="Input JSON file")
    parser.add_argument("--enums", default="enr_enums.json", help="Enum JSON file")
    parser.add_argument("--output", default="context_filled.json", help="Output JSON file")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Llama context size")
    parser.add_argument("--n-threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--n-gpu-layers", type=int, default=35, help="GPU layers to offload")
    parser.add_argument("--max-tokens-node", type=int, default=384, help="Max tokens for node outputs")
    parser.add_argument("--max-tokens-section", type=int, default=512, help="Max tokens for section outputs")
    args = parser.parse_args()

    data = load_json(args.input)
    allowed_node_types, allowed_section_types = load_allowed_enums(args.enums)

    base_llama = Llama(
        model_path=MODEL_PATH,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )

    llm = LLM(model=base_llama, default_seed=42)

    enriched = process_document(
        data=data,
        llm=llm,
        allowed_node_types=allowed_node_types,
        allowed_section_types=allowed_section_types,
    )

    save_json(args.output, enriched)
    print(f"Done. Wrote output to {args.output}")


if __name__ == "__main__":
    main()