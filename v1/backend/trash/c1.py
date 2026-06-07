#!/usr/bin/env python3
"""
Build chunks.json from graph.json + context_filled.json.

Strategy:
- one section chunk per section
- one atomic chunk per non-empty node
- empty container nodes are skipped
- embedding_text uses derived_context whenever present
- citations are built from the node hierarchy IDs/labels
"""


# python ingestion/build_chunk/c1.py --context data/processed/jsons/enriched_jsons/context_filled.json --graph data/processed/jsons/graph_jsons/graph.json --output data/processed/jsons/chunk_jsons/chunks.json 

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


SEMANTIC_EDGE_TYPES = {"REFERS_TO", "EXPLAINS", "ILLUSTRATES"}
ALLOWED_CHUNK_TYPES = {
    "section",
    "subsection",
    "clause",
    "content",
    "explanation",
    "illustration",
    "exception",
    "mixed",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def clean_multiline(value: Optional[str]) -> str:
    """
    Keep paragraph breaks, but normalize internal whitespace.
    """
    if value is None:
        return ""
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in str(value).splitlines()]
    out: List[str] = []
    prev_blank = False
    for ln in lines:
        blank = not ln.strip()
        if blank:
            if out and not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(ln.strip())
            prev_blank = False
    return "\n".join(out).strip()


def parse_label(label: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Convert labels like '(5)', '(a)', 'Explanation', 'Illustration' into a structured form.
    """
    if not label:
        return None

    s = str(label).strip()

    if s.lower() == "body":
        return {"kind": "body", "value": None}

    if s in {"Explanation", "Illustration"}:
        return {"kind": s.lower(), "value": None}

    m = re.fullmatch(r"\((\d+)\)", s)
    if m:
        return {"kind": "subsection", "value": int(m.group(1))}

    m = re.fullmatch(r"\(([a-zA-Z])\)", s)
    if m:
        return {"kind": "clause", "value": m.group(1).lower()}

    m = re.fullmatch(r"\(([ivxlcdm]+)\)", s, flags=re.I)
    if m:
        return {"kind": "clause", "value": m.group(1).lower()}

    m = re.fullmatch(r"(Explanation|Illustration)\s*(\d+)?", s, flags=re.I)
    if m:
        return {"kind": m.group(1).lower(), "value": int(m.group(2)) if m.group(2) else None}

    return {"kind": "label", "value": s}


def build_indices(ctx: dict):
    sections: Dict[str, dict] = {}
    nodes: Dict[str, dict] = {}
    parent: Dict[str, Optional[str]] = {}
    node_to_section: Dict[str, str] = {}
    node_to_chapter: Dict[str, str] = {}
    section_to_chapter: Dict[str, dict] = {}

    def rec(node: dict, chapter_id: str, section_id: str, parent_id: Optional[str] = None):
        nid = node["node_id"]
        nodes[nid] = node
        parent[nid] = parent_id
        node_to_section[nid] = section_id
        node_to_chapter[nid] = chapter_id
        for child in node.get("children", []) or []:
            rec(child, chapter_id, section_id, nid)

    for ch in ctx["chapters"]:
        for sec in ch["sections"]:
            sections[sec["section_id"]] = sec
            section_to_chapter[sec["section_id"]] = ch
            for node in sec.get("nodes", []):
                rec(node, ch["chapter_id"], sec["section_id"], None)

    return sections, nodes, parent, node_to_section, node_to_chapter, section_to_chapter


def section_descendant_nodes(sections: dict, section_id: str) -> List[str]:
    ids: List[str] = []

    def rec(node: dict):
        ids.append(node["node_id"])
        for child in node.get("children", []) or []:
            rec(child)

    for root in sections[section_id].get("nodes", []) or []:
        rec(root)

    return ids


def build_chunks(context_path: Path, graph_path: Path) -> dict:
    ctx = load_json(context_path)
    graph = load_json(graph_path)

    sections, nodes, parent, node_to_section, node_to_chapter, section_to_chapter = build_indices(ctx)
    graph_nodes = {n["id"]: n for n in graph["nodes"]}

    graph_semantic_refs = collections.defaultdict(list)
    for edge in graph.get("edges", []) or []:
        if edge.get("type") in SEMANTIC_EDGE_TYPES:
            graph_semantic_refs[edge["source"]].append(edge)

    def token_for_node(node: dict) -> str:
        parsed = parse_label(node.get("node_label"))
        if not parsed:
            return ""
        kind, value = parsed["kind"], parsed["value"]

        if kind == "subsection" and isinstance(value, int):
            return f"({value})"
        if kind == "clause" and value is not None:
            return f"({value})"
        if kind in {"explanation", "illustration"}:
            return f"{kind.capitalize()}{f' {value}' if isinstance(value, int) else ''}"
        if kind == "body":
            return "Body"
        return str(value or node.get("node_label") or "").strip()

    def ancestry_chain(node_id: str) -> List[str]:
        chain: List[str] = []
        cur = node_id
        while cur:
            chain.append(cur)
            cur = parent.get(cur)
        return list(reversed(chain))

    def build_node_citation(node_id: str) -> dict:
        sec_id = node_to_section[node_id]
        ch = section_to_chapter[sec_id]
        sec = sections[sec_id]

        path = [
            f"Act: {ctx['act']}",
            f"Chapter {ch['chapter_number']}: {ch['chapter_title']}",
            f"Section {sec['section_number']}: {sec['section_title']}",
        ]

        citation_text = f"Section {sec['section_number']}"
        previous_appended = ""

        for nid in ancestry_chain(node_id):
            if nid == sec_id:
                continue
            tok = token_for_node(nodes[nid])
            if not tok:
                continue
            path.append(tok)
            if tok.startswith("(") and previous_appended and previous_appended[-1].isalpha():
                citation_text += f" {tok}"
            elif tok.startswith("("):
                citation_text += tok
            else:
                citation_text += f" {tok}"
            previous_appended = tok

        citation_text += f", {ctx['act']}"
        return {
            "node_id": node_id,
            "citation_text": citation_text,
            "path": path,
            "node_label": nodes[node_id].get("node_label"),
            "parsed_label": parse_label(nodes[node_id].get("node_label")),
        }

    def refs_for_node(node_id: str) -> List[dict]:
        node = nodes[node_id]
        out: List[dict] = []

        # References already stored in context_filled.json
        for ref in node.get("references", []) or []:
            if isinstance(ref, dict):
                target_id = ref.get("target_id")
                target_type = ref.get("target_type")
                relation = ref.get("relation") or "REFERS_TO"
            else:
                target_id = str(ref)
                target_type = graph_nodes.get(target_id, {}).get("type")
                relation = "REFERS_TO"

            if not target_type:
                if target_id == ctx["doc_id"]:
                    target_type = "act"
                elif str(target_id).startswith("chp_"):
                    target_type = "chapter"
                elif "_S" in str(target_id):
                    target_type = "section"
                else:
                    target_type = "node"

            out.append(
                {
                    "target_id": target_id,
                    "target_type": target_type,
                    "relation": relation,
                }
            )

        # Semantic edges from graph.json
        for edge in graph_semantic_refs.get(node_id, []):
            target_id = edge["target"]
            target_type = graph_nodes.get(target_id, {}).get("type")
            if not target_type:
                if target_id == ctx["doc_id"]:
                    target_type = "act"
                elif str(target_id).startswith("chp_"):
                    target_type = "chapter"
                elif "_S" in str(target_id):
                    target_type = "section"
                else:
                    target_type = "node"

            out.append(
                {
                    "target_id": target_id,
                    "target_type": target_type,
                    "relation": edge["type"],
                }
            )

        # De-duplicate while preserving order
        uniq: List[dict] = []
        seen = set()
        for item in out:
            key = (item["target_id"], item["target_type"], item.get("relation"))
            if key not in seen:
                seen.add(key)
                uniq.append(item)
        return uniq

    def chunkable(node: dict) -> bool:
        return bool(normalize_text(node.get("text")))

    # First create one section chunk per section
    chunks: List[dict] = []
    chunk_map: Dict[str, dict] = {}
    node_to_chunk: Dict[str, str] = {}
    section_chunk_id_map: Dict[str, str] = {}

    for sec_id, sec in sections.items():
        ch = section_to_chapter[sec_id]
        section_chunk_id = f"chunk__section__{sec_id}"
        section_chunk_id_map[sec_id] = section_chunk_id

        sec_text = clean_multiline(sec.get("full_text") or "")

        sec_chunk = {
            "chunk_id": section_chunk_id,
            "doc_id": ctx["doc_id"],
            "act": ctx["act"],
            "chunk_type": "section",
            "chunk_order": 0,
            "parent_chunk_id": None,
            "chapter": {
                "chapter_id": ch["chapter_id"],
                "chapter_number": ch["chapter_number"],
                "chapter_title": ch["chapter_title"],
                "chapter_order": ch.get("chapter_order"),
            },
            "section": {
                "section_id": sec["section_id"],
                "section_number": sec["section_number"],
                "section_title": sec["section_title"],
                "section_order": sec.get("section_order"),
                "section_type": sec.get("section_type"),
                "subheading": sec.get("subheading"),
            },
            "citation": {
                "node_id": sec_id,
                "citation_text": f"Section {sec['section_number']}: {sec['section_title']}",
                "path": [
                    f"Act: {ctx['act']}",
                    f"Chapter {ch['chapter_number']}: {ch['chapter_title']}",
                    f"Section {sec['section_number']}: {sec['section_title']}",
                ],
                "node_label": None,
                "parsed_label": None,
            },
            "root_node_id": sec.get("nodes", [{}])[0].get("node_id") if sec.get("nodes") else None,
            "node_ids": section_descendant_nodes(sections, sec_id),
            "text": sec_text,
            "derived_context": f"Act: {ctx['act']}\nChapter {ch['chapter_number']}: {ch['chapter_title']}\nSection {sec['section_number']}: {sec['section_title']}\n{sec_text}".strip(),
            "embedding_text": f"Act: {ctx['act']}\nChapter {ch['chapter_number']}: {ch['chapter_title']}\nSection {sec['section_number']}: {sec['section_title']}\n{sec_text}".strip(),
            "semantic_summary": sec.get("semantic_summary"),
            "plain_english_paraphrase": sec.get("plain_english_paraphrase"),
            "keywords": sec.get("keywords") or [],
            "legal_concepts": sec.get("legal_concepts") or [],
            "references": [],
            "explains_node_id": None,
            "children_chunk_ids": [],
        }

        chunks.append(sec_chunk)
        chunk_map[section_chunk_id] = sec_chunk

    # Then create node chunks in pre-order
    for sec_id, sec in sections.items():
        section_chunk_id = section_chunk_id_map[sec_id]
        section_node_order = 0

        def nearest_chunked_ancestor(node_id: str) -> Optional[str]:
            cur = parent.get(node_id)
            while cur:
                if cur in node_to_chunk:
                    return cur
                cur = parent.get(cur)
            return None

        def rec(node: dict):
            nonlocal section_node_order
            nid = node["node_id"]

            if chunkable(node):
                section_node_order += 1
                chunk_id = f"chunk__node__{nid}"

                parent_node_id = nearest_chunked_ancestor(nid)
                parent_chunk_id = node_to_chunk.get(parent_node_id, section_chunk_id)

                # The nearest chunked ancestor in the same section becomes the root of this chunk tree.
                root_node_id = nid
                cur = parent.get(nid)
                while cur:
                    if cur in node_to_chunk:
                        root_node_id = cur
                        break
                    cur = parent.get(cur)

                node_category = (node.get("node_category") or node.get("node_type") or "mixed").lower()
                chunk_type = node_category if node_category in ALLOWED_CHUNK_TYPES else "mixed"

                n_chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": ctx["doc_id"],
                    "act": ctx["act"],
                    "chunk_type": chunk_type,
                    "chunk_order": section_node_order,
                    "parent_chunk_id": parent_chunk_id,
                    "chapter": {
                        "chapter_id": section_to_chapter[sec_id]["chapter_id"],
                        "chapter_number": section_to_chapter[sec_id]["chapter_number"],
                        "chapter_title": section_to_chapter[sec_id]["chapter_title"],
                        "chapter_order": section_to_chapter[sec_id].get("chapter_order"),
                    },
                    "section": {
                        "section_id": sec["section_id"],
                        "section_number": sec["section_number"],
                        "section_title": sec["section_title"],
                        "section_order": sec.get("section_order"),
                        "section_type": sec.get("section_type"),
                        "subheading": sec.get("subheading"),
                    },
                    "citation": build_node_citation(nid),
                    "root_node_id": root_node_id,
                    "node_ids": [nid],
                    "text": normalize_text(node.get("text")),
                    "derived_context": node.get("derived_context")
                    or f"Act: {ctx['act']}\nChapter {section_to_chapter[sec_id]['chapter_number']}: {section_to_chapter[sec_id]['chapter_title']}\nSection {sec['section_number']}: {sec['section_title']}\n{normalize_text(node.get('text'))}".strip(),
                    "embedding_text": node.get("derived_context")
                    or f"Act: {ctx['act']}\nChapter {section_to_chapter[sec_id]['chapter_number']}: {section_to_chapter[sec_id]['chapter_title']}\nSection {sec['section_number']}: {sec['section_title']}\n{normalize_text(node.get('text'))}".strip(),
                    "semantic_summary": node.get("semantic_summary"),
                    "plain_english_paraphrase": node.get("plain_english_paraphrase"),
                    "keywords": node.get("keywords") or [],
                    "legal_concepts": node.get("legal_concepts") or [],
                    "references": refs_for_node(nid),
                    "explains_node_id": node.get("explains_node_id"),
                    "children_chunk_ids": [],
                }

                chunks.append(n_chunk)
                chunk_map[chunk_id] = n_chunk
                node_to_chunk[nid] = chunk_id

            for child in node.get("children", []) or []:
                rec(child)

        for root in sec.get("nodes", []) or []:
            rec(root)

    # Fill parent->children links
    for chunk in chunks:
        pid = chunk.get("parent_chunk_id")
        if pid and pid in chunk_map:
            chunk_map[pid]["children_chunk_ids"].append(chunk["chunk_id"])

    for chunk in chunks:
        chunk["children_chunk_ids"] = sorted(
            chunk["children_chunk_ids"],
            key=lambda cid: (chunk_map[cid]["chunk_order"], cid),
        )

    # Stable ordering: chapter -> section -> section chunk first -> node chunks by traversal order
    chunks = sorted(
        chunks,
        key=lambda c: (
            c["chapter"]["chapter_number"],
            c["section"]["section_number"],
            0 if c["chunk_type"] == "section" else 1,
            c["chunk_order"],
            c["chunk_id"],
        ),
    )

    return {
        "doc_id": ctx["doc_id"],
        "act": ctx["act"],
        "jurisdiction": ctx.get("jurisdiction"),
        "version_date": ctx.get("version_date"),
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "chunking_strategy": (
            "One parent chunk per section and one atomic chunk per non-empty legal node. "
            "Empty container nodes are skipped. Embedding text uses derived_context whenever available, "
            "with act/chapter/section context prepended as a fallback. Child chunks attach to the nearest "
            "chunked ancestor so the hierarchy stays exact."
        ),
        "source_files": {
            "graph_json": str(graph_path),
            "context_json": str(context_path),
        },
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = build_chunks(args.context, args.graph)
    save_json(args.output, data)


if __name__ == "__main__":
    main()
