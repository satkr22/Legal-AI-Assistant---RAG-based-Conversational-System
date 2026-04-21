#!/usr/bin/env python3
"""
Build chunks.json from graph.json + context_filled.json.

Production-oriented strategy:
- Section chunks are structural only (not embedded).
- Every non-empty legal node becomes an atomic chunk.
- derived_context preserves hierarchical legal context for node chunks.
- embedding_text is an optimized retrieval string derived from derived_context.
- Empty container nodes are skipped, but their children attach to the nearest chunked ancestor.
- Citations are built deterministically from the node hierarchy and labels.



'''

python ingestion/build_chunk/c2.py \
  --context data/processed/jsons/enriched_jsons/context_filled.json \
  --graph data/processed/jsons/graph_jsons/graph.json \
  --output data/processed/jsons/chunk_jsons/chunks2.json



'''



"""




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

# Legal nodes are typically manageable in size; section chunks are structural only.
MAX_EMBED_CHARS = 6000


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(value: Optional[str]) -> str:
    """Collapse all whitespace to a single space."""
    return re.sub(r"\s+", " ", (value or "")).strip()


def clean_multiline(value: Optional[str]) -> str:
    """
    Keep paragraph breaks, but normalize internal whitespace.
    This is safer than collapsing everything into one line for legal text.
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


def dedupe_consecutive_lines(text: str) -> str:
    """Remove consecutive duplicate lines after normalization."""
    lines = text.splitlines()
    out: List[str] = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped == prev and stripped:
            continue
        out.append(line)
        prev = stripped
    return "\n".join(out).strip()


def optimize_for_embedding(node, section, parent_node=None):
    parts = []

    # Section anchor
    parts.append(f"Section {section['section_number']}: {section['section_title']}")

    # Parent context (VERY important for clauses)
    if parent_node:
        parent_text = parent_node.get("text", "").strip()
        if parent_text:
            parts.append(parent_text)

    # Node label
    label = node.get("node_label")
    if label:
        parts.append(label)

    # Node text
    text = node.get("text", "").strip()
    if text:
        parts.append(text)

    return "\n".join(parts)


def parse_label(label: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Convert labels like '(5)', '(a)', 'Explanation', 'Illustration', 'body'
    into a structured form.
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


def infer_label_from_node_id(node_id: str) -> Optional[Dict[str, Any]]:
    """
    Fallback when node_label is missing.

    Handles common forms:
    - BNS_S101_A -> clause(a)
    - BNS_S2_N28_K_I -> clause(i)
    - BNS_S2_N16_II -> clause(ii)
    - BNS_S100_EXP1 -> explanation 1
    - BNS_S100_ILL1 -> illustration 1
    - BNS_S2_BODY -> body
    """
    s = str(node_id).strip()

    if s.endswith("_BODY"):
        return {"kind": "body", "value": None}

    m = re.search(r"_(EXP)(\d+)$", s, flags=re.I)
    if m:
        return {"kind": "explanation", "value": int(m.group(2))}

    m = re.search(r"_(ILL)(\d+)$", s, flags=re.I)
    if m:
        return {"kind": "illustration", "value": int(m.group(2))}

    # A, B, C ... Z
    m = re.search(r"_([A-Z])$", s)
    if m:
        return {"kind": "clause", "value": m.group(1).lower()}

    # Roman numeral children like _I, _II, _III
    m = re.search(r"_(I|II|III|IV|V|VI|VII|VIII|IX|X)$", s, flags=re.I)
    if m:
        return {"kind": "clause", "value": m.group(1).lower()}

    # Numeric children like _1
    m = re.search(r"_(\d+)$", s)
    if m:
        return {"kind": "subsection", "value": int(m.group(1))}

    return None


def label_to_token(parsed: Dict[str, Any], original_label: Optional[str] = None) -> str:
    kind = parsed.get("kind")
    value = parsed.get("value")

    if kind == "subsection" and isinstance(value, int):
        return f"({value})"
    if kind == "clause" and value is not None:
        return f"({value})"
    if kind == "explanation":
        if isinstance(value, int):
            return f"Explanation {value}"
        return "Explanation"
    if kind == "illustration":
        if isinstance(value, int):
            return f"Illustration {value}"
        return "Illustration"
    if kind == "body":
        return "Body"
    if original_label:
        return str(original_label).strip()
    return str(value or "").strip()


def build_indices(ctx: dict):
    """
    Flatten context_filled.json into lookup tables.
    """
    sections: Dict[str, dict] = {}
    nodes: Dict[str, dict] = {}
    parent: Dict[str, Optional[str]] = {}
    node_to_section: Dict[str, str] = {}
    node_to_chapter: Dict[str, str] = {}
    section_to_chapter: Dict[str, dict] = {}

    def rec(node: dict, chapter_id: str, section_id: str, parent_id: Optional[str] = None) -> None:
        nid = node["node_id"]
        if nid in nodes:
            raise ValueError(f"Duplicate node_id encountered: {nid}")

        nodes[nid] = node
        parent[nid] = parent_id
        node_to_section[nid] = section_id
        node_to_chapter[nid] = chapter_id

        for child in node.get("children", []) or []:
            rec(child, chapter_id, section_id, nid)

    for ch in ctx.get("chapters", []) or []:
        for sec in ch.get("sections", []) or []:
            sec_id = sec["section_id"]
            sections[sec_id] = sec
            section_to_chapter[sec_id] = ch
            for node in sec.get("nodes", []) or []:
                rec(node, ch["chapter_id"], sec_id, None)

    return sections, nodes, parent, node_to_section, node_to_chapter, section_to_chapter


def section_descendant_nodes(sections: dict, section_id: str) -> List[str]:
    ids: List[str] = []

    def rec(node: dict) -> None:
        ids.append(node["node_id"])
        for child in node.get("children", []) or []:
            rec(child)

    for root in sections[section_id].get("nodes", []) or []:
        rec(root)

    return ids


def get_target_type(target_id: str, ctx_doc_id: str, graph_nodes: Dict[str, dict]) -> str:
    """
    Robust target type detection:
    1. Prefer graph_nodes metadata
    2. Fallback to ID pattern parsing
    """

    if not target_id:
        return "unknown"

    tid = str(target_id)

    # 🔹 Act
    if tid == ctx_doc_id:
        return "act"

    # 🔹 Chapter
    if tid.startswith("chp_"):
        return "chapter"

    # 🔹 FIRST: Try graph metadata (MOST reliable)
    node_type = graph_nodes.get(tid, {}).get("type")

    if node_type:
        if node_type == "node":
            # refine generic "node" using ID
            pass
        else:
            return node_type

    # 🔴 FALLBACK: ID-based parsing (specific → general)
    tid_upper = tid.upper()

    # Illustration
    if "_ILL" in tid_upper:
        return "illustration"

    # Explanation
    if "_EXP" in tid_upper:
        return "explanation"

    # Clause (ends with single letter)
    parts = tid_upper.split("_")
    if len(parts) >= 3:
        last = parts[-1]

        if last.isalpha() and len(last) == 1:
            return "clause"

        if last.startswith("N"):
            return "subsection"

    # Section (ONLY pure section)
    if tid_upper.startswith("BNS_S") and "_N" not in tid_upper:
        return "section"

    return "node"


def build_chunks(context_path: Path, graph_path: Path) -> dict:
    ctx = load_json(context_path)
    graph = load_json(graph_path)

    sections, nodes, parent, node_to_section, node_to_chapter, section_to_chapter = build_indices(ctx)
    graph_nodes = {n["id"]: n for n in graph.get("nodes", []) or []}

    graph_semantic_refs = collections.defaultdict(list)
    for edge in graph.get("edges", []) or []:
        if edge.get("type") in SEMANTIC_EDGE_TYPES and edge.get("source") and edge.get("target"):
            graph_semantic_refs[edge["source"]].append(edge)

    def ancestry_chain(node_id: str) -> List[str]:
        chain: List[str] = []
        cur = node_id
        while cur:
            chain.append(cur)
            cur = parent.get(cur)
        return list(reversed(chain))

    def section_citation(sec: dict, chapter: dict) -> dict:
        return {
            "node_id": sec["section_id"],
            "citation_text": f"Section {sec['section_number']}: {sec['section_title']}",
            "path": [
                f"Act: {ctx['act']}",
                f"Chapter {chapter['chapter_number']}: {chapter['chapter_title']}",
                f"Section {sec['section_number']}: {sec['section_title']}",
            ],
            "node_label": None,
            "parsed_label": None,
        }

    def node_citation(node_id: str) -> dict:
        sec_id = node_to_section[node_id]
        ch = section_to_chapter[sec_id]
        sec = sections[sec_id]
        node = nodes[node_id]

        path = [
            f"Act: {ctx['act']}",
            f"Chapter {ch['chapter_number']}: {ch['chapter_title']}",
            f"Section {sec['section_number']}: {sec['section_title']}",
        ]

        parts: List[str] = [f"Section {sec['section_number']}"]
        for nid in ancestry_chain(node_id):
            if nid == sec_id:
                continue
            n = nodes[nid]
            parsed = parse_label(n.get("node_label")) or infer_label_from_node_id(nid)
            token = label_to_token(parsed, n.get("node_label")) if parsed else ""
            if not token:
                continue
            path.append(token)
            parts.append(token)

        citation_text = parts[0]
        for token in parts[1:]:
            citation_text += token if token.startswith("(") else f" {token}"
        citation_text += f", {ctx['act']}"

        return {
            "node_id": node_id,
            "citation_text": citation_text,
            "path": path,
            "node_label": node.get("node_label"),
            "parsed_label": parse_label(node.get("node_label")) or infer_label_from_node_id(node_id),
        }

    def refs_for_node(node_id: str) -> List[dict]:
        node = nodes[node_id]
        out: List[dict] = []

        # References explicitly stored in context_filled.json
        for ref in node.get("references", []) or []:
            if isinstance(ref, dict):
                target_id = ref.get("target_id")
                target_type = ref.get("target_type")
                relation = ref.get("relation") or "REFERS_TO"
            else:
                target_id = str(ref)
                target_type = None
                relation = "REFERS_TO"

            if not target_id:
                continue

            if not target_type:
                target_type = get_target_type(target_id, ctx["doc_id"], graph_nodes)

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
            out.append(
                {
                    "target_id": target_id,
                    "target_type": get_target_type(target_id, ctx["doc_id"], graph_nodes),
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

    def node_chunk_type(node: dict) -> str:
        candidates = [
            (node.get("node_category") or "").lower(),
            (node.get("node_type") or "").lower(),
            (node.get("subtype") or "").lower(),
        ]
        for c in candidates:
            if c in ALLOWED_CHUNK_TYPES:
                return c

        parsed = parse_label(node.get("node_label")) or infer_label_from_node_id(node.get("node_id", ""))
        if parsed:
            kind = parsed.get("kind")
            if kind in ALLOWED_CHUNK_TYPES:
                return kind
            if kind == "body":
                return "content"

        return "mixed"

    def nearest_chunked_ancestor(node_id: str, node_to_chunk: Dict[str, str]) -> Optional[str]:
        cur = parent.get(node_id)
        while cur:
            if cur in node_to_chunk:
                return cur
            cur = parent.get(cur)
        return None

    chunks: List[dict] = []
    chunk_map: Dict[str, dict] = {}
    node_to_chunk: Dict[str, str] = {}
    section_chunk_id_map: Dict[str, str] = {}

    # 1) Section chunks: structural only, not embedded.
    for sec_id, sec in sections.items():
        chapter = section_to_chapter[sec_id]
        section_chunk_id = f"chunk__section__{sec_id}"
        section_chunk_id_map[sec_id] = section_chunk_id

        sec_text = clean_multiline(sec.get("full_text") or "")
        descendant_ids = section_descendant_nodes(sections, sec_id)

        first_chunked_descendant = None
        for nid in descendant_ids:
            if normalize_text(nodes[nid].get("text")):
                first_chunked_descendant = nid
                break

        sec_chunk = {
            "chunk_id": section_chunk_id,
            "doc_id": ctx["doc_id"],
            "act": ctx["act"],
            "chunk_type": "section",
            "chunk_order": sec.get("section_order") or 0,
            "parent_chunk_id": None,
            "chapter": {
                "chapter_id": chapter["chapter_id"],
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter["chapter_title"],
                "chapter_order": chapter.get("chapter_order"),
            },
            "section": {
                "section_id": sec["section_id"],
                "section_number": sec["section_number"],
                "section_title": sec["section_title"],
                "section_order": sec.get("section_order"),
                "section_type": sec.get("section_type"),
                "subheading": sec.get("subheading"),
            },
            "citation": section_citation(sec, chapter),
            "root_node_id": first_chunked_descendant,
            "node_ids": descendant_ids,
            "text": sec_text,
            "derived_context": None,
            "embedding_text": None,
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

    # 2) Node chunks: atomic units.
    for sec_id, sec in sections.items():
        chapter = section_to_chapter[sec_id]
        section_chunk_id = section_chunk_id_map[sec_id]
        traversal_order = 0

        def rec(node: dict) -> None:
            nonlocal traversal_order
            nid = node["node_id"]

            if chunkable(node):
                traversal_order += 1
                chunk_id = f"chunk__node__{nid}"

                parent_node_id = nearest_chunked_ancestor(nid, node_to_chunk)
                parent_chunk_id = node_to_chunk.get(parent_node_id, section_chunk_id)

                # The nearest chunked ancestor is the semantic root for grouping.
                root_node_id = nid
                cur = parent.get(nid)
                while cur:
                    if cur in node_to_chunk:
                        root_node_id = cur
                        break
                    cur = parent.get(cur)

                raw_text = clean_multiline(node.get("text") or "")
                if node.get("derived_context"):
                    derived_context = clean_multiline(node.get("derived_context"))
                else:
                    derived_context = (
                        f"Act: {ctx['act']}\n"
                        f"Chapter {chapter['chapter_number']}: {chapter['chapter_title']}\n"
                        f"Section {sec['section_number']}: {sec['section_title']}\n"
                        f"{raw_text}"
                    ).strip()

                parent_node = nodes.get(parent.get(nid)) if parent.get(nid) else None

                embedding_text = optimize_for_embedding(
                    node=node,
                    section=sec,
                    parent_node=parent_node
                )
                embedding_text = embedding_text[:MAX_EMBED_CHARS]
                parsed_label = parse_label(node.get("node_label")) or infer_label_from_node_id(nid)

                # Build citation text once, with compact legal formatting.
                citation_text = f"Section {sec['section_number']}"
                path = [
                    f"Act: {ctx['act']}",
                    f"Chapter {chapter['chapter_number']}: {chapter['chapter_title']}",
                    f"Section {sec['section_number']}: {sec['section_title']}",
                ]
                for x in ancestry_chain(nid):
                    if x == sec_id:
                        continue
                    token = label_to_token(parse_label(nodes[x].get("node_label")) or infer_label_from_node_id(x), nodes[x].get("node_label"))
                    if not token:
                        continue
                    path.append(token)
                    citation_text += token if token.startswith("(") else f" {token}"
                citation_text += f", {ctx['act']}"

                n_chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": ctx["doc_id"],
                    "act": ctx["act"],
                    "chunk_type": node_chunk_type(node),
                    "chunk_order": traversal_order,
                    "parent_chunk_id": parent_chunk_id,
                    "chapter": {
                        "chapter_id": chapter["chapter_id"],
                        "chapter_number": chapter["chapter_number"],
                        "chapter_title": chapter["chapter_title"],
                        "chapter_order": chapter.get("chapter_order"),
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
                        "node_id": nid,
                        "citation_text": citation_text,
                        "path": path,
                        "node_label": node.get("node_label"),
                        "parsed_label": parsed_label,
                    },
                    "root_node_id": root_node_id,
                    "node_ids": [nid],
                    "text": raw_text,
                    "derived_context": derived_context,
                    "embedding_text": embedding_text,
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

    # 3) Fill parent-child links.
    for chunk in chunks:
        pid = chunk.get("parent_chunk_id")
        if pid and pid in chunk_map:
            chunk_map[pid]["children_chunk_ids"].append(chunk["chunk_id"])

    for chunk in chunks:
        chunk["children_chunk_ids"] = sorted(
            chunk["children_chunk_ids"],
            key=lambda cid: (chunk_map[cid]["chunk_order"], cid),
        )

    # 4) Stable output order.
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

    # 5) Basic integrity checks.
    validate_chunks(chunks, chunk_map)

    return {
        "doc_id": ctx["doc_id"],
        "act": ctx["act"],
        "jurisdiction": ctx.get("jurisdiction"),
        "version_date": ctx.get("version_date"),
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "chunking_strategy": (
            "Section chunks are structural only and are not embedded. "
            "Every non-empty legal node becomes an atomic chunk. "
            "Node chunks use derived_context for semantic grounding and embedding_text as the optimized retrieval string. "
            "Empty container nodes are skipped, and children attach to the nearest chunked ancestor."
        ),
        "source_files": {
            "graph_json": str(graph_path),
            "context_json": str(context_path),
        },
        "chunks": chunks,
    }


def validate_chunks(chunks: List[dict], chunk_map: Dict[str, dict]) -> None:
    """
    Fail fast on obvious integrity problems.
    """
    chunk_ids = set()

    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if not cid:
            raise ValueError("Found chunk without chunk_id")
        if cid in chunk_ids:
            raise ValueError(f"Duplicate chunk_id found: {cid}")
        chunk_ids.add(cid)

        required = ["doc_id", "act", "chunk_type", "chapter", "section", "citation", "node_ids", "text"]
        for field in required:
            if field not in chunk:
                raise ValueError(f"Chunk {cid} missing required field: {field}")

        if chunk["chunk_type"] == "section":
            if chunk.get("derived_context") is not None:
                raise ValueError(f"Section chunk {cid} must not have derived_context")
            if chunk.get("embedding_text") is not None:
                raise ValueError(f"Section chunk {cid} must not have embedding_text")
        else:
            if not chunk.get("derived_context"):
                raise ValueError(f"Node chunk {cid} missing derived_context")
            if not chunk.get("embedding_text"):
                raise ValueError(f"Node chunk {cid} missing embedding_text")

        if chunk.get("parent_chunk_id") is not None and chunk["parent_chunk_id"] not in chunk_map:
            raise ValueError(f"Chunk {cid} has invalid parent_chunk_id: {chunk['parent_chunk_id']}")

        if not isinstance(chunk.get("node_ids"), list) or not chunk["node_ids"]:
            raise ValueError(f"Chunk {cid} must have non-empty node_ids list")

    # parent-child consistency
    for chunk in chunks:
        for child_id in chunk.get("children_chunk_ids", []) or []:
            if child_id not in chunk_map:
                raise ValueError(f"Chunk {chunk['chunk_id']} references missing child_chunk_id {child_id}")
            if chunk_map[child_id].get("parent_chunk_id") != chunk["chunk_id"]:
                raise ValueError(
                    f"Parent-child mismatch: {chunk['chunk_id']} -> {child_id}, "
                    f"but child parent is {chunk_map[child_id].get('parent_chunk_id')}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legal chunks.json from graph + context files.")
    parser.add_argument("--context", required=True, type=Path, help="Path to context_filled.json")
    parser.add_argument("--graph", required=True, type=Path, help="Path to graph.json")
    parser.add_argument("--output", required=True, type=Path, help="Path to output chunks.json")
    args = parser.parse_args()

    data = build_chunks(args.context, args.graph)
    save_json(args.output, data)
    
    print(f"Done. Built chunks.json successfully")


if __name__ == "__main__":
    main()
