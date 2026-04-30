#!/usr/bin/env python3
"""
Rebuild BNS JSON metadata from the existing tree structure.

What it does:
- keeps every node, text, label, type, ordering, and children exactly as-is
- ignores old id/path values completely
- regenerates id and path from the CURRENT nesting
- preserves the current tree shape; it does NOT move nodes around

Usage:
    python ingestion/build_structure/str_3.py \
        --input data/processed/jsons/structure_jsons/bns_structured5.json \
        --output data/processed/jsons/structure_jsons/rebuilt.json
"""

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def slugify_label(label: Any, node_type: str) -> str:
    """
    Convert a node label into a stable id token.

    Examples:
        "(28)"           -> "28"
        "(a)"            -> "a"
        "Explanation 1"   -> "explanation_1"
        "Illustration"    -> "illustration"
        "body"            -> "body"
    """
    if node_type == "content":
        return "body"

    if label is None:
        return node_type.lower() if node_type else "node"

    s = str(label).strip()
    if not s:
        return node_type.lower() if node_type else "node"

    # Convert "(28)" -> "28", "(a)" -> "a"
    m = re.fullmatch(r"\((.*?)\)", s)
    if m:
        s = m.group(1).strip()

    # Normalize punctuation and spaces
    s = (
        s.replace("“", "")
        .replace("”", "")
        .replace("’", "")
        .replace("'", "")
        .replace("—", " ")
        .replace("–", " ")
        .replace(".", " ")
        .replace(",", " ")
        .replace(";", " ")
        .replace(":", " ")
    )
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    return s.lower() if s else (node_type.lower() if node_type else "node")


def rebuild_nodes(
    nodes: List[Dict[str, Any]],
    chapter_num: int,
    section_num: int,
    parent_path_labels: List[Any] | None = None,
    parent_id_tokens: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Rebuild all nodes in one list.

    parent_path_labels:
        the actual path labels of the current parent chain, as human-readable labels

    parent_id_tokens:
        the already-generated id tokens for the current parent chain
    """
    parent_path_labels = parent_path_labels or []
    parent_id_tokens = parent_id_tokens or []

    rebuilt: List[Dict[str, Any]] = []
    sibling_counts: Dict[str, int] = defaultdict(int)

    for node in nodes:
        node_copy = copy.deepcopy(node)

        # Drop stale metadata
        node_copy.pop("id", None)
        node_copy.pop("path", None)

        node_type = str(node_copy.get("type", ""))
        label = node_copy.get("label", "")

        base_token = slugify_label(label, node_type)

        # Make ids unique among siblings while still readable
        sibling_counts[base_token] += 1
        occ = sibling_counts[base_token]
        id_token = base_token if occ == 1 else f"{base_token}_{occ}"

        if node_type == "content":
            # Keep existing convention: content nodes get empty path
            node_copy["path"] = []
            node_copy["id"] = f"node_{chapter_num}_{section_num}_{id_token}"

            # Children, if any, should not inherit the content node itself
            child_parent_path = parent_path_labels
            child_parent_id = parent_id_tokens
        else:
            node_path_labels = parent_path_labels + [label]
            node_copy["path"] = node_path_labels
            node_copy["id"] = "node_" + "_".join(
                [str(chapter_num), str(section_num)] + parent_id_tokens + [id_token]
            )

            child_parent_path = node_path_labels
            child_parent_id = parent_id_tokens + [id_token]

        node_copy["children"] = rebuild_nodes(
            node_copy.get("children", []),
            chapter_num,
            section_num,
            child_parent_path,
            child_parent_id,
        )

        rebuilt.append(node_copy)

    return rebuilt


def rebuild_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild the full document while preserving all chapter/section content.
    """
    out = copy.deepcopy(doc)

    for chapter in out.get("chapters", []):
        ch_num = chapter.get("chapter_number")
        for section in chapter.get("sections", []):
            sec_num = section.get("section_number")
            section["nodes"] = rebuild_nodes(section.get("nodes", []), ch_num, sec_num)

    return out


def collect_stats(doc: Dict[str, Any]) -> Tuple[int, int]:
    """
    Return total node count and duplicate id count for a quick sanity check.
    """
    ids: List[str] = []

    def walk(nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            nid = n.get("id")
            if nid is not None:
                ids.append(nid)
            walk(n.get("children", []))

    for chapter in doc.get("chapters", []):
        for section in chapter.get("sections", []):
            walk(section.get("nodes", []))

    dup_count = len(ids) - len(set(ids))
    return len(ids), dup_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="data/processed/jsons/bns_structured5.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/processed/jsons/bns_rebuilt_from_structure.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rebuilt = rebuild_document(data)

    total_nodes, duplicate_ids = collect_stats(rebuilt)
    print(f"Rebuilt successfully.")
    print(f"Total nodes: {total_nodes}")
    print(f"Duplicate ids: {duplicate_ids}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rebuilt, f, ensure_ascii=False, indent=args.indent)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()