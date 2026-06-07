import json
import re
from collections import defaultdict

# python ingestion/build_graph/graph.py --context data/processed/jsons/enriched_jsons/context_filled.json --output data/processed/jsons/graph_jsons/graph2.json


# -----------------------------
# Helpers
# -----------------------------

def add_edge(edges, src, tgt, etype):
    edges.append({
        "source": src,
        "target": tgt,
        "type": etype
    })

def clean_text(text):
    return text if text else ""

# -----------------------------
# Regex for references
# -----------------------------

SECTION_PATTERN = re.compile(r"section\s+(\d+)", re.IGNORECASE)

def extract_section_refs(text):
    refs = []
    for match in SECTION_PATTERN.findall(text or ""):
        refs.append(f"BNS_S{match}")
    return refs

# -----------------------------
# Main builder
# -----------------------------

def build_graph(context_json):
    nodes = []
    edges = []
    node_map = {}

    doc_id = context_json["doc_id"]
    act_name = context_json["act"]

    # -----------------------------
    # Create ACT node
    # -----------------------------
    act_node = {
        "id": doc_id,
        "type": "act",
        "subtype": "act",
        "text": act_name,
        "parent_id": None,
        "chapter_id": None,
        "section_id": None,
        "order": None
    }
    nodes.append(act_node)
    node_map[doc_id] = act_node

    # -----------------------------
    # Chapter → Section → Nodes
    # -----------------------------
    for ch in context_json["chapters"]:
        ch_id = ch["chapter_id"]

        chapter_node = {
            "id": ch_id,
            "type": "chapter",
            "subtype": "chapter",
            "text": ch["chapter_title"],
            "parent_id": doc_id,
            "chapter_id": ch_id,
            "section_id": None,
            "order": ch["chapter_order"]
        }

        nodes.append(chapter_node)
        node_map[ch_id] = chapter_node

        add_edge(edges, doc_id, ch_id, "PARENT_CHILD")
        add_edge(edges, ch_id, doc_id, "PARENT_CHILD")

        for sec in ch["sections"]:
            sec_id = sec["section_id"]

            section_node = {
                "id": sec_id,
                "type": "section",
                "subtype": "section",
                "text": sec["section_title"],
                "parent_id": ch_id,
                "chapter_id": ch_id,
                "section_id": sec_id,
                "order": sec["section_order"]
            }

            nodes.append(section_node)
            node_map[sec_id] = section_node

            add_edge(edges, ch_id, sec_id, "PARENT_CHILD")
            add_edge(edges, sec_id, ch_id, "PARENT_CHILD")

            # -----------------------------
            # Recursive node builder
            # -----------------------------
            def process_node(n, parent_id):
                nid = n["node_id"]

                node = {
                    "id": nid,
                    "type": "node",
                    "subtype": n.get("level", "content"),
                    "text": clean_text(n.get("text")),
                    "parent_id": parent_id,
                    "chapter_id": ch_id,
                    "section_id": sec_id,
                    "order": None
                }

                nodes.append(node)
                node_map[nid] = node

                # parent-child
                add_edge(edges, parent_id, nid, "PARENT_CHILD")
                add_edge(edges, nid, parent_id, "PARENT_CHILD")

                # explanation edge
                if n.get("explains_node_id"):
                    target = n["explains_node_id"]
                    add_edge(edges, nid, target, "EXPLAINS")
                    add_edge(edges, target, nid, "EXPLAINS")

                # illustration edge
                if n.get("node_category") == "illustration":
                    add_edge(edges, nid, parent_id, "ILLUSTRATES")
                    add_edge(edges, parent_id, nid, "ILLUSTRATES")

                # references
                refs = extract_section_refs(n.get("text"))
                for r in refs:
                    add_edge(edges, nid, r, "REFERS_TO")

                # recurse children
                children = n.get("children", [])
                for child in children:
                    process_node(child, nid)

                return nid

            # process top-level section nodes
            section_children = sec.get("nodes", [])
            child_ids = []

            for n in section_children:
                cid = process_node(n, sec_id)
                child_ids.append(cid)

            # -----------------------------
            # SIBLING edges (ordered chain)
            # -----------------------------
            def connect_siblings(ids):
                for i in range(len(ids) - 1):
                    add_edge(edges, ids[i], ids[i+1], "SIBLING")
                    add_edge(edges, ids[i+1], ids[i], "SIBLING")

            connect_siblings(child_ids)

    # -----------------------------
    # Final graph
    # -----------------------------
    graph = {
        "metadata": {
            "doc_id": doc_id,
            "act": act_name
        },
        "nodes": nodes,
        "edges": edges
    }

    return graph


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.context, "r") as f:
        context_json = json.load(f)

    graph = build_graph(context_json)

    with open(args.output, "w") as f:
        json.dump(graph, f, indent=2)

    print("Graph built successfully")