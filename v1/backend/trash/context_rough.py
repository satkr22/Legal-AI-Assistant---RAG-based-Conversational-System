import json


INPUT_FILE = "data/processed/jsons/enriched_jsons/enriched.json"
OUTPUT_FILE = "data/processed/jsons/enriched_jsons/context.json"


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    if not text:
        return ""

    text = text.replace("\n", " ").strip()

    # Remove trailing ingestion noise like headings
    for suffix in ["Definitions.", "General explanations."]:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()

    return text


# -----------------------------
# CLASSIFICATION LOGIC (refined)
# -----------------------------
def classify_node(node):
    text = (node.get("text") or "").strip().lower()
    label = (node.get("node_label") or "").lower()
    node_type = node.get("node_type")

    # NOISY (keep but treat differently)
    if node_type in ["illustration", "explanation"]:
        return "noisy"

    if "illustration" in label or "explanation" in label:
        return "noisy"

    # WEAK CONDITIONS (improved)
    weak_starts = [
        "any", "such", "he", "it", "they", "who",
        "shall", "nothing", "whoever", "where", "when"
    ]

    if len(text.split()) < 12:
        return "weak"

    if any(text.startswith(w) for w in weak_starts):
        return "weak"

    if label.startswith("(") and len(label) <= 4:
        return "weak"

    # STRONG
    return "strong"


# -----------------------------
# BUILD CONTEXT STRING (refined)
# -----------------------------
def build_context(act, chapter, section, context_stack, node_class):
    parts = []

    parts.append(f"Act: {act}")
    parts.append(f"Chapter {chapter['chapter_number']}: {chapter['chapter_title']}")
    parts.append(f"Section {section['section_number']}: {section['section_title']}")

    # Include hierarchical context intelligently
    if node_class in ["weak", "noisy"]:
        context = context_stack[-3:]  # more context for weak/noisy nodes
    else:
        context = context_stack[-1:]  # minimal for strong nodes

    parts.extend(context)

    return "\n".join(parts)


# -----------------------------
# RECURSIVE NODE PROCESSOR
# -----------------------------
def process_node(node, act, chapter, section, context_stack):
    raw_text = node.get("text", "")
    node_text = clean_text(raw_text)

    node_class = classify_node(node)

    current_stack = context_stack.copy()

    # Only push meaningful text
    if node_text:
        current_stack.append(node_text)

    # Skip empty nodes for embeddings
    if node_text:
        node["derived_context"] = build_context(
            act,
            chapter,
            section,
            current_stack,
            node_class
        )
    else:
        node["derived_context"] = None

    # recurse children
    for child in node.get("children", []):
        process_node(
            child,
            act,
            chapter,
            section,
            current_stack
        )


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    act = data.get("act")

    for chapter in data.get("chapters", []):
        for section in chapter.get("sections", []):
            for node in section.get("nodes", []):
                process_node(
                    node,
                    act,
                    chapter,
                    section,
                    context_stack=[]
                )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Done. Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()