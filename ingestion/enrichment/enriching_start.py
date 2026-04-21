import json
import re
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path("data/processed/jsons/structure_jsons/rebuilt.json")
OUTPUT_PATH = Path("data/processed/jsons/enriched_jsons/enriched.json")  # change this if you want a different output file

STOPWORDS = {
    "the", "and", "or", "to", "of", "in", "for", "on", "by", "with", "as", "is",
    "are", "be", "been", "being", "a", "an", "this", "that", "these", "those",
    "it", "its", "at", "from", "into", "shall", "may", "will", "not", "but",
    "any", "every", "all", "such", "which", "who", "whom", "whose", "under",
    "within", "without", "other", "another", "thereof", "hereof", "therein",
    "herein", "than", "then", "when", "where", "wherein", "whereby", "also",
    "same", "more", "less", "has", "have", "had", "does", "do", "did",
    "can", "could", "would", "should", "if", "unless", "otherwise", "no", "nor",
    "so", "very", "part", "section", "subsection", "clause", "clauses",
    "sub-clause", "sub-clauses", "chapter", "chapters", "act", "sanhita",
    "explanation", "illustration", "general", "law", "person", "persons",
}

SECTION_LIST_RE = re.compile(
    r"\bsections?\s+((?:\d+\s*(?:,\s*|\s+and\s+)?)+\d+|\d+)",
    re.IGNORECASE,
)
SUBSECTIONS_OF_SECTION_RE = re.compile(
    r"\bsub-?sections?\s+((?:\([^)]+\)\s*(?:,\s*|\s+and\s+)?)*)\s+of\s+section\s+(\d+)",
    re.IGNORECASE,
)
SUBSECTION_SINGLE_OF_SECTION_RE = re.compile(
    r"\bsub-?section\s+\(([^)]+)\)\s+of\s+section\s+(\d+)",
    re.IGNORECASE,
)
CLAUSES_OF_SECTION_RE = re.compile(
    r"\b(?:sub-)?clauses?\s+((?:\([^)]+\)\s*(?:,\s*|\s+and\s+)?)*)"
    r"(?:\s+of\s+section\s+(\d+))?",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(r"\bchapter\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
EXTERNAL_ACT_HINT_RE = re.compile(
    r"\bAct,\s*\d{4}\b|\bCode,\s*\d{4}\b|\bRules?,\s*\d{4}\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
LABEL_IN_PARENS_RE = re.compile(r"\(([^)]+)\)")
FIRST_INT_RE = re.compile(r"\d+")
ROMAN_RE = re.compile(
    r"^(?=[IVXLCDM]+$)M{0,4}(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)

SPECIAL_CATEGORIES = {"explanation", "illustration", "exception"}

PROVISO_CUE_RE = re.compile(
    r"\bprovided\s+(?:that|also\s+that|further\s+that)\b",
    re.IGNORECASE,
)


def normalize_ws(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def is_roman(s):
    if not s:
        return False
    return bool(ROMAN_RE.match(s.strip().upper()))


def roman_to_int(s):
    if not s:
        return None
    s = s.strip().upper()
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = vals.get(ch)
        if v is None:
            return None
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if total > 0 else None


def canonical_doc_id(raw_doc_id, act):
    if raw_doc_id and str(raw_doc_id).upper().startswith("BNS"):
        return "BNS_2023"
    if act and "bharatiya nyaya sanhita" in act.lower():
        return "BNS_2023"
    return raw_doc_id or "DOC_1"


def normalize_subheading(value):
    s = normalize_ws(value)
    if not s or s.upper() in {"NONE", "NULL"}:
        return None
    return s


def infer_section_type(title, full_text=""):
    title_l = (title or "").lower()
    if "definition" in title_l:
        return "definition"
    if "punishment" in title_l:
        return "punishment"
    if "general explanation" in title_l or "general explanations" in title_l:
        return "general_rule"
    if "short title" in title_l or "commencement" in title_l or "application" in title_l:
        return "general_rule"
    if "procedure" in title_l:
        return "procedure"
    return None


def infer_chapter_semantic_summary(chapter_title):
    title = (chapter_title or "").lower()
    if "preliminary" in title:
        return "Introduces scope, applicability and definitions"
    if "punishment" in title:
        return "Lists punishment-related provisions"
    return None


def extract_number_from_label(label):
    m = FIRST_INT_RE.search(label or "")
    return int(m.group()) if m else None


def normalize_lookup_token(label):
    label = normalize_ws(label)
    if not label:
        return ""

    low = label.lower()
    if low == "body":
        return "body"

    if low.startswith("explanation"):
        m = re.search(r"(\d+)", label)
        return f"exp{m.group(1) if m else 1}"

    if low.startswith("illustration"):
        m = re.search(r"(\d+)", label)
        return f"ill{m.group(1) if m else 1}"

    if low.startswith("exception"):
        m = re.search(r"(\d+)", label)
        return f"ecp{m.group(1) if m else 1}"

    m = re.fullmatch(r"\(([^)]+)\)", label)
    if not m:
        m = re.search(r"\(([^)]+)\)", label)

    token = m.group(1).strip() if m else label.strip()

    if token.isdigit():
        return token
    if is_roman(token):
        return token.lower()

    token = re.sub(r"[^A-Za-z0-9]+", "", token)
    return token.lower()


def normalize_id_token(label):
    label = normalize_ws(label)
    if not label:
        return "X"

    if label.lower() == "body":
        return "BODY"

    if label.lower().startswith("explanation"):
        m = re.search(r"(\d+)", label)
        return f"EXP{m.group(1) if m else 1}"

    if label.lower().startswith("illustration"):
        m = re.search(r"(\d+)", label)
        return f"ILL{m.group(1) if m else 1}"

    if label.lower().startswith("exception"):
        m = re.search(r"(\d+)", label)
        return f"ECP{m.group(1) if m else 1}"

    m = re.fullmatch(r"\(([^)]+)\)", label)
    if m:
        token = m.group(1).strip()
    else:
        token = label.strip()

    token = token.replace("/", "_").upper()
    if token.isdigit():
        return f"N{token}"
    if is_roman(token):
        return token.upper()

    token = re.sub(r"[^A-Za-z0-9]+", "", token)
    return token.upper() or "X"


def normalize_word_token(token):
    token = token.strip()
    if not token:
        return ""
    token = token.strip("().,;:!?“”\"'")
    if not token:
        return ""
    if token.isdigit():
        return token
    if is_roman(token):
        return token.lower()
    return re.sub(r"[^A-Za-z0-9]+", "", token).lower()


def make_node_category(src_type, parent_category):
    t = (src_type or "").lower()

    if t == "content":
        return "content"

    if t == "num":
        if parent_category in SPECIAL_CATEGORIES:
            return parent_category
        return "subsection"

    if t in {"alpha", "roman"}:
        if parent_category in SPECIAL_CATEGORIES:
            return parent_category
        return "clause"

    if t in {"explanation", "illustration", "exception"}:
        return t

    if parent_category in SPECIAL_CATEGORIES:
        return parent_category

    return "clause"


def make_path_token(src_node, node_category, sibling_index):
    src_type = (src_node.get("type") or "").lower()
    label = normalize_ws(src_node.get("label") or "")

    if src_type == "content" or label.lower() == "body":
        return "body"

    if src_type == "num":
        n = extract_number_from_label(label)
        return str(n or sibling_index)

    if src_type in {"alpha", "roman"}:
        tok = normalize_lookup_token(label)
        return tok or str(sibling_index)

    if node_category == "explanation":
        n = extract_number_from_label(label)
        return f"exp{n or sibling_index}"

    if node_category == "illustration":
        n = extract_number_from_label(label)
        return f"ill{n or sibling_index}"

    if node_category == "exception":
        n = extract_number_from_label(label)
        return f"ecp{n or sibling_index}"

    tok = normalize_lookup_token(label)
    return tok or str(sibling_index)


def make_id_token(src_node, node_category, sibling_index):
    src_type = (src_node.get("type") or "").lower()
    label = normalize_ws(src_node.get("label") or "")

    if src_type == "content" or label.lower() == "body":
        return "BODY"

    if src_type == "num":
        n = extract_number_from_label(label)
        return f"N{n or sibling_index}"

    if src_type in {"alpha", "roman"}:
        tok = normalize_id_token(label)
        return tok or f"X{sibling_index}"

    if node_category == "explanation":
        n = extract_number_from_label(label)
        return f"EXP{n or sibling_index}"

    if node_category == "illustration":
        n = extract_number_from_label(label)
        return f"ILL{n or sibling_index}"

    if node_category == "exception":
        n = extract_number_from_label(label)
        return f"ECP{n or sibling_index}"

    tok = normalize_id_token(label)
    return tok or f"X{sibling_index}"


def truncate_text(text, limit=140):
    text = normalize_ws(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def extract_keywords(*texts, limit=10):
    words = []
    for text in texts:
        for tok in WORD_RE.findall(normalize_ws(text).lower()):
            tok = tok.strip("-'")
            if len(tok) <= 3:
                continue
            if tok in STOPWORDS:
                continue
            if tok.isdigit():
                continue
            words.append(tok)
    counts = Counter(words)
    return [w for w, _ in counts.most_common(limit)]


# def build_embedding_text(doc, chapter, section, node):
#     lines = [
#         f"Act: {doc.get('act', '')}",
#         f"Chapter {chapter.get('chapter_number', '')}: {chapter.get('chapter_title', '')}",
#         f"Section {section.get('section_number', '')}: {section.get('section_title', '')}",
#         f"{node.get('node_label', '')}: {normalize_ws(node.get('text', ''))}",
#     ]

#     child_bits = []
#     for child in node.get("children", [])[:8]:
#         c_label = normalize_ws(child.get("label") or "")
#         c_text = normalize_ws(child.get("text") or "")
#         if c_text:
#             child_bits.append(f"{c_label}: {truncate_text(c_text, 130)}")

#     if child_bits:
#         lines.append("Children: " + " || ".join(child_bits))

#     return "\n".join(lines)




def remove_boundary_noise(text):
    noise_phrases = ["Definitions.", "General explanations."]
    for phrase in noise_phrases:
        if text.strip().endswith(phrase):
            text = text[: -len(phrase)].strip()
    return text

def fix_fragment(text, section):
    lower = text.lower()

    if lower.startswith(("who ", "which ", "that ", "such ", "when ", "if ")):
        return f"In the context of Section {section.get('section_number')}, {text}"

    return text




def build_embedding_text(doc, chapter, section, node):
    text = normalize_ws(node.get("text", ""))

    # --- SKIP completely empty nodes ---
    if not text:
        return ""

    # --- Clean boundary noise ---
    text = remove_boundary_noise(text)
    
    
    # --- node type context ---
    node_type = node.get("node_type")

    if node_type == "explanation":
        text = "Explanation: " + text
    elif node_type == "illustration":
        text = "Illustration: " + text

    # --- Fix fragment sentences (light fix, not LLM) ---
    text = fix_fragment(text, section)

    lines = [
        f"Act: {doc.get('act', '')}",
        f"Chapter {chapter.get('chapter_number', '')}: {chapter.get('chapter_title', '')}",
        f"Section {section.get('section_number', '')}: {section.get('section_title', '')}",
        text  # ❗ removed node_label
    ]

    return "\n".join(lines)






def chapter_number_to_roman(n):
    if n <= 0:
        return ""
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def is_probably_external_reference(full_text, match_end):
    window = full_text[match_end: match_end + 90]
    return bool(EXTERNAL_ACT_HINT_RE.search(window))


def parse_int_list(blob):
    return [int(x) for x in re.findall(r"\d+", blob or "")]


def parse_label_list(blob):
    labels = []
    for item in LABEL_IN_PARENS_RE.findall(blob or ""):
        tok = normalize_lookup_token(item)
        if tok:
            labels.append(tok)
    return labels


def dedupe_refs(refs):
    seen = set()
    out = []
    for ref in refs:
        key = (ref.get("target_type"), ref.get("target_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def has_proviso_cue(text):
    return bool(PROVISO_CUE_RE.search(normalize_ws(text)))


def build_reference_registry():
    return {
        "chapter_id_by_num": {},
        "section_id_by_num": {},
        "node_by_id": {},
        "node_meta_by_id": {},
        "children_by_parent": defaultdict(list),
        "path_to_node_id": defaultdict(dict),
        "labels_in_section": defaultdict(lambda: defaultdict(list)),
    }


def build_node(src_node, doc, chapter, section, parent_node_id, parent_category,
               registry, path_tokens, sibling_counters, label_counters):
    src_type = (src_node.get("type") or "").lower()
    label = normalize_ws(src_node.get("label") or "")
    node_category = make_node_category(src_type, parent_category)

    parent_key = parent_node_id or section["section_id"]
    label_key = normalize_lookup_token(label) or normalize_id_token(label)
    label_counters[parent_key][label_key] += 1
    occurrence_index = label_counters[parent_key][label_key]

    sibling_counters[parent_key][node_category] += 1
    sibling_index = sibling_counters[parent_key][node_category]

    id_token = make_id_token(src_node, node_category, sibling_index)
    path_token = make_path_token(src_node, node_category, sibling_index)

    # Preserve the current naming scheme for first occurrences, but make
    # repeated sibling labels unique so proviso-style repeated (i)/(ii)
    # nodes no longer collide.
    if occurrence_index > 1:
        id_token = f"{id_token}_{occurrence_index}"
        path_token = f"{path_token}_{occurrence_index}"

    section_id = section.get("section_id")
    if not section.get("section_id", "").startswith("BNS_"):
        print("ERROR: Wrong section_id ->", section.get("section_id"))
    node_id = (
        f"{section_id}_{id_token}"
        if parent_node_id is None
        else f"{parent_node_id}_{id_token}"
    )

    text = normalize_ws(src_node.get("text"))

    out_node = {
        "node_id": node_id,
        "node_label": label,
        "node_category": node_category,
        "level": node_category,
        "text": text,
        "embedding_text": build_embedding_text(doc, chapter, section, src_node),
        "semantic_summary": None,
        "legal_concepts": [],
        "plain_english_paraphrase": None,
        "keywords": extract_keywords(
            text,
            section.get("section_title", ""),
            chapter.get("chapter_title", ""),
            label,
        ),
        "node_type": node_category if node_category in SPECIAL_CATEGORIES else None,
        "exception": node_category == "exception",
        "parent_node_id": parent_node_id or section["section_id"],
        "children": [],
        "explains_node_id": (parent_node_id or section["section_id"])
        if node_category in SPECIAL_CATEGORIES
        else None,
    }

    registry["node_by_id"][node_id] = out_node
    registry["node_meta_by_id"][node_id] = {
        "section_number": section["section_number"],
        "section_id": section["section_id"],
        "chapter_number": chapter["chapter_number"],
        "chapter_id": chapter["chapter_id"],
        "label": label,
        "norm_label": normalize_lookup_token(label),
        "node_category": node_category,
        "src_type": src_type,
        "parent_node_id": parent_node_id,
        "path_tokens": tuple(path_tokens + [path_token]),
        "text": text,
    }

    registry["children_by_parent"][parent_node_id or section["section_id"]].append(node_id)
    registry["path_to_node_id"][section["section_number"]][tuple(path_tokens + [path_token])] = node_id

    norm_label = normalize_lookup_token(label)
    if norm_label:
        registry["labels_in_section"][section["section_number"]][norm_label].append(node_id)

    child_counts = defaultdict(lambda: defaultdict(int))
    for child in src_node.get("children", []) or []:
        child_out = build_node(
            child,
            doc,
            chapter,
            section,
            node_id,
            node_category,
            registry,
            list(path_tokens + [path_token]),
            child_counts,
            label_counters,
        )
        out_node["children"].append(child_out)

    return out_node



def validate_ids(data):
    seen = set()
    dupes = []

    def walk(node):
        nid = node["node_id"]
        if not nid.startswith("BNS_"):
            print("BAD ID:", nid)
        if nid in seen:
            dupes.append(nid)
        else:
            seen.add(nid)
        for child in node.get("children", []):
            walk(child)

    for ch in data["chapters"]:
        for sec in ch["sections"]:
            for node in sec["nodes"]:
                walk(node)

    if dupes:
        print("DUPLICATE IDS:", dupes[:25], "..." if len(dupes) > 25 else "")






def resolve_path_reference(section_num, labels, registry):
    return registry["path_to_node_id"].get(section_num, {}).get(tuple(labels))


def find_unique_node_by_label_in_section(section_num, label, registry):
    matches = registry["labels_in_section"].get(section_num, {}).get(label, [])
    if len(matches) == 1:
        return matches[0]
    return None


def find_in_subtree(root_id, label, registry):
    matches = []
    stack = list(registry["children_by_parent"].get(root_id, []))
    while stack:
        nid = stack.pop()
        meta = registry["node_meta_by_id"].get(nid, {})
        if meta.get("norm_label") == label:
            matches.append(nid)
        stack.extend(registry["children_by_parent"].get(nid, []))
    return matches


def resolve_references_for_node(node, registry):
    text = normalize_ws(node.get("text"))
    if not text:
        return []

    refs = []
    current_node_id = node["node_id"]
    current_section_num = registry["node_meta_by_id"][current_node_id]["section_number"]

    # Chapter references
    for m in CHAPTER_RE.finditer(text):
        raw = m.group(1).strip()
        if raw.isdigit():
            chapter_num = int(raw)
        else:
            chapter_num = roman_to_int(raw)
        if chapter_num is None:
            continue
        chapter_id = registry["chapter_id_by_num"].get(chapter_num)
        if chapter_id:
            refs.append({"target_id": chapter_id, "target_type": "chapter"})

    # Section references
    for m in SECTION_LIST_RE.finditer(text):
        if is_probably_external_reference(text, m.end()):
            continue
        nums = parse_int_list(m.group(1))
        for num in nums:
            sec_id = registry["section_id_by_num"].get(num) or f"BNS_S{num}"
            refs.append({"target_id": sec_id, "target_type": "section"})

    # "sub-sections (2), (3) and (4) of section 8"
    for m in SUBSECTIONS_OF_SECTION_RE.finditer(text):
        if is_probably_external_reference(text, m.end()):
            continue
        labels = parse_label_list(m.group(1))
        sec_num = int(m.group(2))
        for lbl in labels:
            nid = resolve_path_reference(sec_num, [lbl], registry)
            if nid:
                refs.append({"target_id": nid, "target_type": "node"})

    # "sub-section (1) of section 189"
    for m in SUBSECTION_SINGLE_OF_SECTION_RE.finditer(text):
        if is_probably_external_reference(text, m.end()):
            continue
        lbl = normalize_lookup_token(m.group(1))
        sec_num = int(m.group(2))
        nid = resolve_path_reference(sec_num, [lbl], registry)
        if nid:
            refs.append({"target_id": nid, "target_type": "node"})

    # Clauses / sub-clauses
    for m in CLAUSES_OF_SECTION_RE.finditer(text):
        if is_probably_external_reference(text, m.end()):
            continue

        labels = parse_label_list(m.group(1))
        explicit_sec = m.group(2)
        sec_num = int(explicit_sec) if explicit_sec else current_section_num

        for lbl in labels:
            nid = None

            # Prefer direct subtree resolution when no explicit section is given
            if not explicit_sec:
                direct_matches = find_in_subtree(current_node_id, lbl, registry)
                if len(direct_matches) == 1:
                    nid = direct_matches[0]

            # If explicit section or subtree resolution failed, only use unique label in that section
            if nid is None:
                nid = find_unique_node_by_label_in_section(sec_num, lbl, registry)

            if nid:
                refs.append({"target_id": nid, "target_type": "node"})

    return dedupe_refs(refs)


def set_references_recursive(node, registry):
    node["references"] = resolve_references_for_node(node, registry)
    for child in node.get("children", []):
        set_references_recursive(child, registry)


def transform_doc(doc):
    out = {
        "doc_id": canonical_doc_id(doc.get("doc_id"), doc.get("act")),
        "act": doc.get("act"),
        "jurisdiction": doc.get("jurisdiction"),
        "version_date": doc.get("version_date"),
        "chapters": [],
    }

    registry = build_reference_registry()

    for chapter in doc.get("chapters", []) or []:
        chapter_number = chapter.get("chapter_number")
        chapter_id = chapter.get("chapter_id") or f"chp_{chapter_number}"

        registry["chapter_id_by_num"][chapter_number] = chapter_id

        out_chapter = {
            "chapter_id": chapter_id,
            "chapter_number": chapter_number,
            "chapter_title": chapter.get("chapter_title"),
            "chapter_order": chapter.get("chapter_order", chapter_number),
            "semantic_summary": infer_chapter_semantic_summary(chapter.get("chapter_title")),
            "sections": [],
        }

        for section in chapter.get("sections", []) or []:
            section_number = section.get("section_number")
            section_id = f"BNS_S{section_number}"
            registry["section_id_by_num"][section_number] = section_id
            
            section = {
                **section,
                "section_id": section_id
            }

            out_section = {
                "section_id": section_id,
                "section_number": section_number,
                "section_title": section.get("section_title"),
                "section_order": section.get("section_order", section_number),
                "subheading": normalize_subheading(section.get("subheading")),
                "section_type": infer_section_type(section.get("section_title"), section.get("full_text", "")),
                "semantic_summary": None,
                "parent_chapter_id": chapter_id,
                "full_text": section.get("full_text"),
                "nodes": [],
            }

            sibling_counters = defaultdict(lambda: defaultdict(int))
            label_counters = defaultdict(lambda: defaultdict(int))
            for node in section.get("nodes", []) or []:
                out_node = build_node(
                    node,
                    doc,
                    chapter,
                    section,
                    None,
                    None,
                    registry,
                    [],
                    sibling_counters,
                    label_counters,
                )
                out_section["nodes"].append(out_node)

            out_chapter["sections"].append(out_section)

        out["chapters"].append(out_chapter)

    # Second pass: resolve references using the fully built tree
    for chapter in out["chapters"]:
        for section in chapter["sections"]:
            for node in section["nodes"]:
                set_references_recursive(node, registry)

    return out


def main():
    doc = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    enriched = transform_doc(doc)
    validate_ids(enriched)
    OUTPUT_PATH.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()