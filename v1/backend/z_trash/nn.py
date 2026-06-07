import json
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
FORMATTED_FILE = Path("data/processed/text/formatted.txt")
OUTPUT_FILE    = Path("data/processed/text/output.txt")
OUT_JSON       = Path("data/processed/jsons/bns_structured3.json")


# ─────────────────────────────────────────────────────────────
# ROMAN → INT
# ─────────────────────────────────────────────────────────────
def roman_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    val = {"I": 1, "V": 5, "X": 10, "L": 50,
           "C": 100, "D": 500, "M": 1000}
    s = s.upper()
    result, prev = 0, 0
    for ch in reversed(s):
        v = val.get(ch, 0)
        result += v if v >= prev else -v
        prev = v
    return result


# ─────────────────────────────────────────────────────────────
# STEP 1 — PARSE STRUCTURE
# ─────────────────────────────────────────────────────────────
def parse_formatted(path: Path):
    chapters = []
    current_chapter = None
    current_subheading = "NONE"

    chapter_re = re.compile(r'^CHAPTER\s+((?:[IVXLCDM]+|\d+))\s*[-–]\s*(.+)', re.I)
    subhead_re = re.compile(r'^SUBHEADING\s*[-–]\s*(.+)', re.I)
    section_re = re.compile(r'^SECTION\s+(\d+)\s*[-–]\s*(.+)', re.I)

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if m := chapter_re.match(line):
            current_chapter = {
                "chapter_number": roman_to_int(m.group(1)),
                "chapter_title": m.group(2).strip(),
                "sections": []
            }
            chapters.append(current_chapter)
            current_subheading = "NONE"
            continue

        if m := subhead_re.match(line):
            sub = m.group(1).strip()
            current_subheading = "NONE" if sub.upper() == "NONE" else sub
            continue

        if m := section_re.match(line):
            current_chapter["sections"].append({
                "section_number": int(m.group(1)),
                "section_title": m.group(2).strip(),
                "subheading": current_subheading
            })

    return chapters


# ─────────────────────────────────────────────────────────────
# STEP 2 — SPLIT RAW SECTIONS (FIXED)
# ─────────────────────────────────────────────────────────────
def split_sections(path: Path):
    text = path.read_text(encoding="utf-8")

    header_re = re.compile(
        r'Section\s+(\d+)\.\s+[^\n]+\n─{5,}\n',
        re.MULTILINE
    )

    matches = list(header_re.finditer(text))
    sections = {}

    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        block = text[start:end]

        # 🔥 CLEANING (VERY IMPORTANT)
        block = re.sub(r'CHAPTER\s+[IVXLCDM]+.*', '', block)
        block = re.sub(r'=+', '', block)
        block = re.sub(r'─+', '', block)
        block = re.sub(r'\n+', '\n', block)
        block = re.sub(r'OF\s+[A-Z\s]+$', '', block)
        block = re.sub(r'[A-Z\s]{10,}$', '', block)

        sections[sec_num] = block.strip()

    return sections


# ─────────────────────────────────────────────────────────────
# FIXED LABEL DETECTION (SCHEMA ALIGNED)
# ─────────────────────────────────────────────────────────────
def detect_label(line, prev_label=None, next_line=None):
    line = line.strip()

    # Explanation
    if m := re.match(r'^(Explanation\s*\d*)[\.\-–—:]*\s*(.*)', line, re.I):
        return ("explanation", m.group(1).strip(), m.group(2).strip())

    # Illustration
    if m := re.match(r'^(Illustrations?)[\.\-–—:]*\s*(.*)', line, re.I):
        return ("illustration", "Illustration", m.group(2).strip())

    # Exception
    if m := re.match(r'^(Exception)[\.\-–—:]*\s*(.*)', line, re.I):
        return ("exception", "Exception", m.group(2).strip())

    # (1)
    if m := re.match(r'^\((\d+)\)\s+(.*)', line):
        return ("content", f"({m.group(1)})", m.group(2).strip())

    # 🔥 HANDLE (i) AMBIGUITY
    if m := re.match(r'^\((i)\)\s+(.*)', line, re.I):
        label = "(i)"
        text = m.group(2).strip()

        # ---- LOOK AHEAD ----
        if next_line:
            if re.match(r'^\((ii|iii|iv)\)', next_line, re.I):
                return ("content", label, text)  # roman

            if re.match(r'^\((j)\)', next_line, re.I):
                return ("content", label, text)  # alpha

        # ---- LOOK BACK ----
        if prev_label:
            if prev_label == "(h)":
                return ("content", label, text)  # alpha

        # ---- DEFAULT ----
        return ("content", label, text)  # roman

    # (a)-(z)
    if m := re.match(r'^\(([a-hj-z])\)\s+(.*)', line):
        return ("content", f"({m.group(1)})", m.group(2).strip())

    # (ii), (iii)
    if m := re.match(r'^\(([ivxlcdm]{2,})\)\s+(.*)', line, re.I):
        return ("content", f"({m.group(1)})", m.group(2).strip())

    return ("content", None, line)


# ─────────────────────────────────────────────────────────────
# FIXED PARAGRAPH MERGING
# ─────────────────────────────────────────────────────────────
def _is_strong_label_start(line):
    return bool(
        re.match(r'^\(\d+\)\s+[A-Z“"]', line) or
        re.match(r'^\([a-z]\)\s+\S', line) or
        re.match(r'^\([ivxlcdm]+\)\s+\S', line, re.I) or
        re.match(r'^Explanation', line, re.I) or
        re.match(r'^Illustrations?\b', line, re.I) or
        re.match(r'^Exception', line, re.I)
    )


def split_inline_labels(text):

    
    # 1. DO NOT split legal references
    # ❗ Only block splitting if it's PURE reference text (not clause list)
    # 🔥 Only skip splitting if there are NO clause patterns
    if not re.search(r'\([a-z]\)|\(\d+\)|\([ivxlcdm]+\)', text, re.I):
        return [text]

    # 2. HANDLE "namely" → split into parent + clauses
    if re.search(r'namely[:—\-]', text, re.I):
        parts = re.split(r'(?=\([a-z]\))', text)
        return [p.strip() for p in parts if p.strip()]

    # 3. Only block pure header, not inline content
    if re.fullmatch(r'Illustrations?\s*[\.\:\-–—]?\s*', text.strip(), re.I):
        return [text]

    # 4. DO NOT split explanation header
    if re.fullmatch(r'Explanation\s*\d*\s*[\.\:\-–—]?\s*', text.strip(), re.I):
        return [text]

    # 5. SAFE SPLIT (only semicolon-based)
    parts = re.split(
    r'(?<=[\.;])\s*(?=\([a-z]\)|\(\d+\)|\([ivxlcdm]+\))',
    text,
    flags=re.I
)
    
    # 6. detect numeric clause after semicolon + and/or
    if re.search(r';\s*(and|or)\s*\(\d+\)', text, re.I):
        parts = re.split(r';\s*(?=(?:and|or)\s*\(\d+\))', text, flags=re.I)
    

    return [p.strip() for p in parts if p.strip()]





def merge_paragraphs(text):
    lines = text.splitlines()
    paragraphs = []
    current = []
    in_explanation = False   # True while accumulating an Explanation block

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            if current:
                combined = " ".join(current)
                paragraphs.extend(split_inline_labels(combined))
                current = []
            in_explanation = False
            continue

        # 🔥 ADD THIS BLOCK HERE (EXACT POSITION)
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if re.match(r'^Illustrations?\s*[\.\:\-–—]?\s*$', stripped, re.I) and (
    re.match(r'^\([a-z]\)', next_line) or
    re.match(r'^[A-Z][a-z]+\s', next_line)
):
            if current:
                combined = " ".join(current)
                paragraphs.extend(split_inline_labels(combined))
                current = []
            paragraphs.append(stripped)
            continue

        # EXISTING LOGIC CONTINUES
        is_strong = _is_strong_label_start(stripped)
        if re.match(r'^\(\d+\)\s+(and|or)\b', stripped, re.I):
            is_strong = False

        # Inside an Explanation, sub-clauses like (a)/(b)/(c)/(i)/(ii) must NOT
        # flush the buffer — they are part of the Explanation text, not new nodes.
        # Only a numeric clause "(29)" or a new Explanation/Illustration/Exception
        # is allowed to break out of the Explanation block.
        if in_explanation and is_strong:
            # allow (a)(b)(c) to become separate nodes
            if re.match(r'^\([a-z]\)|^\([ivxlcdm]+\)', stripped, re.I):
                if current:
                    combined = " ".join(current)
                    paragraphs.extend(split_inline_labels(combined))
                current = [stripped]
                continue

            if (re.match(r'^\(\d+\)\s+', stripped) or
                    re.match(r'^Explanation', stripped, re.I) or
                    re.match(r'^Illustration', stripped, re.I) or
                    re.match(r'^Exception', stripped, re.I)):
                if current:
                    combined = " ".join(current)
                    paragraphs.extend(split_inline_labels(combined))
                current = [stripped]
                in_explanation = bool(re.match(r'^Explanation', stripped, re.I))
            else:
                current.append(stripped)
            continue

        if is_strong:
            if current:
                combined = " ".join(current)
                paragraphs.extend(split_inline_labels(combined))
            current = [stripped]
            in_explanation = bool(re.match(r'^Explanation', stripped, re.I))
        else:
            current.append(stripped)

    if current:
        combined = " ".join(current)
        paragraphs.extend(split_inline_labels(combined))

    return paragraphs


# ─────────────────────────────────────────────────────────────
# CORRECT LEVEL (MATCHES SCHEMA)
# ─────────────────────────────────────────────────────────────
def get_level(label, ltype, parent_text=None):

    if label is None:
        return 0

    # SPECIAL RULE: after "namely", treat (a) as child
    if parent_text and re.search(r'namely[:—\-]', parent_text, re.I):
        if re.match(r'^\([a-z]\)$', label):
            return 2   # child level

    # normal cases
    if re.match(r'^\(\d+\)$', label):
        return 1

    if re.match(r'^\([a-z]\)$', label):
        return 1

    if re.match(r'^\([ivxlcdm]+\)$', label, re.I):
        return 1

    if ltype in ("explanation", "illustration", "exception"):
        return 2

    return 1


# ─────────────────────────────────────────────────────────────
# NODE BUILDER (SCHEMA-COMPLIANT IDS)
# ─────────────────────────────────────────────────────────────
def build_tree(ch_num, sec_num, paragraphs):
    root = []
    seen_ids = set()

    active = {
        "num": None,
        "alpha": None,
        "illustration": None,
        "explanation": None,
    }

    def make_id(parent_path, ltype, label):
        base = f"node_{ch_num}_{sec_num}"

        if label:
            clean = re.sub(r'[^a-z0-9]', '', label.lower())
            candidate = f"{base}_{'_'.join(parent_path + [clean])}" if parent_path else f"{base}_{clean}"
        elif ltype:
            candidate = f"{base}_{ltype}"
        else:
            candidate = f"{base}_body"

        if candidate not in seen_ids:
            return candidate

        i = 2
        while f"{candidate}_{i}" in seen_ids:
            i += 1
        return f"{candidate}_{i}"

    nodes_stack = []

    for idx, para in enumerate(paragraphs):
        prev_para = paragraphs[idx - 1] if idx > 0 else None
        next_para = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else None

        prev_label = detect_label(prev_para)[1] if prev_para else None

        ltype, label, text = detect_label(
            para,
            prev_label=prev_label,
            next_line=next_para
        )
        # ----------------------------
        # DETERMINE LABEL TYPE
        # ----------------------------
        if label and re.match(r'^\(\d+\)$', label):
            label_type = "num"
        elif label and re.match(r'^\([a-hj-z]\)$', label):
            label_type = "alpha"
        elif label and re.match(r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)$', label, re.I):
            label_type = "roman"
        else:
            label_type = "text"

        # ----------------------------
        # CONTEXT EXIT (CRITICAL)
        # ----------------------------
        if active["illustration"] and label_type != "alpha":
            active["illustration"] = None

        if active["explanation"] and label_type == "num":
            active["explanation"] = None

        # ----------------------------
        # DETERMINE PARENT
        # ----------------------------
        if label_type == "num":
            parent = None

        elif label_type == "alpha":
            if active["illustration"]:
                parent = active["illustration"]
            elif active["explanation"]:
                parent = active["explanation"]
            else:
                parent = active["num"]

        elif label_type == "roman":
            if active["illustration"]:
                parent = active["illustration"]
            elif active["explanation"]:
                parent = active["explanation"]
            elif active["alpha"]:
                parent = active["alpha"]   # 🔥 KEY FIX (i under k)
            else:
                parent = active["num"]

        elif ltype in ("explanation", "illustration", "exception"):
            parent = active["num"]

        else:
            parent = active["num"]

        # ----------------------------
        # CREATE NODE
        # ----------------------------
        parent_path = parent["path"] if parent else []
        node_id = make_id(parent_path, ltype, label)
        seen_ids.add(node_id)

        node = {
            "id": node_id,
            "label": label if label else "body",
            "type": ltype,
            "text": text.strip(),
            "children": [],
            "path": parent_path + ([label.strip("()")] if label else []),
        }

        # ----------------------------
        # ATTACH NODE
        # ----------------------------
        if parent:
            parent["children"].append(node)
        else:
            root.append(node)

        # ----------------------------
        # UPDATE ACTIVE CONTEXT
        # ----------------------------
        if label_type == "num":
            active["num"] = node
            active["alpha"] = None
            active["illustration"] = None
            active["explanation"] = None

        elif ltype == "illustration":
            active["illustration"] = node
            active["explanation"] = None

        elif ltype == "explanation":
            active["explanation"] = node
            active["illustration"] = None
        
        if label_type == "alpha":
            active["alpha"] = node

        elif label_type == "roman":
            pass  # do not override alpha

    return root



# ─────────────────────────────────────────────────────────────
# MAIN OUTPUT (SCHEMA ROOT FIX)
# ─────────────────────────────────────────────────────────────
def main():
    chapters = parse_formatted(FORMATTED_FILE)
    raw_sections = split_sections(OUTPUT_FILE)

    final = {
        "doc_id": "BNS-23",
        "act": "Bharatiya Nyaya Sanhita, 2023",
        "chapters": []
    }

    for ch in chapters:
        ch_num = ch["chapter_number"]

        chapter_obj = {
            "chapter_number": ch_num,
            "chapter_id": f"chp_{ch_num}",
            "chapter_title": ch["chapter_title"],
            "chapter_order": ch_num,
            "type": "chapter",
            "sections": []
        }

        for idx, sec in enumerate(ch["sections"], start=1):
            sec_num = sec["section_number"]

            if sec_num not in raw_sections:
                continue

            raw_text = raw_sections[sec_num]
            paragraphs = merge_paragraphs(raw_text)

            nodes = build_tree(ch_num, sec_num, paragraphs)

            section_obj = {
                "section_number": sec_num,
                "section_id": f"sec_{ch_num}_{sec_num}",
                "section_title": sec["section_title"],
                "section_order": idx,
                "subheading": sec["subheading"],
                "type": "section",
                "nodes": nodes,
                "full_text": raw_text
            }

            chapter_obj["sections"].append(section_obj)

        final["chapters"].append(chapter_obj)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("PERFECT — Schema-compliant JSON generated.")
    
    
    
if __name__ == "__main__":
    main()






