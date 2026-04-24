import json
import re
from pathlib import Path

FORMATTED_FILE = Path("data/processed/raw_text/formatted.txt")
OUTPUT_FILE    = Path("data/processed/raw_text/output.txt")
OUT_JSON       = Path("data/processed/jsons/structure_jsons/bns_structured_try.json")


# ─────────────────────────────────────────────
# ROMAN → INT
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# DETECT LABEL (WITH MEMORY)
# ─────────────────────────────────────────────
def detect_label(line, prev_label=None, next_line=None, last_label_type=None):
    if not line:
        return ("content", None, "")

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
        return ("num", f"({m.group(1)})", m.group(2).strip())

    # (i) ambiguity
    if m := re.match(r'^\((i)\)\s+(.*)', line, re.I):
        label = "(i)"
        text = m.group(2).strip()

        if next_line:
            if re.match(r'^\((ii|iii|iv|v|vi|vii|viii|ix|x)\)', next_line, re.I):
                return ("roman", label, text)
            if re.match(r'^\([j-z]\)', next_line, re.I):
                return ("alpha", label, text)

        if last_label_type == "alpha":
            return ("alpha", label, text)
        if last_label_type == "roman":
            return ("roman", label, text)

        if prev_label and re.match(r'^\([a-h]\)$', prev_label, re.I):
            return ("alpha", label, text)

        return ("roman", label, text)

    # alpha
    if m := re.match(r'^\(([a-hj-z])\)\s+(.*)', line):
        return ("alpha", f"({m.group(1)})", m.group(2).strip())

    # roman
    if m := re.match(r'^\(([ivxlcdm]{2,})\)\s+(.*)', line, re.I):
        return ("roman", f"({m.group(1)})", m.group(2).strip())

    return ("content", None, line)


# ─────────────────────────────────────────────
# SEQUENCE VALIDATION
# ─────────────────────────────────────────────
def expected_next(label, ltype):
    if not label:
        return None

    if ltype == "num":
        return f"({int(label.strip('()')) + 1})"

    if ltype == "alpha":
        return f"({chr(ord(label[1]) + 1)})"

    if ltype == "roman":
        return f"({label.strip('()')}+)"  # loose

    return None


# ─────────────────────────────────────────────
# BUILD TREE (FINAL)
# ─────────────────────────────────────────────
def build_tree(ch_num, sec_num, paragraphs):
    root = []
    seen_ids = set()

    LEVELS = {
        "num": 1,
        "alpha": 2,
        "roman": 3,
        "illustration": 4,
        "explanation": 4,
        "exception": 4,
        "content": 5
    }

    stack = []
    last = {"label": None, "type": None}

    def make_id(parent_path, ltype, label):
        base = f"node_{ch_num}_{sec_num}"

        if label:
            clean = re.sub(r'[^a-z0-9]', '', label.lower())
            candidate = f"{base}_{'_'.join(parent_path + [clean])}" if parent_path else f"{base}_{clean}"
        else:
            candidate = f"{base}_body"

        if candidate not in seen_ids:
            return candidate

        i = 2
        while f"{candidate}_{i}" in seen_ids:
            i += 1
        return f"{candidate}_{i}"

    for idx, para in enumerate(paragraphs):
        prev_para = paragraphs[idx - 1] if idx > 0 else None
        next_para = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else None

        prev_label = detect_label(prev_para)[1] if prev_para else None

        ltype, label, text = detect_label(
            para,
            prev_label=prev_label,
            next_line=next_para,
            last_label_type=last["type"]
        )
        
        
        
        parent_type = stack[-1]["type"] if stack else None

        # 🔥 KEY FIX
        if parent_type in ["explanation", "illustration", "exception"] and ltype in ["alpha", "roman"]:
            current_level = LEVELS[parent_type] + 1
        else:
            current_level = LEVELS.get(ltype, 5)

        current_level = LEVELS.get(ltype, 5)

        # AUTO CLOSE
        while stack:
            top_type = stack[-1]["type"]

            # DO NOT break explanation hierarchy
            if top_type in ["explanation", "illustration", "exception"] and ltype in ["alpha", "roman"]:
                break

            if LEVELS[top_type] >= current_level:
                stack.pop()
            else:
                break

        parent = stack[-1]["node"] if stack else None
        parent_path = parent["path"] if parent else []

        node = {
            "id": make_id(parent_path, ltype, label),
            "label": label if label else "body",
            "type": ltype,
            "text": text.strip(),
            "children": [],
            "path": parent_path + ([label.strip("()")] if label else [])
        }

        if parent:
            parent["children"].append(node)
        else:
            root.append(node)

        # PUSH
        if ltype in LEVELS:
            stack.append({"type": ltype, "node": node})

        # 🔥 SEQUENCE CHECK
        if last["label"] and label:
            expected = expected_next(last["label"], last["type"])
            if expected and label != expected:
                # soft warning (can log)
                pass

        if label:
            last["label"] = label
            last["type"] = ltype

    return root






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






# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # (reuse your existing parse + split + merge functions)
    # from nn import parse_formatted, split_sections, merge_paragraphs

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

    print("FINAL STRUCTURED JSON GENERATED ✅")


if __name__ == "__main__":
    main()





