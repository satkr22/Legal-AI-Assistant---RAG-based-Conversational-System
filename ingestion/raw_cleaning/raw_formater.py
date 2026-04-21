
"""
format_legal_doc.py
====================
Formats a raw legal text file using a structure map file.

Usage:
    python ingestion/nn.py --raw data/processed/cleaned_raw2.txt --structure data/processed/formatted.txt --output data/processed/output.txt

Structure map format (formatted.txt):
    CHAPTER I - PRELIMINARY
    SUBHEADING - NONE
    SECTION 1 - Short title, commencement and application.
    ...

Raw text format (cleaned_raw2.txt):
    Plain legal text where each section starts with its number, e.g.:
    1. (1) This Act may be called...
    2. In this Sanhita...
"""

import re
import argparse


# ── Helpers ───────────────────────────────────────────────────────────────────

def roman_to_int(s):
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
    result, prev = 0, 0
    for c in reversed(s.upper()):
        v = vals.get(c, 0)
        result -= v if v < prev else -v
        prev = v
    return result


# ── Step 1: Parse structure map ───────────────────────────────────────────────

def parse_structure(filepath):
    """
    Returns a list of tuples: ('CHAPTER'|'SUBHEADING'|'SECTION', num, title)
    """
    structure = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('CHAPTER '):
                m = re.match(r'CHAPTER ([IVXLC]+) - (.+)', line)
                if m:
                    structure.append(('CHAPTER', m.group(1), m.group(2).strip()))
            elif line.startswith('SUBHEADING - '):
                sub = line[len('SUBHEADING - '):].strip()
                structure.append(('SUBHEADING', '', sub))
            elif line.startswith('SECTION '):
                m = re.match(r'SECTION (\d+) - (.+)', line)
                if m:
                    structure.append(('SECTION', m.group(1), m.group(2).strip()))
    return structure


# ── Step 2: Extract section texts from raw file ───────────────────────────────

def extract_sections(filepath, max_section):
    """
    Splits raw text into a dict: {section_number (int): body_text (str)}
    Handles both '45. ' and '45.(' style section openings,
    and sections that may be indented with leading spaces.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    section_texts = {}

    # Primary pass: sections starting at beginning of line (standard format)
    primary_pattern = re.compile(r'(?m)^(\d+)[.\(]')
    matches = list(primary_pattern.finditer(raw))

    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        if not (1 <= sec_num <= max_section):
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        block = raw[start:end].strip()
        # Strip leading section number
        block = re.sub(rf'^\s*{sec_num}\.?\s*', '', block).strip()
        if sec_num not in section_texts:
            section_texts[sec_num] = block

    # Secondary pass: sections with leading spaces (indented format)
    # e.g. "  129. Whoever..."
    secondary_pattern = re.compile(r'(?m)^\s+(\d+)[.\(]')
    for m in secondary_pattern.finditer(raw):
        sec_num = int(m.group(1))
        if not (1 <= sec_num <= max_section):
            continue
        if sec_num in section_texts:
            continue  # already found
        # Find end: next section-like line
        rest = raw[m.start() + 1:]
        next_m = re.search(r'\n\s*\d+[.\(]', rest)
        end = m.start() + 1 + next_m.start() if next_m else len(raw)
        block = raw[m.start():end].strip()
        block = re.sub(rf'^\s*{sec_num}\.?\s*', '', block).strip()
        section_texts[sec_num] = block

    return section_texts


# ── Step 3: Build formatted output ───────────────────────────────────────────

def build_output(structure, section_texts, doc_title=""):
    lines = []

    if doc_title:
        lines.append(doc_title)
        lines.append("=" * 70)
        lines.append("")

    for item in structure:
        t, num, title = item

        if t == 'CHAPTER':
            lines.append("")
            lines.append("=" * 70)
            lines.append(f"CHAPTER {num} — {title.upper()}")
            lines.append("=" * 70)
            lines.append("")

        elif t == 'SUBHEADING':
            if title.upper() != 'NONE':
                lines.append("")
                lines.append(f"    [{title.upper()}]")
                lines.append("")

        elif t == 'SECTION':
            sec_num = int(num)
            body = section_texts.get(sec_num, "[Content not found in source text]")
            lines.append(f"Section {sec_num}.  {title}")
            lines.append("─" * 55)
            lines.append(body)
            lines.append("")

    return "\n".join(lines)


# ── Step 4: Report missing sections ──────────────────────────────────────────

def report_missing(structure, section_texts):
    expected = {int(num) for t, num, _ in structure if t == 'SECTION'}
    found = set(section_texts.keys())
    missing = sorted(expected - found)
    if missing:
        print(f"⚠  Warning: {len(missing)} section(s) not found in raw text: {missing}")
    else:
        print(f"✓  All {len(expected)} sections found.")
    return missing


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Format a raw legal text file using a structure map."
    )
    parser.add_argument('--raw',       required=True,  help="Path to raw text file (e.g. cleaned_raw2.txt)")
    parser.add_argument('--structure', required=True,  help="Path to structure map file (e.g. formatted.txt)")
    parser.add_argument('--output',    required=True,  help="Path for the formatted output file")
    parser.add_argument('--title',     default="",     help="Optional document title to print at the top")
    parser.add_argument('--max-section', type=int, default=10000,
                        help="Maximum section number to extract (default: 10000)")
    args = parser.parse_args()

    print(f"Parsing structure map: {args.structure}")
    structure = parse_structure(args.structure)
    chapters  = sum(1 for t, _, _ in structure if t == 'CHAPTER')
    sections  = sum(1 for t, _, _ in structure if t == 'SECTION')
    print(f"  → {chapters} chapters, {sections} sections found in structure map")

    print(f"\nExtracting sections from raw text: {args.raw}")
    section_texts = extract_sections(args.raw, args.max_section)
    print(f"  → {len(section_texts)} sections extracted from raw text")

    print("\nChecking coverage:")
    report_missing(structure, section_texts)

    print(f"\nBuilding formatted output...")
    output = build_output(structure, section_texts, doc_title=args.title)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"✓  Output written to: {args.output}")
    print(f"   Total lines: {output.count(chr(10))}")


if __name__ == '__main__':
    main()