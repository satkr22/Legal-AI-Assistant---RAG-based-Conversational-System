"""
fix_bns_json.py
───────────────
Post-processing script for bns_structured4.json.

Fixes applied
─────────────
FIX-A  False-positive specials: exception/explanation whose text begins with a
        lowercase letter or punctuation was split from the middle of a sentence.
        Merge text back into the parent node, remove the bogus child.

FIX-B  Illustration / Illustrations heading (empty text) with alpha/roman
        siblings → those siblings should be CHILDREN of the Illustration node.
        Applied at every nesting level.

FIX-C  Misplaced specials under alpha/roman: explanation, illustration, exception
        that logically belongs to the nearest enclosing num/section level is
        hoisted up one level (becomes a sibling of the alpha/roman that contained it).

FIX-D  Swallowed num nodes: a 'content' child whose text matches
        r'^and\\s*\\((\\d+)\\)\\s+' should be split; the leading "and" is appended
        to the parent text and the rest becomes a new num node.

FIX-E  SEC 178 special case: 'Explanation.—For the purposes of this Chapter,—'
        is a *container* for sub-clauses (1)–(5).  Currently (1) is embedded
        in the Explanation text, (2) is a sibling num node, (3)+(4) are absent,
        and (5) is a content-child of (2).  Rebuilt from full_text.

FIX-F  SEC 2 node (28): Explanation.— is a container for (a)/(b)/(c) which
        currently appear as top-level siblings of (28). Reparent them under
        the Explanation node (which is currently an empty child of roman (ii)).

FIX-G  SEC 46: Explanation 3 + second Illustrations heading (and their children)
        are trapped inside alpha (b) and (d) of the first Illustrations group.
        Hoist them to the section root, then apply FIX-B to the second Illustrations.
"""

import json
import re
import copy
from pathlib import Path

INPUT  = Path("data/processed/jsons/structure_jsons/bns_structured4.json")
OUTPUT = Path("data/processed/jsons/structure_jsons/bns_structured5.json")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def is_false_positive(node):
    """True if the node was created by a mid-sentence keyword detection."""
    if node["type"] not in ("exception", "explanation", "illustration"):
        return False
    txt = node["text"].strip()
    if not txt:
        return False
    # starts with lowercase or leading punctuation → mid-sentence split
    return txt[0].islower() or txt[0] in (",", ";", ":")


def merge_fp_into_parent(parent, child):
    """Append child's text to parent text (mid-sentence merge)."""
    parent["text"] = (parent["text"].rstrip() + " " + child["text"]).strip()


# ──────────────────────────────────────────────────────────────────────────────
# FIX-A  False-positive specials
# ──────────────────────────────────────────────────────────────────────────────

def fix_a_false_positives(nodes):
    """Recursive. Merge false-positive special nodes back into parent text."""
    i = 0
    while i < len(nodes):
        node = nodes[i]
        children = node.get("children", [])
        # process children first (depth-first)
        fix_a_false_positives(children)

        # now check *this* node's children for FP
        j = 0
        new_children = []
        while j < len(children):
            ch = children[j]
            if is_false_positive(ch):
                # merge text back and skip this child
                merge_fp_into_parent(node, ch)
                # any grandchildren of the FP node become children of parent
                new_children.extend(ch.get("children", []))
            else:
                new_children.append(ch)
            j += 1
        node["children"] = new_children
        i += 1
    return nodes


# ──────────────────────────────────────────────────────────────────────────────
# FIX-B  Illustration heading → alpha/roman children
# ──────────────────────────────────────────────────────────────────────────────

def fix_b_illustration_headings(nodes):
    """
    Walk `nodes`. Whenever we find an illustration node with empty text
    followed immediately by alpha/roman siblings, absorb those siblings as
    children.  Stop absorbing when we hit a non-alpha/roman node.
    Applied recursively on every child list.
    """
    # Recurse into children first
    for node in nodes:
        node["children"] = fix_b_illustration_headings(node.get("children", []))

    new_nodes = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node["type"] == "illustration" and not node["text"].strip():
            # absorb following alpha/roman as children
            j = i + 1
            while j < len(nodes) and nodes[j]["type"] in ("alpha", "roman"):
                node["children"].append(nodes[j])
                j += 1
            new_nodes.append(node)
            i = j  # skip the absorbed nodes
        else:
            new_nodes.append(node)
            i += 1
    return new_nodes


# ──────────────────────────────────────────────────────────────────────────────
# FIX-C  Misplaced specials under alpha/roman → hoist to parent level
# ──────────────────────────────────────────────────────────────────────────────

def fix_c_hoist_specials(nodes):
    """
    For each node in `nodes`, after recursing into its children, collect any
    explanation/illustration/exception that ended up as a child of an
    alpha/roman node and hoist them to the current level (after the
    alpha/roman that contained them).
    """
    for node in nodes:
        node["children"] = fix_c_hoist_specials(node.get("children", []))

    new_nodes = []
    for node in nodes:
        new_nodes.append(node)
        if node["type"] in ("alpha", "roman"):
            # pull out any specials that shouldn't be here
            kept_children = []
            hoisted = []
            for ch in node.get("children", []):
                if ch["type"] in ("explanation", "illustration", "exception"):
                    hoisted.append(ch)
                else:
                    kept_children.append(ch)
            node["children"] = kept_children
            new_nodes.extend(hoisted)
    return new_nodes


# ──────────────────────────────────────────────────────────────────────────────
# FIX-D  Swallowed num nodes
# ──────────────────────────────────────────────────────────────────────────────

_SWALLOWED_NUM = re.compile(r'^and\s*\((\d+)\)\s+(.*)', re.DOTALL)


def fix_d_swallowed_nums(nodes, ch_num, sec_num, seen_ids):
    """Split 'content' children whose text starts with 'and (N) …' into new num nodes."""
    for node in nodes:
        fix_d_swallowed_nums(node.get("children", []), ch_num, sec_num, seen_ids)

    new_nodes = []
    for node in nodes:
        new_nodes.append(node)
        children_out = []
        for ch in node.get("children", []):
            m = _SWALLOWED_NUM.match(ch["text"])
            if ch["type"] == "content" and m:
                num_label = f"({m.group(1)})"
                num_text  = m.group(2).strip()
                # append "and" to parent text if not already there
                if not node["text"].rstrip().endswith("; and"):
                    node["text"] = node["text"].rstrip() + "; and"
                # build new num node
                base_id   = f"node_{ch_num}_{sec_num}_{num_label.strip('()')}"
                uid = base_id
                suffix = 2
                while uid in seen_ids:
                    uid = f"{base_id}_{suffix}"
                    suffix += 1
                seen_ids.add(uid)
                new_num = {
                    "id":       uid,
                    "label":    num_label,
                    "type":     "num",
                    "text":     num_text,
                    "children": [],
                    "path":     [num_label.strip("()")],
                }
                # add after the current parent in the outer list
                new_nodes.append(new_num)
                # drop the content child
            else:
                children_out.append(ch)
        node["children"] = children_out
    return new_nodes


# ──────────────────────────────────────────────────────────────────────────────
# FIX-E  CH10 SEC178 – Explanation container for (1)–(5)
# ──────────────────────────────────────────────────────────────────────────────

def fix_e_sec178(sec):
    """
    Rebuild the nodes of section 178 from the full_text.
    The Explanation is a heading; (1)–(5) are its children.
    """
    ft = sec["full_text"]

    # Parse (1)–(5) from the full text
    pattern = re.compile(r'\((\d)\)\s+(.*?)(?=\s*\((?:\d)\)|$)', re.DOTALL)
    explanation_block_start = ft.find("Explanation.—")
    if explanation_block_start == -1:
        return  # nothing to fix

    explanation_text = "For the purposes of this Chapter,—"
    explanation_block = ft[explanation_block_start:]

    sub_items = []
    for m in re.finditer(r'\(([1-5])\)\s+(.*?)(?=\s*\([1-5]\)\s|\s*[A-Z][a-z]|$)',
                         explanation_block, re.DOTALL):
        label = f"({m.group(1)})"
        text  = " ".join(m.group(2).split())
        # strip trailing '; and' or ';'
        text  = text.rstrip(";").rstrip(" and").strip()
        sub_items.append((label, text))

    if not sub_items:
        return

    base = f"node_{sec['section_id'].split('_')[1]}_{sec['section_number']}"

    def make_id(label):
        return f"{base}_expl_{label.strip('()')}"

    explanation_node = {
        "id":       f"{base}_explanation",
        "label":    "Explanation",
        "type":     "explanation",
        "text":     explanation_text,
        "children": [],
        "path":     ["Explanation"],
    }
    for label, text in sub_items:
        explanation_node["children"].append({
            "id":       make_id(label),
            "label":    label,
            "type":     "num",
            "text":     text,
            "children": [],
            "path":     ["Explanation", label.strip("()")],
        })

    # Rebuild nodes: keep the body content, replace everything after it
    body_node = next((n for n in sec["nodes"] if n["type"] == "content"), None)
    sec["nodes"] = ([body_node] if body_node else []) + [explanation_node]


# ──────────────────────────────────────────────────────────────────────────────
# FIX-F  SEC2 node (28) – Explanation container for (a)/(b)/(c)
# ──────────────────────────────────────────────────────────────────────────────

def fix_f_sec2_node28(sec):
    """
    In section 2, num node (28) has an Explanation heading (empty text) as a
    child of roman (ii).  The Explanation's alpha children (a)(b)(c) ended up
    as direct children of (28).  Reparent them.
    """
    def find_node_by_label(nodes, label):
        for n in nodes:
            if n["label"] == label:
                return n, nodes
            result = find_node_by_label(n.get("children", []), label)
            if result:
                return result
        return None, None

    n28, _ = find_node_by_label(sec["nodes"], "(28)")
    if not n28:
        return

    # Find the Explanation node (empty text) somewhere inside (28)
    def find_explanation(node):
        for c in node.get("children", []):
            if c["type"] == "explanation" and not c["text"].strip():
                return c, node
            result = find_explanation(c)
            if result[0]:
                return result
        return None, None

    expl_node, expl_parent = find_explanation(n28)
    if not expl_node:
        return

    # The (a)(b)(c) that belong to Explanation are currently direct children of (28)
    # They appear AFTER alpha (k) in the child list
    new_28_children = []
    absorbed = []
    # Find the cutoff: after alpha(k) the (a)(b)(c) start
    past_k = False
    for ch in n28.get("children", []):
        if ch["label"] == "(k)":
            new_28_children.append(ch)
            past_k = True
        elif past_k and ch["type"] == "alpha" and ch["label"] in ("(a)", "(b)", "(c)"):
            absorbed.append(ch)
        else:
            new_28_children.append(ch)

    if absorbed:
        n28["children"] = new_28_children
        expl_node["children"] = absorbed

        # Also: the Illustration child of alpha(c) is correct (stays there)
        # Set a meaningful text on Explanation
        expl_node["text"] = ""  # keep empty; it's a container heading


# ──────────────────────────────────────────────────────────────────────────────
# FIX-G  SEC46 – Explanation 3/4/5 and second Illustrations hoisted from alpha
# ──────────────────────────────────────────────────────────────────────────────

def fix_g_sec46(sec):
    """
    In section 46 the tree currently looks like:
      content, Explanation 1, Explanation 2,
      Illustration(empty) [first],
        (a), (b) [alpha children of first Illustrations]
          (b).children: [Explanation 3, Illustration(empty) [second]]
          [second Illustration has no children yet]
        (a),(b),(c),(d) [alpha of second Illustrations – currently siblings]
          (d).children: [Explanation 4, Illustration, Explanation 5, Illustration]

    Target:
      content, Explanation 1, Explanation 2,
      Illustration [first, children: (a),(b)],
      Explanation 3,
      Illustration [second, children: (a),(b),(c),(d)],
      Explanation 4, Illustration,
      Explanation 5, Illustration
    """
    nodes = sec["nodes"]

    # ── step 1: find the first Illustration heading (empty text, at root)
    first_illus_idx = None
    for i, n in enumerate(nodes):
        if n["type"] == "illustration" and not n["text"].strip():
            first_illus_idx = i
            break
    if first_illus_idx is None:
        return

    # ── step 2: collect hoisted items from alpha(b) children (Exp3 + second Illus)
    alpha_b = None
    for n in nodes:
        if n["type"] == "alpha" and n["label"] == "(b)":
            alpha_b = n
            break
    if alpha_b is None:
        return

    hoisted_from_b = []
    kept_b_children = []
    for ch in alpha_b.get("children", []):
        if ch["type"] in ("explanation", "illustration"):
            hoisted_from_b.append(ch)
        else:
            kept_b_children.append(ch)
    alpha_b["children"] = kept_b_children

    # ── step 3: collect hoisted items from alpha(d) children (Exp4+Illus, Exp5+Illus)
    # alpha(d) is the LAST (d) in root nodes
    alpha_d = None
    for n in nodes:
        if n["type"] == "alpha" and n["label"] == "(d)":
            alpha_d = n
    hoisted_from_d = []
    if alpha_d:
        kept_d_children = []
        for ch in alpha_d.get("children", []):
            if ch["type"] in ("explanation", "illustration"):
                hoisted_from_d.append(ch)
            else:
                kept_d_children.append(ch)
        alpha_d["children"] = kept_d_children

    # ── step 4: the second Illustrations node (from hoisted_from_b) needs
    #            to absorb the second set of alpha siblings currently at root.
    #            Those alphas come AFTER the first (b) in root (second (a)(b)(c)(d)).
    second_illus = next(
        (n for n in hoisted_from_b if n["type"] == "illustration"), None
    )
    # The second (a)(b)(c)(d) alphas in root – they appear after first_illus block
    # After FIX-B runs, the first Illustration will have (a)(b) as children,
    # and the remaining (a)(b)(c)(d) should be children of the second Illustration.
    # We'll handle this by marking them and letting FIX-B do its job later,
    # but since FIX-B runs on the sibling list, we need the second Illus to be
    # *in* the root list before FIX-B runs.

    # ── step 5: rebuild root nodes
    # Keep: content, Explanation 1, Explanation 2, Illustration[first], (a), (b)
    # Then append: Explanation 3, Illustration[second], (a),(b),(c),(d)
    # Then append: Explanation 4, Illustration, Explanation 5, Illustration
    new_nodes = []
    for n in nodes:
        new_nodes.append(n)
    # insert hoisted_from_b right after the last alpha(b) position
    # find insertion point: after alpha(b) in new_nodes
    insert_after = None
    for i, n in enumerate(new_nodes):
        if n is alpha_b:
            insert_after = i
    if insert_after is not None:
        for k, item in enumerate(hoisted_from_b):
            new_nodes.insert(insert_after + 1 + k, item)

    # append hoisted_from_d at the end (before any trailing content)
    new_nodes.extend(hoisted_from_d)

    sec["nodes"] = new_nodes


def fix_h_normalize_specials(nodes, current_num=None):
    """
    FINAL CORRECT FIX:
    - Finds ANY explanation/illustration under alpha/roman at ANY depth
    - Hoists them to nearest num
    - Fixes structure + path
    """

    result = []
    hoisted_global = []

    for node in nodes:
        node_type = node["type"]

        # update current num
        if node_type == "num":
            current_num = node["label"].strip("()")

        # recurse first
        children, hoisted = fix_h_normalize_specials(
            node.get("children", []),
            current_num
        )

        node["children"] = []

        # 🔥 KEY: process children AFTER recursion
        for ch in children:
            if ch["type"] in ("explanation", "illustration"):
                # ALWAYS hoist (no matter where found)
                ch["path"] = [current_num, ch["label"]]
                hoisted_global.append(ch)
            else:
                node["children"].append(ch)

        result.append(node)
        hoisted_global.extend(hoisted)

    # 🔥 attach hoisted after correct num
    final = []
    for node in result:
        final.append(node)

        if node["type"] == "num":
            num = node["label"].strip("()")

            attach_now = []
            remaining = []

            for h in hoisted_global:
                if h["path"][0] == num:
                    attach_now.append(h)
                else:
                    remaining.append(h)

            final.extend(attach_now)
            hoisted_global = remaining

    return final, hoisted_global



def regenerate_ids(nodes, ch_num, sec_num):
    """
    Rebuild IDs based on FINAL structure + path
    """

    def build_id(node, path):
        parts = ["node", str(ch_num), str(sec_num)]

        for p in path:
            parts.append(p.lower().replace(" ", "").replace("(", "").replace(")", ""))

        return "_".join(parts)

    def dfs(node, parent_path):
        label = node.get("label", "")

        if node["type"] == "num":
            clean = label.strip("()")
            path = [clean]

        elif node["type"] == "alpha":
            clean = label.strip("()")
            path = parent_path + [clean]

        elif node["type"] == "roman":
            clean = label.strip("()")
            path = parent_path + [clean]

        elif node["type"] in ("explanation", "illustration", "exception"):
            path = parent_path + [label]

        else:
            path = parent_path

        # 🔥 regenerate id
        node["id"] = build_id(node, path)
        node["path"] = path

        for ch in node.get("children", []):
            dfs(ch, path)

    for n in nodes:
        dfs(n, [])


def validate_structure(nodes, errors=None, parent_num=None):
    if errors is None:
        errors = []

    for node in nodes:
        if node["type"] == "num":
            parent_num = node["label"].strip("()")

        if node["type"] in ("explanation", "illustration"):
            if parent_num and node["path"][0] != parent_num:
                errors.append({
                    "id": node["id"],
                    "issue": "wrong_parent",
                    "path": node["path"]
                })

        validate_structure(node.get("children", []), errors, parent_num)

    return errors





# ──────────────────────────────────────────────────────────────────────────────
# APPLY ALL FIXES TO A SECTION
# ──────────────────────────────────────────────────────────────────────────────

def apply_fixes_to_section(sec, ch_num):
    sec_num = sec["section_number"]
    seen_ids = {n["id"] for n in sec["nodes"]}

    # Special-case fixes first (before generic ones)
    if ch_num == 10 and sec_num == 178:
        fix_e_sec178(sec)

    if ch_num == 1 and sec_num == 2:
        fix_f_sec2_node28(sec)

    if ch_num == 4 and sec_num == 46:
        fix_g_sec46(sec)

    # Generic fixes
    # A: false positives (merge mid-sentence splits back)
    sec["nodes"] = fix_a_false_positives(sec["nodes"])

    # B: illustration headings absorb following alpha/roman as children
    sec["nodes"] = fix_b_illustration_headings(sec["nodes"])
    
    # C: hoist misplaced specials from alpha/roman up to parent level
    sec["nodes"] = fix_c_hoist_specials(sec["nodes"])
    
    # D: swallowed num nodes
    sec["nodes"] = fix_d_swallowed_nums(sec["nodes"], ch_num, sec_num, seen_ids)
    
    # after FIX-C
    sec["nodes"],_ = fix_h_normalize_specials(sec["nodes"])

    regenerate_ids(
        sec["nodes"],
        ch_num,
        sec_num
    )
    
    
    errors = validate_structure(sec["nodes"])
    if errors:
        print(f"Issues in Chapter {ch_num}, Section {sec_num}:")
        for e in errors[:5]:
            print(e)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    total_sections = 0
    for ch in data["chapters"]:
        ch_num = ch["chapter_number"]
        for sec in ch["sections"]:
            apply_fixes_to_section(sec, ch_num)
            total_sections += 1

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅  Done. Processed {total_sections} sections → {OUTPUT}")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────────────────────────
# FIX-F v2  SEC2 node (28) – correct reparenting (replaces earlier version)
# ──────────────────────────────────────────────────────────────────────────────

def fix_f_sec2_node28_v2(sec):
    """
    In section 2, num node (28):
      - alpha(k) has roman(i) and roman(ii) as children
      - roman(ii) has Explanation (empty, container) as child
      - The Explanation's real children (a)(b)(c) are currently direct children
        of (28), appearing after alpha(k) [indices 11,12,13]
    Fix:
      1. Lift Explanation from roman(ii) to be a direct child of (28) after (k)
      2. Move the (a)(b)(c) [last 3 children of (28)] into Explanation.children
    """
    def find_node(nodes, label):
        for n in nodes:
            if n['label'] == label:
                return n
            r = find_node(n.get('children', []), label)
            if r:
                return r
        return None

    n28 = find_node(sec['nodes'], '(28)')
    if not n28:
        return

    nk = find_node(n28.get('children', []), '(k)')
    if not nk:
        return

    # Find Explanation inside (k) subtree (under roman ii)
    expl_node = None
    roman_ii = find_node(nk.get('children', []), '(ii)')
    if roman_ii:
        for c in roman_ii.get('children', []):
            if c['type'] == 'explanation':
                expl_node = c
                roman_ii['children'] = [x for x in roman_ii['children'] if x is not c]
                break

    if not expl_node:
        return

    # The last 3 children of (28) are the (a)(b)(c) for Explanation
    children = n28['children']
    # Find split point: after (k)
    k_idx = next((i for i, c in enumerate(children) if c['label'] == '(k)'), None)
    if k_idx is None:
        return

    explanation_children = children[k_idx + 1:]
    n28['children'] = children[:k_idx + 1]

    expl_node['children'] = explanation_children
    n28['children'].append(expl_node)


# ──────────────────────────────────────────────────────────────────────────────
# FIX-H  Illustration(empty) heading absorb following NUM siblings
#         (handles SEC30 pattern where illustration paragraphs are parsed as num)
# ──────────────────────────────────────────────────────────────────────────────

def fix_h_illustration_absorb_nums(nodes):
    """Like FIX-B but also absorbs num siblings after an empty Illustration heading."""
    for node in nodes:
        node['children'] = fix_h_illustration_absorb_nums(node.get('children', []))

    new_nodes = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if node['type'] == 'illustration' and not node['text'].strip():
            j = i + 1
            while j < len(nodes) and nodes[j]['type'] in ('alpha', 'roman', 'num'):
                node['children'].append(nodes[j])
                j += 1
            new_nodes.append(node)
            i = j
        else:
            new_nodes.append(node)
            i += 1
    return new_nodes


# ──────────────────────────────────────────────────────────────────────────────
# OVERRIDE apply_fixes_to_section with corrected order
# ──────────────────────────────────────────────────────────────────────────────

def apply_fixes_to_section_v2(sec, ch_num):
    sec_num = sec['section_number']
    seen_ids = {n['id'] for n in sec['nodes']}

    # Special-case fixes BEFORE generic ones
    if ch_num == 10 and sec_num == 178:
        fix_e_sec178(sec)

    if ch_num == 1 and sec_num == 2:
        fix_f_sec2_node28_v2(sec)   # v2 runs before fix_c

    if ch_num == 4 and sec_num == 46:
        fix_g_sec46(sec)

    # FIX-A: merge false-positive mid-sentence splits
    sec['nodes'] = fix_a_false_positives(sec['nodes'])

    # FIX-C: hoist misplaced specials from alpha/roman up to parent
    sec['nodes'] = fix_c_hoist_specials(sec['nodes'])

    # FIX-B + FIX-H: illustration headings absorb following alpha/roman/num children
    sec['nodes'] = fix_h_illustration_absorb_nums(sec['nodes'])

    # FIX-D: swallowed num nodes
    sec['nodes'] = fix_d_swallowed_nums(sec['nodes'], ch_num, sec_num, seen_ids)


def main_v2():
    with open(INPUT, encoding='utf-8') as f:
        data = json.load(f)

    total = 0
    for ch in data['chapters']:
        ch_num = ch['chapter_number']
        for sec in ch['sections']:
            apply_fixes_to_section_v2(sec, ch_num)
            total += 1

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f'✅  Done. Processed {total} sections → {OUTPUT}')


if __name__ == '__main__':
    main_v2()