# import fitz
# import re


# # -------------------------------
# # SMART SECTION TITLE FIX (FINAL)
# # -------------------------------
# import re

# import re

# def fix_structure(text):
#     lines = text.split("\n")
#     new_lines = []

#     i = 0
#     while i < len(lines):
#         line = lines[i].strip()

#         # -------------------------------
#         # ✅ CHAPTER FIX (WORKING)
#         # -------------------------------
#         if re.match(r"^CHAPTER\s+[IVXLC]+$", line):
#             chapter = line

#             # find next non-empty = title
#             j = i + 1
#             while j < len(lines) and not lines[j].strip():
#                 j += 1

#             if j < len(lines):
#                 title = lines[j].strip()
#                 new_lines.append(f"{chapter} {title}")
#                 i = j + 1   # ✅ SKIP TITLE LINE HERE
#                 continue

#         # -------------------------------
#         # ✅ SECTION FIX (NO DUPLICATE)
#         # -------------------------------
#         sec_match = re.match(r"^(\d+)\.\s*(.*)", line)
#         if sec_match:
#             sec_num = sec_match.group(1)
#             inline_content = sec_match.group(2).strip()

#             # look ONE line above ONLY (important)
#             title = None
#             if i - 1 >= 0:
#                 prev = lines[i - 1].strip()

#                 if (
#                     prev
#                     and not re.match(r"^CHAPTER\s+", prev)
#                     and not re.match(r"^[A-Z\s]{3,}$", prev)
#                     and not re.match(r"^\d+\.", prev)
#                 ):
#                     title = prev.rstrip(".")

#             if title:
#                 # ✅ REMOVE LAST LINE if it was added
#                 if new_lines and new_lines[-1].strip() == title:
#                     new_lines.pop()

#                 # Step 5
#                 new_lines.append(f"{sec_num}. {title}")

#                 # Step 6
#                 if inline_content:
#                     new_lines.append(inline_content)
#             else:
#                 new_lines.append(line)

#             i += 1
#             continue

#         # -------------------------------
#         # DEFAULT
#         # -------------------------------
#         new_lines.append(line)
#         i += 1

#     return "\n".join(new_lines)

# # -------------------------------
# # MAIN CLEAN FUNCTION
# # -------------------------------
# def extract_clean_text(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = []

#     for page in doc:
#         text.append(page.get_text("text"))

#     text = "\n".join(text)

#     # -------------------------------
#     # REMOVE NOISE
#     # -------------------------------
#     text = re.sub(r"Chapters and Sections HomePage", "", text)
#     text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

#     # -------------------------------
#     # FIX BROKEN WORDS
#     # -------------------------------
#     text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
#     # text = re.sub(r"\n(?=[a-z])", " ", text)

#     # -------------------------------
#     # NORMALIZE WHITESPACE
#     # -------------------------------
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"\n\s+\n", "\n\n", text)

#     # -------------------------------
#     # ⭐ FIX TITLES (WORKING NOW)
#     # -------------------------------
#     text = fix_structure(text)

#     # -------------------------------
#     # STRUCTURE FORMATTING
#     # -------------------------------
#     # text = re.sub(r"(CHAPTER\s+[IVXLC]+)", r"\n\n\1\n", text)
#     # text = re.sub(r"\n(\d+)\.\s", r"\n\n\1. ", text)

#     # text = re.sub(r"\((\d+)\)", r"\n  (\1)", text)
#     # text = re.sub(r"\(([a-z])\)", r"\n    (\1)", text)

#     # -------------------------------
#     # EXPLANATION / ILLUSTRATION
#     # -------------------------------
#     text = re.sub(r"Explanation\.?—", "\nEXPLANATION:\n", text)
#     text = re.sub(r"Illustrations?\.", "\nILLUSTRATION:\n", text)

#     # -------------------------------
#     # FINAL CLEANUP
#     # -------------------------------
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     return text.strip()


# # -------------------------------
# # RUN
# # -------------------------------
# if __name__ == "__main__":
#     pdf_path = "data/raw/BNS_2023.pdf"

#     clean_text = extract_clean_text(pdf_path)

#     with open("data/processed/clean_.txt", "w", encoding="utf-8") as f:
#         f.write(clean_text)

#     print(" DONE — this will now work correctly")



import fitz  # PyMuPDF
import re


def extract_clean_pdf(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        text = page.get_text()

        # -------------------------------
        # REMOVE HEADER (seen in your PDF)
        # -------------------------------
        text = re.sub(r"Chapters and Sections HomePage", "", text)

        # -------------------------------
        # REMOVE PAGE MARKERS (if present)
        # -------------------------------
        text = re.sub(r"<PARSED TEXT FOR PAGE:.*?>", "", text)

        # -------------------------------
        # REMOVE EXTRA SPACES
        # -------------------------------
        text = re.sub(r"[ \t]+", " ", text)

        # -------------------------------
        # NORMALIZE NEWLINES
        # -------------------------------
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        all_text.append(text.strip())

    final_text = "\n\n".join(all_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print("PDF extracted and cleaned (raw format preserved)")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    extract_clean_pdf("data/raw/BNS_2023.pdf", "data/processed/cleaned_raw2.txt")