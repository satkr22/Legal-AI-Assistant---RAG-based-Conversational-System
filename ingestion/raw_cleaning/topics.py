import fitz  # PyMuPDF
import re


def extract_and_format(pdf_path, output_txt="data/processed/formatted.txt"):
    doc = fitz.open(pdf_path)

    lines_all = []

    # -------------------------------
    # STEP 1: Extract in correct order
    # -------------------------------
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for block in blocks:
            text = block[4].strip()
            if text:
                lines = text.split("\n")
                for line in lines:
                    line = line.strip()

                    # Remove page numbers
                    if re.match(r"^\d+$", line):
                        continue

                    # Remove unwanted headers
                    if line.upper() in ["SECTIONS", "ARRANGEMENT OF SECTIONS"]:
                        continue

                    if line:
                        lines_all.append(line)

    doc.close()

    # -------------------------------
    # STEP 2: Parse structure
    # -------------------------------
    formatted_lines = []

    current_chapter = None
    current_subheading = "NONE"

    i = 0
    while i < len(lines_all):
        line = lines_all[i]

        # -------- CHAPTER --------
        chapter_match = re.match(r"^CHAPTER\s+([IVXLC]+)", line)
        if chapter_match:
            chapter_num = chapter_match.group(1)

            # Next line is usually title
            chapter_title = ""
            if i + 1 < len(lines_all):
                chapter_title = lines_all[i + 1]
                i += 1

            current_chapter = f"CHAPTER {chapter_num} - {chapter_title}"
            formatted_lines.append(current_chapter)

            # Reset subheading
            current_subheading = "NONE"
            formatted_lines.append(f"SUBHEADING - {current_subheading}")

            i += 1
            continue

        # -------- SUBHEADING --------
        # Example: "Of sexual offences"
        if line.lower().startswith("of "):
            current_subheading = line.strip()
            formatted_lines.append(f"SUBHEADING - {current_subheading}")
            i += 1
            continue

        # -------- SECTION --------
        section_match = re.match(r"^(\d+)\.\s*(.*)", line)
        if section_match:
            sec_num = section_match.group(1)
            sec_title = section_match.group(2)

            formatted_lines.append(f"SECTION {sec_num} - {sec_title}")
            i += 1
            continue

        i += 1

    # -------------------------------
    # STEP 3: Save output
    # -------------------------------
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines))

    print("✅ Done! Structured raw text saved.")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    extract_and_format("data/raw/bns-topics.pdf")