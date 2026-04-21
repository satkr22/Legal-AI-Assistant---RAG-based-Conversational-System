import re


def structure_text(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    structured_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # -------------------------------
        # CHAPTER HANDLING
        # -------------------------------
        if re.match(r"^CHAPTER\s+[IVXLC]+$", line):
            chapter_number = line

            # next line is title
            if i + 1 < len(lines):
                chapter_title = lines[i + 1].strip()
                structured_lines.append(f"{chapter_number} - {chapter_title}")
                i += 2
                continue

        # # -------------------------------
        # # SECTION HANDLING
        # # -------------------------------
        # section_match = re.match(r"^(\d+)\.", line)

        # if section_match:
        #     section_number = section_match.group(1)

        #     # previous line is title
        #     if structured_lines:
        #         prev_line = structured_lines[-1]

        #         # remove that previous line (title)
        #         structured_lines.pop()

        #         # create new structured line
        #         structured_lines.append(f"{section_number} - {prev_line}")

        #     # add remaining part of section line (after number)
        #     remaining = line[len(section_number)+1:].strip()
        #     if remaining:
        #         structured_lines.append(remaining)

        #     i += 1
        #     continue
        
        
        # # -------------------------------
        # # SECTION HANDLING (FIXED)
        # # -------------------------------
        # section_match = re.match(r"^(\d+)\.", line)

        # if section_match:
        #     section_number = section_match.group(1)

        #     # 🔥 find previous NON-EMPTY line
        #     title = None
        #     idx = len(structured_lines) - 1

        #     while idx >= 0:
        #         prev_line = structured_lines[idx].strip()

        #         if prev_line:  # non-empty
        #             title = prev_line
        #             break
        #         idx -= 1

        #     # remove that title line
        #     if title is not None:
        #         structured_lines.pop(idx)
        #         structured_lines.append(f"{section_number} - {title}")
        #     else:
        #         structured_lines.append(line)  # fallback (rare)

        #     # add remaining content of section line
        #     remaining = line[len(section_number)+1:].strip()
        #     if remaining:
        #         structured_lines.append(remaining)

        #     i += 1
        #     continue
        
        
        # -------------------------------
        # SECTION HANDLING (CORRECT LOGIC - FIXED)
        # -------------------------------
        section_match = re.match(r"^(\d+)\.", line)

        if section_match:
            section_number = section_match.group(1)

            idx = len(structured_lines) - 1

            # -------------------------------
            # STEP 1: skip blanks
            # -------------------------------
            while idx >= 0 and not structured_lines[idx].strip():
                idx -= 1

            # -------------------------------
            # STEP 2: find START of title (YOUR RULE + FALLBACK)
            # -------------------------------
            full_stop_count = 0
            end_idx = idx
            steps = 0

            while end_idx >= 0 and steps < 15:
                line_check = structured_lines[end_idx].strip()

                if not line_check:
                    break

                # HARD STOPS
                if re.match(r"^\d+\.", line_check) or re.match(r"^CHAPTER\s+", line_check):
                    break

                # COUNT FULL STOPS
                full_stop_count += line_check.count(".")

                # YOUR RULE → stop at SECOND full stop
                if full_stop_count >= 2:
                    break

                end_idx -= 1
                steps += 1

            # -------------------------------
            # FALLBACK (no 2 full stops found)
            # -------------------------------
            if full_stop_count < 2:
                temp_idx = idx
                while temp_idx >= 0:
                    line_check = structured_lines[temp_idx].strip()

                    if not line_check:
                        temp_idx -= 1
                        continue

                    if re.match(r"^\d+\.", line_check) or re.match(r"^CHAPTER\s+", line_check):
                        temp_idx -= 1
                        continue

                    end_idx = temp_idx
                    break

            # -------------------------------
            # STEP 3: collect title lines (DOWNWARD)
            # -------------------------------
            title_lines = []
            start_idx = end_idx

            while start_idx <= idx:
                curr = structured_lines[start_idx].strip()

                if curr:
                    title_lines.append(curr)

                start_idx += 1

            # -------------------------------
            # STEP 4: remove those lines
            # -------------------------------
            remove_count = idx - end_idx + 1
            for _ in range(remove_count):
                structured_lines.pop()

            # -------------------------------
            # BUILD TITLE
            # -------------------------------
            full_title = " ".join(title_lines).strip().rstrip(".")

            structured_lines.append(f"{section_number} - {full_title}")

            # -------------------------------
            # ADD CONTENT
            # -------------------------------
            remaining = line[len(section_number)+1:].strip()
            if remaining:
                structured_lines.append(remaining)

            i += 1
            continue

        # -------------------------------
        # NORMAL LINE
        # -------------------------------
        structured_lines.append(line)
        i += 1

    # write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(structured_lines))

    print("✅ Chapters and sections structured successfully")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    structure_text("data/processed/cleaned_raw.txt", "data/processed/structured.txt")