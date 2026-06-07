import json

# ---------------- CLEAN FUNCTION ----------------
def clean_concepts(concepts):
    BAD_WORDS = {
        "what", "how", "when", "why", "which",
        "is", "are", "was", "were", "do", "does", "did",
        "a", "an", "the",
        "sect", "section", "law", "thing", "case", "cases",
        "apply", "paper", "money"
    }

    cleaned = []
    for c in concepts:
        if not isinstance(c, str):
            continue

        c = c.strip().lower()

        # remove empty or too short
        if not c or len(c) <= 2:
            continue

        # remove pure junk words
        if c in BAD_WORDS:
            continue

        # remove if mostly stopwords
        words = c.split()
        meaningful = [w for w in words if w not in BAD_WORDS]

        if not meaningful:
            continue

        cleaned.append(" ".join(meaningful))

    # dedupe
    return list(dict.fromkeys(cleaned))


# ---------------- PROCESS SINGLE RESULT ----------------
def process_result(item):
    if "result" not in item:
        return item

    result = item["result"]

    # Clean main concepts
    if "concepts" in result:
        result["concepts"] = clean_concepts(result["concepts"])

    # Clean sub-query concepts
    if "sub_queries" in result:
        for sq in result["sub_queries"]:
            if "concepts" in sq:
                sq["concepts"] = clean_concepts(sq["concepts"])

    return item


# ---------------- MAIN ----------------
def main():
    INPUT_FILE = "query_analysis/test_queries/output_results.json"
    OUTPUT_FILE = "query_analysis/test_queries/cleaned_result.json"

    print("Loading results...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Cleaning concepts...")

    cleaned_data = []

    cleaned_item = process_result(data)
    cleaned_data.append(cleaned_item)

    print("Saving cleaned results...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()