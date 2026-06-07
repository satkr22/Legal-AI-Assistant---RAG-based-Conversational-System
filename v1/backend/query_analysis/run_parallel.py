import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your analyzer
from query_analysis.aq_d import build_analyzer


# ---------------- CONFIG ----------------
CHUNKS_PATH = "data/processed/artifacts2/chunks.json"
INPUT_JSON = "query_analysis/test_queries/input_query2.json"
OUTPUT_JSON = "query_analysis/test_queries/output_results_2.json"

MODEL = "gpt-4o-mini"
MAX_WORKERS = 5        # Safe parallel limit
RETRY_LIMIT = 3        # Retry on API failure
SLEEP_BETWEEN = 0.2    # Small delay to avoid rate limit


# ---------------- LOAD ----------------
def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "query" in data:
        return [data["query"]]
    else:
        raise ValueError("Invalid input JSON format")


# ---------------- ANALYZE WITH RETRY ----------------
def analyze_with_retry(analyzer, query, idx):
    for attempt in range(RETRY_LIMIT):
        try:
            start = time.time()
            result = analyzer.analyze(query)
            elapsed = round(time.time() - start, 2)

            return {
                "id": idx,
                "query": query,
                "result": result,
                "time_sec": elapsed,
                "status": "success"
            }

        except Exception as e:
            if attempt == RETRY_LIMIT - 1:
                return {
                    "id": idx,
                    "query": query,
                    "error": str(e),
                    "status": "failed"
                }
            time.sleep(1 * (attempt + 1))  # exponential backoff


# ---------------- MAIN ----------------
def main():
    print("Loading analyzer...")
    analyzer = build_analyzer(CHUNKS_PATH, model=MODEL, enable_llm=True)

    print("Loading queries...")
    queries = load_queries(INPUT_JSON)

    print(f"Running {len(queries)} queries with {MAX_WORKERS} workers...\n")

    results = []
    start_total = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for idx, q in enumerate(queries):
            futures.append(executor.submit(analyze_with_retry, analyzer, q, idx))
            time.sleep(SLEEP_BETWEEN)  # avoid burst

        for future in as_completed(futures):
            res = future.result()
            results.append(res)

            if res["status"] == "success":
                print(f"[✓] {res['id']} ({res['time_sec']}s)")
            else:
                print(f"[✗] {res['id']} ERROR")

    total_time = round(time.time() - start_total, 2)

    print(f"\nTotal time: {total_time} sec")

    # Save output
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()