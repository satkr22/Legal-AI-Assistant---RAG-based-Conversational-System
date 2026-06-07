import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from query_analysis.aq_d import build_analyzer, HintRetriever, load_chunks
from retrieval.rq import _normalize_phase8_items, Phase9HybridRetriever
from reasoning.reason_4 import _normalize_phase9_items, CorpusIndex, Phase11OpenAIClient, Phase11Reasoner
from validation.validate_1 import process_json, _save_json

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def _backend_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND_DIR / path


PATH_ARTIFACTS = _backend_path(os.getenv("ARTIFACTS_DIR", "data/processed/artifacts2"))
PATH_CHUNKS = _backend_path(os.getenv("CHUNKS_PATH", "data/processed/artifacts2/chunks.json"))
OUTPUT = _backend_path("output")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

ENABLE_GRAPH = _env_flag("ENABLE_GRAPH", default=True)
ENABLE_RERANK = _env_flag("ENABLE_RERANK", default=True)
ENABLE_SEMRANK = _env_flag("ENABLE_SEMRANK", default=True)
SEMRANK_TOP_K = _env_int("SEMRANK_TOP_K", 20)
SEMRANK_WEIGHT = _env_float("SEMRANK_WEIGHT", 0.18)
SEMRANK_STRONG_SIM = _env_float("SEMRANK_STRONG_SIM", 0.72)
SEMRANK_MIN_SIM = _env_float("SEMRANK_MIN_SIM", 0.62)
SUBQUERY_PRESERVE_TOP_N = _env_int("SUBQUERY_PRESERVE_TOP_N", 1)
SUBQUERY_PRESERVE_MIN_SCORE = _env_float("SUBQUERY_PRESERVE_MIN_SCORE", 0.62)
MAX_CHUNKS_PER_SECTION = _env_int("MAX_CHUNKS_PER_SECTION", 2)


# --------------------------------------------------

def _json_safe(obj):
    """Convert objects into JSON serializable form."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    else:
        return obj
    
def _save_step(folder: Path, name: str, data):
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{name}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)

    print(f"Saved: {path}")

# --------------------------------------------------




class LegalRAGPipeline:
    def __init__(self):
        load_dotenv(BACKEND_DIR / ".env")
        
        print("Loading pipeline components...")
        
        # initialize ONCE
        self.chunks = load_chunks(str(PATH_CHUNKS))
        
        self.llm_client = Phase11OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        
        self.analyzer = build_analyzer(
            chunks_path=str(PATH_CHUNKS),
            model=LLM_MODEL,
            enable_llm=True
        )

        self.hint_retriever = HintRetriever(
            self.chunks,
            embedding_provider=self.llm_client.embed if self.llm_client else None
        )
        
        self.retriever = Phase9HybridRetriever(
            base_dir=PATH_ARTIFACTS,
            embed_model_name=EMBED_MODEL,
            rerank_model_name=RERANK_MODEL,
            enable_graph=ENABLE_GRAPH,
            enable_rerank=ENABLE_RERANK,
            enable_semrank=ENABLE_SEMRANK,
            semrank_top_k=SEMRANK_TOP_K,
            semrank_weight=SEMRANK_WEIGHT,
            semrank_strong_sim=SEMRANK_STRONG_SIM,
            semrank_min_sim=SEMRANK_MIN_SIM,
            subquery_preserve_top_n=SUBQUERY_PRESERVE_TOP_N,
            subquery_preserve_min_score=SUBQUERY_PRESERVE_MIN_SCORE,
            max_chunks_per_section=MAX_CHUNKS_PER_SECTION
        )

        self.corpus_index = CorpusIndex(str(PATH_CHUNKS))

        

        self.reasoner = Phase11Reasoner(
            corpus_index=self.corpus_index,
            llm_client=self.llm_client,
            llm_model=LLM_MODEL
        )
        
        print("Pipeline ready.")

        
    
    def analyze(self, query: str):
        return self.analyzer.analyze(query)

    def retrieve(self, phase8):
        items = _normalize_phase8_items(phase8)
        return self.retriever.retrieve_many(items)

    def reason(self, retrieval):
        items = _normalize_phase9_items(retrieval)
        return self.reasoner.reason_many(items)

    def validate(self, phase11):
        result = process_json(phase11)
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            return result[0]
        return result
    
    
    def run(self, query: str, debug: bool = False):
        timings = {}

        # create unique folder for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # # save query
        _save_step(run_dir, "query", {"query": query})

        # ---------------- PHASE 8 ----------------
        t0 = time.perf_counter()
        p8 = self.analyze(query)
        timings["phase_8"] = time.perf_counter() - t0

        _save_step(run_dir, "phase_8_analysis", p8)

        # ---------------- PHASE 9/10 ----------------
        t0 = time.perf_counter()
        p9_10 = self.retrieve(p8)
        timings["phase_9_10"] = time.perf_counter() - t0

        _save_step(run_dir, "phase_9_10_retrieval", p9_10)

        # ---------------- PHASE 11 ----------------
        t0 = time.perf_counter()
        p11 = self.reason(p9_10)
        timings["phase_11"] = time.perf_counter() - t0

        _save_step(run_dir, "phase_11_reasoning", p11)

        # ---------------- PHASE 12/13 ----------------
        t0 = time.perf_counter()
        p12_13 = self.validate(p11)
        timings["phase_12_13"] = time.perf_counter() - t0

        _save_step(run_dir, "phase_12_13_validation", p12_13)

        # timings
        total_time = sum(timings.values())
        timings["total"] = total_time

        _save_step(run_dir, "timings", timings)

        if debug:
            print("\nPIPELINE TIMINGS")
            for k, v in timings.items():
                print(f"{k}: {v:.3f}s")
            print(f"TOTAL: {total_time:.3f}s\n")

        print(f"\nAll intermediary results saved in:\n{run_dir}")

        return p12_13
 

    
    # def run(self, query: str, debug: bool = False):
    #     timings = {}

    #     # Phase 8
    #     t0 = time.perf_counter()
    #     p8 = self.analyze(query)
    #     timings["phase_8"] = time.perf_counter() - t0

    #     # Phase 9/10
    #     t0 = time.perf_counter()
    #     p9_10 = self.retrieve(p8)
    #     timings["phase_9_10"] = time.perf_counter() - t0

    #     # Phase 11
    #     t0 = time.perf_counter()
    #     p11 = self.reason(p9_10)
    #     timings["phase_11"] = time.perf_counter() - t0

    #     # Phase 12/13
    #     t0 = time.perf_counter()
    #     p12_13 = self.validate(p11)
    #     timings["phase_12_13"] = time.perf_counter() - t0

    #     total_time = sum(timings.values())

    #     if debug:
    #         print("\nPIPELINE TIMINGS")
    #         for k, v in timings.items():
    #             print(f"{k}: {v:.3f}s")
    #         print(f"TOTAL: {total_time:.3f}s\n")

    #     return p12_13

def main():
    query = input("Enter your Query:\n")

    pipeline = LegalRAGPipeline()
    result = pipeline.run(query)

    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    main()
