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

PATH_ARTIFACTS = "data/processed/artifacts2"
PATH_CHUNKS = PATH_ARTIFACTS + "/chunks.json"
OUTPUT = "output/"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-large"
LLM_MODEL = "gpt-4o-mini"

ENABLE_GRAPH = False

class LegalRAGPipeline:
    def __init__(self):
        load_dotenv()
        
        print("Loading pipeline components...")
        
        # initialize ONCE
        self.chunks = load_chunks(PATH_CHUNKS)
        
        self.llm_client = Phase11OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        
        self.analyzer = build_analyzer(
            chunks_path=PATH_CHUNKS,
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
            enable_graph=ENABLE_GRAPH
        )

        self.corpus_index = CorpusIndex(PATH_CHUNKS)

        

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

        # Phase 8
        t0 = time.perf_counter()
        p8 = self.analyze(query)
        timings["phase_8"] = time.perf_counter() - t0

        # Phase 9/10
        t0 = time.perf_counter()
        p9_10 = self.retrieve(p8)
        timings["phase_9_10"] = time.perf_counter() - t0

        # Phase 11
        t0 = time.perf_counter()
        p11 = self.reason(p9_10)
        timings["phase_11"] = time.perf_counter() - t0

        # Phase 12/13
        t0 = time.perf_counter()
        p12_13 = self.validate(p11)
        timings["phase_12_13"] = time.perf_counter() - t0

        total_time = sum(timings.values())

        if debug:
            print("\nPIPELINE TIMINGS")
            for k, v in timings.items():
                print(f"{k}: {v:.3f}s")
            print(f"TOTAL: {total_time:.3f}s\n")

        return p12_13

def main():
    query = input("Enter your Query:\n")

    pipeline = LegalRAGPipeline()
    result = pipeline.run(query)

    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    main()