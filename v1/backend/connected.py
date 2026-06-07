import os
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from query_analysis.aq_d import build_analyzer
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

# phase 8
def analyzeQuery(user_query: str) -> Dict[str, Any]:
    analyzer = build_analyzer(chunks_path=PATH_CHUNKS, model=LLM_MODEL, enable_llm=True)
    result = analyzer.analyze(user_query)
    return result

# phase 9 and 10
def retrieveChunks(analyzedjson: Dict[str, Any]) -> List[Dict[str, Any]]:
    phase8_obj = analyzedjson
    phase8_items = _normalize_phase8_items(phase8_obj)
    retriever = Phase9HybridRetriever(
        base_dir=PATH_ARTIFACTS,
        embed_model_name=EMBED_MODEL,
        rerank_model_name=RERANK_MODEL,
        enable_graph=ENABLE_GRAPH
    )
    output = retriever.retrieve_many(phase8_items)
    return output
    
# pahse 11
def reasonOver(retreivedjson: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    retrieval_obj = retreivedjson
    items = _normalize_phase9_items(retrieval_obj)
    corpus_index = CorpusIndex(PATH_CHUNKS)
    
    llm_client = Phase11OpenAIClient(api_key=os.getenv("OPENAI_API_KEY", ""))
    reasoner = Phase11Reasoner(
        corpus_index=corpus_index,
        llm_client=llm_client,
        llm_model=LLM_MODEL
    )
    output = reasoner.reason_many(items)
    return output

# phase 12 and 13
def verify(reasonedjson: List[Dict[str, Any]], save: bool = False) -> List[Dict[str, Any]]:
    payload = reasonedjson
    result = process_json(payload)
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = F"output_{timestamp}.json"
        file_path = OUTPUT + filename
        _save_json(file_path, result)
        
    return result
    
    
def main():
    
    # User query
    query = input("Enter your Query:\n")
    
    load_dotenv()
    
    phase_8 = analyzeQuery(query)
    phase_9_10 = retrieveChunks(phase_8)
    phase_11 = reasonOver(phase_9_10)
    phase_12_13 = verify(phase_11, save=True)
    
    print("executed successfully")
    

if __name__ == "__main__":
    main()
    