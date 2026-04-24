from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from main import LegalRAGPipeline


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    session_id: Optional[str] = None
    result: Dict[str, Any]


def _normalize_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result

    if isinstance(result, list):
        if not result:
            return {}
        if len(result) == 1 and isinstance(result[0], dict):
            return result[0]
        return {"items": result}

    return {"value": result}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading LegalRAGPipeline once at startup...")
    app.state.pipeline = LegalRAGPipeline()
    print("Pipeline ready.")
    yield
    app.state.pipeline = None


app = FastAPI(title="Legal RAG API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request) -> Dict[str, Any]:
    pipeline = getattr(req.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")

    try:
        result = pipeline.run(request.query, debug=True)
        result = _normalize_result(result)

        return {
            "query": request.query,
            "session_id": request.session_id,
            "result": result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))