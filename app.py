from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import json
import os
import uuid
import traceback
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client

from main import LegalRAGPipeline








load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# Keep this ON if you want auth to be used when an Authorization header is sent.
# Set to false only if you want to temporarily run without auth.
ALLOW_ANONYMOUS_CHATS = os.getenv("ALLOW_ANONYMOUS_CHATS", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "data/processed/artifacts2/chunks.json"))


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # backward compatibility


class ChatResponse(BaseModel):
    query: str
    session_id: str
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


def _chat_title(query: str) -> str:
    title = " ".join(query.strip().split())
    if not title:
        return "New chat"
    return title[:57] + "..." if len(title) > 60 else title


def load_chunk_lookup() -> Dict[str, Dict[str, Any]]:
    chunks_file = CHUNKS_PATH if CHUNKS_PATH.is_absolute() else Path(__file__).parent / CHUNKS_PATH
    if not chunks_file.exists():
        raise RuntimeError(f"Chunks file is missing: {chunks_file}")

    with chunks_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", payload if isinstance(payload, list) else [])
    lookup: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key in (
            chunk.get("chunk_id"),
            chunk.get("root_node_id"),
            (chunk.get("citation") or {}).get("node_id"),
        ):
            if key:
                lookup[str(key)] = chunk

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for node_id in chunk.get("node_ids") or []:
            lookup.setdefault(str(node_id), chunk)

    return lookup


def public_chunk_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    citation = chunk.get("citation") or {}
    section = chunk.get("section") or {}
    chapter = chunk.get("chapter") or {}
    return {
        "chunk_id": chunk.get("chunk_id"),
        "chunk_type": chunk.get("chunk_type"),
        "act": chunk.get("act"),
        "citation": citation.get("citation_text"),
        "path": citation.get("path") or [],
        "chapter_title": chapter.get("chapter_title"),
        "section_number": section.get("section_number"),
        "section_title": section.get("section_title"),
        "text": chunk.get("text") or "",
        "semantic_summary": chunk.get("semantic_summary"),
        "plain_english_paraphrase": chunk.get("plain_english_paraphrase"),
    }


def is_valid_uuid(val: Optional[str]) -> bool:
    if not val:
        return False
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, TypeError):
        return False


def get_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return parts[1]


def verify_token(token: str) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase env vars are missing")

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {token}",
            },
            timeout=10,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=401, detail=f"Unable to validate token: {str(e)}")

    if resp.status_code != 200:
        detail = "Invalid token"
        try:
            error_body = resp.json()
            detail = error_body.get("msg") or error_body.get("message") or detail
        except ValueError:
            pass
        raise HTTPException(status_code=401, detail=detail)

    user = resp.json()
    user_id = user.get("id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user id")

    return {"sub": user_id, "user": user}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading pipeline...")
    app.state.pipeline = LegalRAGPipeline()

    print("Loading chunks...")
    app.state.chunk_lookup = load_chunk_lookup()

    print("Connecting Supabase...")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase env vars are missing")

    app.state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    yield

    app.state.pipeline = None
    app.state.supabase = None
    app.state.chunk_lookup = None


app = FastAPI(title="Legal RAG API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def create_session(supabase: Client, user_id: Optional[str], title: str = "New chat") -> str:
    payload = {
        "user_id": user_id,
        "title": title,
    }
    res = supabase.table("chat_sessions").insert(payload).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Failed to create chat session")
    return res.data[0]["id"]


def save_message(
    supabase: Client,
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    supabase.table("messages").insert(payload).execute()


def get_user_id_from_request(authorization: Optional[str], fallback_user_id: Optional[str] = None) -> Optional[str]:
    """
    Priority:
    1) Valid Supabase JWT from Authorization header
    2) Explicit fallback_user_id (backward compatibility)
    3) None if anonymous chats are allowed
    """
    token = get_bearer_token(authorization)

    if token:
        payload = verify_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing user id")
        return user_id

    if fallback_user_id:
        return fallback_user_id

    if ALLOW_ANONYMOUS_CHATS:
        return None

    raise HTTPException(status_code=401, detail="Authentication required")


@app.get("/sessions")
def sessions(req: Request, authorization: Optional[str] = Header(None)):
    supabase = getattr(req.app.state, "supabase", None)
    if supabase is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    user_id = get_user_id_from_request(authorization, fallback_user_id=None)
    if user_id is None:
        return {"sessions": []}

    res = (
        supabase.table("chat_sessions")
        .select("id, title, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return {"sessions": res.data or []}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, req: Request, authorization: Optional[str] = Header(None)):
    pipeline = getattr(req.app.state, "pipeline", None)
    supabase = getattr(req.app.state, "supabase", None)

    if pipeline is None or supabase is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        user_id = get_user_id_from_request(authorization, request.user_id)

        session_id = request.session_id
        if not is_valid_uuid(session_id):
            session_id = create_session(supabase, user_id, _chat_title(request.query))

        save_message(
            supabase=supabase,
            session_id=session_id,
            role="user",
            content=request.query,
            metadata={
                "user_id": user_id,
            },
        )

        result = pipeline.run(request.query, debug=True)
        normalized = _normalize_result(result)

        save_message(
            supabase=supabase,
            session_id=session_id,
            role="assistant",
            content=json.dumps(normalized, ensure_ascii=False, default=str),
            metadata={
                "user_id": user_id,
            },
        )

        return {
            "query": request.query,
            "session_id": session_id,
            "result": normalized,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
def history(session_id: str, req: Request, authorization: Optional[str] = Header(None)):
    supabase = getattr(req.app.state, "supabase", None)
    if supabase is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not is_valid_uuid(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        user_id = get_user_id_from_request(authorization, fallback_user_id=None)
    except HTTPException:
        # If auth is bad, reject. If auth is missing and anonymous is allowed,
        # get_user_id_from_request returns None.
        raise

    # If authenticated, ensure the session belongs to the current user.
    if user_id is not None:
        session_res = (
            supabase.table("chat_sessions")
            .select("id, user_id, title, created_at")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not session_res.data:
            raise HTTPException(status_code=403, detail="Forbidden")

    res = (
        supabase.table("messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )

    return {
        "session_id": session_id,
        "messages": res.data,
    }


@app.get("/chunks/{chunk_id}")
def chunk_detail(chunk_id: str, req: Request, authorization: Optional[str] = Header(None)):
    get_user_id_from_request(authorization, fallback_user_id=None)

    chunk_lookup = getattr(req.app.state, "chunk_lookup", None)
    if chunk_lookup is None:
        raise HTTPException(status_code=503, detail="Chunks are not ready")

    chunk = chunk_lookup.get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    return public_chunk_payload(chunk)
