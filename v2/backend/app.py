from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
import json
import os
import re
import uuid
import traceback
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field
from supabase import create_client, Client

from main import LegalRAGPipeline


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


load_dotenv(BACKEND_DIR / ".env")


def _backend_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND_DIR / path


def _split_env_list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]

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
CHUNKS_PATH = _backend_path(os.getenv("CHUNKS_PATH", "data/processed/artifacts2/chunks.json"))
 
# frontend url
FRONTEND_ORIGINS = _split_env_list( 
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173, https://residence-ambassador-order-collapse.trycloudflare.com",  
)
FRONTEND_ORIGIN_REGEX = os.getenv("FRONTEND_ORIGIN_REGEX", r"https://.*\.vercel\.app").strip() or None


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


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _first_pipeline_item(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    if isinstance(payload, dict):
        return payload
    return {}


def _retrieval_rows(payload: Any, key: str) -> list[Dict[str, Any]]:
    item = _first_pipeline_item(payload)
    retrieval = item.get("retrieval") or {}
    rows = retrieval.get(key) or []
    return [row for row in rows if isinstance(row, dict)]


def _chunk_preview(row: Dict[str, Any], selected_ids: Optional[set[str]] = None) -> Dict[str, Any]:
    source_scores = row.get("source_scores") or {}
    rerank_score = row.get("score")
    semrank_scores = [
        value
        for key, value in source_scores.items()
        if "semrank" in str(key).lower() and isinstance(value, (int, float))
    ]
    retrieval_scores = [
        value
        for key, value in source_scores.items()
        if any(name in str(key).lower() for name in ("faiss", "bm25", "graph")) and isinstance(value, (int, float))
    ]
    text = " ".join(str(row.get("text") or "").split())
    chunk_id = str(row.get("chunk_id") or "")
    return {
        "rank": row.get("rank"),
        "chunk_id": chunk_id,
        "source": row.get("act") or "BNS_2023",
        "section_number": row.get("section_number"),
        "citation": row.get("citation"),
        "chunk_type": row.get("chunk_type"),
        "retrieval_score": round(max(retrieval_scores), 4) if retrieval_scores else None,
        "rerank_score": round(float(rerank_score), 4) if isinstance(rerank_score, (int, float)) else None,
        "semrank_score": round(max(semrank_scores), 4) if semrank_scores else None,
        "selected": chunk_id in selected_ids if selected_ids is not None else False,
        "preview": text[:260] + ("..." if len(text) > 260 else ""),
    }


def _selected_chunk_ids(phase11_payload: Any) -> list[str]:
    item = _first_pipeline_item(phase11_payload)
    phase11 = item.get("phase11") if isinstance(item.get("phase11"), dict) else item
    ids = phase11.get("selected_chunk_ids") or []
    return [str(chunk_id) for chunk_id in ids if chunk_id]


def _chat_title(query: str) -> str:
    title = " ".join(query.strip().split())
    if not title:
        return "New chat"
    return title[:57] + "..." if len(title) > 60 else title


def load_chunk_lookup() -> Dict[str, Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"Chunks file is missing: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", payload if isinstance(payload, list) else [])
    lookup: Dict[str, Dict[str, Any]] = {}

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key in (chunk.get("chunk_id"), (chunk.get("citation") or {}).get("node_id")):
            if key:
                lookup[str(key)] = chunk

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        root_node_id = chunk.get("root_node_id")
        if root_node_id:
            lookup.setdefault(str(root_node_id), chunk)

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for node_id in chunk.get("node_ids") or []:
            lookup.setdefault(str(node_id), chunk)

    return lookup


def load_chunk_source_files() -> list[Dict[str, str]]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    source_files = payload.get("source_files", {}) if isinstance(payload, dict) else {}
    out = [{"label": "Chunks", "path": str(CHUNKS_PATH)}]
    if isinstance(source_files, dict):
        out.extend(
            {"label": str(label).replace("_", " ").title(), "path": str(path)}
            for label, path in source_files.items()
            if path
        )
    return out


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


SECTION_REF_RE = re.compile(r"\b(?:section|sec\.?|s\.)\s*(\d+[a-z]?)\b", flags=re.IGNORECASE)
DIRECT_LOOKUP_RE = re.compile(
    r"(?:"
    r"\b(?:show|display|open|view|read)\s+(?:me\s+)?(?:the\s+)?(?:full\s+|exact\s+|complete\s+)?"
    r"(?:text\s+of\s+)?(?:section|sec\.?|s\.)\s*\d+[a-z]?\b"
    r"|"
    r"\b(?:give|provide)\s+(?:me\s+)?(?:the\s+)?(?:full\s+|exact\s+|complete\s+)?"
    r"text\s+(?:of|for)\s+(?:section|sec\.?|s\.)\s*\d+[a-z]?\b"
    r"|"
    r"\b(?:section|sec\.?|s\.)\s*\d+[a-z]?\s+(?:full\s+|exact\s+|complete\s+)?text\b"
    r")",
    flags=re.IGNORECASE,
)
SCENARIO_CUES = (
    " if ",
    " when ",
    " apply",
    " applies",
    " applicable",
    " compare",
    " comparison",
    " difference",
    " vs ",
    " versus",
    " liable",
    " liability",
    "punishment for",
    "what happens",
    "what will happen",
)


def detect_direct_section_lookup(query: str) -> Optional[str]:
    text = " ".join(str(query or "").strip().split())
    if not text:
        return None

    matches = [match.group(1).upper() for match in SECTION_REF_RE.finditer(text)]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        return None

    low = f" {text.lower()} "
    if any(cue in low for cue in SCENARIO_CUES):
        return None
    if not (DIRECT_LOOKUP_RE.search(text) or SECTION_REF_RE.fullmatch(text)):
        return None
    return unique[0]


def _chunk_citation(chunk: Dict[str, Any]) -> str:
    return str((chunk.get("citation") or {}).get("citation_text") or "").strip()


def _chunk_label(chunk: Dict[str, Any]) -> str:
    citation = chunk.get("citation") or {}
    parsed = citation.get("parsed_label") or {}
    return str(citation.get("node_label") or parsed.get("value") or chunk.get("chunk_type") or "").strip()


def direct_section_lookup_result(section_number: str, chunk_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    section_token = str(section_number or "").upper()
    section_chunk_id = f"chunk__section__BNS_S{section_token}"
    section_chunk = chunk_lookup.get(section_chunk_id)

    if section_chunk is None:
        message = (
            f"I could not find Section {section_token} in the indexed BNS corpus. "
            "Please check the section number and try again."
        )
        return {
            "answer_type": "direct_section_lookup_not_found",
            "answer": message,
            "summary_answer": message,
            "detailed_answer": message,
            "final_answer": message,
            "section_number": section_token,
            "section_title": "",
            "citation": "",
            "text": "",
            "parts": [],
            "citations": [],
            "selected_chunk_ids": [],
            "confidence": {"score": 1.0, "label": "high"},
            "retrieval_confidence": {
                "score": 0.0,
                "label": "low",
                "explanation": "The requested section was not found in the indexed corpus.",
            },
            "risk_level": "low",
            "risk_reason": "No legal reasoning was performed because the requested section was not found.",
            "validation": {"should_show": True, "answer_verified": True, "completeness": "complete"},
        }

    section = section_chunk.get("section") or {}
    section_title = str(section.get("section_title") or "").strip()
    citation = _chunk_citation(section_chunk)
    parts = []
    selected_chunk_ids = [section_chunk_id]
    citations = [citation] if citation else []
    seen_part_ids = {section_chunk_id}
    for node_id in section_chunk.get("node_ids") or []:
        child = chunk_lookup.get(str(node_id))
        child_id = str(child.get("chunk_id") or "") if child else ""
        if not child or not child_id or child_id in seen_part_ids:
            continue
        seen_part_ids.add(child_id)
        child_citation = _chunk_citation(child)
        parts.append(
            {
                "chunk_id": child_id,
                "chunk_type": child.get("chunk_type"),
                "citation": child_citation,
                "label": _chunk_label(child),
                "text": child.get("text") or "",
            }
        )
        selected_chunk_ids.append(child_id)
        if child_citation:
            citations.append(child_citation)

    heading = f"Section {section_token}{f': {section_title}' if section_title else ''}"
    summary = f"{heading} from the Bharatiya Nyaya Sanhita, 2023."
    detailed = summary
    return {
        "answer_type": "direct_section_lookup",
        "answer": detailed,
        "summary_answer": summary,
        "detailed_answer": detailed,
        "final_answer": summary,
        "section_number": section_token,
        "section_title": section_title,
        "citation": citation,
        "text": "",
        "parts": parts,
        "citations": citations,
        "selected_chunk_ids": selected_chunk_ids,
        "confidence": {"score": 1.0, "label": "high"},
        "retrieval_confidence": {
            "score": 1.0,
            "label": "high",
            "components": {
                "exact_section_lookup": 1.0,
                "selected_chunk_count": len(selected_chunk_ids),
            },
            "explanation": "The requested section was resolved directly from the indexed corpus.",
        },
        "risk_level": "low",
        "risk_reason": "This is a direct corpus lookup, so no retrieval or legal reasoning was needed.",
        "validation": {"should_show": True, "answer_verified": True, "completeness": "complete"},
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
    app.state.chunk_source_files = load_chunk_source_files()

    print("Connecting Supabase...")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase env vars are missing")

    app.state.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    yield

    app.state.pipeline = None
    app.state.supabase = None
    app.state.chunk_lookup = None
    app.state.chunk_source_files = None


app = FastAPI(title="Legal RAG API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_origin_regex=FRONTEND_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": "Legal RAG API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


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
    chunk_lookup = getattr(req.app.state, "chunk_lookup", None)

    if pipeline is None or supabase is None or chunk_lookup is None:
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

        direct_section = detect_direct_section_lookup(request.query)
        if direct_section:
            result = direct_section_lookup_result(direct_section, chunk_lookup)
        else:
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


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, req: Request, authorization: Optional[str] = Header(None)):
    pipeline = getattr(req.app.state, "pipeline", None)
    supabase = getattr(req.app.state, "supabase", None)
    chunk_lookup = getattr(req.app.state, "chunk_lookup", None)
    chunk_source_files = getattr(req.app.state, "chunk_source_files", None) or [{"label": "Chunks", "path": str(CHUNKS_PATH)}]

    if pipeline is None or supabase is None or chunk_lookup is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    user_id = get_user_id_from_request(authorization, request.user_id)
    session_id = request.session_id
    if not is_valid_uuid(session_id):
        session_id = create_session(supabase, user_id, _chat_title(request.query))

    save_message(
        supabase=supabase,
        session_id=session_id,
        role="user",
        content=request.query,
        metadata={"user_id": user_id},
    )

    def event_stream():
        try:
            yield _sse("session", {"session_id": session_id})
            yield _sse(
                "phase",
                {
                    "id": "files",
                    "label": "Loading corpus files",
                    "status": "completed",
                    "progress": 0.05,
                    "detail": "Using indexed BNS chunks and retrieval artifacts.",
                },
            )
            yield _sse(
                "files",
                {"files": [*chunk_source_files, {"label": "Model", "path": LLM_MODEL}]},
            )

            direct_section = detect_direct_section_lookup(request.query)
            if direct_section:
                yield _sse(
                    "phase",
                    {
                        "id": "direct_lookup",
                        "label": "Direct section lookup",
                        "status": "running",
                        "progress": 0.35,
                        "detail": f"Looking up Section {direct_section} in chunk index.",
                    },
                )
                result = direct_section_lookup_result(direct_section, chunk_lookup)
                yield _sse(
                    "phase",
                    {
                        "id": "direct_lookup",
                        "label": "Direct section lookup",
                        "status": "completed",
                        "progress": 0.9,
                        "detail": f"Resolved {len(result.get('parts') or [])} section parts.",
                    },
                )
                normalized = _normalize_result(result)
                save_message(
                    supabase=supabase,
                    session_id=session_id,
                    role="assistant",
                    content=json.dumps(normalized, ensure_ascii=False, default=str),
                    metadata={"user_id": user_id},
                )
                yield _sse("final", {"query": request.query, "session_id": session_id, "result": normalized})
                yield _sse("done", {"session_id": session_id})
                return

            yield _sse(
                "phase",
                {
                    "id": "intent",
                    "label": "Intent discovery",
                    "status": "running",
                    "progress": 0.12,
                    "detail": "Analyzing query intent, targets, concepts, and decomposition.",
                },
            )
            phase8 = pipeline.analyze(request.query)
            phase8_item = _first_pipeline_item(phase8)
            intent = phase8_item.get("intent") or {}
            concepts = phase8_item.get("concepts") or []
            yield _sse(
                "phase",
                {
                    "id": "intent",
                    "label": "Intent discovered",
                    "status": "completed",
                    "progress": 0.22,
                    "detail": f"Primary intent: {intent.get('primary', 'unknown')}; concepts: {len(concepts)}.",
                    "data": {"intent": intent, "concepts": concepts[:8]},
                },
            )

            yield _sse(
                "phase",
                {
                    "id": "retrieval",
                    "label": "Retrieving chunks",
                    "status": "running",
                    "progress": 0.32,
                    "detail": "Searching semantic, keyword, graph, and SemRank evidence.",
                },
            )
            retrieval = pipeline.retrieve(phase8)
            raw_rows = _retrieval_rows(retrieval, "results_without_global_rerank")
            yield _sse(
                "chunks",
                {
                    "stage": "retrieved",
                    "label": f"Retrieved {len(raw_rows)} chunks",
                    "chunks": [_chunk_preview(row) for row in raw_rows[:20]],
                },
            )
            yield _sse(
                "phase",
                {
                    "id": "retrieval",
                    "label": "Retrieved chunks",
                    "status": "completed",
                    "progress": 0.48,
                    "detail": f"Retrieved {len(raw_rows)} candidate chunks.",
                },
            )

            yield _sse(
                "phase",
                {
                    "id": "rerank",
                    "label": "Re-ranking chunks",
                    "status": "running",
                    "progress": 0.56,
                    "detail": "Applying global ranking and evidence ordering.",
                },
            )
            reranked_rows = _retrieval_rows(retrieval, "results_with_global_rerank")
            yield _sse(
                "chunks",
                {
                    "stage": "reranked",
                    "label": f"Re-ranked {len(reranked_rows)} chunks",
                    "chunks": [_chunk_preview(row) for row in reranked_rows[:20]],
                },
            )
            yield _sse(
                "phase",
                {
                    "id": "rerank",
                    "label": "Re-ranking complete",
                    "status": "completed",
                    "progress": 0.62,
                    "detail": f"Prepared {len(reranked_rows)} globally ranked chunks.",
                },
            )

            yield _sse(
                "phase",
                {
                    "id": "reasoning",
                    "label": "Selecting prompt evidence",
                    "status": "running",
                    "progress": 0.7,
                    "detail": "Selecting final chunks and generating grounded answer.",
                },
            )
            phase11 = pipeline.reason(retrieval)
            selected_ids = _selected_chunk_ids(phase11)
            selected_set = set(selected_ids)
            yield _sse(
                "selection",
                {
                    "selected_chunk_ids": selected_ids,
                    "chunks": [_chunk_preview(row, selected_set) for row in reranked_rows[:20]],
                },
            )
            yield _sse(
                "phase",
                {
                    "id": "reasoning",
                    "label": "Selected top prompt chunks",
                    "status": "completed",
                    "progress": 0.78,
                    "detail": f"Selected {len(selected_ids)} chunks for grounded answer generation.",
                },
            )

            yield _sse(
                "phase",
                {
                    "id": "guardrails",
                    "label": "Applying guardrails",
                    "status": "running",
                    "progress": 0.84,
                    "detail": "Checking grounding, citations, and answer safety.",
                },
            )
            final = _normalize_result(pipeline.validate(phase11))
            validation = final.get("validation") or {}
            yield _sse(
                "phase",
                {
                    "id": "guardrails",
                    "label": "Guardrails complete",
                    "status": "completed",
                    "progress": 0.9,
                    "detail": "Grounding and display checks completed.",
                    "data": validation,
                },
            )

            confidence = final.get("confidence") or {}
            retrieval_confidence = final.get("retrieval_confidence") or {}
            yield _sse(
                "confidence",
                {
                    "confidence": confidence,
                    "retrieval_confidence": retrieval_confidence,
                    "risk_level": final.get("risk_level"),
                    "risk_reason": final.get("risk_reason"),
                },
            )
            yield _sse(
                "phase",
                {
                    "id": "validation",
                    "label": "Validation complete",
                    "status": "completed",
                    "progress": 1.0,
                    "detail": (
                        f"Risk: {final.get('risk_level') or 'unknown'}; "
                        f"answer confidence: {confidence.get('score', 'n/a')}; "
                        f"retrieval confidence: {retrieval_confidence.get('score', 'n/a')}."
                    ),
                },
            )

            save_message(
                supabase=supabase,
                session_id=session_id,
                role="assistant",
                content=json.dumps(final, ensure_ascii=False, default=str),
                metadata={"user_id": user_id},
            )
            yield _sse("final", {"query": request.query, "session_id": session_id, "result": final})
            yield _sse("done", {"session_id": session_id})
        except Exception as e:
            traceback.print_exc()
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
