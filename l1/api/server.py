"""
InferenceToken — OpenAI-compatible API server
==============================================
Drop-in replacement for the OpenAI API, backed by the on-chain
inference marketplace. Zero code changes needed in existing apps:

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="0xYOUR_ETHEREUM_PRIVATE_KEY",
    )
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role": "user", "content": "Hello"}],
    )

The api_key header is the caller's Ethereum private key; it funds
inference jobs directly without any custodial layer.

Usage:
    pip install -r requirements.txt
    RPC_URL=https://sepolia.base.org CONTRACT_ADDRESS=0x... python server.py
    # OR
    python server.py --deployment ../deployment.json --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from web3 import Web3

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk"))
from inference_sdk import InferenceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

# ── Prometheus metrics ────────────────────────────────────────────────────────

REQUESTS_TOTAL   = Counter("api_requests_total",   "Total API requests",  ["endpoint", "status"])
INFERENCE_JOBS   = Counter("api_inference_jobs_total", "Jobs submitted")
INFERENCE_ERRORS = Counter("api_inference_errors_total", "Job errors")
ACTIVE_JOBS      = Gauge  ("api_active_jobs",       "Currently running jobs")
JOB_DURATION     = Histogram("api_job_duration_seconds", "End-to-end job duration",
                              buckets=[5, 15, 30, 60, 120, 300, 600])

# ── Web3 connection pool ──────────────────────────────────────────────────────

class ContractPool:
    """
    Caches one InferenceClient per (rpc_url, contract_address, private_key) triple.
    Avoids creating a new Web3 connection on every request.
    """
    def __init__(self):
        self._clients: dict[str, InferenceClient] = {}

    def get(self, rpc_url: str, address: str, abi: list, private_key: str) -> InferenceClient:
        key = f"{rpc_url}:{address}:{private_key[:10]}"
        if key not in self._clients:
            self._clients[key] = InferenceClient(
                rpc_url=rpc_url,
                contract_address=address,
                contract_abi=abi,
                private_key=private_key,
            )
        return self._clients[key]

pool = ContractPool()

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── Deployment config ─────────────────────────────────────────────────────────

DEPLOYMENT: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("InferenceToken API server starting  contract=%s", DEPLOYMENT.get("address", "?"))
    yield
    log.info("API server shut down")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InferenceToken API",
    description="OpenAI-compatible API backed by on-chain AI inference",
    version="0.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenAI schema models ──────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    # Accepted but ignored for compatibility
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[list[str] | str] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

class CompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "inference-token-network"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client(authorization: Optional[str]) -> InferenceClient:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    key = authorization.removeprefix("Bearer ").strip()
    if not (key.startswith("0x") and len(key) == 66):
        raise HTTPException(
            status_code=401,
            detail="api_key must be a 0x-prefixed 32-byte Ethereum private key",
        )
    if not DEPLOYMENT.get("address"):
        raise HTTPException(status_code=503, detail="Contract not configured — set CONTRACT_ADDRESS")

    return pool.get(
        rpc_url=DEPLOYMENT["rpc_url"],
        address=DEPLOYMENT["address"],
        abi=DEPLOYMENT.get("abi", []),
        private_key=key,
    )

def _messages_to_prompt(messages: list[Message]) -> str:
    """Convert OpenAI message list to Mistral/Llama instruct format."""
    parts: list[str] = []
    for i, msg in enumerate(messages):
        if msg.role == "system":
            parts.append(f"<s>[INST] <<SYS>>\n{msg.content}\n<</SYS>>\n\n")
        elif msg.role == "user":
            prefix = "" if parts else "<s>[INST] "
            parts.append(f"{prefix}{msg.content} [/INST]")
        elif msg.role == "assistant":
            parts.append(f" {msg.content} </s><s>[INST] ")
    return "".join(parts).strip()

def _count_tokens(text: str) -> int:
    return max(1, len(text) // 3)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":   "ok",
        "contract": DEPLOYMENT.get("address", "not configured"),
        "rpc":      DEPLOYMENT.get("rpc_url", "?"),
    }

@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/v1/models", response_model=ModelList)
@limiter.limit("120/minute")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    client = _get_client(authorization)
    try:
        model_ids = await asyncio.get_event_loop().run_in_executor(None, client.list_models)
    except Exception as exc:
        log.error("list_models error: %s", exc)
        model_ids = []
    REQUESTS_TOTAL.labels("list_models", "200").inc()
    return ModelList(data=[ModelObject(id=m) for m in model_ids])

@app.post("/v1/chat/completions")
@limiter.limit("30/minute")
async def chat_completions(
    request: Request,
    req: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    client  = _get_client(authorization)
    prompt  = _messages_to_prompt(req.messages)
    max_tok = req.max_tokens or 512
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    log.info("POST /v1/chat/completions model=%s prompt_len=%d", req.model, len(prompt))
    INFERENCE_JOBS.inc()
    ACTIVE_JOBS.inc()
    t0 = time.monotonic()

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.infer(
                model=req.model, prompt=prompt, max_tokens=max_tok,
                poll_interval=5, timeout=900,
            ),
        )
    except TimeoutError:
        INFERENCE_ERRORS.inc()
        REQUESTS_TOTAL.labels("chat", "504").inc()
        raise HTTPException(status_code=504, detail="No miner responded within 15 minutes")
    except Exception as exc:
        INFERENCE_ERRORS.inc()
        REQUESTS_TOTAL.labels("chat", "500").inc()
        log.error("inference error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        ACTIVE_JOBS.dec()
        JOB_DURATION.observe(time.monotonic() - t0)

    output = result.text
    p_tok  = _count_tokens(prompt)
    c_tok  = _count_tokens(output)

    REQUESTS_TOTAL.labels("chat", "200").inc()

    if req.stream:
        async def _stream():
            chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": output}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            done = {
                "id": cmpl_id, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_stream(), media_type="text/event-stream")

    return ChatResponse(
        id=cmpl_id, created=int(time.time()), model=req.model,
        choices=[CompletionChoice(index=0, message=Message(role="assistant", content=output), finish_reason="stop")],
        usage=Usage(prompt_tokens=p_tok, completion_tokens=c_tok, total_tokens=p_tok + c_tok),
    )

@app.post("/v1/completions")
@limiter.limit("30/minute")
async def completions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    body   = await request.json()
    client = _get_client(authorization)
    prompt    = body.get("prompt", "")
    max_tok   = body.get("max_tokens", 512)
    model     = body.get("model", "")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.infer(model=model, prompt=prompt, max_tokens=max_tok),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}", "object": "text_completion",
        "created": int(time.time()), "model": model,
        "choices": [{"text": result.text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": _count_tokens(prompt),
            "completion_tokens": _count_tokens(result.text),
            "total_tokens": _count_tokens(prompt) + _count_tokens(result.text),
        },
    }

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="InferenceToken OpenAI-compatible API")
    parser.add_argument("--deployment", default="../deployment.json")
    parser.add_argument("--rpc",        default="")
    parser.add_argument("--contract",   default="")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--workers",    type=int, default=1)
    args = parser.parse_args()

    dep_path = Path(args.deployment)
    if dep_path.exists():
        DEPLOYMENT.update(json.load(open(dep_path)))

    # Environment variables override file
    if args.rpc or os.environ.get("RPC_URL"):
        DEPLOYMENT["rpc_url"] = args.rpc or os.environ["RPC_URL"]
    if args.contract or os.environ.get("CONTRACT_ADDRESS"):
        DEPLOYMENT["address"] = args.contract or os.environ["CONTRACT_ADDRESS"]
    if not DEPLOYMENT.get("rpc_url"):
        DEPLOYMENT["rpc_url"] = os.environ.get("RPC_URL", "https://sepolia.base.org")

    log.info("Starting  contract=%s  rpc=%s  port=%d",
             DEPLOYMENT.get("address"), DEPLOYMENT.get("rpc_url"), args.port)

    uvicorn.run(
        "server:app", host=args.host, port=args.port,
        workers=args.workers, log_level="info",
    )

if __name__ == "__main__":
    main()
