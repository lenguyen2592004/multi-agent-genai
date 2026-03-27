import time
from pathlib import Path
from typing import Dict

from fastapi import Depends, FastAPI

from api.deps import InMemoryRateLimiter, parse_rate_limit_from_env, require_optional_bearer
from api.runtime import get_runtime_services
from api.schemas import BasicResponse, IngestRequest, QueryRequest, QueryResponse, TraceStepModel
from observability.logging_config import configure_logging


BASE_DIR = Path(__file__).resolve().parents[1]
configure_logging(BASE_DIR / "logs" / "app.log")

app = FastAPI(title="Local-First Multi-Agent GenAI Platform", version="0.1.0")

rate_limiter = InMemoryRateLimiter(max_requests=parse_rate_limit_from_env(), window_seconds=60)


@app.on_event("startup")
async def startup_event() -> None:
    get_runtime_services()


@app.get("/health", response_model=Dict[str, str])
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "service": "Local-First Multi-Agent GenAI Platform",
        "status": "ok",
        "docs": "/docs",
        "query_endpoint": "/query",
        "query_endpoint_compat": "/api/query",
    }


@app.get("/api/health", response_model=Dict[str, str])
async def api_health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    payload: QueryRequest,
    _: None = Depends(require_optional_bearer),
) -> QueryResponse:
    rate_limiter.check(payload.user_id)
    services = get_runtime_services()

    start = time.perf_counter()
    result = services.orchestrator.run(
        user_id=payload.user_id,
        query=payload.query,
        top_k=payload.top_k,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    services.metrics.record_request(
        latency_ms=latency_ms,
        success=result.get("valid", False),
        token_estimate=max(1, len(payload.query.split()) + len(result.get("answer", "").split())),
    )
    services.metrics.record_tool_usage(result.get("tool_results", []))

    trace = [TraceStepModel(**step) for step in result.get("trace", [])]

    return QueryResponse(
        trace_id=result["trace_id"],
        answer=result["answer"],
        valid=result.get("valid", False),
        reason=result.get("reason", ""),
        latency_ms=latency_ms,
        plan=result.get("plan", []),
        tools=result.get("tools", []),
        tool_results=result.get("tool_results", []),
        trace=trace,
    )


@app.post("/api/ingest", response_model=BasicResponse)
@app.post("/ingest", response_model=BasicResponse)
async def ingest_endpoint(
    payload: IngestRequest,
    _: None = Depends(require_optional_bearer),
) -> BasicResponse:
    services = get_runtime_services()
    metadata = {"user_id": payload.user_id}
    metadata.update(payload.metadata)

    chunks = services.rag_pipeline.ingest_text(
        document_id=payload.document_id,
        text=payload.text,
        metadata=metadata,
    )

    return BasicResponse(
        status="success",
        message="Document ingested",
        details={"chunks": chunks},
    )


@app.get("/api/metrics", response_model=Dict[str, float])
@app.get("/metrics", response_model=Dict[str, float])
async def metrics_endpoint() -> Dict[str, float]:
    services = get_runtime_services()
    return services.metrics.snapshot()
