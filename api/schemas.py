from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=3, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=10)


class IngestRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    document_id: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceStepModel(BaseModel):
    agent: str
    latency_ms: int
    output: Dict[str, Any]


class QueryResponse(BaseModel):
    trace_id: str
    answer: str
    valid: bool
    reason: str
    latency_ms: int
    plan: List[str]
    tools: List[str]
    tool_results: List[Dict[str, Any]]
    trace: List[TraceStepModel]


class BasicResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
