from typing import Any, Dict, List, TypedDict


class TraceStep(TypedDict):
    agent: str
    latency_ms: int
    output: Dict[str, Any]


class AgentState(TypedDict, total=False):
    user_id: str
    query: str
    top_k: int
    trace_id: str
    plan: List[str]
    tools: List[str]
    needs_retrieval: bool
    needs_tools: bool
    retrieved_docs: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    draft_answer: str
    final_answer: str
    critic_feedback: Dict[str, Any]
    retry_count: int
    should_retry: bool
    trace: List[TraceStep]
