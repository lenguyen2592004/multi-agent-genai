import time
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from agents.critic import CriticAgent
from agents.planner import PlannerAgent
from agents.retrieval import RetrievalAgent
from agents.state import AgentState, TraceStep
from agents.synthesizer import SynthesizerAgent
from agents.tool_executor import ToolExecutionAgent
from llm.ollama_client import OllamaClient
from rag.pipeline import RAGPipeline
from tools.registry import ToolRegistry


class AgentOrchestrator:
    def __init__(
        self,
        llm_client: OllamaClient,
        rag_pipeline: RAGPipeline,
        tool_registry: ToolRegistry,
    ) -> None:
        self.planner = PlannerAgent(llm_client=llm_client)
        self.retrieval = RetrievalAgent(rag_pipeline=rag_pipeline)
        self.tool_executor = ToolExecutionAgent(tool_registry=tool_registry)
        self.synthesizer = SynthesizerAgent(llm_client=llm_client)
        self.critic = CriticAgent()
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("tool_execution", self._tools_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("repair", self._repair_node)

        workflow.set_entry_point("planner")

        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "retrieval": "retrieval",
                "tool_execution": "tool_execution",
                "synthesizer": "synthesizer",
            },
        )

        workflow.add_conditional_edges(
            "retrieval",
            self._route_after_retrieval,
            {
                "tool_execution": "tool_execution",
                "synthesizer": "synthesizer",
            },
        )

        workflow.add_edge("tool_execution", "synthesizer")
        workflow.add_edge("synthesizer", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "retry": "repair",
                "end": END,
            },
        )
        workflow.add_edge("repair", "synthesizer")

        return workflow

    def _append_trace(
        self,
        state: AgentState,
        agent: str,
        latency_ms: int,
        output: Dict[str, Any],
    ) -> List[TraceStep]:
        trace = list(state.get("trace", []))
        trace.append({"agent": agent, "latency_ms": latency_ms, "output": output})
        return trace

    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        plan = self.planner.plan(state["query"])
        latency = int((time.perf_counter() - start) * 1000)
        trace = self._append_trace(state, "planner", latency, plan)

        tools = list(plan.get("tools", []))
        return {
            "plan": list(plan.get("steps", [])),
            "needs_retrieval": bool(plan.get("needs_retrieval", False)),
            "tools": tools,
            "needs_tools": bool(tools),
            "trace": trace,
        }

    def _retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        docs = self.retrieval.retrieve(state["query"], top_k=int(state.get("top_k", 4)))
        latency = int((time.perf_counter() - start) * 1000)
        trace = self._append_trace(
            state,
            "retrieval",
            latency,
            {"documents": len(docs)},
        )
        return {"retrieved_docs": docs, "trace": trace}

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        tools = list(state.get("tools", []))
        results = self.tool_executor.execute(tools=tools, query=state["query"])
        latency = int((time.perf_counter() - start) * 1000)
        trace = self._append_trace(
            state,
            "tools",
            latency,
            {"tools": [item.get("tool", "unknown") for item in results]},
        )
        return {"tool_results": results, "trace": trace}

    def _synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        answer = self.synthesizer.synthesize(
            query=state["query"],
            plan=list(state.get("plan", [])),
            docs=list(state.get("retrieved_docs", [])),
            tool_results=list(state.get("tool_results", [])),
        )
        latency = int((time.perf_counter() - start) * 1000)
        trace = self._append_trace(
            state,
            "synthesizer",
            latency,
            {"answer_chars": len(answer)},
        )
        return {"draft_answer": answer, "trace": trace}

    def _critic_node(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()
        feedback = self.critic.validate(
            query=state["query"],
            answer=state.get("draft_answer", ""),
            docs=list(state.get("retrieved_docs", [])),
            tool_results=list(state.get("tool_results", [])),
        )
        latency = int((time.perf_counter() - start) * 1000)

        retry_count = int(state.get("retry_count", 0))
        should_retry = (not feedback.get("valid", False)) and retry_count < 1
        if should_retry:
            retry_count += 1

        final_answer = state.get("draft_answer", "")
        if not feedback.get("valid", False):
            final_answer = f"{final_answer}\n\n[Validator notice] {feedback.get('reason', 'unknown validation issue')}"

        trace = self._append_trace(
            state,
            "critic",
            latency,
            {"valid": bool(feedback.get("valid", False)), "reason": feedback.get("reason", "")},
        )

        return {
            "critic_feedback": feedback,
            "final_answer": final_answer,
            "should_retry": should_retry,
            "retry_count": retry_count,
            "trace": trace,
        }

    def _repair_node(self, state: AgentState) -> Dict[str, Any]:
        start = time.perf_counter()

        docs = list(state.get("retrieved_docs", []))
        if not docs:
            docs = self.retrieval.retrieve(state["query"], top_k=max(5, int(state.get("top_k", 4))))

        tools = list(state.get("tools", []))
        tool_results = list(state.get("tool_results", []))
        if tools and not tool_results:
            tool_results = self.tool_executor.execute(tools=tools, query=state["query"])

        latency = int((time.perf_counter() - start) * 1000)
        trace = self._append_trace(
            state,
            "repair",
            latency,
            {"documents": len(docs), "tool_results": len(tool_results)},
        )

        return {
            "retrieved_docs": docs,
            "tool_results": tool_results,
            "trace": trace,
        }

    def _route_after_planner(self, state: AgentState) -> str:
        if state.get("needs_retrieval", False):
            return "retrieval"
        if state.get("needs_tools", False):
            return "tool_execution"
        return "synthesizer"

    def _route_after_retrieval(self, state: AgentState) -> str:
        if state.get("needs_tools", False):
            return "tool_execution"
        return "synthesizer"

    def _route_after_critic(self, state: AgentState) -> str:
        if state.get("should_retry", False):
            return "retry"
        return "end"

    def run(self, user_id: str, query: str, top_k: int = 4) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())

        initial_state: AgentState = {
            "user_id": user_id,
            "query": query,
            "top_k": top_k,
            "trace_id": trace_id,
            "retry_count": 0,
            "trace": [],
            "plan": [],
            "tools": [],
            "retrieved_docs": [],
            "tool_results": [],
        }

        result: AgentState = self.graph.invoke(initial_state)
        feedback = result.get("critic_feedback", {"valid": False, "reason": "No critic feedback"})

        return {
            "trace_id": trace_id,
            "answer": result.get("final_answer", result.get("draft_answer", "")),
            "valid": bool(feedback.get("valid", False)),
            "reason": str(feedback.get("reason", "")),
            "plan": list(result.get("plan", [])),
            "tools": list(result.get("tools", [])),
            "tool_results": list(result.get("tool_results", [])),
            "trace": list(result.get("trace", [])),
        }
