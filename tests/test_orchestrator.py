from api.runtime import get_runtime_services


def test_orchestrator_returns_trace_and_answer() -> None:
    services = get_runtime_services()
    result = services.orchestrator.run(
        user_id="u1",
        query="Summarize this document and extract action items",
        top_k=3,
    )

    assert result["trace_id"]
    assert isinstance(result["answer"], str) and result["answer"].strip()
    assert isinstance(result["trace"], list) and len(result["trace"]) >= 2
    assert any(step["agent"] == "planner" for step in result["trace"])
    assert any(step["agent"] == "critic" for step in result["trace"])
