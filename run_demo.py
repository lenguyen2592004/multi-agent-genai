import json

from api.runtime import get_runtime_services


if __name__ == "__main__":
    services = get_runtime_services()
    result = services.orchestrator.run(
        user_id="demo-user",
        query="Summarize this document and extract action items",
        top_k=4,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
