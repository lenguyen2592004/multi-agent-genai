import json
from difflib import SequenceMatcher
from pathlib import Path
import sys
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.runtime import get_runtime_services


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def run_eval(dataset_path: Path) -> Dict[str, Any]:
    services = get_runtime_services()

    rows: List[Dict[str, Any]] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    exact_match = 0
    semantic_scores: List[float] = []
    tool_correct = 0
    tool_total = 0

    for item in rows:
        result = services.orchestrator.run(user_id="eval", query=item["question"], top_k=4)
        answer = result.get("answer", "")
        expected = str(item.get("expected", ""))

        if expected.lower() in answer.lower():
            exact_match += 1

        semantic_scores.append(similarity(answer, expected))

        expected_tool = item.get("expected_tool")
        if expected_tool:
            tool_total += 1
            if any(
                row.get("tool") == expected_tool and row.get("status") == "success"
                for row in result.get("tool_results", [])
            ):
                tool_correct += 1

    total = max(1, len(rows))
    return {
        "samples": len(rows),
        "exact_match": exact_match / total,
        "semantic_similarity": sum(semantic_scores) / max(1, len(semantic_scores)),
        "tool_correctness": tool_correct / max(1, tool_total),
    }


if __name__ == "__main__":
    dataset = Path(__file__).resolve().parent / "dataset.jsonl"
    report = run_eval(dataset)
    print(json.dumps(report, indent=2, ensure_ascii=True))
