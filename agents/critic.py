from typing import Any, Dict, List


class CriticAgent:
    def validate(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        q = query.lower()

        if not answer.strip() or len(answer.strip()) < 40:
            reasons.append("Answer is too short or empty")

        if "action item" in q and "- " not in answer:
            reasons.append("Action items requested but bullet list is missing")

        if docs and "no retrieved documents" in answer.lower():
            reasons.append("Answer ignores available retrieved context")

        failed_tools = [item for item in tool_results if item.get("status") != "success"]
        if failed_tools and len(failed_tools) == len(tool_results):
            reasons.append("All requested tools failed")

        valid = not reasons
        return {
            "valid": valid,
            "reason": "OK" if valid else "; ".join(reasons),
            "failed_checks": reasons,
        }
