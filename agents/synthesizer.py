from typing import Any, Dict, List

from llm.ollama_client import OllamaClient


class SynthesizerAgent:
    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm_client = llm_client

    def synthesize(
        self,
        query: str,
        plan: List[str],
        docs: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        prompt = self._build_prompt(query, plan, docs, tool_results)
        llm_answer = self.llm_client.generate(
            prompt=(
                "You are an enterprise assistant. Answer with grounded facts only. "
                "If context is limited, state assumptions briefly."
            ),
            user_input=prompt,
        )
        if llm_answer.strip():
            return llm_answer.strip()
        return self._fallback_answer(query, plan, docs, tool_results)

    def _build_prompt(
        self,
        query: str,
        plan: List[str],
        docs: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        doc_block = "\n".join(
            f"- ({doc.get('score', 0):.3f}) {doc.get('text', '')[:400]}" for doc in docs[:5]
        )
        tool_block = "\n".join(
            f"- {item.get('tool', 'unknown')} [{item.get('status', 'unknown')}]: {str(item.get('result', ''))[:300]}"
            for item in tool_results
        )
        plan_block = "\n".join(f"- {step}" for step in plan)

        return (
            f"User query:\n{query}\n\n"
            f"Plan:\n{plan_block or '- none'}\n\n"
            f"Retrieved context:\n{doc_block or '- none'}\n\n"
            f"Tool outputs:\n{tool_block or '- none'}\n\n"
            "Produce a concise, complete answer. If action items are requested, include a bullet list."
        )

    def _fallback_answer(
        self,
        query: str,
        plan: List[str],
        docs: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = ["Summary:"]

        if docs:
            lines.append("Grounded points from retrieved documents:")
            for doc in docs[:3]:
                lines.append(f"- {doc.get('text', '')[:180]}")
        else:
            lines.append("- No retrieved documents were available.")

        if tool_results:
            lines.append("Tool outputs:")
            for item in tool_results:
                status = item.get("status", "unknown")
                result = str(item.get("result", ""))[:180]
                lines.append(f"- {item.get('tool', 'unknown')} [{status}]: {result}")

        if "action item" in query.lower():
            lines.append("Action items:")
            actions = self._extract_actions(docs)
            if actions:
                for action in actions:
                    lines.append(f"- {action}")
            else:
                lines.append("- Review the source document and assign owners with due dates.")

        if plan:
            lines.append("Executed plan:")
            for step in plan:
                lines.append(f"- {step}")

        return "\n".join(lines)

    def _extract_actions(self, docs: List[Dict[str, Any]]) -> List[str]:
        action_signals = ("must", "should", "action", "deadline", "owner", "follow-up")
        actions: List[str] = []
        for doc in docs:
            sentences = doc.get("text", "").replace("\n", " ").split(".")
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue
                if any(signal in sentence_clean.lower() for signal in action_signals):
                    actions.append(sentence_clean)
                if len(actions) >= 5:
                    return actions
        return actions
