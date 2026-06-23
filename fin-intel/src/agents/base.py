"""Base agent: a small unit that runs tools, then asks the LLM to synthesize."""
from __future__ import annotations

import json
from typing import Any

from src.llm.client import LLMClient


class BaseAgent:
    name: str = "agent"
    role_prompt: str = "You are a financial analyst agent."

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, findings: dict) -> str:
        """Ask the LLM to turn structured findings into analyst narrative."""
        user = (
            "Summarize the following analysis for a financial-operations audience. "
            "Be concise, lead with the most material point, and avoid speculation "
            "beyond the data.\n\nFINDINGS:" + json.dumps(findings, default=str)
        )
        return self.llm.complete(self.role_prompt, user).text

    def run(self, state: dict) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError
