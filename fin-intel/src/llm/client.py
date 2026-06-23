"""
Provider-agnostic LLM client.

Primary backend: OpenAI Chat Completions API (matches the platform's documented stack).
Fallback backend: a deterministic, offline "synthesis engine" that turns the
structured quantitative findings produced by the tools into readable narrative.

Design intent
-------------
The agents do the *quantitative* work with real tools (anomaly detection, VaR,
concentration, etc.). The LLM layer is responsible only for *synthesis* — turning
structured findings into analyst-grade narrative and triage. Because of that split,
the platform produces meaningful output even with no API key, and swapping in a real
key simply upgrades the quality of the narrative. Set OPENAI_API_KEY to enable the
live backend.
"""
from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    backend: str  # "openai" or "offline"


class LLMClient:
    """A thin wrapper that prefers OpenAI and degrades gracefully to offline mode."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._client = None
        if self._api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except Exception:  # pragma: no cover - defensive
                self._client = None

    @property
    def backend(self) -> str:
        return "openai" if self._client else "offline"

    def complete(self, system: str, user: str) -> LLMResponse:
        """Single-shot completion. Falls back to offline synthesis on any failure."""
        if self._client is not None:
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return LLMResponse(text=resp.choices[0].message.content.strip(),
                                   backend="openai")
            except Exception as exc:  # pragma: no cover - network/credentials
                return LLMResponse(
                    text=self._offline_synthesis(system, user, error=str(exc)),
                    backend="offline",
                )
        return LLMResponse(text=self._offline_synthesis(system, user), backend="offline")

    # ------------------------------------------------------------------ #
    # Offline deterministic synthesis
    # ------------------------------------------------------------------ #
    @staticmethod
    def _offline_synthesis(system: str, user: str, error: str | None = None) -> str:
        """
        Turn the structured findings embedded in the user prompt into narrative.

        Agents pass their findings as a JSON block tagged FINDINGS:{...}. We parse
        that and produce a templated-but-substantive analyst summary, so the offline
        mode is genuinely useful rather than a stub.
        """
        findings = LLMClient._extract_findings(user)
        role = LLMClient._extract_role(system)

        if not findings:
            return (f"[offline] {role}: no structured findings were supplied; "
                    "unable to synthesize a narrative.")

        lines: list[str] = []
        headline = findings.get("headline")
        if headline:
            lines.append(headline)

        for section, payload in findings.items():
            if section in ("headline", "recommendation"):
                continue
            lines.append(LLMClient._render_section(section, payload))

        rec = findings.get("recommendation")
        if rec:
            lines.append(f"Recommendation: {rec}")

        body = "\n".join(l for l in lines if l)
        return textwrap.dedent(body).strip()

    @staticmethod
    def _render_section(section: str, payload: Any) -> str:
        title = section.replace("_", " ").capitalize()
        if isinstance(payload, list):
            if not payload:
                return f"{title}: none detected."
            bullets = "\n".join(f"  - {LLMClient._stringify(item)}" for item in payload[:8])
            more = "" if len(payload) <= 8 else f"\n  - (+{len(payload) - 8} more)"
            return f"{title}:\n{bullets}{more}"
        if isinstance(payload, dict):
            parts = ", ".join(f"{k}={LLMClient._stringify(v)}" for k, v in payload.items())
            return f"{title}: {parts}"
        return f"{title}: {LLMClient._stringify(payload)}"

    @staticmethod
    def _stringify(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:,.2f}"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{k}: {LLMClient._stringify(x)}" for k, x in v.items()) + "}"
        return str(v)

    @staticmethod
    def _extract_findings(user: str) -> dict:
        marker = "FINDINGS:"
        if marker not in user:
            return {}
        raw = user.split(marker, 1)[1].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to recover the first JSON object in the string.
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(raw[start:end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

    @staticmethod
    def _extract_role(system: str) -> str:
        for line in system.splitlines():
            if line.lower().startswith("you are"):
                return line.strip().rstrip(".")
        return "Analyst agent"
