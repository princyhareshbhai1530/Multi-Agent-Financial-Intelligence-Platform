"""
Memory management.

Two layers:
  1. Run state  — carried through the LangGraph execution (see graph/orchestrator.py).
  2. Persistent memory — a JSON-backed store that survives across runs so the
     platform can accumulate context: repeat-offender accounts, prior risk levels,
     and a rolling log of run summaries the reporting agent can reference.

This is deliberately simple (file-backed) so the proof-of-concept has zero infra
dependencies; the same interface could be backed by Redis/DynamoDB in production.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any


class MemoryStore:
    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        self._data: dict[str, Any] = {
            "runs": [],
            "flagged_accounts": {},  # account -> times flagged
            "last_risk_level": None,
        }
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    # ---- writers -------------------------------------------------------- #
    def record_flagged_accounts(self, accounts: list[str]) -> None:
        counts = self._data.setdefault("flagged_accounts", {})
        for acct in accounts:
            counts[acct] = counts.get(acct, 0) + 1

    def record_run(self, summary: dict) -> None:
        self._data.setdefault("runs", []).append({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **summary,
        })
        self._data["last_risk_level"] = summary.get("risk_level")

    # ---- readers -------------------------------------------------------- #
    def repeat_offenders(self, min_count: int = 2) -> list[dict]:
        counts = self._data.get("flagged_accounts", {})
        return [{"account": a, "times_flagged": c}
                for a, c in sorted(counts.items(), key=lambda kv: -kv[1])
                if c >= min_count]

    def prior_risk_level(self) -> str | None:
        return self._data.get("last_risk_level")

    def run_count(self) -> int:
        return len(self._data.get("runs", []))

    def context_brief(self) -> dict:
        """A compact memory snapshot the reporting agent can fold into its narrative."""
        return {
            "runs_to_date": self.run_count(),
            "prior_risk_level": self.prior_risk_level(),
            "repeat_offenders": self.repeat_offenders(),
        }
