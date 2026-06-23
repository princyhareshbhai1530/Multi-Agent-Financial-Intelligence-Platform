"""Reporting agent: synthesizes the executive report and updates persistent memory."""
from __future__ import annotations

from datetime import datetime, timezone

from src.agents.base import BaseAgent


class ReportingAgent(BaseAgent):
    name = "reporting"
    role_prompt = ("You are an automated-reporting agent. You write a concise "
                   "executive brief for financial-operations leadership, integrating "
                   "fraud, market, and risk findings plus historical context.")

    def run(self, state: dict) -> dict:
        memory = state["memory"]
        fraud = state.get("fraud", {})
        market = state.get("market", {})
        risk = state.get("risk", {})

        # Update persistent memory before reporting so the brief reflects it.
        memory.record_flagged_accounts(fraud.get("flagged_accounts", []))
        memory.record_run({
            "risk_level": risk.get("level"),
            "flagged_accounts": len(fraud.get("flagged_accounts", [])),
            "regime": market.get("regime"),
        })
        memory.save()

        findings = {
            "headline": f"Executive brief — composite risk {risk.get('level', 'N/A')}.",
            "fraud_summary": {
                "flagged_accounts": fraud.get("flagged_accounts", []),
                "severity_score": fraud.get("severity_score", 0),
            },
            "market_summary": {
                "regime": market.get("regime"),
                "avg_annualized_vol_pct": market.get("avg_annualized_vol_pct"),
            },
            "risk_summary": {
                "level": risk.get("level"),
                "score": risk.get("score"),
                "var_95_pct": risk.get("value_at_risk", {}).get("var_pct"),
            },
            "historical_context": memory.context_brief(),
            "recommendation": self._tail_recommendation(risk.get("narrative", "")),
        }

        narrative = self.synthesize(findings)
        report = self._assemble_markdown(state, narrative, memory)
        return {"report": {"markdown": report, "generated_at":
                           datetime.now(timezone.utc).isoformat(timespec="seconds")}}

    @staticmethod
    def _tail_recommendation(narrative: str) -> str:
        if not narrative:
            return "See risk section."
        last = narrative.strip().splitlines()[-1]
        return last.split("Recommendation:", 1)[-1].strip()

    @staticmethod
    def _assemble_markdown(state: dict, narrative: str, memory) -> str:
        fraud = state.get("fraud", {})
        market = state.get("market", {})
        risk = state.get("risk", {})
        ctx = memory.context_brief()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Financial Intelligence — Executive Brief",
            f"_Generated {now} · run #{ctx['runs_to_date']}_",
            "",
            "## Synthesis",
            narrative,
            "",
            "## Fraud Detection",
            f"- Flagged accounts: **{len(fraud.get('flagged_accounts', []))}** "
            f"({', '.join(fraud.get('flagged_accounts', [])) or 'none'})",
            f"- Severity score: **{fraud.get('severity_score', 0)}**",
            "",
            "## Market Intelligence",
            f"- Regime: **{market.get('regime', 'n/a')}**",
            f"- Avg annualized volatility: **{market.get('avg_annualized_vol_pct', 0)}%**",
            "",
            "## Risk Assessment",
            f"- Composite level: **{risk.get('level', 'n/a')}** "
            f"(score {risk.get('score', 0)}/100)",
            f"- 95% Value at Risk: **{risk.get('value_at_risk', {}).get('var_pct', 0)}%**",
            f"- Concentration: **{risk.get('concentration', {}).get('interpretation', 'n/a')}** "
            f"(largest: {risk.get('concentration', {}).get('largest_position', 'n/a')})",
            "",
            "## Memory / Historical Context",
            f"- Prior risk level: **{ctx['prior_risk_level'] or 'n/a'}**",
            f"- Repeat-offender accounts: "
            f"{', '.join(o['account'] for o in ctx['repeat_offenders']) or 'none'}",
        ]
        return "\n".join(lines)
