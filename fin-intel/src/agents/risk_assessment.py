"""
Risk assessment agent.

This agent is the fusion point: it depends on the fraud agent's severity and the
market agent's regime, then adds portfolio-level risk (VaR + concentration) to
produce a single composite risk level.
"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.tools import financial_tools as T


class RiskAssessmentAgent(BaseAgent):
    name = "risk_assessment"
    role_prompt = ("You are a risk-assessment agent. You fuse fraud signals, the "
                   "market regime, and portfolio risk metrics into a single composite "
                   "risk level with a clear rationale.")

    def run(self, state: dict) -> dict:
        positions = state["positions"]
        portfolio_returns = state.get("portfolio_returns", [])

        var = T.compute_value_at_risk(portfolio_returns, confidence=0.95)
        concentration = T.compute_portfolio_concentration(positions)

        fraud_severity = state.get("fraud", {}).get("severity_score", 0)
        regime = state.get("market", {}).get("regime", "risk_on")

        # Composite scoring (0-100), transparent and auditable.
        score = 0
        score += min(fraud_severity * 4, 40)
        score += 20 if regime == "risk_off" else 0
        score += min(var["var_pct"] * 2, 25)
        score += {"well_diversified": 0, "moderately_concentrated": 8,
                  "highly_concentrated": 15}[concentration["interpretation"]]
        score = min(round(score), 100)
        level = "LOW" if score < 30 else "MEDIUM" if score < 60 else "HIGH"

        findings = {
            "headline": f"Composite risk level: {level} (score {score}/100).",
            "drivers": {
                "fraud_severity": fraud_severity,
                "market_regime": regime,
                "value_at_risk_95_pct": var["var_pct"],
                "concentration": concentration["interpretation"],
                "largest_position": concentration["largest_position"],
            },
            "recommendation": {
                "LOW": "Routine monitoring; no action required.",
                "MEDIUM": "Increase review cadence; pre-stage hedges.",
                "HIGH": "Escalate to risk committee; de-risk and freeze flagged accounts.",
            }[level],
        }

        return {
            "risk": {
                "level": level,
                "score": score,
                "value_at_risk": var,
                "concentration": concentration,
                "narrative": self.synthesize(findings),
            }
        }
