"""Fraud detection agent: runs anomaly tools and triages the results."""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.tools import financial_tools as T


class FraudDetectionAgent(BaseAgent):
    name = "fraud_detection"
    role_prompt = ("You are a fraud-detection agent for a financial intelligence "
                   "platform. You triage anomaly signals and call out the highest-risk "
                   "transactions and accounts.")

    def run(self, state: dict) -> dict:
        txns = state["transactions"]

        amount_hits = T.detect_amount_anomalies(txns)
        velocity_hits = T.detect_velocity_anomalies(txns)
        geo_hits = T.detect_geo_anomalies(txns)

        flagged_accounts = sorted({h["account"] for h in amount_hits}
                                  | {h["account"] for h in velocity_hits}
                                  | {h["account"] for h in geo_hits})

        severity = len(amount_hits) + 2 * len(velocity_hits) + 3 * len(geo_hits)
        findings = {
            "headline": f"{len(flagged_accounts)} account(s) flagged across "
                        f"{len(txns)} transactions (severity score {severity}).",
            "amount_anomalies": amount_hits,
            "velocity_anomalies": velocity_hits,
            "impossible_travel": geo_hits,
            "recommendation": ("Escalate flagged accounts for manual review and "
                               "apply step-up authentication." if flagged_accounts
                               else "No anomalies; continue passive monitoring."),
        }

        return {
            "fraud": {
                "flagged_accounts": flagged_accounts,
                "severity_score": severity,
                "signals": {
                    "amount": amount_hits,
                    "velocity": velocity_hits,
                    "geo": geo_hits,
                },
                "narrative": self.synthesize(findings),
            }
        }
