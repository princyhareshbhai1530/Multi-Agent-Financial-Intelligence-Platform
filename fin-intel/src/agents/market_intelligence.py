"""Market intelligence agent: pulls indicators and characterizes the regime."""
from __future__ import annotations

from statistics import mean

from src.agents.base import BaseAgent
from src.tools import financial_tools as T


class MarketIntelligenceAgent(BaseAgent):
    name = "market_intelligence"
    role_prompt = ("You are a market-intelligence agent. You summarize market "
                   "indicators and characterize the prevailing regime (risk-on vs "
                   "risk-off) for downstream risk assessment.")

    def run(self, state: dict) -> dict:
        market_data = state["market_data"]
        symbols = list(market_data.keys())

        indicators = [T.fetch_market_indicator(s, market_data) for s in symbols]
        valid = [i for i in indicators if i.get("status") != "no_data"]
        avg_vol = mean(i["annualized_vol_pct"] for i in valid) if valid else 0.0
        up = sum(1 for i in valid if i["trend"] == "up")
        regime = "risk_on" if up >= len(valid) / 2 and avg_vol < 30 else "risk_off"

        findings = {
            "headline": f"Market regime: {regime.replace('_', '-')} "
                        f"(avg annualized vol {avg_vol:.1f}%).",
            "indicators": indicators,
            "breadth": {"symbols": len(valid), "trending_up": up},
            "recommendation": ("Maintain exposure with normal hedges."
                               if regime == "risk_on"
                               else "Reduce gross exposure and tighten stop levels."),
        }

        return {
            "market": {
                "regime": regime,
                "avg_annualized_vol_pct": round(avg_vol, 2),
                "indicators": indicators,
                "narrative": self.synthesize(findings),
            }
        }
