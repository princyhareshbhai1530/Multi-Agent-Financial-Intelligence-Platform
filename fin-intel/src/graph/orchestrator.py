"""
LangGraph orchestration.

Builds a StateGraph that coordinates the four agents. Topology:

    ingest ─► fraud ─┐
                     ├─► risk ─► reporting ─► END
    ingest ─► market ┘

`risk` is a fan-in: it runs only after both `fraud` and `market` have written to
shared state. LangGraph handles this via the edge wiring below — both fraud and
market route into risk, and because state updates are merged, risk sees both.

Shared state is a TypedDict; each node returns a partial dict that LangGraph merges
into the running state. This is the "agent coordination + memory management" layer.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.fraud_detection import FraudDetectionAgent
from src.agents.market_intelligence import MarketIntelligenceAgent
from src.agents.reporting import ReportingAgent
from src.agents.risk_assessment import RiskAssessmentAgent
from src.llm.client import LLMClient
from src.memory.store import MemoryStore


class PlatformState(TypedDict, total=False):
    # Inputs
    transactions: list[dict]
    market_data: dict
    positions: list[dict]
    portfolio_returns: list[float]
    memory: Any
    # Agent outputs (merged in as agents run)
    fraud: dict
    market: dict
    risk: dict
    report: dict


def build_platform(llm: LLMClient | None = None, memory_path: str = "memory_store.json"):
    """Compile and return the multi-agent LangGraph app plus its memory store."""
    llm = llm or LLMClient()
    memory = MemoryStore(memory_path)

    fraud_agent = FraudDetectionAgent(llm)
    market_agent = MarketIntelligenceAgent(llm)
    risk_agent = RiskAssessmentAgent(llm)
    reporting_agent = ReportingAgent(llm)

    graph = StateGraph(PlatformState)
    graph.add_node("fraud", fraud_agent.run)
    graph.add_node("market", market_agent.run)
    graph.add_node("risk", risk_agent.run)
    graph.add_node("reporting", reporting_agent.run)

    # Fan-out from START to the two independent analysis agents.
    graph.add_edge(START, "fraud")
    graph.add_edge(START, "market")
    # Fan-in: risk waits for both fraud and market.
    graph.add_edge("fraud", "risk")
    graph.add_edge("market", "risk")
    graph.add_edge("risk", "reporting")
    graph.add_edge("reporting", END)

    app = graph.compile()
    return app, memory, llm


def run_platform(inputs: dict, llm: LLMClient | None = None,
                 memory_path: str = "memory_store.json") -> dict:
    """Convenience runner: build, inject memory, invoke, return final state."""
    app, memory, llm = build_platform(llm, memory_path)
    state: PlatformState = {
        "transactions": inputs["transactions"],
        "market_data": inputs["market_data"],
        "positions": inputs["positions"],
        "portfolio_returns": inputs.get("portfolio_returns", []),
        "memory": memory,
    }
    final = app.invoke(state)
    final["_backend"] = llm.backend
    return final
