"""Test suite. Runs fully offline (no API key needed)."""
import os
import tempfile

from src.data.generate import build_inputs
from src.graph.orchestrator import run_platform
from src.memory.store import MemoryStore
from src.tools import financial_tools as T


def test_tools_catch_injected_anomalies():
    txns = build_inputs()["transactions"]
    amount = {h["account"] for h in T.detect_amount_anomalies(txns)}
    velocity = {h["account"] for h in T.detect_velocity_anomalies(txns)}
    geo = {h["account"] for h in T.detect_geo_anomalies(txns)}
    assert "ACCT-002" in amount      # injected amount outlier
    assert "ACCT-005" in velocity    # injected velocity burst
    assert "ACCT-003" in geo         # injected impossible travel


def test_var_and_concentration():
    var = T.compute_value_at_risk([-0.06, -0.02, 0.01, 0.03, -0.05] * 20)
    assert var["var_pct"] > 0
    conc = T.compute_portfolio_concentration(
        [{"symbol": "A", "market_value": 900}, {"symbol": "B", "market_value": 100}])
    assert conc["interpretation"] == "highly_concentrated"
    assert conc["largest_position"] == "A"


def test_full_graph_produces_report():
    with tempfile.TemporaryDirectory() as tmp:
        final = run_platform(build_inputs(),
                             memory_path=os.path.join(tmp, "mem.json"))
        for key in ("fraud", "market", "risk", "report"):
            assert key in final
        assert final["risk"]["level"] in ("LOW", "MEDIUM", "HIGH")
        assert "Executive Brief" in final["report"]["markdown"]


def test_memory_persists_across_runs():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mem.json")
        run_platform(build_inputs(), memory_path=path)
        run_platform(build_inputs(), memory_path=path)
        mem = MemoryStore(path)
        assert mem.run_count() == 2
        assert len(mem.repeat_offenders()) >= 1
