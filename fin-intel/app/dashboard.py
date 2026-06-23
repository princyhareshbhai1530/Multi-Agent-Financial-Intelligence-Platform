"""
Streamlit dashboard — real-time financial insights.

Run with:
    streamlit run app/dashboard.py

Click "Run analysis" to execute the multi-agent pipeline live and render the
fraud / market / risk views plus the auto-generated executive brief.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# Make the project root importable when launched via `streamlit run app/dashboard.py`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.generate import build_inputs            # noqa: E402
from src.graph.orchestrator import run_platform        # noqa: E402

st.set_page_config(page_title="Financial Intelligence Platform",
                   page_icon="📊", layout="wide")

st.title("📊 Multi-Agent Financial Intelligence Platform")
st.caption("LangGraph · specialized agents for fraud, market, risk & reporting")

with st.sidebar:
    st.header("Controls")
    backend_hint = "OpenAI (live)" if os.getenv("OPENAI_API_KEY") else "Offline synthesis"
    st.info(f"LLM backend: **{backend_hint}**")
    run = st.button("▶ Run analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("Set `OPENAI_API_KEY` before launching to enable live LLM narrative.")


@st.cache_data(show_spinner="Running multi-agent pipeline…")
def _run():
    final = run_platform(build_inputs(),
                         memory_path=os.path.join("outputs", "memory_store.json"))
    return final


if run or st.session_state.get("_has_run"):
    st.session_state["_has_run"] = True
    final = _run()

    fraud, market, risk = final["fraud"], final["market"], final["risk"]

    # --- KPI row ------------------------------------------------------- #
    c1, c2, c3, c4 = st.columns(4)
    level = risk["level"]
    color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[level]
    c1.metric("Composite Risk", f"{color} {level}", f"score {risk['score']}/100")
    c2.metric("Flagged Accounts", len(fraud["flagged_accounts"]),
              f"severity {fraud['severity_score']}")
    c3.metric("95% Value at Risk", f"{risk['value_at_risk']['var_pct']}%")
    c4.metric("Market Regime", market["regime"].replace("_", "-"),
              f"vol {market['avg_annualized_vol_pct']}%")

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("🚨 Fraud signals")
        signals = fraud["signals"]
        rows = []
        for s in signals["amount"]:
            rows.append({"account": s["account"], "type": "amount outlier",
                         "detail": f"${s['amount']:,.0f} (z={s['modified_z']})"})
        for s in signals["velocity"]:
            rows.append({"account": s["account"], "type": "velocity spike",
                         "detail": f"{s['count_in_window']} txns / {s['window_min']}m"})
        for s in signals["geo"]:
            rows.append({"account": s["account"], "type": "impossible travel",
                         "detail": f"{s['from']}→{s['to']} @ {s['implied_kmh']:,} km/h"})
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(
            [{"account": "—", "type": "none", "detail": "no anomalies"}]),
            use_container_width=True, hide_index=True)

        st.subheader("📈 Market indicators")
        st.dataframe(pd.DataFrame(market["indicators"]),
                     use_container_width=True, hide_index=True)

    with right:
        st.subheader("🧮 Risk drivers")
        conc = risk["concentration"]
        st.write(f"**Concentration:** {conc['interpretation']} "
                 f"(HHI {conc['hhi']}, largest {conc['largest_position']} "
                 f"@ {conc['largest_weight_pct']}%)")
        st.progress(min(risk["score"], 100) / 100,
                    text=f"Composite risk score {risk['score']}/100")

        st.subheader("🧠 Risk narrative")
        st.write(risk["narrative"])

    st.markdown("---")
    st.subheader("📝 Executive brief (auto-generated)")
    st.markdown(final["report"]["markdown"])
else:
    st.info("Press **Run analysis** in the sidebar to execute the pipeline.")
