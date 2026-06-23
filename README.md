# Multi-Agent Financial Intelligence Platform

A proof-of-concept business-intelligence platform that coordinates four specialized
AI agents over a **LangGraph** state graph to analyze financial activity end-to-end:
fraud detection, market intelligence, risk assessment, and automated reporting.

The quantitative work (anomaly detection, Value at Risk, concentration) is done by
real, testable tools. The **OpenAI**-backed LLM layer is responsible only for
*synthesis* - turning structured findings into analyst-grade narrative - so the
platform produces meaningful output even with no API key, and a key simply upgrades
the narrative quality.

> Tech: Python · LangGraph · OpenAI API · pandas/numpy · Streamlit · Docker

---

## Architecture

```
                 ┌──────────────────────────────┐
                 │        LangGraph StateGraph    │
   inputs ──►    │                                │
 (transactions,  │   START ─┬─► fraud  ─┐         │
  market feed,   │          └─► market ─┤         │
  positions)     │                      ▼         │
                 │                    risk ──► reporting ──► report
                 └──────────────────────────────┘                │
                          │ shared state (TypedDict)              ▼
                          └────────► MemoryStore (JSON) ◄─── repeat offenders,
                                                              prior risk levels
```

- **Agent coordination** - `fraud` and `market` run in parallel from `START`; `risk`
  is a fan-in that waits for both, then `reporting` synthesizes everything.
- **Tool-calling** - agents invoke deterministic tools (`detect_amount_anomalies`,
  `compute_value_at_risk`, `compute_portfolio_concentration`, …). The same registry
  can be exposed to OpenAI function-calling.
- **Memory management** - in-graph shared state during a run, plus a persistent
  JSON `MemoryStore` that accumulates cross-run context (repeat-offender accounts,
  prior risk level) that the reporting agent folds into its brief.

### Agents
| Agent | Responsibility |
|---|---|
| `FraudDetectionAgent` | Amount outliers (modified z-score), velocity bursts, impossible travel |
| `MarketIntelligenceAgent` | Per-symbol indicators, volatility, risk-on/off regime |
| `RiskAssessmentAgent` | Fuses fraud + market + portfolio VaR/concentration → composite risk level |
| `ReportingAgent` | Executive brief + persists run to memory |

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) Batch run - prints the executive brief, writes outputs/
python run.py

# 2) Dashboard - interactive, real-time view
streamlit run app/dashboard.py
```

By default the platform runs on its **offline synthesis engine** (no key needed).
To enable the live LLM backend:

```bash
cp .env.example .env          # add your key, then:
export OPENAI_API_KEY=sk-...
python run.py                 # backend prints "openai"
```

### Tests
```bash
pytest -q
```

---

## Project layout
```
fin-intel/
├── run.py                      # CLI entry point
├── app/dashboard.py            # Streamlit dashboard
├── src/
│   ├── graph/orchestrator.py   # LangGraph StateGraph (coordination)
│   ├── agents/                 # fraud, market, risk, reporting + base
│   ├── tools/financial_tools.py# quantitative tools (testable)
│   ├── memory/store.py         # persistent cross-run memory
│   ├── llm/client.py           # OpenAI client + offline fallback
│   └── data/generate.py        # synthetic dataset w/ injected anomalies
├── tests/test_pipeline.py
├── Dockerfile
└── requirements.txt
```

---

## Deploying to AWS (cloud services + dashboard)

The container is the deployable unit. Two common targets:

- **ECS Fargate / App Runner** - `docker build -t fin-intel . && docker push` to ECR,
  then run the image; App Runner exposes the Streamlit port (8501) with a managed URL.
- **Batch / scheduled analysis** - run the same image with `python run.py` on an
  EventBridge schedule; swap the JSON `MemoryStore` for DynamoDB and write reports to S3.

For production you'd put `OPENAI_API_KEY` in Secrets Manager and front the dashboard
with an ALB. These steps require your own AWS account and credentials.

---

## Notes & honest scope

This is a **proof-of-concept**. The data is synthetic (with deliberately injected
anomalies so the agents have real signal to find), and "cloud deployment" here means
the container + the AWS steps above rather than a live hosted environment. Everything
in `run.py`, the agents, the LangGraph orchestration, the tools, the memory layer, and
the dashboard is fully implemented and runs locally as-is.
