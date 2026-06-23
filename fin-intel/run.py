"""
CLI entry point.

Runs the full multi-agent pipeline on the synthetic dataset (or your own inputs),
prints the executive brief, and writes machine-readable results to ./outputs/ for
the dashboard to consume.

Usage:
    python run.py
    OPENAI_API_KEY=sk-... python run.py      # enables the live LLM backend
"""
from __future__ import annotations

import csv
import json
import os

from src.data.generate import build_inputs
from src.graph.orchestrator import run_platform

OUT_DIR = "outputs"


def _write_sample_csv(transactions: list[dict]) -> None:
    os.makedirs("src/data", exist_ok=True)
    path = "src/data/sample_transactions.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(transactions[0].keys()))
        writer.writeheader()
        writer.writerows(transactions)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    inputs = build_inputs()
    _write_sample_csv(inputs["transactions"])

    final = run_platform(inputs, memory_path=os.path.join(OUT_DIR, "memory_store.json"))

    # Persist results for the dashboard.
    results = {
        "backend": final.get("_backend"),
        "fraud": final.get("fraud"),
        "market": final.get("market"),
        "risk": final.get("risk"),
        "report": final.get("report"),
    }
    with open(os.path.join(OUT_DIR, "latest_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "latest_report.md"), "w", encoding="utf-8") as fh:
        fh.write(final["report"]["markdown"])

    print(f"\nLLM backend: {final.get('_backend')}\n")
    print(final["report"]["markdown"])
    print(f"\nResults written to ./{OUT_DIR}/latest_results.json and latest_report.md")


if __name__ == "__main__":
    main()
