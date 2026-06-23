"""
Financial analysis tools.

These are the real, deterministic computations the agents invoke. Keeping the
quantitative logic in plain, testable functions (rather than asking the LLM to
"do math") is both better engineering and the reason the platform produces
trustworthy output. Each tool has a docstring and a typed signature so it can be
exposed to OpenAI function-calling as well.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Any

# Rough lat/long for the cities used in the synthetic dataset, for the
# impossible-travel heuristic.
_CITY_COORDS = {
    "Toronto": (43.65, -79.38),
    "New York": (40.71, -74.01),
    "London": (51.51, -0.13),
    "Singapore": (1.35, 103.82),
    "Mumbai": (19.08, 72.88),
    "Sydney": (-33.87, 151.21),
    "San Francisco": (37.77, -122.42),
    "Frankfurt": (50.11, 8.68),
}


# --------------------------------------------------------------------------- #
# Fraud / anomaly tools
# --------------------------------------------------------------------------- #
def detect_amount_anomalies(transactions: list[dict], z_threshold: float = 3.0) -> list[dict]:
    """Flag transactions whose amount is a statistical outlier (modified z-score)."""
    amounts = [t["amount"] for t in transactions]
    if len(amounts) < 3:
        return []
    med = _median(amounts)
    abs_dev = [abs(a - med) for a in amounts]
    mad = _median(abs_dev) or 1e-9
    flagged = []
    for t in transactions:
        score = 0.6745 * (t["amount"] - med) / mad
        if abs(score) >= z_threshold:
            flagged.append({
                "txn_id": t["txn_id"],
                "account": t["account"],
                "amount": round(t["amount"], 2),
                "modified_z": round(score, 2),
                "reason": "amount_outlier",
            })
    return sorted(flagged, key=lambda x: -abs(x["modified_z"]))


def detect_velocity_anomalies(transactions: list[dict], window_min: int = 5,
                              max_in_window: int = 4) -> list[dict]:
    """Flag accounts making an unusually rapid burst of transactions."""
    by_account: dict[str, list[dict]] = defaultdict(list)
    for t in transactions:
        by_account[t["account"]].append(t)

    flagged = []
    for account, txns in by_account.items():
        txns = sorted(txns, key=lambda x: x["timestamp"])
        times = [_parse_ts(t["timestamp"]) for t in txns]
        for i in range(len(times)):
            window = [j for j in range(len(times))
                      if 0 <= (times[i] - times[j]).total_seconds() <= window_min * 60]
            if len(window) > max_in_window:
                flagged.append({
                    "account": account,
                    "count_in_window": len(window),
                    "window_min": window_min,
                    "reason": "velocity_spike",
                })
                break
    return flagged


def detect_geo_anomalies(transactions: list[dict], max_kmh: float = 900.0) -> list[dict]:
    """Flag 'impossible travel': consecutive txns implying a speed above a jet."""
    by_account: dict[str, list[dict]] = defaultdict(list)
    for t in transactions:
        by_account[t["account"]].append(t)

    flagged = []
    for account, txns in by_account.items():
        txns = sorted(txns, key=lambda x: x["timestamp"])
        for a, b in zip(txns, txns[1:]):
            if a["city"] == b["city"]:
                continue
            dist = _haversine_km(a["city"], b["city"])
            hours = (_parse_ts(b["timestamp"]) - _parse_ts(a["timestamp"])).total_seconds() / 3600
            if hours <= 0:
                continue
            speed = dist / hours
            if speed > max_kmh:
                flagged.append({
                    "account": account,
                    "from": a["city"],
                    "to": b["city"],
                    "implied_kmh": round(speed),
                    "reason": "impossible_travel",
                })
    return flagged


# --------------------------------------------------------------------------- #
# Market / risk tools
# --------------------------------------------------------------------------- #
def fetch_market_indicator(symbol: str, market_data: dict) -> dict:
    """Return the latest indicator snapshot for a symbol from the market feed."""
    series = market_data.get(symbol)
    if not series:
        return {"symbol": symbol, "status": "no_data"}
    closes = [p["close"] for p in series]
    returns = _pct_returns(closes)
    return {
        "symbol": symbol,
        "last_close": round(closes[-1], 2),
        "period_return_pct": round((closes[-1] / closes[0] - 1) * 100, 2),
        "annualized_vol_pct": round(_annualized_vol(returns) * 100, 2),
        "trend": "up" if closes[-1] > mean(closes) else "down",
    }


def compute_value_at_risk(returns: list[float], confidence: float = 0.95) -> dict:
    """Historical-simulation Value at Risk (one-period, as a positive loss fraction)."""
    if not returns:
        return {"var_pct": 0.0, "confidence": confidence}
    ordered = sorted(returns)
    idx = max(0, int((1 - confidence) * len(ordered)) - 1)
    var = -ordered[idx]
    return {"var_pct": round(var * 100, 2), "confidence": confidence,
            "observations": len(returns)}


def compute_portfolio_concentration(positions: list[dict]) -> dict:
    """Herfindahl-Hirschman concentration index over portfolio weights."""
    total = sum(p["market_value"] for p in positions) or 1e-9
    weights = {p["symbol"]: p["market_value"] / total for p in positions}
    hhi = sum(w ** 2 for w in weights.values())
    top = max(weights.items(), key=lambda kv: kv[1])
    return {
        "hhi": round(hhi, 4),
        "interpretation": _hhi_label(hhi),
        "largest_position": top[0],
        "largest_weight_pct": round(top[1] * 100, 2),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _pct_returns(closes: list[float]) -> list[float]:
    return [(closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes))]


def _annualized_vol(returns: list[float], periods: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * math.sqrt(periods)


def _hhi_label(hhi: float) -> str:
    if hhi < 0.15:
        return "well_diversified"
    if hhi < 0.25:
        return "moderately_concentrated"
    return "highly_concentrated"


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def _haversine_km(city_a: str, city_b: str) -> float:
    a = _CITY_COORDS.get(city_a)
    b = _CITY_COORDS.get(city_b)
    if not a or not b:
        return 0.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


# Registry consumed by the agents (and convertible to OpenAI tool schemas).
TOOL_REGISTRY: dict[str, Any] = {
    "detect_amount_anomalies": detect_amount_anomalies,
    "detect_velocity_anomalies": detect_velocity_anomalies,
    "detect_geo_anomalies": detect_geo_anomalies,
    "fetch_market_indicator": fetch_market_indicator,
    "compute_value_at_risk": compute_value_at_risk,
    "compute_portfolio_concentration": compute_portfolio_concentration,
}
