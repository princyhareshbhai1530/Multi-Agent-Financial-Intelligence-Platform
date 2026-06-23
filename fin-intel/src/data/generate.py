"""
Synthetic data generator.

Produces a self-contained, reproducible dataset with *deliberately injected*
anomalies so the agents have genuine signal to find:
  - one amount outlier
  - one velocity burst
  - one impossible-travel pair
It also generates a small market feed, a portfolio, and a return series for VaR.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

CITIES = ["Toronto", "New York", "London", "Singapore", "Mumbai",
          "San Francisco", "Frankfurt"]
MERCHANTS = ["GroceryCo", "AirTravel", "ElectroMart", "FuelStop", "CloudSaaS",
             "LuxuryGoods", "Pharmacy", "RideShare"]
ACCOUNTS = [f"ACCT-{i:03d}" for i in range(1, 9)]
SYMBOLS = ["AAPL", "TSLA", "JPM", "XOM", "BTC"]


def generate_transactions(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    base = datetime(2026, 6, 1, 9, 0, 0)
    # Each account has a stable home city; legitimate activity stays local, so the
    # geo detector only fires on genuinely anomalous travel.
    home_city = {acct: rng.choice(CITIES[:4]) for acct in ACCOUNTS}
    home_city["ACCT-003"] = "Toronto"  # pinned: makes its injected travel anomaly clean
    txns: list[dict] = []
    tid = 0
    for _ in range(60):
        tid += 1
        acct = rng.choice(ACCOUNTS)
        txns.append({
            "txn_id": f"T{tid:05d}",
            "account": acct,
            "amount": round(rng.uniform(8, 320), 2),
            "merchant": rng.choice(MERCHANTS),
            "city": home_city[acct],
            "timestamp": (base + timedelta(minutes=rng.randint(0, 6000))).isoformat(),
        })

    # --- Injected anomalies -------------------------------------------- #
    # 1) Amount outlier (kept in home city so it's purely an amount signal)
    tid += 1
    txns.append({"txn_id": f"T{tid:05d}", "account": "ACCT-002", "amount": 48750.00,
                 "merchant": "LuxuryGoods", "city": home_city["ACCT-002"],
                 "timestamp": (base + timedelta(minutes=1200)).isoformat()})
    # 2) Velocity burst on ACCT-005 (6 txns within ~4 minutes, home city)
    burst = base + timedelta(minutes=2000)
    for k in range(6):
        tid += 1
        txns.append({"txn_id": f"T{tid:05d}", "account": "ACCT-005",
                     "amount": round(rng.uniform(40, 90), 2), "merchant": "CloudSaaS",
                     "city": home_city["ACCT-005"],
                     "timestamp": (burst + timedelta(seconds=40 * k)).isoformat()})
    # 3) Impossible travel on ACCT-003 (Toronto -> Singapore in 30 min)
    t0 = base + timedelta(minutes=3000)
    tid += 1
    txns.append({"txn_id": f"T{tid:05d}", "account": "ACCT-003", "amount": 120.0,
                 "merchant": "FuelStop", "city": "Toronto", "timestamp": t0.isoformat()})
    tid += 1
    txns.append({"txn_id": f"T{tid:05d}", "account": "ACCT-003", "amount": 95.0,
                 "merchant": "AirTravel", "city": "Singapore",
                 "timestamp": (t0 + timedelta(minutes=30)).isoformat()})

    return txns


def generate_market_data(seed: int = 11) -> dict:
    rng = random.Random(seed)
    data: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        price = rng.uniform(80, 400)
        vol = rng.uniform(0.01, 0.05)  # daily vol; BTC ends up choppier
        series = []
        day = datetime(2026, 5, 1)
        for d in range(40):
            price *= (1 + rng.gauss(0.0003, vol))
            series.append({"date": (day + timedelta(days=d)).date().isoformat(),
                           "close": round(max(price, 1), 2)})
        data[sym] = series
    return data


def generate_portfolio(seed: int = 3) -> tuple[list[dict], list[float]]:
    rng = random.Random(seed)
    positions = [
        {"symbol": "AAPL", "market_value": 420000},
        {"symbol": "TSLA", "market_value": 95000},
        {"symbol": "JPM", "market_value": 180000},
        {"symbol": "XOM", "market_value": 60000},
        {"symbol": "BTC", "market_value": 240000},  # drives concentration up
    ]
    returns = [rng.gauss(0.0004, 0.018) for _ in range(250)]
    # Inject a few tail losses so VaR is non-trivial.
    for idx in (40, 120, 200):
        returns[idx] = -0.06
    return positions, returns


def build_inputs() -> dict:
    positions, returns = generate_portfolio()
    return {
        "transactions": generate_transactions(),
        "market_data": generate_market_data(),
        "positions": positions,
        "portfolio_returns": returns,
    }
