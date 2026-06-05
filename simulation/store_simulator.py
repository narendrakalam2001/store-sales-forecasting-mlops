# ============================================================
# STORE SIMULATOR — Store Sales Forecasting System
# ============================================================
# Sends synthetic forecast requests to the API.
# 3 scenarios:  random | high_promo | holiday
# ============================================================

import requests
import random
import time
import os

API_URL = os.getenv("FORECAST_API_URL", "http://localhost:8000") + "/forecast"

FAMILIES = [
    "GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY",
    "MEATS", "BREAD/BAKERY", "PERSONAL CARE", "FROZEN FOODS",
    "DELI", "EGGS", "PREPARED FOODS", "SEAFOOD", "BEAUTY",
]

DATES_VAL = [f"2017-08-{str(d).zfill(2)}" for d in range(1, 16)]


def generate_request(scenario: str = "random") -> dict:
    if scenario == "high_promo":
        return {
            "store_nbr":           random.randint(1, 54),
            "family":              random.choice(["GROCERY I", "BEVERAGES", "CLEANING"]),
            "date":                random.choice(DATES_VAL),
            "onpromotion":         random.randint(50, 300),
            "dcoilwtico":          random.uniform(40, 55),
            "is_national_holiday": 0,
        }
    elif scenario == "holiday":
        return {
            "store_nbr":           random.randint(1, 54),
            "family":              random.choice(FAMILIES),
            "date":                random.choice(DATES_VAL),
            "onpromotion":         random.randint(0, 50),
            "dcoilwtico":          random.uniform(40, 55),
            "is_national_holiday": 1,
        }
    else:
        return {
            "store_nbr":           random.randint(1, 54),
            "family":              random.choice(FAMILIES),
            "date":                random.choice(DATES_VAL),
            "onpromotion":         random.randint(0, 100),
            "dcoilwtico":          random.uniform(35, 80),
            "is_national_holiday": random.choice([0, 0, 0, 1]),
        }


def send_request(req: dict, idx: int) -> None:
    try:
        resp = requests.post(API_URL, json=req, timeout=15)
        if resp.status_code == 200:
            r = resp.json()
            print(
                f"[{idx+1:3d}]  Store={req['store_nbr']:2d}  "
                f"Family={req['family'][:15]:<15}  "
                f"Date={req['date']}  "
                f"Promo={req['onpromotion']:3d}  "
                f"→  pred={r.get('predicted_sales',0):8.2f}  "
                f"conf={r.get('confidence_band','?'):<6}  "
                f"rule={r.get('rule_triggered') or 'None'}"
            )
        else:
            print(f"[{idx+1}] API error: {resp.status_code}")
    except Exception as e:
        print(f"[{idx+1}] Error: {e}")


def simulate(n: int = 20, scenario: str = "random") -> None:
    print(f"\nSimulating {n} requests  |  scenario={scenario}")
    print("-" * 80)
    for i in range(n):
        req = generate_request(scenario)
        send_request(req, i)
        time.sleep(0.3)
    print("-" * 80)
    print("Simulation complete\n")


if __name__ == "__main__":
    simulate(20, scenario="random")
