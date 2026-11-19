"""Tools shared by the agents: data access and calculation functions.

Agents never touch the CSVs directly; they call these tools. Any tool can raise,
and the Coordinator is responsible for retry / fallback.
"""
from __future__ import annotations

import datetime as dt
import math
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
SIM_TODAY = dt.date.fromisoformat(os.getenv("SIM_DATE", "2026-11-19"))

PRIORITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}
LATE_PENALTY_RATE = 0.08      # assumption: SLA penalty = 8% of order value per unfulfilled share
FREIGHT_BASE_RATE = 0.04      # assumption: sea freight = 4% of shipment value
ORIGIN_REGION = {
    "Shanghai": "China", "Shenzhen": "China", "Dubai": "UAE",
    "Mumbai": "India", "Istanbul": "Turkey", "Karachi": "Pakistan",
}


def to_date(x) -> dt.date:
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x)[:10])


def _num(x):
    return None if x is None else (int(x) if float(x).is_integer() else float(x))


class DataStore:
    FILES = ("shipments", "inventory", "suppliers", "routes", "orders")

    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)
        for name in self.FILES:
            setattr(self, name, self._read(name))
        for s in self.shipments:
            s["eta"] = to_date(s["eta"])
        for o in self.orders:
            o["deadline"] = to_date(o["deadline"])

    def _read(self, name):
        p = self.dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run `python generate_data.py` first")
        df = pd.read_csv(p)
        df = df.astype(object).where(pd.notna(df), None)
        return df.to_dict("records")

    # ---------- lookups ----------
    def locations(self):
        locs = {s["origin"] for s in self.shipments} | {s["destination"] for s in self.shipments}
        locs |= {r["destination"] for r in self.routes} | {"Gwadar"}
        return sorted(locs)

    def supplier_name(self, sid):
        for s in self.suppliers:
            if s["supplier_id"] == sid:
                return s["name"]
        return None

    def supplier_ids(self):
        return sorted({s["supplier_id"] for s in self.suppliers})

    def inventory_for(self, sku):
        return next((i for i in self.inventory if i["sku"] == sku), None)

    def suppliers_for(self, sku, exclude=()):
        return [dict(s) for s in self.suppliers
                if s["sku"] == sku and s["status"] == "Active" and s["supplier_id"] not in exclude]

    def routes_for(self, origin, destination):
        region = ORIGIN_REGION.get(origin)
        return [dict(r) for r in self.routes if r["origin_region"] == region and r["destination"] == destination]

    # ---------- mutations used for what-if / mid-run events ----------
    def cut_capacity(self, supplier_id, new_capacity, sku=None):
        n = 0
        for s in self.suppliers:
            if s["supplier_id"] == supplier_id and (sku is None or s["sku"] == sku):
                s["capacity_units"] = new_capacity
                n += 1
        return n

    def delay_supplier(self, supplier_id, days):
        for s in self.suppliers:
            if s["supplier_id"] == supplier_id:
                s["lead_time_days"] = int(s["lead_time_days"]) + int(days)

    # ---------- mock external tool ----------
    def weather_check(self, location):
        """Mock weather API (swap for OpenWeatherMap in production)."""
        stormy = {"Karachi": "Cyclone warning - port and airport operations restricted"}
        return {"location": location, "alert": stormy.get(location, "No active alerts")}


# ---------- disruption physics ----------
def shipment_delay(ev, s):
    """Returns (delay_days, reason) if event `ev` delays shipment `s`, else None."""
    t, dur = ev["type"], int(ev.get("duration_days") or 0)
    if t == "port_strike":
        if s["destination"] == ev["location"] and s["mode"] == "Sea":
            return dur, f"port strike at {ev['location']}"
    elif t == "weather":
        if ev["location"] in (s["origin"], s["destination"]):
            d = dur if s["mode"] == "Sea" else math.ceil(dur / 2)
            return max(d, 1), f"severe weather at {ev['location']}"
    elif t in ("supplier_failure", "supplier_delay"):
        if s["supplier_id"] == ev.get("supplier_id"):
            label = "supplier failure" if t == "supplier_failure" else "supplier delay"
            return dur, f"{label} ({ev['supplier_id']})"
    return None


def route_blocked(ev, route):
    t, loc = ev["type"], ev.get("location")
    if t == "port_strike":
        return route["via"] == loc and route["mode"] == "Sea"
    if t == "weather":
        return str(route["via"]).startswith(str(loc))
    return False


def freight_cost(shipment, route):
    return FREIGHT_BASE_RATE * shipment["value_usd"] * route["cost_multiplier"]


# ---------- calculators ----------
def days_of_cover(stock, daily_demand):
    return None if not daily_demand else round(stock / daily_demand, 1)


def simulate_fulfillment(orders, stock_by_sku, arrivals_by_sku):
    """Allocates supply to orders by priority then deadline.

    arrivals_by_sku: {sku: [(date, qty), ...]}.  Orders are treated as the committed
    demand (daily_demand is used separately by the inventory view).
    """
    allocated = defaultdict(float)
    results = []
    ranked = sorted(orders, key=lambda o: (-PRIORITY_WEIGHT.get(o["priority"], 1), o["deadline"]))
    for o in ranked:
        sku = o["sku"]
        by_deadline = sum(q for d, q in arrivals_by_sku.get(sku, []) if d <= o["deadline"])
        available = stock_by_sku.get(sku, 0) + by_deadline - allocated[sku]
        fulfilled = max(0, min(o["qty"], available))
        allocated[sku] += fulfilled
        results.append({
            "order_id": o["order_id"], "customer_id": o["customer_id"], "priority": o["priority"],
            "sku": sku, "qty": o["qty"], "deadline": o["deadline"],
            "order_value_usd": o["order_value_usd"], "fulfilled_qty": fulfilled,
            "on_time": fulfilled >= o["qty"],
            "fill_rate": round(fulfilled / o["qty"], 3) if o["qty"] else 1.0,
        })
    return results


def service_metrics(sim):
    tot_w = sum(PRIORITY_WEIGHT.get(r["priority"], 1) for r in sim) or 1
    service = sum(PRIORITY_WEIGHT.get(r["priority"], 1) * r["fill_rate"] for r in sim) / tot_w
    penalty = sum(LATE_PENALTY_RATE * r["order_value_usd"] * (1 - r["fill_rate"]) for r in sim)
    high = [r for r in sim if r["priority"] == "High"]
    return {
        "service_score": round(service, 3),
        "late_penalty": round(penalty, 2),
        "on_time_orders": sum(r["on_time"] for r in sim),
        "total_orders": len(sim),
        "high_on_time": sum(r["on_time"] for r in high),
        "high_total": len(high),
    }


def risk_score(weighted_reliabilities):
    """weighted_reliabilities: [(weight, reliability)] -> risk in [0,1]."""
    tot = sum(w for w, _ in weighted_reliabilities)
    if not tot:
        return 0.5
    return round(1 - sum(w * r for w, r in weighted_reliabilities) / tot, 3)
