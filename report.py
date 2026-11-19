"""Builds the human-readable markdown report and the JSON export from a run."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json


def _money(x):
    return f"${x:,.0f}"


def build_markdown(run) -> str:
    R = run["results"]
    sc = run["scenario"]
    L = [f"# Recovery Report: {sc['title']}", "", f"_{sc['description']}_", "",
         f"**Severity:** {run['severity'].upper()}  |  **LLM:** {run['llm_status']}", ""]

    L += ["## Detected disruptions", ""]
    for e in run["events"]:
        L.append(f"- **{e['id']}** {e['type'].replace('_', ' ')} - "
                 f"{e.get('location') or e.get('supplier_id')} - {e['duration_days']} days "
                 f"(severity {e['severity']}, confidence {e['confidence']:.0%}, source: {e['source']})")
    imp = R.get("impact")
    if imp and imp.data:
        d = imp.data
        L += ["", "## Impact", "", imp.summary, "",
              "| Shipment | SKU | Qty | Original ETA | New ETA | Delay | Cause |", "|---|---|---|---|---|---|---|"]
        for a in d["affected_shipments"]:
            L.append(f"| {a['shipment_id']} | {a['sku']} | {a['qty']} | {a['orig_eta']} | {a['new_eta']} | "
                     f"{a['delay_days']}d | {', '.join(a['reasons'])} |")
    inv = R.get("inventory")
    if inv and inv.data:
        L += ["", "## Inventory", "", inv.summary, "",
              "| SKU | Stock | Demand/day | Days cover | Stock-out | Resupply | Replace qty | Status |",
              "|---|---|---|---|---|---|---|---|"]
        for sku, v in inv.data["per_sku"].items():
            L.append(f"| {sku} | {v['stock']} | {v['daily_demand']} | {v['days_of_cover']} | {v['stockout_date']} | "
                     f"{v['resupply_date']} | {v['replacement_qty']} | {v['status']} |")
    for key, title in (("supplier", "Supplier options"), ("logistics", "Logistics options")):
        r = R.get(key)
        if r:
            L += ["", f"## {title}", "", r.summary]

    pl = R.get("planner")
    if pl and pl.data:
        d = pl.data
        L += ["", "## Recovery options", "",
              "| Option | Orders on time | High-priority on time | Procurement | Extra freight | Late penalties | **Total cost** | Risk | Score |",
              "|---|---|---|---|---|---|---|---|---|"]
        b = d["baseline"]
        L.append(f"| {b['label']} | {b['on_time_orders']}/{b['total_orders']} | {b['high_on_time']}/{b['high_total']} | "
                 f"{_money(b['procurement_cost'])} | {_money(b['extra_freight'])} | {_money(b['late_penalty'])} | "
                 f"**{_money(b['total_cost'])}** | {b['risk']:.0%} | - |")
        for o in d["options"]:
            star = " (recommended)" if o["recommended"] else ""
            L.append(f"| {o['label']}{star} | {o['on_time_orders']}/{o['total_orders']} | {o['high_on_time']}/{o['high_total']} | "
                     f"{_money(o['procurement_cost'])} | {_money(o['extra_freight'])} | {_money(o['late_penalty'])} | "
                     f"**{_money(o['total_cost'])}** | {o['risk']:.0%} | {o['score']} |")
        L += ["", "## Recommendation", "", pl.summary, "", "**Actions**", ""]
        L += [f"{i}. {a}" for i, a in enumerate(d["actions"], 1)]
        if d["escalation_reasons"]:
            L += ["", "**Escalated to human because:**", ""] + [f"- {r}" for r in d["escalation_reasons"]]
        L += ["", "**Risks**", ""] + [f"- {r}" for r in d["risks"]]
        L += ["", f"**Overall confidence:** {pl.confidence:.0%}"]
    ap = run.get("approval")
    if ap:
        L += ["", f"**Approval status:** {ap['decision']}" + (f" - {ap['note']}" if ap["note"] else "")]

    notes = sorted({a for r in R.values() if r for a in r.assumptions})
    warns = sorted({w for r in R.values() if r for w in r.warnings})
    if notes:
        L += ["", "## Assumptions (incomplete information)", ""] + [f"- {n}" for n in notes]
    if warns:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in warns]
    L += ["", "## Agent activity", "", "| Agent | Time (s) | Status | LLM used |", "|---|---|---|---|"]
    for name, secs, status, used in run["timeline"]:
        L.append(f"| {name} | {secs} | {status} | {'yes' if used else 'no'} |")
    return "\n".join(L) + "\n"


def _jsonable(o):
    if dataclasses.is_dataclass(o):
        return _jsonable(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    return o


def build_json(run) -> str:
    return json.dumps(_jsonable(run), indent=2)
