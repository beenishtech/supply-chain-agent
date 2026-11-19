"""The six specialist agents. The Coordinator lives in orchestrator.py.

Pattern for every agent:
  1. tools compute the facts (deterministic, auditable)
  2. the LLM (if available) turns the facts into reasoning text
  3. the agent returns an AgentResult with data, confidence and assumptions
"""
from __future__ import annotations

import datetime as dt
import json
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

import tools as T
from tools import SIM_TODAY, DataStore


# ======================================================================
# Infrastructure
# ======================================================================
class MessageBus:
    """Every agent-to-agent message goes through here (also feeds the UI later)."""

    def __init__(self, printer=None):
        self.messages = []
        self.printer = printer

    def send(self, sender, receiver, kind, content):
        m = {"t": time.strftime("%H:%M:%S"), "from": sender, "to": receiver, "kind": kind, "content": content}
        self.messages.append(m)
        if self.printer:
            self.printer(m)
        return m


@dataclass
class AgentResult:
    agent: str
    data: dict = field(default_factory=dict)
    summary: str = ""
    confidence: float = 1.0
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    used_llm: bool = False
    degraded: bool = False
    seconds: float = 0.0


class BaseAgent:
    name = "Agent"
    role = ""

    def __init__(self, store: DataStore, llm, bus: MessageBus):
        self.store, self.llm, self.bus = store, llm, bus
        self.fail_mode = None   # "once" | "always" (used by the failure-injection demo)

    def run(self, ctx) -> AgentResult:
        t0 = time.time()
        if self.fail_mode:
            if self.fail_mode == "once":
                self.fail_mode = None
            raise RuntimeError(f"{self.name}: simulated tool outage")
        res = self.execute(ctx)
        res.seconds = round(time.time() - t0, 2)
        return res

    def execute(self, ctx) -> AgentResult:  # pragma: no cover
        raise NotImplementedError

    def fallback(self, ctx, error) -> AgentResult:
        return AgentResult(self.name, {}, f"{self.name} unavailable ({error}); continuing without it.",
                           confidence=0.3, warnings=[f"{self.name} failed: {error}"], degraded=True)

    def narrate(self, facts: dict, fallback_text: str):
        """Returns (text, used_llm)."""
        system = (f"You are the {self.name} in a supply chain crisis response team. Role: {self.role} "
                  "Write 2-3 concise sentences of assessment using ONLY the facts given. "
                  "Do not invent numbers, names or dates. No markdown.")
        out = self.llm.complete(system, json.dumps(facts, default=str))
        return (out, True) if out else (fallback_text, False)


# ======================================================================
# 1. Disruption Monitor
# ======================================================================
class MonitorAgent(BaseAgent):
    name = "Disruption Monitor Agent"
    role = "Detects disruption events from raw signals, classifies type and severity."

    DEFAULT_DURATION = {"port_strike": 5, "weather": 3, "supplier_failure": 30,
                        "supplier_delay": 7, "capacity_cut": 14}

    def execute(self, ctx):
        signals = ctx["signals"]
        events, assumptions, warnings, confs = [], [], [], []
        for i, sig in enumerate(signals, 1):
            ev = self._parse_rules(sig["text"])
            if ev is None or ev["confidence"] < 0.8:
                ev = self._parse_llm(sig["text"], ev) or ev
            if ev is None:
                warnings.append(f"Could not classify signal: {sig['text'][:60]}")
                continue
            ev["id"] = f"EV-{len(ctx.get('events', [])) + i}"
            ev["source"] = sig.get("source", "unknown")
            ev["raw"] = sig["text"]
            ev["severity"] = self._severity(ev)
            assumptions += ev.pop("assumptions", [])
            confs.append(ev["confidence"])
            events.append(ev)
            self.bus.send(self.name, "Coordinator", "EVENT",
                          f"{ev['id']} {ev['type']} @ {ev.get('location') or ev.get('supplier_id')} "
                          f"duration={ev['duration_days']}d severity={ev['severity']} conf={ev['confidence']}")
        order = ["low", "medium", "high", "critical"]
        sev = max((e["severity"] for e in events), key=order.index, default="low")
        facts = {"events": [{k: e[k] for k in ("type", "location", "supplier_id", "duration_days", "severity")
                             if k in e} for e in events]}
        fb = "; ".join(f"{e['type'].replace('_', ' ')} ({e['severity']})" for e in events) or "no events detected"
        text, used = self.narrate(facts, f"Detected {len(events)} disruption(s): {fb}.")
        return AgentResult(self.name, {"events": events, "severity": sev}, text,
                           confidence=min(confs) if confs else 0.0, assumptions=assumptions,
                           warnings=warnings, used_llm=used)

    # ---- rule based parser ----
    def _parse_rules(self, text):
        t = text.lower()
        if "capacity" in t:
            etype = "capacity_cut"
        elif any(k in t for k in ("bankrupt", "insolven", "shut down", "shutdown", "collapse", "ceased")):
            etype = "supplier_failure"
        elif any(k in t for k in ("strike", "walkout")):
            etype = "port_strike"
        elif any(k in t for k in ("cyclone", "storm", "flood", "hurricane", "weather")):
            etype = "weather"
        elif "delay" in t:
            etype = "supplier_delay"
        else:
            return None
        ev = {"type": etype, "confidence": 0.9, "assumptions": []}
        # location
        loc = next((l for l in self.store.locations() if l.lower() in t), None)
        ev["location"] = None if etype in ("supplier_failure", "supplier_delay", "capacity_cut") else loc
        # supplier
        m = re.search(r"sup-\d+", t)
        sid = m.group(0).upper() if m else None
        if not sid:
            for s in self.store.suppliers:
                if s["name"].lower() in t:
                    sid = s["supplier_id"]
                    break
        ev["supplier_id"] = sid
        # duration
        m = re.search(r"(\d+)\s*(day|week)", t)
        if m:
            ev["duration_days"] = int(m.group(1)) * (7 if m.group(2) == "week" else 1)
        else:
            ev["duration_days"] = self.DEFAULT_DURATION[etype]
            ev["confidence"] = 0.6
            ev["assumptions"].append(
                f"{etype.replace('_', ' ')}: duration not stated, assumed {ev['duration_days']} days (low confidence)")
        # capacity cut: new capacity
        if etype == "capacity_cut":
            m = re.search(r"(?:to|at)\s+(\d+)\s*units", t)
            ev["new_capacity"] = int(m.group(1)) if m else 100
            if not m:
                ev["assumptions"].append("capacity cut size not stated, assumed 100 units")
                ev["confidence"] = min(ev["confidence"], 0.6)
        # required fields
        if etype in ("port_strike", "weather") and not ev["location"]:
            ev["confidence"] = 0.4
            ev["assumptions"].append("location could not be identified from the signal")
        if etype in ("supplier_failure", "supplier_delay", "capacity_cut") and not ev["supplier_id"]:
            ev["confidence"] = 0.4
            ev["assumptions"].append("supplier could not be identified from the signal")
        return ev

    def _parse_llm(self, text, partial):
        """Ask the LLM to fill gaps; validate strictly, otherwise ignore."""
        system = ("Extract a supply chain disruption as JSON with keys: type (port_strike|weather|"
                  "supplier_failure|supplier_delay|capacity_cut), location, supplier_id, duration_days (int). "
                  "Use null when unknown. Return JSON only.")
        known = f"Known locations: {self.store.locations()}. Known suppliers: {self.store.supplier_ids()}.\nSignal: {text}"
        out = self.llm.complete(system, known, max_tokens=200)
        if not out:
            return None
        try:
            data = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            if data.get("type") not in self.DEFAULT_DURATION:
                return None
            dur = data.get("duration_days")
            ev = dict(partial or {"confidence": 0.7, "assumptions": []})
            ev.update({"type": data["type"], "location": data.get("location") or ev.get("location"),
                       "supplier_id": data.get("supplier_id") or ev.get("supplier_id"),
                       "duration_days": int(dur) if dur else ev.get("duration_days") or self.DEFAULT_DURATION[data["type"]]})
            ev["confidence"] = max(ev.get("confidence", 0.5), 0.7)
            return ev
        except Exception:
            return None

    @staticmethod
    def _severity(ev):
        t, d, raw = ev["type"], ev["duration_days"], ev.get("raw", "")
        if t == "supplier_failure":
            return "high"
        if t == "port_strike":
            return "high" if d >= 5 else "medium"
        if t == "weather":
            return "high" if "cyclone" in raw.lower() or d >= 5 else "medium"
        if t == "supplier_delay":
            return "medium" if d >= 7 else "low"
        return "medium"


# ======================================================================
# 2. Impact Analysis
# ======================================================================
class ImpactAgent(BaseAgent):
    name = "Impact Analysis Agent"
    role = "Finds affected shipments, orders and customers and quantifies the exposure."

    def execute(self, ctx):
        events = ctx["events"]
        affected = {}
        for s in self.store.shipments:
            if s["status"] == "Delivered":
                continue
            for ev in events:
                hit = T.shipment_delay(ev, s)
                if not hit:
                    continue
                delay, reason = hit
                cur = affected.get(s["shipment_id"])
                if cur is None:
                    affected[s["shipment_id"]] = {
                        "shipment_id": s["shipment_id"], "sku": s["sku"], "qty": s["qty"],
                        "origin": s["origin"], "destination": s["destination"], "mode": s["mode"],
                        "status": s["status"], "value_usd": s["value_usd"], "customer_id": s["customer_id"],
                        "supplier_id": s["supplier_id"], "orig_eta": s["eta"], "delay_days": delay,
                        "new_eta": s["eta"] + dt.timedelta(days=delay), "reasons": [reason],
                        "event_ids": [ev["id"]], "event_types": [ev["type"]],
                    }
                else:   # several events hit the same shipment: the worst delay wins
                    cur["reasons"].append(reason)
                    cur["event_ids"].append(ev["id"])
                    cur["event_types"].append(ev["type"])
                    if delay > cur["delay_days"]:
                        cur["delay_days"] = delay
                        cur["new_eta"] = s["eta"] + dt.timedelta(days=delay)

        stock = {i["sku"]: i["stock"] for i in self.store.inventory}
        base_arrivals = defaultdict(list)
        for s in self.store.shipments:
            eta = affected[s["shipment_id"]]["new_eta"] if s["shipment_id"] in affected else s["eta"]
            base_arrivals[s["sku"]].append((eta, s["qty"]))
        orig_arrivals = defaultdict(list)
        for s in self.store.shipments:
            orig_arrivals[s["sku"]].append((s["eta"], s["qty"]))

        before = T.simulate_fulfillment(self.store.orders, stock, orig_arrivals)
        after = T.simulate_fulfillment(self.store.orders, stock, base_arrivals)
        before_late = {r["order_id"] for r in before if not r["on_time"]}
        newly_late = [r for r in after if not r["on_time"] and r["order_id"] not in before_late]
        late_all = [r for r in after if not r["on_time"]]
        high_late = [r for r in late_all if r["priority"] == "High"]
        value_at_risk = sum(a["value_usd"] for a in affected.values())
        revenue_at_risk = sum(r["order_value_usd"] * (1 - r["fill_rate"]) for r in late_all)
        customers = sorted({r["customer_id"] for r in late_all})

        data = {
            "affected_shipments": sorted(affected.values(), key=lambda a: a["shipment_id"]),
            "baseline_orders": after,
            "late_orders": late_all, "newly_late_orders": newly_late, "high_priority_late": high_late,
            "value_at_risk_usd": value_at_risk, "revenue_at_risk_usd": round(revenue_at_risk, 2),
            "customers_affected": customers,
            "late_penalty_usd": T.service_metrics(after)["late_penalty"],
        }
        assumptions, conf = [], 1.0
        if not affected:
            assumptions.append("no shipment matched the disruption(s) in the shipment table")
            conf = 0.5
        unknown_suppliers = {a["supplier_id"] for a in affected.values()} - set(self.store.supplier_ids())
        if unknown_suppliers:
            assumptions.append(f"supplier record missing for {sorted(unknown_suppliers)}; reliability unknown")
            conf = min(conf, 0.8)
        facts = {"affected_shipments": len(affected), "value_at_risk_usd": value_at_risk,
                 "orders_late": [(r["order_id"], r["priority"], r["fill_rate"]) for r in late_all],
                 "revenue_at_risk_usd": round(revenue_at_risk)}
        fb = (f"{len(affected)} shipment(s) worth ${value_at_risk:,.0f} are affected; "
              f"{len(late_all)} order(s) would miss deadlines ({len(high_late)} high priority), "
              f"putting about ${revenue_at_risk:,.0f} of revenue at risk.")
        text, used = self.narrate(facts, fb)
        self.bus.send(self.name, "Coordinator", "RESULT", fb)
        return AgentResult(self.name, data, text, conf, assumptions, used_llm=used)


# ======================================================================
# 3. Inventory
# ======================================================================
class InventoryAgent(BaseAgent):
    name = "Inventory Agent"
    role = "Computes days of cover, stock-out dates and replacement quantities per SKU."
    HORIZON = 14

    def execute(self, ctx):
        impact = ctx["results"]["impact"].data
        affected = {a["shipment_id"]: a for a in impact["affected_shipments"]}
        late_skus = {r["sku"] for r in impact["late_orders"]}
        skus = sorted({a["sku"] for a in affected.values()} | late_skus)
        per_sku, assumptions = {}, []
        for sku in skus:
            inv = self.store.inventory_for(sku)
            if not inv:
                assumptions.append(f"{sku}: no inventory record, skipped")
                continue
            doc = T.days_of_cover(inv["stock"], inv["daily_demand"])
            stockout = SIM_TODAY + dt.timedelta(days=int(doc)) if doc is not None else None
            incoming = []
            for s in self.store.shipments:
                if s["sku"] != sku:
                    continue
                eta = affected[s["shipment_id"]]["new_eta"] if s["shipment_id"] in affected else s["eta"]
                incoming.append({"shipment_id": s["shipment_id"], "eta": eta, "qty": s["qty"],
                                 "affected": s["shipment_id"] in affected})
            resupply = min((i["eta"] for i in incoming), default=None)
            gap = max(0, (resupply - SIM_TODAY).days) if resupply else self.HORIZON
            need = inv["daily_demand"] * gap
            safe_incoming = sum(i["qty"] for i in incoming if not i["affected"] and i["eta"] <= (resupply or SIM_TODAY))
            demand_deficit = max(0, need - inv["stock"] - safe_incoming)
            order_shortfall = sum(r["qty"] - r["fulfilled_qty"] for r in impact["late_orders"] if r["sku"] == sku)
            replacement = int(max(demand_deficit, order_shortfall))
            high = [r for r in impact["high_priority_late"] if r["sku"] == sku]
            deadline = min((r["deadline"] for r in high), default=None) or \
                min((r["deadline"] for r in impact["late_orders"] if r["sku"] == sku), default=None)
            status = "critical" if (stockout and resupply and stockout < resupply) else \
                     ("watch" if replacement > 0 else "ok")
            per_sku[sku] = {
                "item": inv["item_name"], "stock": inv["stock"], "daily_demand": inv["daily_demand"],
                "days_of_cover": doc, "stockout_date": stockout, "resupply_date": resupply,
                "gap_days": gap, "demand_deficit": int(demand_deficit), "order_shortfall": int(order_shortfall),
                "replacement_qty": replacement, "need_by": deadline, "status": status,
                "below_reorder_level": inv["stock"] < inv["reorder_level"],
            }
        needs = {k: v for k, v in per_sku.items() if v["replacement_qty"] > 0}
        for sku, v in needs.items():
            self.bus.send(self.name, "Supplier Agent", "REQUEST",
                          f"need {v['replacement_qty']} x {sku} by {v['need_by'] or 'ASAP'} "
                          f"(stock-out {v['stockout_date']}, status {v['status']})")
        facts = {"skus": {k: {"cover_days": v["days_of_cover"], "stockout": v["stockout_date"],
                              "replace_qty": v["replacement_qty"], "status": v["status"]} for k, v in per_sku.items()}}
        crit = [k for k, v in per_sku.items() if v["status"] == "critical"]
        fb = (f"{len(crit)} SKU(s) will stock out before resupply ({', '.join(crit) or 'none'}); "
              f"{len(needs)} SKU(s) need emergency replacement: "
              + ", ".join(f"{k} x{v['replacement_qty']}" for k, v in needs.items()) + ".")
        text, used = self.narrate(facts, fb)
        return AgentResult(self.name, {"per_sku": per_sku, "needs": needs}, text, 1.0 if per_sku else 0.5,
                           assumptions, used_llm=used)


# ======================================================================
# 4. Supplier
# ======================================================================
class SupplierAgent(BaseAgent):
    name = "Supplier Agent"
    role = "Finds and ranks alternative suppliers, builds multi-supplier sourcing plans."
    WEIGHTS = {  # (price, lead time, reliability)
        "cost": (0.70, 0.20, 0.10),
        "balanced": (0.35, 0.40, 0.25),
        "speed": (0.10, 0.70, 0.20),
    }

    def execute(self, ctx):
        needs = ctx["results"]["inventory"].data.get("needs", {})
        excluded = {e["supplier_id"] for e in ctx["events"] if e["type"] == "supplier_failure" and e.get("supplier_id")}
        assumptions, warnings, plans = [], [], {k: {} for k in self.WEIGHTS}
        no_alt = []
        for sku, need in needs.items():
            for strat in self.WEIGHTS:
                allocs, unfilled, notes = self.allocate(sku, need["replacement_qty"], strat, need["need_by"], excluded)
                plans[strat][sku] = allocs
                if strat == "balanced":
                    assumptions += notes
                    if unfilled > 0:
                        warnings.append(f"{sku}: only {need['replacement_qty'] - unfilled}/{need['replacement_qty']} "
                                        f"units can be sourced from known alternatives")
                    if not allocs:
                        no_alt.append(sku)
        assumptions = sorted(set(assumptions))
        conf = 1.0
        if any("estimated" in a or "unknown capacity" in a for a in assumptions):
            conf = 0.75
        if no_alt:
            conf = 0.4
            warnings.append(f"No alternative supplier found for: {', '.join(no_alt)}")
        for sku, allocs in plans["balanced"].items():
            if allocs:
                self.bus.send(self.name, "Logistics Agent", "INFO",
                              f"{sku}: sourcing from " + ", ".join(f"{a['supplier_id']} x{a['qty']} (arrives {a['arrival']})"
                                                                    for a in allocs))
        facts = {"excluded_suppliers": sorted(excluded),
                 "balanced_plan": {sku: [(a["supplier_id"], a["qty"], a["unit_price"], a["lead_days"]) for a in al]
                                   for sku, al in plans["balanced"].items()},
                 "no_alternative": no_alt}
        fb = "Balanced sourcing: " + ("; ".join(
            f"{sku} -> " + " + ".join(f"{a['name']} x{a['qty']} ({a['lead_days']}d)" for a in al)
            for sku, al in plans["balanced"].items() if al) or "none") + (
            f". No alternatives for {', '.join(no_alt)}." if no_alt else ".")
        text, used = self.narrate(facts, fb)
        return AgentResult(self.name, {"plans": plans, "excluded": sorted(excluded), "no_alternative": no_alt},
                           text, conf, assumptions, warnings, used_llm=used)

    def replan(self, ctx, skus, strategy):
        """Called by the Planner (feedback loop): re-source only `skus` with another strategy."""
        needs = ctx["results"]["inventory"].data.get("needs", {})
        excluded = {e["supplier_id"] for e in ctx["events"] if e["type"] == "supplier_failure" and e.get("supplier_id")}
        out = {}
        for sku in skus:
            if sku in needs:
                out[sku], _, _ = self.allocate(sku, needs[sku]["replacement_qty"], strategy, needs[sku]["need_by"], excluded)
        return out

    def allocate(self, sku, qty, strategy, need_by, excluded):
        notes = []
        cands = self.store.suppliers_for(sku, exclude=excluded)
        if not cands:
            return [], qty, notes
        known_prices = [c["unit_price_usd"] for c in cands if c["unit_price_usd"] is not None]
        est = statistics.median(known_prices) if known_prices else None
        for c in cands:
            c["price_estimated"] = c["unit_price_usd"] is None
            if c["price_estimated"]:
                if est is None:
                    c["unit_price_usd"] = 0.0
                    notes.append(f"{c['supplier_id']}/{sku}: price unknown and no benchmark, cost excluded")
                else:
                    c["unit_price_usd"] = est
                    notes.append(f"{c['supplier_id']}/{sku}: price estimated at peer median ${est:.2f} (unverified)")
            c["arrival"] = SIM_TODAY + dt.timedelta(days=int(c["lead_time_days"]))
            c["in_time"] = bool(need_by and c["arrival"] <= need_by)

        def norm(vals, v, invert=True):
            lo, hi = min(vals), max(vals)
            x = 0.5 if hi == lo else (v - lo) / (hi - lo)
            return 1 - x if invert else x

        wp, wl, wr = self.WEIGHTS[strategy]
        prices = [c["unit_price_usd"] for c in cands]
        leads = [c["lead_time_days"] for c in cands]
        rels = [c["reliability"] for c in cands]
        for c in cands:
            c["score"] = wp * norm(prices, c["unit_price_usd"]) + wl * norm(leads, c["lead_time_days"]) \
                + wr * norm(rels, c["reliability"], invert=False)
        if strategy == "cost":
            ranked = sorted(cands, key=lambda c: -c["score"])
        else:   # deadline-aware: suppliers that can arrive in time go first
            ranked = sorted(cands, key=lambda c: (not c["in_time"], -c["score"]))

        remaining, allocs = qty, []
        for c in ranked:
            if remaining <= 0:
                break
            cap = c["capacity_units"]
            if cap is None:
                notes.append(f"{c['supplier_id']}/{sku}: unknown capacity, assumed able to supply {remaining} units")
                cap = remaining
            take = int(min(remaining, cap))
            if take <= 0:
                continue
            allocs.append({
                "sku": sku, "supplier_id": c["supplier_id"], "name": c["name"], "qty": take,
                "unit_price": round(c["unit_price_usd"], 2), "price_estimated": c["price_estimated"],
                "lead_days": int(c["lead_time_days"]), "arrival": c["arrival"], "reliability": c["reliability"],
                "cost": round(take * c["unit_price_usd"], 2), "in_time": c["in_time"],
            })
            remaining -= take
        return allocs, max(0, remaining), notes


# ======================================================================
# 5. Logistics
# ======================================================================
class LogisticsAgent(BaseAgent):
    name = "Logistics Optimization Agent"
    role = "Finds alternative routes/modes for affected shipments and trades cost against time."

    def execute(self, ctx):
        impact = ctx["results"]["impact"].data
        events, assumptions, warnings = ctx["events"], [], []
        decisions = {"cost": {}, "balanced": {}, "speed": {}}
        options_by_ship = {}
        for a in impact["affected_shipments"]:
            ship = next(s for s in self.store.shipments if s["shipment_id"] == a["shipment_id"])
            opts = [{"action": "wait", "route": "Wait for disruption to clear", "eta": a["new_eta"],
                     "extra_cost": 0.0, "reliability": 0.6}]
            note = None
            if a["status"] == "Stuck at Port":
                note = "already at the affected port: cannot be rerouted"
            elif "supplier_failure" in a["event_types"]:
                note = "supplier failed: shipment cannot be rerouted, replacement sourcing needed"
            else:
                routes = self.store.routes_for(a["origin"], a["destination"])
                if not routes:
                    note = "no alternative route in the route table"
                    assumptions.append(f"{a['shipment_id']}: no routes known for {a['origin']}->{a['destination']}, must wait")
                for r in routes:
                    if any(T.route_blocked(e, r) for e in events):
                        continue
                    if r["mode"] == "Sea":   # the default sea lane is the one that is delayed
                        continue
                    opts.append({"action": "reroute", "route": f"{r['mode']} via {r['via']}",
                                 "eta": SIM_TODAY + dt.timedelta(days=int(r["transit_days"])),
                                 "extra_cost": round(T.freight_cost(ship, r), 2), "reliability": r["reliability"]})
                if len(opts) > 1:
                    assumptions.append(f"{a['shipment_id']}: reroute ETA assumes full transit time from today (conservative)")
            required_by = min((o["deadline"] for o in self.store.orders
                               if o["sku"] == a["sku"] and o["priority"] in ("High", "Medium")), default=None)
            options_by_ship[a["shipment_id"]] = {"options": opts, "note": note, "required_by": required_by,
                                                 "sku": a["sku"]}
            wait = opts[0]
            reroutes = opts[1:]
            decisions["cost"][a["shipment_id"]] = wait
            fastest = min(reroutes, key=lambda o: o["eta"]) if reroutes else wait
            decisions["speed"][a["shipment_id"]] = fastest if fastest["eta"] < wait["eta"] else wait
            ok = [o for o in reroutes if required_by and o["eta"] <= required_by and o["reliability"] >= 0.7]
            if ok and wait["eta"] > required_by:
                decisions["balanced"][a["shipment_id"]] = min(ok, key=lambda o: o["extra_cost"])
            elif reroutes and required_by and wait["eta"] > required_by and fastest["eta"] < wait["eta"]:
                decisions["balanced"][a["shipment_id"]] = fastest
                warnings.append(f"{a['shipment_id']}: even the fastest reroute misses {required_by}")
            else:
                decisions["balanced"][a["shipment_id"]] = wait
        rer = [(sid, d["route"], d["eta"], d["extra_cost"]) for sid, d in decisions["balanced"].items()
               if d["action"] == "reroute"]
        for sid, route, eta, cost in rer:
            self.bus.send(self.name, "Planner Agent", "INFO", f"{sid}: reroute {route}, ETA {eta}, +${cost:,.0f}")
        stuck = [sid for sid, v in options_by_ship.items() if v["note"]]
        facts = {"balanced_reroutes": rer, "shipments_with_no_option": stuck}
        fb = (f"{len(rer)} shipment(s) can be usefully rerouted: "
              + ("; ".join(f"{sid} via {route} (ETA {eta}, +${cost:,.0f})" for sid, route, eta, cost in rer) or "none")
              + f". {len(stuck)} shipment(s) have no reroute option and must wait.")
        text, used = self.narrate(facts, fb)
        return AgentResult(self.name, {"decisions": decisions, "options": options_by_ship}, text,
                           0.85, sorted(set(assumptions)), warnings, used_llm=used)

    def fallback(self, ctx, error):
        """Safe degraded mode: nobody is rerouted, everything waits."""
        impact = ctx["results"]["impact"].data
        dec = {k: {a["shipment_id"]: {"action": "wait", "route": "Wait (logistics tool unavailable)",
                                      "eta": a["new_eta"], "extra_cost": 0.0, "reliability": 0.6}
                   for a in impact["affected_shipments"]} for k in ("cost", "balanced", "speed")}
        return AgentResult(self.name, {"decisions": dec, "options": {}},
                           "Logistics tool unavailable: defaulting to 'wait' for every affected shipment.",
                           confidence=0.4, assumptions=["no reroutes evaluated (logistics degraded mode)"],
                           warnings=[f"Logistics Agent failed: {error}"], degraded=True)


# ======================================================================
# 6. Recovery Planner
# ======================================================================
class PlannerAgent(BaseAgent):
    name = "Recovery Planner Agent"
    role = "Combines all agent outputs into ranked recovery options and recommends one."
    APPROVAL_COST_LIMIT = 25_000
    LABELS = {
        "cost": ("Cost-Saver", "Cheapest sourcing, shipments wait out the disruption."),
        "balanced": ("Balanced", "Deadline-aware sourcing plus targeted reroutes where they save a deadline."),
        "speed": ("Speed-First", "Fastest suppliers and fastest available reroutes, accepting higher cost."),
    }
    SEVERITY_WEIGHTS = {  # (service, cost, risk)
        "critical": (0.60, 0.15, 0.25), "high": (0.50, 0.25, 0.25),
        "medium": (0.40, 0.40, 0.20), "low": (0.30, 0.50, 0.20),
    }

    def execute(self, ctx):
        R = ctx["results"]
        impact = R["impact"].data
        sup = R.get("supplier").data if R.get("supplier") else {}
        log = R.get("logistics").data if R.get("logistics") else {}
        sup_plans = sup.get("plans") or {}
        log_dec = log.get("decisions") or {}
        assumptions, warnings = [], []

        baseline = self._build("baseline", "Do Nothing (baseline)", "Wait for the disruption to clear.", {}, {}, ctx)
        options = [self._build(k, *self.LABELS[k], sup_plans.get(k, {}), log_dec.get(k, {}), ctx,
                               prune=(k == "balanced"))
                   for k in ("cost", "balanced", "speed")]

        # --- feedback loop: Planner asks Supplier Agent for a faster plan where High orders still fail ---
        best_prelim = max(options, key=lambda o: self._score(o, options, ctx))
        risky_skus = sorted({r["sku"] for r in best_prelim["orders"] if r["priority"] == "High" and not r["on_time"]})
        if risky_skus and R.get("supplier") and "supplier_agent" in ctx:
            self.bus.send(self.name, "Supplier Agent", "REQUEST",
                          f"Balanced plan still misses High-priority orders for {risky_skus}; re-source with 'speed'")
            fast = ctx["supplier_agent"].replan(ctx, risky_skus, "speed")
            hybrid_plan = {**sup_plans.get("balanced", {}), **{k: v for k, v in fast.items() if v}}
            hybrid = self._build("hybrid", "Balanced + Expedite",
                                 "Balanced plan, but fastest suppliers for SKUs where High-priority orders are at risk.",
                                 hybrid_plan, log_dec.get("balanced", {}), ctx, prune=True)
            self.bus.send("Supplier Agent", self.name, "REPLY",
                          f"expedited sourcing for {risky_skus}: High orders on time "
                          f"{hybrid['high_on_time']}/{hybrid['high_total']} (was {best_prelim['high_on_time']}/{best_prelim['high_total']})")
            if hybrid["service_score"] > best_prelim["service_score"]:
                options.append(hybrid)

        for o in options:
            o["score"] = round(self._score(o, options + [baseline], ctx), 3)
        options.sort(key=lambda o: -o["score"])
        rec = options[0]
        rec["recommended"] = True

        # --- escalation rules ---
        reasons = []
        if rec["high_on_time"] < rec["high_total"]:
            reasons.append(f"{rec['high_total'] - rec['high_on_time']} High-priority order(s) cannot be fully met "
                           "even with the best option - customer negotiation needed")
        if rec["total_cost"] > self.APPROVAL_COST_LIMIT:
            reasons.append(f"total cost ${rec['total_cost']:,.0f} exceeds the ${self.APPROVAL_COST_LIMIT:,} auto-approval limit")
        unresolved = []
        for sku, need in (R["inventory"].data.get("needs", {}) if R.get("inventory") else {}).items():
            covered = sum(a["qty"] for a in rec["supplier_plan"].get(sku, []))
            if covered < need["replacement_qty"]:
                inv = R["inventory"].data["per_sku"][sku]
                unresolved.append({"sku": sku, "short_qty": need["replacement_qty"] - covered,
                                   "stockout_date": inv["stockout_date"]})
                reasons.append(f"{sku}: {need['replacement_qty'] - covered} units cannot be sourced "
                               f"(stock-out {inv['stockout_date']}) - no available alternative")
        inv_needs = R["inventory"].data.get("needs") if R.get("inventory") else {}
        if inv_needs and rec["procurement_cost"] == 0 and rec["extra_freight"] == 0:
            reasons.append("no actionable alternative found - every option collapses to waiting")

        partial = [r for r in rec["orders"] if 0 < r["fill_rate"] < 1]
        for r in partial:
            warnings.append(f"{r['order_id']} ({r['customer_id']}) only {r['fill_rate']:.0%} fulfilled on time: "
                            "propose partial delivery / revised date to the customer")
        risks = ["Delayed original shipments will still arrive later and create surplus stock - "
                 "consider cancelling or redirecting them."]
        if any(a["price_estimated"] for al in rec["supplier_plan"].values() for a in al):
            risks.append("Some supplier prices are estimates and must be confirmed before ordering.")
        if any(d["action"] == "reroute" for d in rec["logistics_plan"].values()):
            risks.append("Reroute ETAs assume full transit time from today (conservative).")

        # overall confidence = weakest contributing agent
        confs = [r.confidence for r in R.values() if r is not None]
        conf = round(min(confs), 2) if confs else 0.0
        if any(r.degraded for r in R.values() if r is not None):
            reasons.append("one or more agents ran in degraded mode")
        for r in R.values():
            if r is not None:
                assumptions += r.assumptions
                warnings += r.warnings

        data = {"baseline": baseline, "options": options, "recommended_key": rec["key"],
                "escalation_reasons": reasons, "requires_approval": bool(reasons),
                "risks": risks, "unresolved": unresolved, "actions": self._actions(rec, unresolved)}
        facts = {"severity": ctx.get("severity"), "recommended": rec["label"],
                 "metrics": {k: rec[k] for k in ("total_cost", "service_score", "on_time_orders", "total_orders",
                                                 "high_on_time", "high_total")},
                 "baseline_late_orders": baseline["total_orders"] - baseline["on_time_orders"],
                 "escalation": reasons}
        fb = (f"Recommend '{rec['label']}': {rec['on_time_orders']}/{rec['total_orders']} orders on time "
              f"(baseline {baseline['on_time_orders']}/{baseline['total_orders']}), total cost ${rec['total_cost']:,.0f}, "
              f"risk {rec['risk']:.0%}." + (" Human approval required." if reasons else ""))
        text, used = self.narrate(facts, fb)
        self.bus.send(self.name, "Coordinator", "RESULT", fb)
        return AgentResult(self.name, data, text, conf, sorted(set(assumptions)), sorted(set(warnings)), used_llm=used)

    # ---- helpers ----
    def _prune_reroutes(self, sup_plan, log_plan, ctx):
        """Drop reroutes that do not improve order service once sourcing is in place (saves freight)."""
        plan = dict(log_plan)
        current = self._build("t", "t", "", sup_plan, plan, ctx)["service_score"]
        for sid in sorted([s for s, d in plan.items() if d["action"] == "reroute"],
                          key=lambda s: -plan[s]["extra_cost"]):
            trial = {k: v for k, v in plan.items() if k != sid}
            if self._build("t", "t", "", sup_plan, trial, ctx)["service_score"] >= current - 1e-9:
                plan = trial
        return plan

    def _build(self, key, label, desc, sup_plan, log_plan, ctx, prune=False):
        if prune and log_plan:
            log_plan = self._prune_reroutes(sup_plan, log_plan, ctx)
        store = self.store
        impact = ctx["results"]["impact"].data
        affected = {a["shipment_id"]: a for a in impact["affected_shipments"]}
        arrivals, rel, extra_freight = defaultdict(list), [], 0.0
        for s in store.shipments:
            eta = affected[s["shipment_id"]]["new_eta"] if s["shipment_id"] in affected else s["eta"]
            d = log_plan.get(s["shipment_id"])
            rel_s = 0.6 if s["shipment_id"] in affected else None
            if d and d["action"] == "reroute":
                eta, extra_freight, rel_s = d["eta"], extra_freight + d["extra_cost"], d["reliability"]
            arrivals[s["sku"]].append((eta, s["qty"]))
            if rel_s is not None:
                rel.append((s["value_usd"], rel_s))
        procurement, last_arrival = 0.0, SIM_TODAY
        for sku, allocs in sup_plan.items():
            for a in allocs:
                arrivals[sku].append((a["arrival"], a["qty"]))
                procurement += a["cost"]
                rel.append((max(a["cost"], 1.0), a["reliability"]))
                last_arrival = max(last_arrival, a["arrival"])
        stock = {i["sku"]: i["stock"] for i in store.inventory}
        sim = T.simulate_fulfillment(store.orders, stock, arrivals)
        m = T.service_metrics(sim)
        return {
            "key": key, "label": label, "description": desc, "recommended": False,
            "procurement_cost": round(procurement, 2), "extra_freight": round(extra_freight, 2),
            "late_penalty": m["late_penalty"],
            "total_cost": round(procurement + extra_freight + m["late_penalty"], 2),
            "service_score": m["service_score"], "on_time_orders": m["on_time_orders"], "total_orders": m["total_orders"],
            "high_on_time": m["high_on_time"], "high_total": m["high_total"],
            "risk": T.risk_score(rel) if rel else 0.5, "recovery_date": last_arrival,
            "orders": sim, "supplier_plan": sup_plan,
            "logistics_plan": {k: v for k, v in log_plan.items() if v["action"] == "reroute"},
        }

    def _score(self, o, pool, ctx):
        ws, wc, wr = self.SEVERITY_WEIGHTS.get(ctx.get("severity", "medium"), self.SEVERITY_WEIGHTS["medium"])
        max_cost = max((p["total_cost"] for p in pool), default=1) or 1
        return ws * o["service_score"] + wc * (1 - o["total_cost"] / max_cost) + wr * (1 - o["risk"])

    @staticmethod
    def _actions(rec, unresolved=()):
        acts = []
        for u in unresolved:
            acts.append(f"URGENT manual sourcing: {u['sku']} short by {u['short_qty']} units, stock-out on "
                        f"{u['stockout_date']}. No known alternative supplier - try spot market, ration stock "
                        f"to priority customers, or negotiate with the failed supplier's receiver.")
        for sku, allocs in rec["supplier_plan"].items():
            for a in allocs:
                est = " (price estimated - confirm first)" if a["price_estimated"] else ""
                acts.append(f"Place emergency order: {a['qty']} x {sku} with {a['name']} ({a['supplier_id']}) at "
                            f"${a['unit_price']}/unit = ${a['cost']:,.0f}, arrives {a['arrival']}{est}.")
        for sid, d in rec["logistics_plan"].items():
            acts.append(f"Reroute shipment {sid}: {d['route']}, new ETA {d['eta']}, extra freight ${d['extra_cost']:,.0f}.")
        for r in rec["orders"]:
            if not r["on_time"]:
                acts.append(f"Contact {r['customer_id']} about {r['order_id']} ({r['priority']}): "
                            f"{r['fill_rate']:.0%} can ship by {r['deadline']}; agree partial delivery or new date.")
        return acts or ["No action required."]
