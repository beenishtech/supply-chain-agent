"""Coordinator: runs the agents in dependency order, in parallel where possible, and handles
retries, degraded mode, mid-run disruptions (replanning), and human approval."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import tools as T
from agents import (ImpactAgent, InventoryAgent, LogisticsAgent, MessageBus, MonitorAgent,
                    PlannerAgent, SupplierAgent)

SEVERITY_ORDER = ["low", "medium", "high", "critical"]


class Coordinator:
    def __init__(self, llm, printer=None, approval="pending", inject_failure=None, data_dir=None):
        self.llm = llm
        self.bus = MessageBus(printer)
        self.approval = approval            # pending | approve | ask | reject | callable(plan)->bool
        self.inject_failure = inject_failure
        self.store = T.DataStore(data_dir) if data_dir else T.DataStore()
        b = (self.store, llm, self.bus)
        self.agents = {
            "monitor": MonitorAgent(*b), "impact": ImpactAgent(*b), "inventory": InventoryAgent(*b),
            "supplier": SupplierAgent(*b), "logistics": LogisticsAgent(*b), "planner": PlannerAgent(*b),
        }
        if inject_failure:
            name, _, mode = inject_failure.partition(":")
            if name in self.agents:
                self.agents[name].fail_mode = "always" if mode == "hard" else "once"
        self.ctx = {"results": {}, "events": [], "severity": "low", "supplier_agent": self.agents["supplier"]}
        self.timeline = []

    # ------------------------------------------------------------------
    def say(self, msg, kind="INFO"):
        self.bus.send("Coordinator", "all", kind, msg)

    def _run_agent(self, key):
        """Run with one retry, then fallback to degraded mode."""
        agent = self.agents[key]
        for attempt in (1, 2):
            try:
                res = agent.run(self.ctx)
                self.ctx["results"][key] = res
                if attempt == 2:
                    self.say(f"{agent.name} recovered on retry.", "RECOVERY")
                self.timeline.append((agent.name, res.seconds, "ok", res.used_llm))
                return res
            except Exception as e:
                self.say(f"{agent.name} failed (attempt {attempt}): {e}", "ERROR")
        res = agent.fallback(self.ctx, "repeated failure")
        self.ctx["results"][key] = res
        self.say(f"{agent.name} unavailable -> degraded mode, confidence lowered.", "FALLBACK")
        self.timeline.append((agent.name, 0, "degraded", False))
        return res

    def _ingest(self, signals):
        self.ctx["signals"] = signals
        res = self._run_agent("monitor")
        new = res.data.get("events", [])
        self.ctx["events"] += new
        for ev in new:
            if ev["type"] == "supplier_delay":
                self.store.delay_supplier(ev["supplier_id"], ev["duration_days"])
            elif ev["type"] == "capacity_cut":
                self.store.cut_capacity(ev["supplier_id"], ev["new_capacity"])
        return new

    def _escalate_severity(self):
        imp = self.ctx["results"]["impact"].data
        sev = max((e["severity"] for e in self.ctx["events"]), key=SEVERITY_ORDER.index, default="low")
        if imp["value_at_risk_usd"] > 100_000 or len(imp["high_priority_late"]) >= 2:
            new = "critical"
        else:
            new = sev
        if new != self.ctx["severity"]:
            self.say(f"Severity set to {new.upper()} (value at risk ${imp['value_at_risk_usd']:,.0f}, "
                     f"{len(imp['high_priority_late'])} high-priority orders late).", "DECISION")
        self.ctx["severity"] = new

    def _analysis_phase(self, full=True):
        if full:
            self._run_agent("impact")
            self._escalate_severity()
            self._run_agent("inventory")
        self.say("Dispatching Supplier and Logistics agents in parallel.", "DECISION")
        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(self._run_agent, "supplier")
            f2 = ex.submit(self._run_agent, "logistics")
            f1.result(); f2.result()

    # ------------------------------------------------------------------
    def run(self, scenario):
        self.say(f"Scenario: {scenario['title']}", "START")
        self._ingest(scenario["signals"])
        if not self.ctx["events"]:
            self.say("No actionable disruption detected - nothing to do.", "DONE")
            return self._package(scenario)
        self.ctx["severity"] = max((e["severity"] for e in self.ctx["events"]), key=SEVERITY_ORDER.index)
        self._analysis_phase(full=True)

        # --- a new disruption arrives while we are still planning ---
        mid = scenario.get("mid_run_signals")
        if mid:
            self.say("NEW SIGNAL received mid-run - re-evaluating.", "ALERT")
            new = self._ingest(mid)
            if any(e["type"] in ("port_strike", "weather", "supplier_failure", "supplier_delay") for e in new):
                self.say("New event changes shipments -> full replan.", "REPLAN")
                self._analysis_phase(full=True)
            elif new:
                self.say("New event only changes supplier constraints -> replanning sourcing.", "REPLAN")
                self._run_agent("supplier")

        plan = self._run_agent("planner")
        self._approval_gate(plan)
        self.say("Run complete.", "DONE")
        return self._package(scenario)

    # ------------------------------------------------------------------
    def _approval_gate(self, plan):
        d = plan.data
        if not d:
            return
        rec = next(o for o in d["options"] if o["key"] == d["recommended_key"])
        self.ctx["approval"] = {"required": d["requires_approval"], "decision": "not required",
                                "chosen": rec["key"], "note": ""}
        if not d["requires_approval"]:
            self.say(f"Auto-approved: {rec['label']} (no escalation triggers).", "DECISION")
            self.ctx["approval"]["decision"] = "auto-approved"
            return
        self.say("HUMAN APPROVAL REQUIRED: " + " | ".join(d["escalation_reasons"]), "ESCALATE")
        if callable(self.approval):
            ok = self.approval(plan)
        elif self.approval == "reject":
            ok = False
        elif self.approval == "pending":
            ok = None
        elif self.approval == "ask":
            ans = input(f"\nApprove '{rec['label']}' (cost ${rec['total_cost']:,.0f})? [y/n] ").strip().lower()
            ok = ans.startswith("y")
        else:   # "approve"
            ok = True
        if ok is None:
            self.ctx["approval"]["decision"] = "PENDING human approval"
            self.say(f"Plan '{rec['label']}' prepared and waiting for a human decision.", "ESCALATE")
            return
        if ok:
            self.ctx["approval"]["decision"] = "approved by human"
            self.say(f"Plan '{rec['label']}' approved.", "DECISION")
        else:
            alt = next((o for o in d["options"] if o["key"] != rec["key"]), None)
            self.ctx["approval"]["decision"] = "rejected"
            if alt:
                self.ctx["approval"]["chosen"] = alt["key"]
                self.ctx["approval"]["note"] = f"Rejected; fallback to next best option '{alt['label']}'."
                self.say(f"Plan rejected -> next best option: '{alt['label']}' queued for review.", "DECISION")

    def _package(self, scenario):
        return {"scenario": scenario, "events": self.ctx["events"], "severity": self.ctx["severity"],
                "results": self.ctx["results"], "approval": self.ctx.get("approval"),
                "messages": self.bus.messages, "timeline": self.timeline, "llm_status": self.llm.status()}
