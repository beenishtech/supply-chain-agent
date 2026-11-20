"""CrewAI engine for the Supply Chain Crisis system.

How it works
------------
* Every specialist is a real CrewAI ``Agent`` (role / goal / backstory / tool / LLM) with its own ``Task``.
* The agent's CrewAI **tool** runs the tested calculation code (tools.py + agents.py). Numbers always come
  from deterministic code, because an LLM is unreliable at arithmetic. The LLM decides to call the tool,
  reads the findings and writes the assessment, and it also sees what the teammates found before it.
* Supplier and Logistics run as CrewAI **async tasks** (parallel), then a Crisis Coordinator agent merges them.
* The Python Coordinator (orchestrator.py) still owns retries, degraded mode, mid-run replanning and human approval.
* If CrewAI or its LLM fails at any point, that step falls back to the native engine, so a run never dies.

Run:  python main.py --engine crewai --scenario port_strike
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from concurrent.futures import ThreadPoolExecutor

from crewai import LLM as CrewLLM
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from llm import LLM
from orchestrator import Coordinator

STYLE = ("Write 2-3 plain sentences (no markdown, no bullet points). Use ONLY facts returned by your tool "
         "or reported by teammates. Never invent numbers, names or dates.")

SPECS = {
    "monitor": dict(
        role="Disruption Monitor",
        goal="Detect supply chain disruptions from raw signals and classify type, duration and severity.",
        backstory="You watch news feeds, supplier emails and weather alerts for a logistics company.",
        tool="Run Disruption Monitor",
        doc="Parses the incoming raw signals into structured disruption events with severity and confidence.",
        task="Run your tool to parse the incoming signals.\nSignals:\n{signals}\nThen summarise the disruption(s) detected.",
    ),
    "impact": dict(
        role="Impact Analyst",
        goal="Quantify which shipments, orders and customers are hit and how much value is at risk.",
        backstory="You are a supply chain risk analyst who turns an event into business exposure.",
        tool="Run Impact Analysis",
        doc="Finds affected shipments and late orders for the detected disruptions and computes value at risk.",
        task="Run your tool to analyse the impact of the detected disruption(s), then summarise the exposure.",
    ),
    "inventory": dict(
        role="Inventory Controller",
        goal="Work out days of cover, stock-out dates and the replacement quantity needed per SKU.",
        backstory="You manage stock levels and know exactly when each SKU runs out.",
        tool="Run Inventory Check",
        doc="Computes days of cover, stock-out dates and replacement quantities per SKU.",
        task="Run your tool to check stock positions, then summarise which SKUs are critical and how much is needed.",
    ),
    "supplier": dict(
        role="Supplier Sourcing Specialist",
        goal="Find and rank alternative suppliers and build a sourcing plan that covers the shortages.",
        backstory="You negotiate with suppliers and balance price, lead time and reliability.",
        tool="Run Supplier Sourcing",
        doc="Ranks alternative suppliers and builds cost, balanced and speed sourcing plans.",
        task="Run your tool to build the sourcing plans, then summarise the recommended balanced sourcing and any gap.",
    ),
    "logistics": dict(
        role="Logistics Optimizer",
        goal="Find alternative routes and modes for affected shipments and weigh cost against time.",
        backstory="You plan freight across sea, air and road and know which routes are blocked.",
        tool="Run Logistics Optimization",
        doc="Evaluates reroute options (mode, route, cost, ETA) for every affected shipment.",
        task="Run your tool to evaluate reroutes, then summarise which shipments can be rerouted and which must wait.",
    ),
    "planner": dict(
        role="Recovery Planner",
        goal="Combine all findings into ranked recovery options and recommend the best one.",
        backstory="You are the decision maker who balances service level, cost and risk, and knows when to escalate.",
        tool="Run Recovery Planner",
        doc="Builds Cost-Saver, Balanced and Speed-First recovery options, scores them and recommends one.",
        task="Run your tool to build and score the recovery options, then state which option you recommend, "
             "why, and whether human approval is needed.",
    ),
}
ORDER = ["monitor", "impact", "inventory", "supplier", "logistics", "planner"]


class CrewCoordinator(Coordinator):
    def __init__(self, llm: LLM, **kwargs):
        self.max_rpm = int(os.getenv("CREW_MAX_RPM", "5"))
        # native agents use rule-based text; CrewAI supplies the LLM reasoning (avoids double LLM calls)
        super().__init__(LLM(enabled=False), **kwargs)
        self.model_name = None
        self.crew_llm = self._build_llm(llm)
        self.crew_ok = self.crew_llm is not None
        self.engine_label = (f"CrewAI engine ({self.model_name})" if self.crew_ok
                             else "CrewAI engine unavailable (no LLM key) - native engine used")
        self._fresh = {}
        self._tools = {k: self._make_tool(k) for k in SPECS}

    # ------------------------------------------------------------------ setup helpers
    def _build_llm(self, llm: LLM):
        if llm.provider == "gemini":
            self.model_name = os.getenv("CREW_MODEL", f"gemini/{llm.gemini_model}")
            key = llm.gemini_key
        elif llm.provider == "groq":
            self.model_name = os.getenv("CREW_MODEL", f"groq/{llm.groq_model}")   # follows GROQ_MODEL
            key = llm.groq_key
        else:
            return None
        return CrewLLM(model=self.model_name, api_key=key, temperature=0.2, max_tokens=1500, num_retries=3)

    def _make_tool(self, key):
        runner, spec = self, SPECS[key]

        def run_agent_tool(request: str) -> str:
            """placeholder docstring (replaced below)"""
            res = Coordinator._run_agent(runner, key)      # native logic incl. retry + degraded mode
            runner._fresh[key] = res
            return runner._facts(key, res)

        run_agent_tool.__doc__ = (f"{spec['doc']}\n\nArgs:\n    request: a short reason for running this tool "
                                  "(any text).\n\nReturns: findings as JSON.")
        return tool(spec["tool"])(run_agent_tool)

    def _facts(self, key, res):
        d = {"agent": res.agent, "findings": res.summary, "confidence": res.confidence,
             "assumptions": res.assumptions[:6], "warnings": res.warnings[:6], "degraded": res.degraded}
        if key == "monitor" and res.data:
            d["events"] = [{k: e.get(k) for k in ("id", "type", "location", "supplier_id", "duration_days",
                                                   "severity", "confidence")} for e in res.data.get("events", [])]
        if key == "planner" and res.data:
            d["options"] = [{"option": o["label"], "orders_on_time": f"{o['on_time_orders']}/{o['total_orders']}",
                             "total_cost_usd": o["total_cost"], "risk": o["risk"], "score": o.get("score")}
                            for o in res.data["options"]]
            d["recommended"] = next(o["label"] for o in res.data["options"] if o["key"] == res.data["recommended_key"])
            d["escalation_reasons"] = res.data["escalation_reasons"]
        return json.dumps(d, default=str)

    def _team_context(self):
        lines = [f"- {r.agent}: {r.summary[:300]}" for k in ORDER
                 if (r := self.ctx["results"].get(k)) is not None]
        return "Findings from teammates so far:\n" + "\n".join(lines) if lines else ""

    def _build(self, key, async_exec=False):
        spec = SPECS[key]
        agent = Agent(role=spec["role"], goal=spec["goal"], backstory=spec["backstory"],
                      tools=[self._tools[key]], llm=self.crew_llm, verbose=False,
                      allow_delegation=False, max_iter=4, max_rpm=self.max_rpm)
        signals = "\n".join(f"- [{s.get('source', '?')}] {s['text']}" for s in self.ctx.get("signals", []))
        desc = spec["task"].format(signals=signals) + "\n\n" + self._team_context() + "\n\n" + STYLE
        task = Task(description=desc, expected_output="2-3 plain sentences summarising the findings.",
                    agent=agent, async_execution=async_exec)
        return agent, task

    @staticmethod
    def _clean(text):
        text = (text or "").strip()
        bad = ("stopped due to", "iteration limit", "i now can give a great answer")
        return None if (not text or any(b in text.lower() for b in bad)) else text

    def _apply_text(self, key, res, text):
        text = self._clean(text)
        if text and res is not None:
            res.summary, res.used_llm = text, True

    # ------------------------------------------------------------------ single-agent crew
    def _run_agent(self, key):
        if not self.crew_ok:
            return super()._run_agent(key)
        self._fresh.pop(key, None)
        text = None
        try:
            agent, task = self._build(key)
            out = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False, cache=False,
                       max_rpm=self.max_rpm).kickoff()
            text = getattr(out, "raw", None) or str(out)
        except Exception as e:
            self.say(f"CrewAI step '{key}' failed ({str(e)[:100]}) - using native agent.", "FALLBACK")
        res = self._fresh.get(key)
        if res is None:      # the LLM never called the tool (or CrewAI failed) -> run the native agent
            res = Coordinator._run_agent(self, key)
        self._apply_text(key, res, text)
        return res

    # ------------------------------------------------------------------ parallel stage (async tasks)
    def _analysis_phase(self, full=True):
        if full:
            self._run_agent("impact")
            self._escalate_severity()
            self._run_agent("inventory")
        if not self.crew_ok:
            return self._parallel_native()
        self.say("Dispatching Supplier and Logistics agents in parallel (CrewAI async tasks).", "DECISION")
        for k in ("supplier", "logistics"):
            self._fresh.pop(k, None)
        texts = {}
        try:
            sup_agent, sup_task = self._build("supplier", async_exec=True)
            log_agent, log_task = self._build("logistics", async_exec=True)
            boss = Agent(role="Crisis Response Coordinator",
                         goal="Merge the Supplier and Logistics findings into one handoff for the Recovery Planner.",
                         backstory="You run the crisis room and make sure the planner gets one clear picture.",
                         llm=self.crew_llm, verbose=False, allow_delegation=False, max_rpm=self.max_rpm)
            merge = Task(description="Using the Supplier and Logistics reports, write a 2-sentence handoff for the "
                                     "Recovery Planner: which sourcing and routing options exist and the main "
                                     "trade-off. " + STYLE,
                         expected_output="Two plain sentences.", agent=boss, context=[sup_task, log_task])
            out = Crew(agents=[sup_agent, log_agent, boss], tasks=[sup_task, log_task, merge],
                       process=Process.sequential, verbose=False, cache=False, max_rpm=self.max_rpm).kickoff()
            handoff = self._clean(getattr(out, "raw", None) or str(out))
            if handoff:
                self.bus.send("Crisis Response Coordinator (CrewAI)", "Recovery Planner Agent", "INFO", handoff)
            for k, t in (("supplier", sup_task), ("logistics", log_task)):
                texts[k] = getattr(getattr(t, "output", None), "raw", None)
        except Exception as e:
            self.say(f"CrewAI parallel stage failed ({str(e)[:100]}) - using native agents.", "FALLBACK")
        missing = [k for k in ("supplier", "logistics") if k not in self._fresh]
        if missing:
            self._parallel_native(missing)
        for k in ("supplier", "logistics"):
            self._apply_text(k, self._fresh.get(k) or self.ctx["results"].get(k), texts.get(k))

    def _parallel_native(self, keys=("supplier", "logistics")):
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(Coordinator._run_agent, self, k) for k in keys]
            for f in futs:
                f.result()

    # ------------------------------------------------------------------ misc
    def run(self, scenario):
        if not self.crew_ok:
            self.say("CrewAI engine needs an LLM key (GEMINI_API_KEY or GROQ_API_KEY) - running native engine.",
                     "FALLBACK")
        else:
            self.say(f"Engine: CrewAI | model {self.model_name} | max {self.max_rpm} requests/min per agent", "START")
        return super().run(scenario)

    def _package(self, scenario):
        pkg = super()._package(scenario)
        pkg["llm_status"] = self.engine_label
        return pkg