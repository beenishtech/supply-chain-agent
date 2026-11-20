# Supply Chain Crisis Management: Multi-Agent System

An AI multi-agent system for a logistics company. It receives a disruption signal (news, supplier email,
weather alert), analyzes the impact, finds alternative suppliers and routes, and recommends a recovery plan.
Expensive or risky decisions are escalated to a human for approval.

## Architecture

```
Signals -> Monitor -> Impact -> Inventory -> [Supplier || Logistics] (parallel) -> Planner -> Human approval
                                                  ^                                     |
                                                  +---- Planner asks Supplier to replan -+
```

The **Coordinator** controls the flow: dependency order, parallel execution, retries, degraded mode,
mid-run replanning, and the human approval gate. All agent-to-agent messages are logged on a message bus.

## Agents and tools

| Agent | Responsibility | Tool (what it computes) |
|---|---|---|
| Disruption Monitor | Turns a raw text signal into a structured event (type, location/supplier, duration, severity) | Rule parser + LLM fallback for unclear signals |
| Impact Analysis | Finds affected shipments, late orders, customers and value at risk | Shipment and order simulator |
| Inventory | Days of cover, stock-out date, replacement quantity per SKU | Inventory calculator |
| Supplier | Ranks alternative suppliers (price, lead time, reliability) and builds sourcing plans | Supplier database and scoring |
| Logistics Optimization | Alternative routes and modes, cost vs time | Route table and freight cost model |
| Recovery Planner | Builds Cost-Saver / Balanced / Speed-First options, scores them, recommends one, decides escalation | Option scoring and escalation rules |
| Coordinator | Orchestration, retries, fallbacks, replanning, approval | Python orchestrator |

## Two engines

| Engine | What it is |
|---|---|
| **Native** | Built-in Python orchestrator (`orchestrator.py`). Works offline, with or without an LLM. |
| **CrewAI** | Every specialist is a CrewAI `Agent` with its own `Task` and `Tool`. Supplier and Logistics run as async tasks, and a Crisis Response Coordinator agent merges their output for the Planner. Needs an LLM key. |

In both engines the numbers (cost, stock-out dates, order fulfillment) come from deterministic Python tools,
because LLMs make arithmetic mistakes. The LLM reasons over the tool output and writes the assessment.
If CrewAI or the LLM fails at any step, that step falls back to the native agent so a run never dies.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set ONE key (if both are set, Gemini is used first):

```
GROQ_API_KEY=your_key          # free at console.groq.com
# GEMINI_API_KEY=your_key      # free at aistudio.google.com
```

Then generate the data and run:

```powershell
python generate_data.py
streamlit run ui.py                                     # dashboard (choose Engine: Native or CrewAI)
python main.py --scenario port_strike                   # terminal, native engine
python main.py --scenario port_strike --engine crewai   # terminal, CrewAI engine
```

Without a key (or with `--no-llm`) the Native engine runs in offline mode with rule-based text.

## Demo scenarios

| Scenario | What it shows |
|---|---|
| `port_strike` | Karachi port strike, stock-out risk, multi-supplier sourcing, partial fulfilment |
| `supplier_bankruptcy` | Main supplier fails, alternatives are ranked, high cost triggers approval |
| `mixed_crisis` | Cyclone with unknown duration (assumption) + supplier delay + a **mid-run capacity cut that forces a replan** |
| `no_alternative` | Sole-source supplier fails, no alternative exists, human escalation |

## Handling unexpected situations and incomplete data

- Missing duration: default assumption, confidence lowered, listed under "Assumptions"
- Missing supplier price: estimated from peer median and flagged "confirm first"
- Unknown supplier capacity or missing supplier record: assumption or warning
- Agent crash: one retry, then degraded mode (`--inject-failure logistics` or `logistics:hard`)
- LLM down or rate limited: circuit breaker, falls back to rule-based reasoning
- New disruption mid-run: only the affected agents run again
- No alternative available: "URGENT manual sourcing" action and human escalation
- Human-in-the-loop: approve or reject buttons for expensive or risky plans

## Useful commands

```powershell
python main.py --scenario all --quiet
python main.py --scenario mixed_crisis --approve ask       # y/n approval prompt
python main.py --scenario port_strike --approve reject     # rejected -> next best option
python main.py --scenario port_strike --inject-failure logistics:hard
```

Reports are written to `outputs/` (`<scenario>_report.md` and `<scenario>_run.json`) by the terminal commands.

## Project structure

```
agents.py          the six specialist agents (calculation + reasoning)
tools.py           data access and calculation helpers
orchestrator.py    Coordinator: order, parallelism, retry, replan, approval
crew_engine.py     CrewAI engine (Agents, Tasks, Tools, async tasks)
llm.py             LLM client (Gemini or Groq) with rate limiting and offline fallback
scenarios.py       demo scenarios
generate_data.py   creates the mock CSV data in data/
report.py          markdown and JSON report builder
main.py            command-line interface
ui.py              Streamlit dashboard
```

## Limitations (honest list)

- Data is simulated (CSV). The weather tool is a mock; a real API can be plugged into `tools.weather_check`.
- Freight cost (4% of shipment value x route multiplier) and late penalty (8% of order value) are assumptions.
- Reroute ETAs assume the full transit time from today (conservative).
- The CrewAI engine makes roughly 15-20 LLM calls per run, so free-tier rate limits can apply.