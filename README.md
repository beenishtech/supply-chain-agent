# Supply Chain Crisis Management - Multi-Agent System

Ek disruption signal (news / supplier email / weather alert) aata hai, 6 agents milkar impact analyze karte hain,
alternatives dhoondhte hain, aur ek recovery plan recommend karte hain. Critical decisions par human approval maanga jata hai.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
# .env mein GEMINI_API_KEY paste karein (key: aistudio.google.com)
python generate_data.py
python main.py --list
python main.py --scenario port_strike
```

Key na ho (ya `--no-llm`) to system **offline mode** mein chalta hai: saare numbers aur faislay tools se aate hain,
LLM sirf reasoning text aur free-text signal parsing ke liye hai. Groq use karna ho to `.env` mein `GROQ_API_KEY` daalein.

## Agents

| Agent | Kaam |
|---|---|
| Disruption Monitor | Raw text signal se event type, location/supplier, duration, severity nikalta hai (rules, phir LLM agar confidence kam ho) |
| Impact Analysis | Kaun se shipments/orders/customers affected, value at risk, kaun se orders late honge |
| Inventory | Days of cover, stock-out date, kitna replacement chahiye per SKU |
| Supplier | Alternative suppliers rank karta hai (price / lead time / reliability), multi-supplier sourcing plan |
| Logistics Optimization | Affected shipments ke liye alternate route/mode, cost vs time |
| Recovery Planner | Sab ka output jod kar Cost-Saver / Balanced / Speed-First options, scoring, recommendation, escalation |
| Coordinator | Order, parallel execution, retry, fallback, mid-run replanning, human approval |

## Flow

```
Signals -> Monitor -> Impact -> Inventory -> [Supplier || Logistics] (parallel) -> Planner -> Human approval
                                                  ^                                     |
                                                  +---- Planner asks Supplier to replan -+
```

## Demo scenarios

| Scenario | Kya dikhata hai |
|---|---|
| `port_strike` | Karachi port strike, stock-out, multi-supplier sourcing, partial fulfilment (ORD-3 93%) |
| `supplier_bankruptcy` | Main supplier band, alternatives rank hote hain, high cost -> approval |
| `mixed_crisis` | Cyclone (duration unknown -> assumption) + supplier delay + **mid-run capacity cut -> replan** |
| `no_alternative` | Sole-source supplier fail, koi alternative nahi -> human escalation |

## Unexpected scenarios / incomplete info

- Duration missing -> default assumption, confidence 60%, report mein "Assumptions" mein likha jata hai
- Supplier price missing -> peer median se estimate, "confirm first" flag
- Supplier capacity unknown -> assumption flag
- Supplier record hi missing (SUP-5) -> warning, reliability unknown
- Agent crash -> 1 retry, phir degraded mode (`--inject-failure logistics` / `logistics:hard`)
- LLM down / rate limited -> circuit breaker, offline reasoning
- Naya disruption beech mein -> sirf zaroori agents dobara chalte hain
- Koi alternative nahi -> "URGENT manual sourcing" action + human escalation

## Useful commands

```bash
python main.py --scenario all --quiet
python main.py --scenario mixed_crisis --approve ask       # y/n approval prompt
python main.py --scenario port_strike --approve reject     # rejected -> next best option
python main.py --scenario port_strike --inject-failure logistics:hard
```

Outputs `outputs/` folder mein: `<scenario>_report.md` (readable) aur `<scenario>_run.json` (UI ke liye, saare agent messages ke saath).

## Simplifications (honest list)

- Data simulated hai (CSV). Weather tool mock hai; real API (OpenWeatherMap) `tools.weather_check` mein plug ho sakti hai.
- Orders ko committed demand maana gaya hai; `daily_demand` alag se inventory view mein use hota hai.
- Freight cost = shipment value x 4% x route multiplier; late penalty = 8% of order value. Dono assumptions hain.
- Reroute ETA poora transit time maanta hai (conservative).
- Framework: custom orchestrator (koi CrewAI/AutoGen dependency nahi). Agent classes aur tools framework-agnostic hain,
  CrewAI mein port karna aasan hai.
