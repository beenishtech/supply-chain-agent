"""Streamlit dashboard for the Supply Chain Crisis multi-agent system.
Run:  streamlit run ui.py
"""
import html

import pandas as pd
import streamlit as st

from llm import LLM
from orchestrator import Coordinator
from report import build_markdown
from scenarios import SCENARIOS

st.set_page_config(page_title="Supply Chain Crisis Command Center", page_icon="🚚", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 100%); color:#fff; padding:22px 28px;
       border-radius:14px; margin-bottom:18px;}
.hero h1 {margin:0; font-size:1.7rem; color:#fff;} .hero p {margin:4px 0 0; color:#cbd5e1;}
.kpi {background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 16px;}
.kpi .l {font-size:.74rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em;}
.kpi .v {font-size:1.55rem; font-weight:700; color:#0f172a; line-height:1.25;}
.kpi .s {font-size:.78rem; color:#64748b;}
.badge {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.72rem; font-weight:600; color:#fff;}
.msg {border-left:4px solid #94a3b8; background:#f8fafc; padding:7px 12px; margin:5px 0; border-radius:6px;
      font-size:.86rem; color:#0f172a;}
.msg .meta {color:#64748b; font-size:.74rem;}
.sev-low{background:#16a34a} .sev-medium{background:#d97706} .sev-high{background:#ea580c} .sev-critical{background:#dc2626}
</style>
""", unsafe_allow_html=True)

KIND_COLOR = {"EVENT": "#0d9488", "RESULT": "#16a34a", "DECISION": "#2563eb", "REQUEST": "#64748b",
              "REPLY": "#64748b", "INFO": "#64748b", "ERROR": "#dc2626", "FALLBACK": "#dc2626",
              "RECOVERY": "#16a34a", "ESCALATE": "#d97706", "ALERT": "#7c3aed", "REPLAN": "#7c3aed",
              "START": "#0f172a", "DONE": "#0f172a"}
AGENT_ORDER = ["monitor", "impact", "inventory", "supplier", "logistics", "planner"]


def money(x):
    return f"${x:,.0f}"


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def df(rows, cols=None):
    d = pd.DataFrame(rows)
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    return d.astype(str) if not d.empty else d


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Input", ["Demo scenario", "Custom signal"], horizontal=True)
    if mode == "Demo scenario":
        key = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k]["title"])
        st.caption(SCENARIOS[key]["description"])
        scenario = SCENARIOS[key]
    else:
        text = st.text_area("Disruption signal", height=130,
                            value="Cyclone warning for Karachi: port and airport operations restricted for 4 days.")
        scenario = {"title": "Custom disruption", "description": "Signal entered manually",
                    "signals": [{"source": "Manual input", "text": text}]}
    engine = st.radio("Engine", ["Native", "CrewAI"], horizontal=True,
                      help="CrewAI needs `pip install crewai` and a Gemini/Groq key in .env")
    use_llm = st.toggle("Use LLM (Gemini / Groq)", value=True)
    fail_label = st.selectbox("Failure injection (resilience demo)",
                              ["None", "Logistics fails once (retry)", "Logistics down (fallback)"])
    fail = {"None": None, "Logistics fails once (retry)": "logistics",
            "Logistics down (fallback)": "logistics:hard"}[fail_label]
    run_btn = st.button("Run agents", type="primary")

st.markdown('<div class="hero"><h1>🚚 Supply Chain Crisis Command Center</h1>'
            '<p>Multi-agent system: detect, analyze, source alternatives, optimize logistics, recommend recovery.</p></div>',
            unsafe_allow_html=True)

if run_btn:
    with st.spinner("Agents are working..."):
        try:
            llm = LLM(enabled=use_llm)
            if engine == "CrewAI":
                try:
                    from crew_engine import CrewCoordinator
                    coord = CrewCoordinator(llm, printer=None, approval="pending", inject_failure=fail)
                except ImportError as e:
                    st.warning(f"CrewAI engine could not load: {e}. Running the native engine instead.")
                    coord = Coordinator(llm, printer=None, approval="pending", inject_failure=fail)
            else:
                coord = Coordinator(llm, printer=None, approval="pending", inject_failure=fail)
            st.session_state.run = coord.run(scenario)
            st.session_state.decision = None
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()

if "run" not in st.session_state:
    st.info("Select a scenario from the sidebar and click **Run agents**.")
    st.stop()

run = st.session_state.run
R = run["results"]
planner = R.get("planner")

st.subheader(run["scenario"]["title"])
st.caption(run["scenario"]["description"] + "  |  LLM: " + run["llm_status"])

if not planner or not planner.data:
    st.warning("No actionable disruption was detected from the signal. Try rephrasing it "
               "(mention a strike, cyclone, supplier name or SUP-id, and a duration).")
    st.stop()

pd_ = planner.data
rec = next(o for o in pd_["options"] if o["key"] == pd_["recommended_key"])
base = pd_["baseline"]
impact = R["impact"].data

# ------------------------------------------------------------------ KPIs
c = st.columns(5)
kpi(c[0], "Severity", f'<span class="badge sev-{run["severity"]}">{run["severity"].upper()}</span>',
    f'{len(run["events"])} event(s) detected')
kpi(c[1], "Value at risk", money(impact["value_at_risk_usd"]), f'{len(impact["affected_shipments"])} shipments affected')
kpi(c[2], "Orders on time", f'{base["on_time_orders"]}/{base["total_orders"]} → {rec["on_time_orders"]}/{rec["total_orders"]}',
    "baseline → recommended")
kpi(c[3], "Recovery cost", money(rec["total_cost"]), rec["label"])
kpi(c[4], "Confidence", f"{planner.confidence:.0%}", "weakest agent input")
st.write("")

tabs = st.tabs(["Recommendation", "Agent Activity", "Impact & Inventory", "Options", "Assumptions & Risks"])

# ------------------------------------------------------------------ tab 1
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### Recommended: {rec['label']}")
        st.write(planner.summary)
        st.markdown("**Action plan**")
        for i, a in enumerate(pd_["actions"], 1):
            st.markdown(f"{i}. {a}")
    with right:
        st.markdown("### Human approval")
        decision = st.session_state.get("decision")
        if not pd_["requires_approval"]:
            st.success(f"Auto-approved: {rec['label']} (no escalation triggers).")
        else:
            st.warning("Escalated to a human because:")
            for r in pd_["escalation_reasons"]:
                st.markdown(f"- {r}")
            if decision is None:
                b1, b2 = st.columns(2)
                if b1.button("✅ Approve plan", type="primary"):
                    st.session_state.decision = "approved"
                    st.rerun()
                if b2.button("❌ Reject plan"):
                    st.session_state.decision = "rejected"
                    st.rerun()
            elif decision == "approved":
                st.success(f"Plan '{rec['label']}' approved. Execute the action plan.")
            else:
                alt = next((o for o in pd_["options"] if o["key"] != rec["key"]), None)
                st.error("Plan rejected.")
                if alt:
                    st.info(f"Next best option for review: **{alt['label']}** "
                            f"(cost {money(alt['total_cost'])}, {alt['on_time_orders']}/{alt['total_orders']} orders on time).")
                if st.button("Undo decision"):
                    st.session_state.decision = None
                    st.rerun()
    st.markdown("### Order outcome under the recommended plan")
    rows = [{"Order": o["order_id"], "Customer": o["customer_id"], "Priority": o["priority"], "SKU": o["sku"],
             "Qty": o["qty"], "Deadline": str(o["deadline"]), "Fulfilled": int(o["fulfilled_qty"]),
             "Fill rate": round(o["fill_rate"] * 100),
             "Status": "On time" if o["on_time"] else ("Partial" if o["fill_rate"] > 0 else "Late")}
            for o in rec["orders"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"Fill rate": st.column_config.ProgressColumn("Fill rate", min_value=0, max_value=100,
                                                                             format="%.0f%%")})
    st.download_button("Download full report (.md)", build_markdown(run),
                       file_name=f"{run['scenario']['title'].replace(' ', '_')}_report.md")

# ------------------------------------------------------------------ tab 2
with tabs[1]:
    a, b = st.columns([2, 3])
    with a:
        st.markdown("### Agents")
        for k in AGENT_ORDER:
            r = R.get(k)
            if not r:
                continue
            icon = "⚠️" if r.degraded else "✅"
            with st.expander(f"{icon} {r.agent}  ·  conf {r.confidence:.0%}  ·  {r.seconds}s  ·  "
                             f"{'LLM' if r.used_llm else 'rules'}"):
                st.write(r.summary)
    with b:
        st.markdown("### Agent-to-agent message log")
        for m in run["messages"]:
            col = KIND_COLOR.get(m["kind"], "#64748b")
            st.markdown(
                f'<div class="msg" style="border-left-color:{col}"><span class="meta">{m["t"]} · '
                f'{html.escape(m["from"])} → {html.escape(m["to"])} · '
                f'<span class="badge" style="background:{col}">{m["kind"]}</span></span><br>'
                f'{html.escape(str(m["content"]))}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 3
with tabs[2]:
    st.markdown("### Detected disruptions")
    st.dataframe(df(run["events"], ["id", "type", "location", "supplier_id", "duration_days", "severity",
                                    "confidence", "source"]), hide_index=True)
    st.markdown("### Affected shipments")
    st.dataframe(df(impact["affected_shipments"], ["shipment_id", "sku", "qty", "origin", "destination", "mode",
                                                   "status", "orig_eta", "new_eta", "delay_days", "value_usd",
                                                   "reasons"]), hide_index=True)
    inv = R.get("inventory")
    if inv and inv.data:
        st.markdown("### Inventory position")
        rows = [{"sku": k, **v} for k, v in inv.data["per_sku"].items()]
        st.dataframe(df(rows, ["sku", "item", "stock", "daily_demand", "days_of_cover", "stockout_date",
                               "resupply_date", "replacement_qty", "status"]), hide_index=True)

# ------------------------------------------------------------------ tab 4
with tabs[3]:
    allopts = [base] + pd_["options"]
    st.markdown("### Option comparison")
    comp = pd.DataFrame([{
        "Option": o["label"] + (" ★" if o.get("recommended") else ""),
        "Orders on time": f'{o["on_time_orders"]}/{o["total_orders"]}',
        "High-priority on time": f'{o["high_on_time"]}/{o["high_total"]}',
        "Procurement": money(o["procurement_cost"]), "Extra freight": money(o["extra_freight"]),
        "Late penalties": money(o["late_penalty"]), "Total cost": money(o["total_cost"]),
        "Risk": f'{o["risk"]:.0%}', "Score": o.get("score", "-")} for o in allopts])
    st.dataframe(comp, hide_index=True)
    st.markdown("### Cost breakdown")
    chart = pd.DataFrame({"Procurement": [o["procurement_cost"] for o in allopts],
                          "Extra freight": [o["extra_freight"] for o in allopts],
                          "Late penalties": [o["late_penalty"] for o in allopts]},
                         index=[o["label"] for o in allopts])
    st.bar_chart(chart)
    st.markdown("### Sourcing and routing detail")
    pick = st.selectbox("Option", [o["label"] for o in pd_["options"]])
    o = next(x for x in pd_["options"] if x["label"] == pick)
    sup_rows = [a for al in o["supplier_plan"].values() for a in al]
    if sup_rows:
        st.markdown("**Emergency sourcing**")
        st.dataframe(df(sup_rows, ["sku", "supplier_id", "name", "qty", "unit_price", "lead_days", "arrival",
                                   "reliability", "cost", "price_estimated"]), hide_index=True)
    else:
        st.caption("No emergency sourcing in this option.")
    if o["logistics_plan"]:
        st.markdown("**Reroutes**")
        st.dataframe(df([{"shipment": k, **v} for k, v in o["logistics_plan"].items()],
                        ["shipment", "route", "eta", "extra_cost", "reliability"]), hide_index=True)
    else:
        st.caption("No reroutes in this option.")

# ------------------------------------------------------------------ tab 5
with tabs[4]:
    x, y = st.columns(2)
    with x:
        st.markdown("### Assumptions (incomplete information)")
        notes = sorted({n for r in R.values() if r for n in r.assumptions})
        for n in notes or ["None"]:
            st.markdown(f"- {n}")
        st.markdown("### Warnings")
        warns = sorted({w for r in R.values() if r for w in r.warnings})
        for w in warns or ["None"]:
            st.markdown(f"- {w}")
    with y:
        st.markdown("### Risks")
        for r in pd_["risks"]:
            st.markdown(f"- {r}")
        if pd_.get("unresolved"):
            st.markdown("### Unresolved shortages")
            for u in pd_["unresolved"]:
                st.error(f"{u['sku']}: short by {u['short_qty']} units, stock-out {u['stockout_date']}")