"""CLI entry point.

  python main.py --list
  python main.py --scenario port_strike
  python main.py --scenario all
  python main.py --scenario mixed_crisis --approve ask      (pending | approve | ask | reject)
  python main.py --scenario port_strike --inject-failure logistics        (agent fails once, retry works)
  python main.py --scenario port_strike --inject-failure logistics:hard   (agent stays down, fallback)
  python main.py --scenario port_strike --no-llm                          (force offline mode)
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llm import LLM
from orchestrator import Coordinator
from report import build_json, build_markdown
from scenarios import SCENARIOS

OUT = Path(__file__).parent / "outputs"


def printer(m):
    print(f"[{m['t']}] {m['from']} -> {m['to']} | {m['kind']}: {m['content']}", flush=True)


def run_one(name, args):
    sc = SCENARIOS[name]
    print("\n" + "=" * 100 + f"\nSCENARIO: {sc['title']}\n" + "=" * 100)
    coord = Coordinator(LLM(enabled=not args.no_llm), printer=None if args.quiet else printer,
                        approval=args.approve, inject_failure=args.inject_failure)
    run = coord.run(sc)
    OUT.mkdir(exist_ok=True)
    md = build_markdown(run)
    (OUT / f"{name}_report.md").write_text(md, encoding="utf-8")
    (OUT / f"{name}_run.json").write_text(build_json(run), encoding="utf-8")
    pl = run["results"].get("planner")
    if pl and pl.data:
        rec = next(o for o in pl.data["options"] if o["key"] == pl.data["recommended_key"])
        print(f"\n>>> RECOMMENDED: {rec['label']} | on time {rec['on_time_orders']}/{rec['total_orders']} | "
              f"cost ${rec['total_cost']:,.0f} | confidence {pl.confidence:.0%} | approval: {run['approval']['decision']}")
    print(f">>> Report: {OUT / (name + '_report.md')}")
    return run


def main():
    ap = argparse.ArgumentParser(description="Supply Chain Crisis Management - multi-agent system")
    ap.add_argument("--scenario", default="port_strike", help="name or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--approve", choices=["pending", "approve", "ask", "reject"], default="pending")
    ap.add_argument("--inject-failure", help="e.g. logistics | logistics:hard | supplier")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="hide the agent message log")
    args = ap.parse_args()
    if args.list:
        for k, v in SCENARIOS.items():
            print(f"{k:22s} {v['title']} - {v['description']}")
        return
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for n in names:
        if n not in SCENARIOS:
            sys.exit(f"Unknown scenario '{n}'. Use --list.")
        run_one(n, args)


if __name__ == "__main__":
    main()
