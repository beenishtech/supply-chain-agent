"""Demo scenarios. Signals are raw free text, exactly what a news/ERP feed would give."""

SCENARIOS = {
    "port_strike": {
        "title": "Karachi port strike",
        "description": "Dockworkers strike halts Karachi port; 3 sea shipments are stuck or delayed.",
        "signals": [
            {"source": "News feed", "text": "BREAKING: Dockworkers at Karachi port begin a strike, terminal "
                                             "operations halted, expected to last about 7 days."},
        ],
    },
    "supplier_bankruptcy": {
        "title": "Main supplier bankruptcy",
        "description": "Shanghai Textiles (SUP-1) declares bankruptcy; SKU-A and SKU-C supply is cut.",
        "signals": [
            {"source": "Supplier portal", "text": "Shanghai Textiles (SUP-1) has filed for bankruptcy and ceased "
                                                  "operations. Recovery not expected for 30 days."},
        ],
    },
    "mixed_crisis": {
        "title": "Cyclone + supplier delay + mid-run capacity cut",
        "description": "Cyclone at Karachi (duration NOT stated) and a Mumbai Fabrics delay; while the plan is "
                       "being built, Lahore Local Mills cuts capacity, forcing a replan.",
        "signals": [
            {"source": "Weather API", "text": "Cyclone warning for Karachi: port and airport operations restricted."},
            {"source": "Supplier email", "text": "Mumbai Fabrics (SUP-4) reports a production delay of 10 days."},
        ],
        "mid_run_signals": [
            {"source": "Supplier email", "text": "Lahore Local Mills (SUP-6) reports a labour shortage, capacity cut "
                                                  "to 100 units for the next 2 weeks."},
        ],
    },
    "no_alternative": {
        "title": "Sole-source supplier failure",
        "description": "Dubai Chem Trading (SUP-2) is the only dye supplier. No alternatives exist -> human escalation.",
        "signals": [
            {"source": "Supplier portal", "text": "Dubai Chem Trading (SUP-2) has shut down due to insolvency, "
                                                  "expected to remain closed for 30 days."},
        ],
    },
}
