"""Generates the mock datasets in ./data (deliberately contains gaps and traps)."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)


def save(name, rows, cols):
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(DATA, name), index=False)


# status: On Time / In Transit / Delayed / Stuck at Port
save("shipments.csv", [
    ["SH001", "Shanghai", "Karachi", "Sea", "Delayed", "2026-11-25", 45000, "SKU-A", 500, "SUP-1", "CUST-1"],
    ["SH002", "Dubai", "Lahore", "Air", "In Transit", "2026-11-21", 12000, "SKU-B", 120, "SUP-2", "CUST-2"],
    ["SH003", "Shenzhen", "Karachi", "Sea", "Stuck at Port", "2026-11-28", 80000, "SKU-C", 900, "SUP-1", "CUST-3"],
    ["SH004", "Istanbul", "Islamabad", "Air", "In Transit", "2026-11-22", 30000, "SKU-D", 300, "SUP-3", "CUST-1"],
    ["SH005", "Mumbai", "Karachi", "Sea", "Stuck at Port", "2026-11-27", 22000, "SKU-A", 250, "SUP-4", "CUST-4"],
    # SUP-5 is intentionally missing from suppliers.csv (incomplete information test)
    ["SH006", "Karachi", "Faisalabad", "Road", "On Time", "2026-11-20", 9000, "SKU-E", 80, "SUP-5", "CUST-2"],
], ["shipment_id", "origin", "destination", "mode", "status", "eta", "value_usd", "sku", "qty", "supplier_id", "customer_id"])

save("inventory.csv", [
    ["SKU-A", "Cotton Fabric", 300, 60, 200],
    ["SKU-B", "Dyes", 90, 10, 40],
    ["SKU-C", "Machine Parts", 150, 40, 120],
    ["SKU-D", "Packaging", 600, 50, 200],
    ["SKU-E", "Thread", 40, 20, 50],
], ["sku", "item_name", "stock", "daily_demand", "reorder_level"])

# one row per supplier-sku. Blank price / capacity = unknown (incomplete information test)
save("suppliers.csv", [
    ["SUP-1", "Shanghai Textiles", "China", "SKU-A", 5.0, 25, 0.75, "Active", 1000],
    ["SUP-1", "Shanghai Textiles", "China", "SKU-C", 88.0, 25, 0.75, "Active", 1500],
    ["SUP-2", "Dubai Chem Trading", "UAE", "SKU-B", 12.0, 5, 0.90, "Active", 200],
    ["SUP-3", "Istanbul Pack Co", "Turkey", "SKU-D", 3.0, 8, 0.85, "Active", 500],
    ["SUP-4", "Mumbai Fabrics", "India", "SKU-A", 4.5, 14, 0.70, "Active", 600],
    ["SUP-6", "Lahore Local Mills", "Pakistan", "SKU-A", 6.5, 4, 0.88, "Active", 300],
    ["SUP-6", "Lahore Local Mills", "Pakistan", "SKU-E", 2.0, 3, 0.88, "Active", 400],
    ["SUP-7", "Dhaka Fabrics", "Bangladesh", "SKU-A", None, 12, 0.80, "Active", None],
    ["SUP-8", "Guangzhou Machinery", "China", "SKU-C", 95.0, 18, 0.82, "Active", 800],
    ["SUP-9", "Gulf Parts FZE", "UAE", "SKU-C", 110.0, 6, 0.91, "Active", 500],
], ["supplier_id", "name", "country", "sku", "unit_price_usd", "lead_time_days", "reliability", "status", "capacity_units"])

# via = the port/airport the route depends on
save("routes.csv", [
    ["China", "Karachi", "Sea", "Karachi", 25, 1.0, 0.70],
    ["China", "Karachi", "Air", "Karachi Airport", 4, 6.5, 0.92],
    ["China", "Karachi", "Sea+Road", "Gwadar", 30, 1.3, 0.80],
    ["UAE", "Karachi", "Sea", "Karachi", 6, 1.2, 0.85],
    ["UAE", "Karachi", "Air", "Karachi Airport", 2, 5.0, 0.95],
    ["UAE", "Karachi", "Sea+Road", "Gwadar", 8, 1.4, 0.82],
    ["India", "Karachi", "Sea", "Karachi", 5, 1.0, 0.75],
    ["India", "Karachi", "Air", "Karachi Airport", 2, 6.0, 0.93],
    ["India", "Karachi", "Sea+Road", "Gwadar", 9, 1.4, 0.70],
], ["origin_region", "destination", "mode", "via", "transit_days", "cost_multiplier", "reliability"])

save("orders.csv", [
    ["ORD-1", "CUST-1", "High", "2026-11-26", "SKU-A", 400, 18000],
    ["ORD-2", "CUST-2", "Medium", "2026-11-26", "SKU-B", 60, 4000],
    ["ORD-3", "CUST-3", "High", "2026-11-29", "SKU-C", 700, 65000],
    ["ORD-4", "CUST-4", "Low", "2026-12-05", "SKU-A", 200, 9000],
], ["order_id", "customer_id", "priority", "deadline", "sku", "qty", "order_value_usd"])

print("Data generated in", DATA)
