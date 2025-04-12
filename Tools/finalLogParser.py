import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_FILE = "logs/final.log"  # <--- Update this

def split_log_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()

    # Extract sandbox JSON entries
    sandbox_jsons = re.findall(r'{\s*"sandboxLog":.*?}', text, re.DOTALL)

    # Extract activity CSV section
    activities_match = re.search(r'Activities log:(.*?)Trade History:', text, re.DOTALL)
    activities_csv = activities_match.group(1).strip().splitlines() if activities_match else []

    # Extract trade history JSON array
    trade_match = re.search(r'Trade History:\s*(\[[^\]]*\])', text, re.DOTALL)
    trade_json = json.loads(trade_match.group(1)) if trade_match else []

    return sandbox_jsons, activities_csv, trade_json

def parse_trades_and_pnl(sandbox_logs, trade_history):
    pnl_data = defaultdict(list)
    trade_data = defaultdict(list)
    cumulative_pnl = defaultdict(float)

    for line in sandbox_logs:
        try:
            entry = json.loads(line)
            timestamp = entry.get("timestamp", 0)
            lambda_log = json.loads(entry.get("lambdaLog", "[]"))
            trades = lambda_log[1]

            for trade in trades:
                product, price, quantity = trade
                pnl = -price * quantity
                cumulative_pnl[product] += pnl
                pnl_data[product].append((timestamp, cumulative_pnl[product]))
                trade_data[product].append((timestamp, price, quantity))

        except Exception as e:
            print("Sandbox parse error:", e)

    for trade in trade_history:
        try:
            symbol = trade["symbol"]
            price = trade["price"]
            qty = trade["quantity"]
            ts = trade["timestamp"]
            pnl = -price * qty
            cumulative_pnl[symbol] += pnl
            pnl_data[symbol].append((ts, cumulative_pnl[symbol]))
            trade_data[symbol].append((ts, price, qty))
        except Exception as e:
            print("Trade history parse error:", e)

    return pnl_data, trade_data

def plot_pnl(pnl_data, trade_data):
    for product in pnl_data:
        times, pnls = zip(*sorted(pnl_data[product]))
        plt.figure(figsize=(10, 5))
        plt.plot(times, pnls, label='PnL', linewidth=2)

        for t, price, qty in trade_data[product]:
            color = 'green' if qty > 0 else 'red'
            plt.scatter(t, pnl_data[product][-1][1], color=color, label='Buy' if qty > 0 else 'Sell', s=50, alpha=0.7)

        plt.title(f"{product} – PnL and Trade Executions")
        plt.xlabel("Time")
        plt.ylabel("Cumulative PnL")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === RUN ===
sandbox_logs, activity_csv, trade_json = split_log_file(LOG_FILE)
pnl_data, trade_data = parse_trades_and_pnl(sandbox_logs, trade_json)
plot_pnl(pnl_data, trade_data)
