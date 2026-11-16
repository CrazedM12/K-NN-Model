import pandas as pd

from data_loader import load_bars
from features import FEATURES, to_daily, build_features
from model import train_knn

SYMBOLS = ["AAPL", "NVDA", "MSFT", "META", "AMD"]
START, END = "2025-01-01", "2025-04-01"  # END is exclusive
THRESHOLD = 0.60  # "strong enough" vote

frames, latest_rows = [], {}
for sym in SYMBOLS:
    bars = load_bars(sym, START, END)
    if bars.empty:
        print(f"No data for {sym} in this range, skipping.")
        continue
    labeled, latest = build_features(to_daily(bars))
    labeled = labeled.assign(symbol=sym)
    frames.append(labeled)
    latest_rows[sym] = latest

data = pd.concat(frames).sort_index(kind="stable")
print(f"Total labeled rows: {len(data)}")

# Chronological split: train on the past, test on the future
split = int(len(data) * 0.8)
train, test = data.iloc[:split], data.iloc[split:]

model, best = train_knn(train[FEATURES], train["target"])
print("Best params:", best)

proba_up = model.predict_proba(test[FEATURES])[:, 1]
signal = (proba_up >= THRESHOLD).astype(int)

accuracy = ((proba_up >= 0.5).astype(int) == test["target"]).mean()
strategy_ret = (signal * test["next_ret"]).mean()
buy_hold_ret = test["next_ret"].mean()

print(f"Test accuracy: {accuracy:.2%}")
print(f"Trades taken: {signal.sum()} of {len(test)} test rows")
print(f"Avg return per row, strategy: {strategy_ret:.3%} | buy & hold: {buy_hold_ret:.3%}")

for sym, latest in latest_rows.items():
    print(f"{sym} P(up): {model.predict_proba(latest[FEATURES])[0, 1]:.2f}")
    
import joblib
joblib.dump(model, "knn_model.joblib")