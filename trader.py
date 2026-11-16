import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

from features import FEATURES, build_features

load_dotenv()
KEY = os.getenv("PKBR3J4CLDEM6PFNNMTKK6QQ2S")
SECRET = os.getenv("88wZ9XC1jhoaKJnAMwTGrjvHVL8yZ5gJ6gakf3Ynntmu")

SYMBOLS = ["AAPL", "NVDA", "MSFT", "META", "AMD"]
THRESHOLD = 0.60          # buy if P(up) is at least this
POSITION_USD = 500        # dollars per trade
MAX_ORDERS_PER_RUN = 3    # safety cap
DRY_RUN = True            # True = print only, False = place paper orders

trading = TradingClient(KEY, SECRET, paper=True)  # paper=True: never touches real money
data = StockHistoricalDataClient(KEY, SECRET)
model = joblib.load("knn_model.joblib")


def recent_daily_bars(symbol, days=90):
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days),
        feed=DataFeed.IEX,  # free-tier feed
    )
    df = data.get_stock_bars(req).df.xs(symbol)
    df.index = pd.to_datetime(df.index.date)
    return df[["open", "high", "low", "close", "volume"]]


def main():
    clock = trading.get_clock()
    if clock.is_open:
        print("Market is open: today's daily bar is incomplete. Run after the close.")
        return

    held = {p.symbol for p in trading.get_all_positions()}
    orders_placed = 0

    for sym in SYMBOLS:
        _, latest = build_features(recent_daily_bars(sym))
        p_up = model.predict_proba(latest[FEATURES])[0, 1]
        print(f"{sym}: P(up) = {p_up:.2f}")

        if p_up >= THRESHOLD and sym not in held and orders_placed < MAX_ORDERS_PER_RUN:
            print(f"  -> BUY ${POSITION_USD} of {sym}")
            if not DRY_RUN:
                trading.submit_order(MarketOrderRequest(
                    symbol=sym, notional=POSITION_USD,
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            orders_placed += 1
        elif p_up < 0.5 and sym in held:
            print(f"  -> CLOSE position in {sym}")
            if not DRY_RUN:
                trading.close_position(sym)


if __name__ == "__main__":
    main()