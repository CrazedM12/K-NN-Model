import pandas as pd

FEATURES = ["rsi", "ret_1d", "ret_5d", "vol_ratio", "volatility"]


def to_daily(bars):
    """Collapse 5-minute bars (from data_loader.load_bars) into daily bars."""
    df = bars.set_index("ts_et").between_time("09:30", "15:59")
    daily = df.groupby(df.index.date).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def build_features(daily):
    df = daily.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["volatility"] = df["ret_1d"].rolling(20).std()
    df["rsi"] = rsi(df["close"])

    # Target: does tomorrow close higher than today? (1 = UP, 0 = DOWN)
    df["next_ret"] = df["ret_1d"].shift(-1)
    df["target"] = (df["next_ret"] > 0).astype(int)

    df = df.dropna(subset=FEATURES)
    labeled = df.dropna(subset=["next_ret"])  # for training/testing
    latest = df.tail(1)                       # today's row, for live prediction
    return labeled, latest