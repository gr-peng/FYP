import numpy as np
import pandas as pd


def features(bars: pd.DataFrame) -> dict[str, float | None]:
    """Caller supplies completed bars only; this function never looks up future bars."""
    if bars.empty:
        raise ValueError("price history is empty")
    close = bars.close.astype(float)
    volume = bars.volume.astype(float)
    returns = close.pct_change()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    values = {
        "last_close": close.iloc[-1],
        "return_1d": returns.iloc[-1],
        "return_5d": close.pct_change(5).iloc[-1],
        "momentum_5": close.pct_change(5).iloc[-1],
        "ma5": close.rolling(5).mean().iloc[-1],
        "ma20": close.rolling(20).mean().iloc[-1],
        "volatility_20": returns.rolling(20).std(ddof=1).iloc[-1],
        "volume_ratio_5_20": (volume.rolling(5).mean() / volume.rolling(20).mean()).iloc[-1],
        "macd": macd.iloc[-1],
        "macd_signal": signal.iloc[-1],
        "macd_hist": (macd - signal).iloc[-1],
    }
    return {
        key: round(float(value), 8) if np.isfinite(value) else None for key, value in values.items()
    }


def historical_features(bars: pd.DataFrame, session: str) -> dict:
    return features(bars.loc[bars.session_date < session].sort_values("session_date"))
