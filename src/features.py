import pandas as pd
import numpy as np
import ta

DIP_THRESHOLD = 0.02   # 2% drop
DIP_HORIZON   = 6      # hours ahead


def label_dips(btc: pd.DataFrame) -> pd.Series:
    """1 if BTC close drops >=2% within the next 6 hours, else 0."""
    future_min = btc["close"].shift(-DIP_HORIZON).rolling(DIP_HORIZON).min()
    pct_change = (future_min - btc["close"]) / btc["close"]
    return (pct_change <= -DIP_THRESHOLD).astype(int)


# ─── Hourly BTC features ───────────────────────────────────────────────────────

def add_btc_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    df["return_1h"]     = c.pct_change(1)
    df["return_3h"]     = c.pct_change(3)
    df["return_6h"]     = c.pct_change(6)
    df["log_return_1h"] = np.log(c / c.shift(1))

    df["volatility_6h"]  = df["log_return_1h"].rolling(6).std()
    df["volatility_24h"] = df["log_return_1h"].rolling(24).std()

    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    macd = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    bb = ta.volatility.BollingerBands(c, window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / c

    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    df["ema_12"]   = ta.trend.EMAIndicator(c, window=12).ema_indicator()
    df["ema_26"]   = ta.trend.EMAIndicator(c, window=26).ema_indicator()
    df["ema_cross"] = df["ema_12"] - df["ema_26"]

    df["volume_change"]   = v.pct_change(1)
    df["volume_ma_ratio"] = v / v.rolling(24).mean()

    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["momentum_6h"]  = c - c.shift(6)
    df["momentum_12h"] = c - c.shift(12)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"close_lag_{lag}"]  = c.shift(lag)
        df[f"return_lag_{lag}"] = df["return_1h"].shift(lag)

    return df


# ─── Hourly ETH features ───────────────────────────────────────────────────────

def add_eth_features(btc: pd.DataFrame, eth: pd.DataFrame) -> pd.DataFrame:
    eth_aligned = eth.reindex(btc.index, method="nearest")

    btc["eth_close"]        = eth_aligned["close"]
    btc["eth_return_1h"]    = eth_aligned["close"].pct_change(1)
    btc["eth_return_6h"]    = eth_aligned["close"].pct_change(6)
    btc["eth_volatility_6h"] = np.log(eth_aligned["close"] / eth_aligned["close"].shift(1)).rolling(6).std()
    btc["eth_rsi_14"]       = ta.momentum.RSIIndicator(eth_aligned["close"], window=14).rsi()

    btc["btc_eth_ratio"]      = btc["close"] / eth_aligned["close"]
    btc["btc_eth_corr_24h"]   = btc["return_1h"].rolling(24).corr(btc["eth_return_1h"])

    return btc


# ─── Daily feature engineering (7 years of context) ───────────────────────────

def add_daily_features(df_hourly: pd.DataFrame,
                       btc_daily: pd.DataFrame,
                       eth_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute long-term daily features and forward-fill them onto the hourly frame.
    This gives the model 7 years of market regime context for each hourly prediction.
    """
    bd = btc_daily.copy()
    ed = eth_daily.copy()

    # ── BTC daily features ──
    c = bd["close"]
    v = bd["volume"]

    # Long-term moving averages (market regime signals)
    bd["d_sma_7"]   = c.rolling(7).mean()
    bd["d_sma_30"]  = c.rolling(30).mean()
    bd["d_sma_90"]  = c.rolling(90).mean()
    bd["d_sma_200"] = c.rolling(200).mean()

    # Price position relative to MAs (above/below regime)
    bd["d_price_vs_sma50"]  = c / c.rolling(50).mean() - 1
    bd["d_price_vs_sma200"] = c / c.rolling(200).mean() - 1

    # Bull/bear regime: 1 if above SMA-200, 0 if below
    bd["d_bull_regime"] = (c > bd["d_sma_200"]).astype(float)

    # Long-term returns
    bd["d_return_7d"]  = c.pct_change(7)
    bd["d_return_30d"] = c.pct_change(30)
    bd["d_return_90d"] = c.pct_change(90)

    # Long-term volatility
    bd["d_volatility_7d"]  = np.log(c / c.shift(1)).rolling(7).std()
    bd["d_volatility_30d"] = np.log(c / c.shift(1)).rolling(30).std()

    # Daily RSI (longer horizon)
    bd["d_rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    bd["d_rsi_21"] = ta.momentum.RSIIndicator(c, window=21).rsi()

    # Daily MACD
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    bd["d_macd_diff"] = macd.macd_diff()

    # Daily Bollinger Band position
    bb = ta.volatility.BollingerBands(c, window=20)
    bd["d_bb_pct"] = bb.bollinger_pband()

    # Volume trend
    bd["d_volume_ma_ratio_7"]  = v / v.rolling(7).mean()
    bd["d_volume_ma_ratio_30"] = v / v.rolling(30).mean()

    # ── ETH daily features ──
    ec = ed["close"]
    bd["d_eth_return_7d"]  = ec.reindex(bd.index, method="nearest").pct_change(7)
    bd["d_eth_return_30d"] = ec.reindex(bd.index, method="nearest").pct_change(30)
    bd["d_eth_rsi_14"]     = ta.momentum.RSIIndicator(
        ec.reindex(bd.index, method="nearest"), window=14
    ).rsi()

    # 30-day BTC/ETH return correlation
    btc_dr = np.log(c / c.shift(1))
    eth_dr = np.log(ec.reindex(bd.index, method="nearest") /
                    ec.reindex(bd.index, method="nearest").shift(1))
    bd["d_btc_eth_corr_30d"] = btc_dr.rolling(30).corr(eth_dr)

    # ── Replace inf, drop NaN ──
    bd = bd.replace([float("inf"), float("-inf")], float("nan"))

    # ── Map daily features onto the hourly frame by date ──
    daily_feat_cols = [col for col in bd.columns if col.startswith("d_")]
    daily_feats = bd[daily_feat_cols].copy()

    # Ensure daily index is timezone-naive dates at midnight
    daily_feats.index = pd.to_datetime(daily_feats.index).tz_localize(None).normalize()
    daily_feats = daily_feats[~daily_feats.index.duplicated(keep="last")].sort_index()

    # Normalize hourly timestamps to midnight for the lookup
    hourly_dates = pd.to_datetime(df_hourly.index).tz_localize(None).normalize()

    for col in daily_feat_cols:
        df_hourly[col] = hourly_dates.map(daily_feats[col])

    # Forward-fill any gaps (weekends / holidays / dates before daily data starts)
    df_hourly[daily_feat_cols] = df_hourly[daily_feat_cols].ffill()

    return df_hourly


# ─── Master build function ─────────────────────────────────────────────────────

def build_features(btc: pd.DataFrame,
                   eth: pd.DataFrame,
                   btc_daily: pd.DataFrame,
                   eth_daily: pd.DataFrame) -> pd.DataFrame:
    df = btc.copy()
    df = add_btc_features(df)
    df = add_eth_features(df, eth)
    df = add_daily_features(df, btc_daily, eth_daily)
    df["dip_label"] = label_dips(btc)

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna()
    return df


# ─── Feature column list ───────────────────────────────────────────────────────

HOURLY_FEATURES = [
    "return_1h", "return_3h", "return_6h", "log_return_1h",
    "volatility_6h", "volatility_24h",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "bb_pct", "bb_width", "atr_14",
    "ema_cross", "volume_change", "volume_ma_ratio",
    "stoch_k", "stoch_d", "momentum_6h", "momentum_12h",
    "close_lag_1", "close_lag_2", "close_lag_3",
    "close_lag_6", "close_lag_12", "close_lag_24",
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_lag_6", "return_lag_12", "return_lag_24",
    "eth_return_1h", "eth_return_6h", "eth_volatility_6h", "eth_rsi_14",
    "btc_eth_ratio", "btc_eth_corr_24h",
]

DAILY_FEATURES = [
    "d_sma_7", "d_sma_30", "d_sma_90", "d_sma_200",
    "d_price_vs_sma50", "d_price_vs_sma200", "d_bull_regime",
    "d_return_7d", "d_return_30d", "d_return_90d",
    "d_volatility_7d", "d_volatility_30d",
    "d_rsi_14", "d_rsi_21", "d_macd_diff",
    "d_bb_pct",
    "d_volume_ma_ratio_7", "d_volume_ma_ratio_30",
    "d_eth_return_7d", "d_eth_return_30d", "d_eth_rsi_14",
    "d_btc_eth_corr_30d",
]

FEATURE_COLS = HOURLY_FEATURES + DAILY_FEATURES
