import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Yahoo Finance only serves hourly data for the last 730 days
PERIOD = "729d"
DAILY_START = "2017-01-01"


def _clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, lowercase, strip timezone."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "timestamp"
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _fetch_binance_hourly(symbol: str, days: int = 60) -> pd.DataFrame:
    """Binance public API — hourly OHLCV for the last N days. Fast, no auth."""
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    rows = []

    while start_ms < end_ms:
        resp = requests.get(url, params={
            "symbol":    symbol,
            "interval":  "1h",
            "startTime": start_ms,
            "limit":     1000,
        }, timeout=15)
        data = resp.json()
        if not data or not isinstance(data, list):
            break
        rows.extend(data)
        start_ms = data[-1][6] + 1
        if len(data) < 1000:
            break

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]].astype(float).dropna()


def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    """Hourly OHLCV — last 729 days (Yahoo Finance limit)."""
    df = yf.download(ticker, period=PERIOD, interval="1h", auto_adjust=True, progress=False)
    df = _clean_yf(df)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_ohlcv_fast(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Fast hourly fetch for the app — goes straight to Binance (no Yahoo retries).
    Returns last N days of hourly OHLCV in seconds.
    """
    binance_sym = _BINANCE_SYMBOLS.get(ticker)
    if not binance_sym:
        raise RuntimeError(f"No Binance symbol for {ticker}")
    return _fetch_binance_hourly(binance_sym, days=days)


_BINANCE_SYMBOLS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}


def _fetch_binance_daily(symbol: str) -> pd.DataFrame:
    """
    Binance public klines API — no auth, no rate limit for daily data.
    Paginates automatically to cover 2017-present.
    """
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(DAILY_START).timestamp() * 1000)
    end_ms   = int(pd.Timestamp.now().timestamp() * 1000)
    rows = []

    while start_ms < end_ms:
        resp = requests.get(url, params={
            "symbol":    symbol,
            "interval":  "1d",
            "startTime": start_ms,
            "limit":     1000,
        }, timeout=15)
        data = resp.json()
        if not data or not isinstance(data, list):
            break
        rows.extend(data)
        start_ms = data[-1][6] + 1   # close_time + 1 ms
        if len(data) < 1000:
            break

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df.dropna()


def fetch_daily_ohlcv(ticker: str) -> pd.DataFrame:
    """
    Daily OHLCV from 2017. Tries Yahoo Finance first, falls back to Binance
    public API if Yahoo is rate-limited.
    """
    for attempt in range(3):
        df = yf.download(ticker, start=DAILY_START, interval="1d", auto_adjust=True, progress=False)
        df = _clean_yf(df)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) > 100:
            return df
        wait = 10 * (attempt + 1)
        print(f"  Yahoo rate limited - waiting {wait}s (attempt {attempt + 1}/3)...")
        time.sleep(wait)

    # Binance fallback
    binance_sym = _BINANCE_SYMBOLS.get(ticker)
    if not binance_sym:
        raise RuntimeError(f"No Binance symbol mapping for {ticker}")
    print(f"  Yahoo unavailable - downloading from Binance public API...")
    return _fetch_binance_daily(binance_sym)


def load_raw_data(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load hourly BTC/ETH (cached as CSV)."""
    btc_path = DATA_DIR / "btc_hourly.csv"
    eth_path = DATA_DIR / "eth_hourly.csv"

    if not force_download and btc_path.exists() and eth_path.exists():
        print("Loading cached hourly data...")
        btc = pd.read_csv(btc_path, index_col="timestamp", parse_dates=True)
        eth = pd.read_csv(eth_path, index_col="timestamp", parse_dates=True)
        return btc, eth

    print("Downloading BTC-USD hourly data (last 729 days)...")
    btc = fetch_ohlcv("BTC-USD")
    print("Downloading ETH-USD hourly data (last 729 days)...")
    eth = fetch_ohlcv("ETH-USD")

    btc.to_csv(btc_path)
    eth.to_csv(eth_path)
    print(f"Saved hourly data to {DATA_DIR}")
    return btc, eth


def load_daily_data(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load daily BTC/ETH from 2017 (cached as CSV)."""
    btc_path = DATA_DIR / "btc_daily.csv"
    eth_path = DATA_DIR / "eth_daily.csv"

    if not force_download and btc_path.exists() and eth_path.exists():
        btc_d = pd.read_csv(btc_path, index_col="timestamp", parse_dates=True)
        eth_d = pd.read_csv(eth_path, index_col="timestamp", parse_dates=True)
        # Discard empty cached files (from a previously rate-limited download)
        if len(btc_d) > 0 and len(eth_d) > 0:
            print(f"Loading cached daily data ({len(btc_d)} BTC rows)...")
            return btc_d, eth_d
        print("Cached daily files are empty - re-downloading...")

    print("Downloading BTC-USD daily data (2017-present)...")
    btc_d = fetch_daily_ohlcv("BTC-USD")
    print("Downloading ETH-USD daily data (2017-present)...")
    eth_d = fetch_daily_ohlcv("ETH-USD")

    btc_d.to_csv(btc_path)
    eth_d.to_csv(eth_path)
    print(f"Saved daily data to {DATA_DIR}")
    return btc_d, eth_d
