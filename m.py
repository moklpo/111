"""
Telegram Breakout Bot — India (IST)

What this does
- Runs as a Telegram bot and posts intraday breakout/boost candidates till 3:00 PM IST.
- Scans a universe (e.g., NIFTY50, F&O list, or your custom watchlist).
- Strategies included: Previous Day High/Low breakout + volume surge, Opening Range Breakout (first 15m), 200 EMA trend filter, RSI heat-check, VWAP proximity.
- Schedules checks every N minutes between 09:20–15:00 Asia/Kolkata on trading days (Mon–Fri). Holiday skipping is a TODO.

Quick start
1) Python 3.10+
2) pip install -r requirements (see list below)
3) Set environment variables (or edit CONFIG below):
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID (your own chat id or channel id where bot is admin)
4) Choose data provider: start with MockProvider for dry-run; later plug Angel One / Upstox.
5) Run: python bot.py

Minimal requirements (put in requirements.txt if you want)
- python-telegram-bot==21.6
- pandas>=2.2
- numpy>=1.26
- pytz>=2024.1
- APScheduler==3.10.4
- requests>=2.32

Optional (when you wire a real broker/API)
- smartapi-python  # Angel One SmartAPI
- upstox-api      # Upstox

"""
from __future__ import annotations
import os
import sys
import time
import math
import json
import enum
import pytz
import queue
import random
import logging
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot

# ----------------------- CONFIG -----------------------
class CONFIG:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("7699186659:AAF9nf26Y2h4sDgAB0P0LevH9vO8hvkOSjw")
    TELEGRAM_CHAT_ID = os.getenv("1250330319", "1250330319")

    # Universe (edit to your liking). You can also load from a file.
    UNIVERSE: List[str] = [
        # NIFTY50 sample (tradingview format-like; adjust to your provider symbol scheme)
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "SBIN",
        "BHARTIARTL", "ITC", "HINDUNILVR", "KOTAKBANK", "BAJFINANCE", "AXISBANK",
        "HCLTECH", "MARUTI", "M&M", "ASIANPAINT", "ULTRACEMCO", "WIPRO", "POWERGRID"
    ]

    # Scan cadence (minutes)
    SCAN_EVERY_MIN = int(os.getenv("SCAN_EVERY_MIN", 5))

    # Market hours (IST). Scans will run inside this window.
    MARKET_TZ = "Asia/Kolkata"
    MARKET_OPEN = dt.time(9, 20)   # start scanning after first candles settle
    MARKET_CLOSE = dt.time(15, 0)  # stop at 3:00 PM IST as requested

    # Strategy params
    ORB_MINUTES = 15
    VOL_SURGE_MULTIPLIER = 1.5  # volume > 1.5x of 20-bar average
    RSI_OVERHEAT = 70
    RSI_COLD = 30
    EMA_FAST = 20
    EMA_SLOW = 200

    # Message throttling: minimum minutes between re-alerts for the same symbol & reason
    REPEAT_MINS = 20

    # Data provider: "mock", "angel", "upstox" (implementations below)
    DATA_PROVIDER = os.getenv("DATA_PROVIDER", "mock").lower()

    # Logging
    LOGLEVEL = os.getenv("LOGLEVEL", "INFO")

# ----------------------- UTILITIES -----------------------
logging.basicConfig(
    level=getattr(logging, CONFIG.LOGLEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("breakout-bot")

IST = pytz.timezone(CONFIG.MARKET_TZ)


def now_ist() -> dt.datetime:
    return dt.datetime.now(tz=IST)


def in_market_hours(ts: Optional[dt.datetime] = None) -> bool:
    ts = ts or now_ist()
    t = ts.time()
    return (t >= CONFIG.MARKET_OPEN) and (t <= CONFIG.MARKET_CLOSE) and (ts.weekday() < 5)


# ----------------------- INDICATORS -----------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(window=period).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(window=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(method="bfill").fillna(50)


def vwap(df: pd.DataFrame) -> pd.Series:
    # typical price * volume cumulative / cumulative volume
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vp = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (cum_vp / cum_vol).fillna(method="bfill").fillna(df["close"])  # fallback


# ----------------------- SIGNAL LOGIC -----------------------
@dataclass
class Signal:
    symbol: str
    reason: str
    price: float
    extra: Dict[str, str]


def compute_signals(symbol: str, df: pd.DataFrame) -> List[Signal]:
    """Given a minute DataFrame with columns: datetime, open, high, low, close, volume
    return a list of signals for the latest bar.
    """
    if df.empty or len(df) < 40:
        return []

    df = df.copy()
    df["ema20"] = ema(df["close"], CONFIG.EMA_FAST)
    df["ema200"] = ema(df["close"], CONFIG.EMA_SLOW)
    df["rsi14"] = rsi(df["close"], 14)
    df["vwap"] = vwap(df)
    df["avg_vol20"] = df["volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev_day = df[df["datetime"].dt.date == (df["datetime"].iloc[-1].date() - dt.timedelta(days=1))]
    pdh = prev_day["high"].max() if not prev_day.empty else np.nan
    pdl = prev_day["low"].min() if not prev_day.empty else np.nan

    signals: List[Signal] = []

    # Trend filter: price above 200 EMA considered uptrend
    uptrend = last["close"] > last["ema200"] if not math.isnan(last["ema200"]) else False
    downtrend = last["close"] < last["ema200"] if not math.isnan(last["ema200"]) else False

    # Volume surge check
    vol_ok = (last["volume"] > CONFIG.VOL_SURGE_MULTIPLIER * last["avg_vol20"]) if not math.isnan(last["avg_vol20"]) else False

    # 1) Previous Day High breakout
    if pd.notna(pdh) and last["high"] >= pdh and uptrend and vol_ok:
        signals.append(Signal(
            symbol=symbol,
            reason="PDH Breakout + Vol",
            price=float(last["close"]),
            extra={"PDH": f"{pdh:.2f}", "EMA200": f"{last['ema200']:.2f}", "RSI": f"{last['rsi14']:.1f}"}
        ))

    # 2) Opening Range Breakout (first N mins)
    session_date = df["datetime"].iloc[-1].date()
    session = df[df["datetime"].dt.date == session_date]
    if not session.empty:
        session_start = session["datetime"].iloc[0]
        orb_cutoff = session_start + dt.timedelta(minutes=CONFIG.ORB_MINUTES)
        opening_range = session[(session["datetime"] >= session_start) & (session["datetime"] < orb_cutoff)]
        if len(opening_range) >= 2:
            orb_high = opening_range["high"].max()
            orb_low = opening_range["low"].min()
            # latest candle breaks ORH with vol surge and price above VWAP
            if last["high"] >= orb_high and last["close"] > last["vwap"] and vol_ok:
                signals.append(Signal(
                    symbol=symbol,
                    reason=f"{CONFIG.ORB_MINUTES}m ORB Up + Vol + VWAP",
                    price=float(last["close"]),
                    extra={"ORH": f"{orb_high:.2f}", "VWAP": f"{last['vwap']:.2f}"}
                ))
            # ORB down (optional): uncomment if you want shorts
            # if last["low"] <= orb_low and last["close"] < last["vwap"] and vol_ok:
            #     signals.append(Signal(
            #         symbol=symbol,
            #         reason=f"{CONFIG.ORB_MINUTES}m ORB Down + Vol + VWAP",
            #         price=float(last["close"]),
            #         extra={"ORL": f"{orb_low:.2f}", "VWAP": f"{last['vwap']:.2f}"}
            #     ))

    # 3) Boost/Momentum: price > EMA20 > EMA200 and RSI between 55–75
    if uptrend and (last["ema20"] > last["ema200"]) and (55 <= last["rsi14"] <= 75) and vol_ok:
        signals.append(Signal(
            symbol=symbol,
            reason="Momentum Boost (EMA20>EMA200, RSI 55–75, Vol)",
            price=float(last["close"]),
            extra={"EMA20": f"{last['ema20']:.2f}", "EMA200": f"{last['ema200']:.2f}", "RSI": f"{last['rsi14']:.1f}"}
        ))

    return signals


# ----------------------- DATA PROVIDERS -----------------------
class BaseProvider:
    def fetch_intraday(self, symbol: str, interval: str = "1m", lookback_minutes: int = 400) -> pd.DataFrame:
        """Return DataFrame with columns: datetime (tz-aware, IST), open, high, low, close, volume.
        """
        raise NotImplementedError


class MockProvider(BaseProvider):
    """Generates semi-realistic intraday minute candles for testing the bot end-to-end."""
    def fetch_intraday(self, symbol: str, interval: str = "1m", lookback_minutes: int = 400) -> pd.DataFrame:
        end = now_ist().replace(second=0, microsecond=0)
        idx = pd.date_range(end=end, periods=lookback_minutes, freq="1min", tz=IST)
        base = 100 + random.random() * 100
        noise = np.random.normal(0, 0.15, size=len(idx)).cumsum()
        close = base + noise + np.linspace(0, random.choice([1, 2, -1, 0.5]), len(idx))
        high = close + np.random.rand(len(idx)) * 0.2
        low = close - np.random.rand(len(idx)) * 0.2
        open_ = close + np.random.normal(0, 0.05, size=len(idx))
        vol = np.random.randint(5000, 50000, size=len(idx))
        df = pd.DataFrame({
            "datetime": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        })
        return df


class AngelProvider(BaseProvider):
    """Skeleton for Angel One SmartAPI. Fill your credentials and implement API calls.
    Notes:
      - Use SmartAPI REST for historical intraday (or build minute candles from ticks via WebSocket).
      - Ensure symbol mapping (Angel tokens) is correct.
    """
    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY", "")
        self.client_code = os.getenv("ANGEL_CLIENT_CODE", "")
        self.password = os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")
        # TODO: init SmartConnect etc.

    def fetch_intraday(self, symbol: str, interval: str = "1m", lookback_minutes: int = 400) -> pd.DataFrame:
        # TODO: call SmartAPI historical endpoint and return standardized DataFrame
        raise NotImplementedError("Implement Angel One historical fetch")


class UpstoxProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY", "")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
        # TODO: init Upstox client

    def fetch_intraday(self, symbol: str, interval: str = "1m", lookback_minutes: int = 400) -> pd.DataFrame:
        # TODO: call Upstox historical API and return standardized DataFrame
        raise NotImplementedError("Implement Upstox historical fetch")


# Provider factory
PROVIDERS = {
    "mock": MockProvider,
    "angel": AngelProvider,
    "upstox": UpstoxProvider,
}


def get_provider() -> BaseProvider:
    provider_cls = PROVIDERS.get(CONFIG.DATA_PROVIDER, MockProvider)
    return provider_cls()


# ----------------------- TELEGRAM SENDER -----------------------
class TelegramClient:
    def __init__(self, token: str, chat_id: str):
        if not token or not chat_id or token.startswith("PASTE_"):
            logger.warning("Telegram token/chat_id missing. Messages will be logged only.")
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=token) if token and not token.startswith("PASTE_") else None

    def send(self, text: str, disable_web_page_preview: bool = True):
        if self.bot:
            try:
                self.bot.send_message(chat_id=self.chat_id, text=text, disable_web_page_preview=disable_web_page_preview)
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
        else:
            logger.info(f"[TELEGRAM MOCK]\n{text}")


# ----------------------- SCANNER ENGINE -----------------------
class Scanner:
    def __init__(self, provider: BaseProvider, tg: TelegramClient):
        self.provider = provider
        self.tg = tg
        self.last_alert_at: Dict[Tuple[str, str], dt.datetime] = {}

    def should_alert(self, symbol: str, reason: str) -> bool:
        key = (symbol, reason)
        last = self.last_alert_at.get(key)
        if not last:
            return True
        mins = (now_ist() - last).total_seconds() / 60.0
        return mins >= CONFIG.REPEAT_MINS

    def mark_alerted(self, symbol: str, reason: str):
        self.last_alert_at[(symbol, reason)] = now_ist()

    def scan_once(self):
        if not in_market_hours():
            logger.info("Outside market hours; skipping scan.")
            return

        lines = []
        for sym in CONFIG.UNIVERSE:
            try:
                df = self.provider.fetch_intraday(sym, interval="1m", lookback_minutes=500)
                # ensure columns
                needed = {"datetime", "open", "high", "low", "close", "volume"}
                if not needed.issubset(df.columns):
                    logger.warning(f"{sym}: missing columns {needed - set(df.columns)}")
                    continue
                df = df.sort_values("datetime")
                sigs = compute_signals(sym, df)
                for s in sigs:
                    if self.should_alert(s.symbol, s.reason):
                        self.mark_alerted(s.symbol, s.reason)
                        extra = ", ".join([f"{k}:{v}" for k, v in s.extra.items()]) if s.extra else ""
                        lines.append(f"• {s.symbol} — {s.reason} @ {s.price:.2f} ({extra})")
            except Exception as e:
                logger.error(f"Scan error for {sym}: {e}")

        if lines:
            header = f"🚀 Intraday Scanner {now_ist().strftime('%d-%b %H:%M IST')}\n"
            footer = "\n— timeframe: 1m, filters: Vol surge, EMA200 trend, VWAP/RSI; re-alert ≥{m}m".format(m=CONFIG.REPEAT_MINS)
            msg = header + "\n".join(lines) + footer
            self.tg.send(msg)
        else:
            logger.info("No signals this run.")


# ----------------------- SCHEDULER -----------------------

def start_scheduler(scanner: Scanner):
    sched = BackgroundScheduler(timezone=CONFIG.MARKET_TZ)

    # Cron trigger every SCAN_EVERY_MIN between open/close on weekdays
    # Note: CronTrigger doesn't support time ranges directly with tz-naive times, so we run every N minutes
    # and check in_market_hours() inside scan_once().
    trig = CronTrigger(minute=f"*/{CONFIG.SCAN_EVERY_MIN}", day_of_week="mon-fri")
    sched.add_job(scanner.scan_once, trig, name="intraday-scan")
    sched.start()
    logger.info(f"Scheduler started: every {CONFIG.SCAN_EVERY_MIN} min, Mon–Fri, IST window enforced in code.")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()


# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    provider = get_provider()
    tg = TelegramClient(CONFIG.TELEGRAM_BOT_TOKEN, CONFIG.TELEGRAM_CHAT_ID)
    scanner = Scanner(provider, tg)

    # Run once at start if within market
    scanner.scan_once()

    # Start periodic schedule
    start_scheduler(scanner)
