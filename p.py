"""
Telegram Momentum + News Bot (NSE)
----------------------------------
Features
- 24/7 bot that switches modes by India time (IST)
  - 09:15–15:30: Intraday momentum scanner on a liquid universe (default: NIFTY50)
  - Off-hours: Important market news (RSS) + rotating trading knowledge bites
- Sends concise alerts to a Telegram chat via Bot API
- Simple, modular design so you can swap data source later (Angel One SmartAPI, Zerodha, etc.)

Quick Start
1) Create a Telegram bot with @BotFather → get BOT_TOKEN
2) Get your chat ID (forward any message to @userinfobot, or add the bot to a private group and use a /start handler)
3) Copy .env.example → .env and fill values
4) `pip install -r requirements.txt`
5) `python main.py`

Deploy 24/7
- Railway.app / Render.com / Fly.io / VPS all work
- Use the provided Procfile & Dockerfile (optional) for one-click deploys

Note
- Default data source is Yahoo Finance (yfinance) with .NS tickers. For production-grade real-time, replace with Angel One SmartAPI (sample adapter included below, commented).
"""

import os
import time
import math
import asyncio
import logging
import textwrap
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pytz
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot
from telegram.constants import ParseMode

# ---------------------- CONFIG ----------------------
IST = pytz.timezone("Asia/Kolkata")
TRADING_START = datetime.strptime("09:15", "%H:%M").time()
TRADING_END = datetime.strptime("15:30", "%H:%M").time()

BOT_TOKEN = os.getenv("7845992222:AAEmNnAc_YOisquuMS94qGy2L1dhpBMbc14", "")
CHAT_ID = os.getenv("1250330319", "")  # numeric ID or @channelusername

# Universe: NIFTY50 tickers on Yahoo Finance (.NS suffix)
UNIVERSE = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","LT.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","AXISBANK.NS",
    "HINDUNILVR.NS","BAJFINANCE.NS","ASIANPAINT.NS","KOTAKBANK.NS","MARUTI.NS","SUNPHARMA.NS","NTPC.NS","NESTLEIND.NS","ULTRACEMCO.NS","POWERGRID.NS",
    "TITAN.NS","ONGC.NS","WIPRO.NS","M&M.NS","TATASTEEL.NS","JSWSTEEL.NS","HCLTECH.NS","BAJAJFINSV.NS","COALINDIA.NS","BAJAJ-AUTO.NS",
    "HDFCLIFE.NS","TATAMOTORS.NS","ADANIENT.NS","ADANIPORTS.NS","HEROMOTOCO.NS","TECHM.NS","LTIM.NS","BRITANNIA.NS","GRASIM.NS","DRREDDY.NS",
    "CIPLA.NS","APOLLOHOSP.NS","BPCL.NS","EICHERMOT.NS","INDUSINDBK.NS","TATACONSUM.NS","DIVISLAB.NS","HINDALCO.NS","UPL.NS","SBILIFE.NS"
]

# Momentum parameters
INTERVAL = "1m"  # yfinance minute data (may be delayed). Swap to broker API for real-time.
LOOKBACK_MIN = 45  # build indicators on last N minutes
REL_VOL_LOOKBACK = 20
REL_VOL_MIN = 1.5
ADX_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 200
RSI_PERIOD = 14

# News feeds (add/remove to taste)
RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.business-standard.com/rss/home_page_top_stories.rss",
    "https://www.rbi.org.in/Scripts/RSSDisplay.aspx?rss=PressReleases",
    "https://www.nseindia.com/api/marketStatus"
]

KNOWLEDGE_BITS = [
    "Opening Range Breakout (ORB): 15–30 min nu high/low breakout volume sathe aave to conviction vadhe.",
    "Risk first: pratyek trade ma max 1–2% capital risk karo—long run ma aa j bachaave.",
    "VWAP trend: price consistently VWAP thi upar/below hoy to big players no bias soojhe.",
    "Avoid mid-day chop: 11:30–13:30 vachche low momentum; event nathi to overtrade na karo.",
    "Journal rakho: setup, reason, SL, TP, result—mahine ek vaar review karo.",
]

# ---------------------- LOGGING ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("momentum_bot")

# ---------------------- UTILS ----------------------

def is_trading_time(now_ist: datetime) -> bool:
    t = now_ist.time()
    return (t >= TRADING_START) and (t <= TRADING_END) and (now_ist.weekday() < 5)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    atr = true_range(high, low, close).ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9))
    dx = ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9) ) * 100
    return dx.ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    # df columns: [Open, High, Low, Close, Volume]
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).cumsum()
    cum_vol = df['Volume'].cumsum() + 1e-9
    return cum_tp_vol / cum_vol


def relative_volume(vol: pd.Series, lookback: int = 20) -> pd.Series:
    avg_vol = vol.rolling(lookback).mean()
    return vol / (avg_vol + 1e-9)

# ---------------------- DATA ADAPTERS ----------------------

class YahooAdapter:
    def __init__(self, interval: str = INTERVAL):
        self.interval = interval

    def fetch_recent(self, symbol: str, minutes: int = LOOKBACK_MIN) -> pd.DataFrame:
        end = datetime.now(tz=IST)
        start = end - timedelta(minutes=minutes + 10)
        try:
            df = yf.download(symbol, start=start, end=end, interval=self.interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns=str.title)[['Open','High','Low','Close','Volume']]
            df.index = df.index.tz_localize(None)  # yfinance returns tz-aware sometimes
            return df.tail(minutes)
        except Exception as e:
            logger.warning(f"Yahoo fetch failed for {symbol}: {e}")
            return pd.DataFrame()

# Placeholder for Angel One SmartAPI (real-time). Uncomment and implement when ready.
"""
from SmartApi import SmartConnect
import pyotp
class AngelAdapter:
    def __init__(self):
        self.api_key = os.getenv('ANGEL_API_KEY','')
        self.client_code = os.getenv('ANGEL_CLIENT_CODE','')
        self.password = os.getenv('ANGEL_PASSWORD','')
        self.totp_secret = os.getenv('ANGEL_TOTP_SECRET','')
        self.sc = None

    def login(self):
        self.sc = SmartConnect(self.api_key)
        token = pyotp.TOTP(self.totp_secret).now()
        data = self.sc.generateSession(self.client_code, self.password, token)
        # store feed token for websockets if needed

    def fetch_recent(self, symbol: str, minutes: int = 45) -> pd.DataFrame:
        # Use historical data endpoint with exchange+token mapping.
        # Return DataFrame with columns Open,High,Low,Close,Volume indexed by minute
        raise NotImplementedError
"""

# ---------------------- SCANNER LOGIC ----------------------

def momentum_signal(df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """Return (signal, metrics) where signal in {"LONG","SHORT",""} based on rules."""
    if df is None or df.empty or len(df) < max(EMA_SLOW, RSI_PERIOD, ADX_PERIOD) // 2:
        return "", {}

    close = df['Close']
    high = df['High']
    low = df['Low']
    vol = df['Volume']

    ema_fast = ema(close, EMA_FAST)
    ema_slow = ema(close, EMA_SLOW)
    rsi14 = rsi(close, RSI_PERIOD)
    adx14 = adx(high, low, close, ADX_PERIOD)
    vwap_series = vwap(df)
    relvol = relative_volume(vol, REL_VOL_LOOKBACK)

    last = df.index[-1]
    price = float(close.iloc[-1])

    # Breakout filters
    day_high = float(high.max())
    day_low = float(low.min())

    cond_long = (
        price > float(vwap_series.iloc[-1]) and
        price > float(ema_fast.iloc[-1]) and
        ema_slow.iloc[-2] < ema_slow.iloc[-1] and  # slow EMA rising
        rsi14.iloc[-1] > 55 and rsi14.iloc[-1] < 80 and
        adx14.iloc[-1] > 20 and
        relvol.iloc[-1] >= REL_VOL_MIN and
        price >= day_high * 0.999  # near/new day high
    )

    cond_short = (
        price < float(vwap_series.iloc[-1]) and
        price < float(ema_fast.iloc[-1]) and
        ema_slow.iloc[-2] > ema_slow.iloc[-1] and  # slow EMA falling
        rsi14.iloc[-1] < 45 and rsi14.iloc[-1] > 20 and
        adx14.iloc[-1] > 20 and
        relvol.iloc[-1] >= REL_VOL_MIN and
        price <= day_low * 1.001  # near/new day low
    )

    metrics = {
        "price": price,
        "rsi": float(rsi14.iloc[-1]),
        "adx": float(adx14.iloc[-1]),
        "relvol": float(relvol.iloc[-1]),
        "above_vwap": float(price > float(vwap_series.iloc[-1])),
        "above_20ema": float(price > float(ema_fast.iloc[-1])),
    }

    if cond_long:
        return "LONG", metrics
    if cond_short:
        return "SHORT", metrics
    return "", metrics

# ---------------------- MESSAGING ----------------------

class Notifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id

    async def send(self, text: str, disable_preview: bool = True):
        if not self.chat_id:
            logger.error("CHAT_ID missing.")
            return
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=disable_preview)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

# ---------------------- JOBS ----------------------

class MomentumBot:
    def __init__(self, adapter, notifier: Notifier):
        self.adapter = adapter
        self.notifier = notifier
        self.last_alert: Dict[str, str] = {}  # symbol -> last signal

    async def scan_universe(self):
        now = datetime.now(tz=IST)
        if not is_trading_time(now):
            return
        logger.info("Scanning universe...")
        found: List[str] = []
        for sym in UNIVERSE:
            df = self.adapter.fetch_recent(sym, minutes=LOOKBACK_MIN)
            if df.empty:
                continue
            signal, metrics = momentum_signal(df)
            if signal:
                # Avoid duplicate spam: only alert on change
                if self.last_alert.get(sym) != signal:
                    txt = (
                        f"<b>{sym}</b> | <b>{signal}</b> Momentum\n"
                        f"Price: {metrics['price']:.2f} | RSI: {metrics['rsi']:.1f} | ADX: {metrics['adx']:.1f} | RelVol: {metrics['relvol']:.2f}\n"
                        f"VWAP: {'Above' if metrics['above_vwap'] else 'Below'} | 20EMA: {'Above' if metrics['above_20ema'] else 'Below'}\n"
                        f"Idea: {'ORB/Breakout' if signal=='LONG' else 'Breakdown'} — Tight SL under/over last swing."
                    )
                    await self.notifier.send(txt)
                    self.last_alert[sym] = signal
                    found.append(sym)
        if not found:
            logger.info("No fresh signals this scan.")

    async def market_open_ping(self):
        await self.notifier.send("🟢 <b>Market Live (09:15–15:30 IST)</b> — Momentum alerts will appear here.")

    async def market_close_ping(self):
        await self.notifier.send("🔵 <b>Market Closed</b> — I will share important news & trading gyan off-hours.")

    async def send_news_digest(self, limit_per_feed: int = 3):
        now = datetime.now(tz=IST)
        if is_trading_time(now):
            return
        items = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for e in feed.entries[:limit_per_feed]:
                    title = e.get('title', '').strip()
                    link = e.get('link', '').strip()
                    if title and link:
                        items.append(f"• <a href='{link}'> {title} </a>")
            except Exception as e:
                logger.warning(f"RSS fail: {url} | {e}")
        if items:
            block = "\n".join(items[:12])
            await self.notifier.send(f"🗞️ <b>Market News (Digest)</b>\n{block}", disable_preview=False)

    async def send_knowledge(self):
        now = datetime.now(tz=IST)
        if is_trading_time(now):
            return
        tip = KNOWLEDGE_BITS[int(time.time()) % len(KNOWLEDGE_BITS)]
        await self.notifier.send(f"📚 <b>Trading Knowledge</b>\n{tip}")

# ---------------------- SCHEDULER ----------------------

async def main():
    if not BOT_TOKEN or not CHAT_ID:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment.")

    notifier = Notifier(BOT_TOKEN, CHAT_ID)
    adapter = YahooAdapter()
    bot = MomentumBot(adapter, notifier)

    scheduler = AsyncIOScheduler(timezone=IST)

    # Market open/close pings (Mon–Fri)
    scheduler.add_job(bot.market_open_ping, CronTrigger(day_of_week='mon-fri', hour=9, minute=15))
    scheduler.add_job(bot.market_close_ping, CronTrigger(day_of_week='mon-fri', hour=15, minute=31))

    # Scanner every 2 minutes during market hours
    scheduler.add_job(bot.scan_universe, IntervalTrigger(minutes=2))

    # Off-hours news digests (every 2 hours)
    scheduler.add_job(bot.send_news_digest, CronTrigger(hour='16-23,0-8', minute=0))

    # Off-hours knowledge bite (twice a day)
    scheduler.add_job(bot.send_knowledge, CronTrigger(hour='10,18', minute=30))

    scheduler.start()

    await notifier.send("🤖 Bot started. Mode will auto-switch by IST.")

    # Keep alive
    while True:
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
