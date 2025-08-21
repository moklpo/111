import asyncio
import feedparser
import yfinance as yf
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot
from telegram.constants import ParseMode

# --- Telegram Setup ---
TELEGRAM_BOT_TOKEN = "7845992222:AAEmNnAc_YOisquuMS94qGy2L1dhpBMbc14"
TELEGRAM_CHAT_ID = "1250330319"
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# --- Functions ---
async def check_stocks():
    try:
        stock = yf.Ticker("RELIANCE.NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"📊 Reliance Price Update: {price}"
        )
    except Exception as e:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ Error: {e}")

async def fetch_news():
    try:
        feed = feedparser.parse("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")
        latest_news = feed.entries[0].title
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"📰 Market News: {latest_news}"
        )
    except Exception as e:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ News Error: {e}")

# --- Main Scheduler ---
async def main():
    scheduler = AsyncIOScheduler()

    # 9:15 to 3:30 stock check every 5 min
    scheduler.add_job(
        check_stocks,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute="*/5")
    )

    # Outside market hours - news every 30 min
    scheduler.add_job(
        fetch_news,
        CronTrigger(day_of_week="mon-fri", hour="16-23", minute="*/30")
    )

    scheduler.start()
    print("✅ Scheduler Started...")

    # keep running forever
    await asyncio.get_event_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())
