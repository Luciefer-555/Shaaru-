"""
Pipeline 2 — Trend Watch Scheduler
Runs automatically on a schedule.
No human needed after initial setup.
"""

import asyncio
import schedule
import time
import logging
import os
import sys
from datetime import datetime

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from pipeline.trend_watch.detector import detect_trends
from pipeline.trend_watch.extractor import extract_for_trend
from pipeline.trend_watch.trend_store import get_active_trends

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TREND_WATCH] %(message)s"
)
logger = logging.getLogger(__name__)


async def run_trend_detection():
    logger.info("Starting trend detection scan...")
    try:
        trends = await detect_trends()
        logger.info(f"Detected {len(trends)} trends")
        for trend in trends:
            logger.info(
                "Trend: %s (confidence: %.2f, velocity: %s)",
                trend["trend_name"],
                trend.get("confidence", 0),
                trend.get("velocity", "unknown")
            )
        return trends
    except Exception as e:
        logger.error("Trend detection failed: %s", e)
        return []


async def run_trend_extraction():
    logger.info("Starting trend extraction...")
    active_trends = get_active_trends()
    if not active_trends:
        logger.info("No active trends to extract")
        return
        
    for trend in active_trends[:3]:
        await extract_for_trend(trend)
        await asyncio.sleep(10)
    logger.info("Trend extraction complete")


def daily_morning_job():
    logger.info("Daily morning scan starting...")
    asyncio.run(run_trend_detection())


def daily_evening_job():
    logger.info("Daily evening extraction starting...")
    asyncio.run(run_trend_extraction())


def weekly_fashion_week_job():
    logger.info("Weekly fashion week scan starting...")
    from pipeline.trend_watch.sources.fashion_week import scan_fashion_week
    signals = scan_fashion_week()
    logger.info(f"Fashion week signals collected: {len(signals)}")


def start_scheduler():
    logger.info("Trend Watch Scheduler started")
    logger.info("Schedule:")
    logger.info("  Daily 06:00 — trend detection scan")
    logger.info("  Daily 23:00 — trend extraction")
    logger.info("  Monday 09:00 — fashion week scan")
    
    schedule.every().day.at("06:00").do(daily_morning_job)
    schedule.every().day.at("23:00").do(daily_evening_job)
    schedule.every().monday.at("09:00").do(weekly_fashion_week_job)
    
    asyncio.run(run_trend_detection())
    
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    start_scheduler()
