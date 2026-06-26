"""
background_profiler.py — APScheduler-based background job runner.

Runs periodic tasks:
  - Product scraping (every 24h)
  - Profile refresh / gap analysis (every 6h)
  - Trend refresh (every 12h)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.profiler")


def scrape_products_job():
    """Scrape all brands and update products_seed.json."""
    log.info("[JOB] Starting product scrape...")
    try:
        from brand_scraper import scrape_all_brands
        products = scrape_all_brands()
        log.info(f"[JOB] Product scrape complete: {len(products)} products")
    except Exception as e:
        log.error(f"[JOB] Product scrape failed: {e}")


def profile_refresh_job():
    """Refresh wardrobe gap analysis for all users."""
    log.info("[JOB] Starting profile refresh...")
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "shaaru")
        db = MongoClient(uri, serverSelectionTimeoutMS=3000)[db_name]

        users = db["comfort_profiles"].find({}, {"user_id": 1})
        refreshed = 0

        for user_doc in users:
            user_id = user_doc.get("user_id")
            if not user_id:
                continue
            try:
                from opportunity_detector import detect_gaps
                gaps = detect_gaps(user_id)
                if gaps:
                    db["comfort_profiles"].update_one(
                        {"user_id": user_id},
                        {"$set": {"wardrobe_gaps": gaps}},
                    )
                    refreshed += 1
            except Exception as e:
                log.warning(f"[JOB] Gap refresh failed for {user_id}: {e}")

        log.info(f"[JOB] Profile refresh complete: {refreshed} users updated")
    except Exception as e:
        log.error(f"[JOB] Profile refresh failed: {e}")


def trend_refresh_job():
    """Refresh trending aesthetics data."""
    log.info("[JOB] Starting trend refresh...")
    try:
        from trend_watcher import fetch_trends, get_trending_aesthetics
        trends = fetch_trends()
        aesthetics = get_trending_aesthetics()
        log.info(
            f"[JOB] Trend refresh complete: {len(trends)} articles, "
            f"{len(aesthetics)} trending aesthetics"
        )
    except Exception as e:
        log.error(f"[JOB] Trend refresh failed: {e}")


def start_scheduler():
    """Initialize and start the APScheduler with all jobs."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError:
        log.error("[SCHEDULER] APScheduler not installed. pip install apscheduler")
        return None

    scheduler = BackgroundScheduler()

    # Product scrape: every 24 hours
    scheduler.add_job(
        scrape_products_job,
        "interval",
        hours=24,
        id="scrape_products",
        name="Scrape fashion brand products",
    )

    # Profile refresh: every 6 hours
    scheduler.add_job(
        profile_refresh_job,
        "interval",
        hours=6,
        id="profile_refresh",
        name="Refresh user profiles and wardrobe gaps",
    )

    # Trend refresh: every 12 hours
    scheduler.add_job(
        trend_refresh_job,
        "interval",
        hours=12,
        id="trend_refresh",
        name="Refresh fashion trends",
    )

    scheduler.start()
    log.info("[SCHEDULER] Background scheduler started with 3 jobs")
    return scheduler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running all jobs once...")
    scrape_products_job()
    profile_refresh_job()
    trend_refresh_job()
    print("Done.")
