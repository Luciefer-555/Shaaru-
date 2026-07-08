import os
import sys
import time
import schedule
import asyncio
import datetime
import logging

pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir = os.path.abspath(os.path.join(pipeline_dir, ".."))
if pipeline_dir not in sys.path: sys.path.append(pipeline_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from pipeline.trend_watch.trend_detector import TrendDetector
from pipeline.api.on_demand_extractor import OnDemandExtractor

log = logging.getLogger("shaaru.scheduler")


class TrendScheduler:
    """
    Runs trend detection automatically.
    No human needed.
    """
    
    def __init__(self):
        self.detector = TrendDetector()
        self.extractor = OnDemandExtractor()
    
    def _run_trend_ingestion(self):
        log.info("[SCHEDULER] Firing scheduled trend/styling ingestion (trend_ingestion.py)...")
        try:
            from trend_ingestion import run_pipeline
            res = run_pipeline()
            log.info(f"[SCHEDULER] Scheduled trend ingestion completed successfully: {res}")
        except Exception as e:
            log.error(f"[SCHEDULER] Scheduled trend ingestion failed: {e}", exc_info=True)

    def start(self, stop_event=None):
        log.info("[SCHEDULER] Initializing TrendScheduler jobs...")
        # Clear existing jobs on start to prevent duplicate schedules on restart/redeploy
        schedule.clear()

        # Run daily at 6 AM
        schedule.every().day.at("06:00").do(
            lambda: asyncio.run(self.daily_trend_scan())
        )
        schedule.every().day.at("06:00").do(self._run_trend_ingestion)
        
        # Check for fashion week (weekly)
        schedule.every().monday.at("09:00").do(
            lambda: asyncio.run(self.fashion_week_scan())
        )
        schedule.every().monday.at("09:00").do(self._run_trend_ingestion)
        
        log.info("[OK] Proactive TrendScheduler booted — monitoring editorial, Shopify arrivals & trend ingestion")
        while True:
            if stop_event and stop_event.is_set():
                log.info("[SCHEDULER] Stop event detected, exiting scheduler loop cleanly.")
                break
            try:
                schedule.run_pending()
            except Exception as e:
                log.error(f"[SCHEDULER] Unhandled error during schedule.run_pending(): {e}", exc_info=True)
            # Sleep in 1-second increments up to 60s for responsive shutdown
            for _ in range(60):
                if stop_event and stop_event.is_set():
                    break
                time.sleep(1)
            
    def _check_neo4j(self, trend: dict) -> bool:
        try:
            from knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            if kg.is_connected:
                with kg.driver.session() as s:
                    res = s.run(
                        "MATCH (t:Technique) WHERE toLower(t.name) = toLower($name) RETURN t LIMIT 1",
                        name=trend["trend_name"]
                    )
                    return bool(list(res))
        except Exception as e:
            log.warning(f"[TrendScheduler] Neo4j check error: {e}")
        return False

    def _log_trend(self, trend: dict):
        try:
            from shaaru_brain import _get_db
            db = _get_db()
            if db:
                db['trend_history'].insert_one({
                    **trend,
                    "processed_at": datetime.datetime.now(datetime.timezone.utc)
                })
        except Exception as e:
            log.error(f"[TrendScheduler] Log error: {e}")

    async def fashion_week_scan(self):
        log.info("Running weekly fashion week scan...")
        trends = await self.detector.detect_trends()
        for t in trends[:5]:
            self._log_trend(t)

    async def daily_trend_scan(self):
        log.info("Running daily trend scan...")
        
        # Detect what's trending
        trends = await self.detector.detect_trends()
        
        # For each trend with confidence > 0.7
        for trend in trends:
            if trend.get("confidence", 0) < 0.7:
                continue
            
            # Check if Shaaru already knows about it
            already_known = self._check_neo4j(trend)
            if already_known:
                continue
            
            # Extract products for this trend
            for designer in trend.get("suggested_designers", []):
                await self.extractor.extract_for_trend(
                    trend=trend,
                    designer_id=designer,
                    count=trend.get("suggested_extraction_count", 5)
                )
            
            # Log to trend_history
            self._log_trend(trend)
        
        log.info(f"Trend scan complete — {len(trends)} trends processed")


if __name__ == "__main__":
    scheduler = TrendScheduler()
    scheduler.start()
