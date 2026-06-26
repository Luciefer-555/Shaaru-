import os
import sys
import time
import schedule
import asyncio
import datetime

pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir = os.path.abspath(os.path.join(pipeline_dir, ".."))
if pipeline_dir not in sys.path: sys.path.append(pipeline_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

from pipeline.trend_watch.trend_detector import TrendDetector
from pipeline.api.on_demand_extractor import OnDemandExtractor


class TrendScheduler:
    """
    Runs trend detection automatically.
    No human needed.
    """
    
    def __init__(self):
        self.detector = TrendDetector()
        self.extractor = OnDemandExtractor()
    
    def start(self):
        # Run daily at 6 AM
        schedule.every().day.at("06:00").do(
            lambda: asyncio.run(self.daily_trend_scan())
        )
        
        # Check for fashion week (weekly)
        schedule.every().monday.at("09:00").do(
            lambda: asyncio.run(self.fashion_week_scan())
        )
        
        print("[OK] Proactive TrendScheduler booted — monitoring editorial & Shopify arrivals")
        while True:
            schedule.run_pending()
            time.sleep(60)
            
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
            print(f"[TrendScheduler] Neo4j check error: {e}")
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
            print(f"[TrendScheduler] Log error: {e}")

    async def fashion_week_scan(self):
        print("Running weekly fashion week scan...")
        trends = await self.detector.detect_trends()
        for t in trends[:5]:
            self._log_trend(t)

    async def daily_trend_scan(self):
        print("Running daily trend scan...")
        
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
        
        print(f"Trend scan complete — {len(trends)} trends processed")


if __name__ == "__main__":
    scheduler = TrendScheduler()
    scheduler.start()
