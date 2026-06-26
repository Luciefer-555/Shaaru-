"""
Extracts products for detected trends.
Uses on-demand extractor under the hood.
"""

import asyncio
from pipeline.on_demand.extractor import handle_user_query
from pipeline.trend_watch.trend_store import (
    save_trend, mark_extracted, trend_already_known
)


async def extract_for_trend(trend: dict):
    """
    Extracts products for one detected trend.
    Runs on-demand extractor for each suggested product.
    """
    trend_name = trend["trend_name"]
    
    if trend_already_known(trend_name):
        print(f"Already know about: {trend_name}")
        return
    
    save_trend(trend)
    print(f"Extracting for trend: {trend_name}")
    
    count = trend.get("suggested_extraction_count", 5)
    designers = trend.get("suggested_designers", [])
    techniques = trend.get("techniques_involved", [])
    aesthetic = trend.get("aesthetics_involved", [None])[0]
    
    extracted = 0
    for designer_id in designers:
        if extracted >= count:
            break
        
        for i in range(min(3, count - extracted)):
            try:
                result = await handle_user_query(
                    query=trend_name,
                    techniques=techniques,
                    aesthetic=aesthetic,
                    designer_id=designer_id
                )
                
                if result.get("status") in ("freshly_extracted", "instant"):
                    extracted += 1
                    print(f"  [OK] Extracted {extracted}/{count} for {trend_name}")
                
                await asyncio.sleep(5)
            except Exception as e:
                print(f"  [FAILED] Extraction failed: {e}")
    
    mark_extracted(trend_name)
    print(f"Trend '{trend_name}' fully extracted: {extracted} products")
