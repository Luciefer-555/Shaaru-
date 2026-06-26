"""
Aggregates signals from all sources.
Scores and ranks trends by confidence and velocity.
Uses LLM to extract structured trend data from raw signals.
"""

import asyncio
import json
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from pipeline.trend_watch.sources.editorial import scan_editorial
from pipeline.trend_watch.sources.new_arrivals import scan_new_arrivals
from pipeline.trend_watch.sources.fashion_week import scan_fashion_week

AESTHETIC_CATEGORIES = [
    "Heritage Couture", "Handloom Minimal",
    "Folk Maximalist", "Bollywood Glam",
    "Contemporary Indie", "Artisan Craft",
    "Festive Occasion", "Avant-Garde",
    "Graphic Pop Indian", "Mirror Maximalism"
]

KNOWN_TECHNIQUES = [
    "mirror work", "sheesha", "zardozi", "resham",
    "kantha", "block print", "natural dye", "gota patti",
    "dabka", "chikankari", "phulkari", "sequin",
    "crystal", "zari", "bandhani", "ikat"
]

KNOWN_DESIGNERS = [
    "Abhinav Mishra", "Sabyasachi", "Raw Mango",
    "House of Masaba", "Torani", "Anavila",
    "Pero", "Rimzim Dadu", "Anita Dongre", "Injiri"
]


async def detect_trends() -> list:
    """
    Main trend detection function.
    Aggregates all sources and returns scored trends.
    """
    print("Scanning editorial sources...")
    editorial_signals = scan_editorial()
    
    print("Scanning new arrivals...")
    arrivals_signals = scan_new_arrivals()
    
    print("Scanning fashion week coverage...")
    week_signals = scan_fashion_week()
    
    all_signals = editorial_signals + arrivals_signals + week_signals
    print(f"Total signals collected: {len(all_signals)}")
    
    trends = await _extract_trends_from_signals(all_signals)
    scored = _score_trends(trends, all_signals)
    return scored


async def _extract_trends_from_signals(signals: list) -> list:
    """
    Uses LLM to extract structured trend data from raw editorial and arrival signals.
    """
    signal_text = ""
    for s in signals[:30]:
        signal_text += f"""
Source: {s.get('source', '')}
Title: {s.get('title', '')}
Content: {s.get('content', '')[:300]}
---
"""
    
    prompt = f"""
You are analyzing Indian fashion signals to detect trends.

Known aesthetic categories: {json.dumps(AESTHETIC_CATEGORIES)}
Known techniques: {json.dumps(KNOWN_TECHNIQUES)}
Known designers: {json.dumps(KNOWN_DESIGNERS)}

Analyze these fashion signals and extract emerging trends:
{signal_text}

Return a JSON array of detected trends.
Start with [ and end with ].
No markdown. No preamble.

Each trend object:
{{
  "trend_name": "specific descriptive name",
  "trend_type": "technique|silhouette|aesthetic|collection|color",
  "description": "one sentence what this trend is",
  "techniques_involved": [],
  "aesthetics_involved": [],
  "designers_involved": [],
  "occasions": [],
  "velocity": "rising|peak|declining",
  "confidence": 0.0-1.0,
  "why_trending": "one sentence explanation",
  "suggested_extraction_count": 5,
  "suggested_designers": []
}}
"""
    try:
        from trend_ingestion import _call_nvidia, _parse_json_response
        response = await asyncio.to_thread(_call_nvidia, prompt, 2000, "meta/llama-3.1-8b-instruct")
        parsed = _parse_json_response(response)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "trends" in parsed:
            return parsed["trends"]
        elif isinstance(parsed, dict):
            return [parsed]
    except Exception as e:
        print(f"Trend extraction LLM failed: {e}")
        
    return _extract_from_arrivals(signals)


def _extract_from_arrivals(signals: list) -> list:
    trends = []
    for signal in signals:
        if signal.get("signal_type") != "new_collection":
            continue
        
        trends.append({
            "trend_name": f"new {signal['designer_name']} collection",
            "trend_type": "collection",
            "description": f"{signal['designer_name']} launched {signal['new_product_count']} new products",
            "techniques_involved": signal.get("techniques_detected", []),
            "aesthetics_involved": [signal.get("aesthetic_hint", "")],
            "designers_involved": [signal["designer_name"]],
            "velocity": "rising",
            "confidence": signal.get("confidence", 0.7),
            "suggested_extraction_count": 10,
            "suggested_designers": [signal["designer_id"]]
        })
    return trends


def _score_trends(trends: list, signals: list) -> list:
    for trend in trends:
        trend_name_lower = trend["trend_name"].lower()
        supporting_signals = sum(
            1 for s in signals
            if trend_name_lower in (s.get("content", "") + s.get("title", "")).lower()
        )
        
        trend["supporting_signal_count"] = supporting_signals
        if supporting_signals >= 3:
            trend["confidence"] = min(trend.get("confidence", 0.5) + 0.15, 1.0)
        elif supporting_signals >= 2:
            trend["confidence"] = min(trend.get("confidence", 0.5) + 0.08, 1.0)
            
    trends.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return [t for t in trends if t.get("confidence", 0) > 0.6]
