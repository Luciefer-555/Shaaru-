"""
Stores trend history in MongoDB.
Tracks what's trending, when it peaked,
and what Shaaru extracted for it.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

_trends_col = None

def _get_col():
    global _trends_col
    if _trends_col is None:
        from pymongo import MongoClient
        client = MongoClient(os.environ["MONGODB_URI"])
        db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
        _trends_col = db["trend_history"]
    return _trends_col


def save_trend(trend: dict):
    col = _get_col()
    trend["detected_at"] = datetime.utcnow().isoformat()
    trend["extracted"] = False
    col.insert_one(trend)


def mark_extracted(trend_name: str):
    col = _get_col()
    col.update_one(
        {"trend_name": trend_name},
        {"$set": {
            "extracted": True,
            "extracted_at": datetime.utcnow().isoformat()
        }}
    )


def trend_already_known(trend_name: str) -> bool:
    col = _get_col()
    return col.find_one({
        "trend_name": trend_name,
        "extracted": True
    }) is not None


def get_active_trends() -> list:
    col = _get_col()
    return list(col.find(
        {"velocity": {"$in": ["rising", "peak"]}},
        {"_id": 0}
    ).sort("confidence", -1).limit(20))
