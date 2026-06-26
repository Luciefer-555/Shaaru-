"""
database.py — MongoDB connection singleton.

Provides a get_db() helper used throughout the SHAARU backend.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.db")

_db = None


def get_db():
    """
    Get the MongoDB database object (singleton).

    Uses MONGODB_URI and MONGODB_DB from environment.
    Returns None if connection fails.
    """
    global _db
    if _db is None:
        try:
            from pymongo import MongoClient
            uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGODB_DB", "shaaru")
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            _db = client[db_name]
            log.info(f"[DB] Connected to MongoDB: {db_name}")
        except Exception as e:
            log.error(f"[DB] MongoDB connection failed: {e}")
            return None
    return _db
