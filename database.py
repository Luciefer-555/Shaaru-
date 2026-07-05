"""
database.py — MongoDB connection singleton.
"""
import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.db")

_client = None

def get_db():
    global _client
    if _client is None:
        try:
            uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGODB_DB", "shaaru")
            _client = MongoClient(
                uri,
                maxPoolSize=50,
                minPoolSize=5,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000
            )
            log.info(f"[DB] Connected to MongoDB pool: {db_name}")
            return _client[db_name]
        except Exception as e:
            log.error(f"[DB] MongoDB connection failed: {e}")
            return None
    db_name = os.getenv("MONGODB_DB", "shaaru")
    return _client[db_name]
