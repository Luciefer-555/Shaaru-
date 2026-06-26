import os
from pymongo import MongoClient

_fabric_cache = None
_embellishment_cache = None

def get_fabric_list():
    global _fabric_cache
    if _fabric_cache is not None:
        return _fabric_cache
        
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    _fabric_cache = db.fabric_intelligence.distinct("fabric_id")
    return _fabric_cache

def get_embellishment_list():
    global _embellishment_cache
    if _embellishment_cache is not None:
        return _embellishment_cache
        
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB")]
    _embellishment_cache = db.embellishment_sourcing.distinct("embellishment_id")
    return _embellishment_cache
