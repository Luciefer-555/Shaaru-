"""
tailor_db.py
SHAARU — Tailor session database helpers
"""
from datetime import datetime, timezone
from bson import ObjectId

def get_session(session_id: str, db) -> dict | None:
    try:
        return db["tailor_sessions"].find_one({"_id": ObjectId(session_id)})
    except Exception:
        return None

def update_session(session_id: str, updates: dict, db) -> bool:
    try:
        db["tailor_sessions"].update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {**updates, "updated_at": datetime.now(timezone.utc)}}
        )
        return True
    except Exception as e:
        print(f"[FAIL] update_session: {e}")
        return False

def save_project(user_id: str, brief: dict, session_id: str, db) -> str:
    """Saves completed tailor brief as a Project in MongoDB."""
    project = {
        "user_id": user_id,
        "session_id": session_id,
        "garment_name": brief.get("garment_name"),
        "status": "pending_tailor",
        "brief": brief,
        "created_at": datetime.now(timezone.utc)
    }
    result = db["projects"].insert_one(project)
    return str(result.inserted_id)
