import json
from datetime import datetime, timezone
from shaaru_brain import _get_db, nvidia_call, _get_client

def record_style_snapshot(user_id: str, profile: dict) -> bool:
    db = _get_db()
    if db is None:
        return False
        
    doc = {
        "user_id": user_id,
        "snapshot_at": datetime.now(timezone.utc),
        "style_equation": profile.get("style_equation", {}),
        "active_aesthetics": profile.get("active_aesthetics", []),
        "top_saved_categories": profile.get("top_saved", []),
        "occasion_focus": profile.get("occasion_focus", "")
    }
    
    try:
        db["style_evolution"].insert_one(doc)
        return True
    except Exception as e:
        print(f"[FAIL] Failed to save style snapshot: {e}")
        return False

def detect_style_drift(user_id: str) -> dict:
    db = _get_db()
    if db is None:
        return {"drift_detected": False, "reason": "db_error"}
        
    cursor = db["style_evolution"].find({"user_id": user_id}).sort("snapshot_at", -1).limit(5)
    snapshots = list(cursor)
    
    if len(snapshots) < 2:
        return {"drift_detected": False, "reason": "insufficient_history"}
        
    for s in snapshots:
        if "_id" in s:
            s["_id"] = str(s["_id"])
        if "snapshot_at" in s and isinstance(s["snapshot_at"], datetime):
            s["snapshot_at"] = s["snapshot_at"].isoformat()
            
    snapshots_json = json.dumps(snapshots, indent=2)
    
    prompt = f"""Compare these sequential style snapshots for a fashion user and detect meaningful evolution.
Snapshots (newest first):
{snapshots_json}

Return ONLY valid JSON:
{{
  "drift_detected": true,
  "drift_type": "gradual_shift",
  "from_aesthetic": "",
  "to_aesthetic": "",
  "confidence": 0.0,
  "riley_note": "one sentence Riley would say to the user about their evolving style"
}}
Set drift_detected to false if no meaningful drift.
"""

    try:
        client = _get_client()
        response_text = nvidia_call(
            client=client,
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
        result = json.loads(json_str)
        
        if result.get("drift_detected"):
            print(f"[DRIFT DETECTED] for user {user_id}")
            
        return result
    except Exception as e:
        print(f"[FAIL] Style drift detection failed: {e}")
        return {"drift_detected": False, "reason": "llm_error"}

def get_evolution_summary(user_id: str) -> str:
    db = _get_db()
    if db is None:
        return ""
        
    cursor = db["style_evolution"].find({"user_id": user_id}).sort("snapshot_at", -1).limit(3)
    snapshots = list(cursor)
    
    if len(snapshots) < 2:
        return ""
        
    drift_result = detect_style_drift(user_id)
    if drift_result.get("drift_detected"):
        from_aes = drift_result.get("from_aesthetic", "unknown")
        to_aes = drift_result.get("to_aesthetic", "unknown")
        drift_type = drift_result.get("drift_type", "shift")
        note = drift_result.get("riley_note", "")
        return f"Style journey: {from_aes} -> {to_aes} ({drift_type}). {note}"
        
    return ""
