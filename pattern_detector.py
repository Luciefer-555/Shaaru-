"""
pattern_detector.py
SHAARU — RSI Global Pattern Detection

Runs every 500 interactions across all users.
Finds behavioral patterns in Neo4j graph.
Writes patterns back as weighted edges.

Called automatically by signal_collector.py.
Run manually: python pattern_detector.py
"""

import logging
from datetime import datetime, timezone
from shaaru_brain import _get_db
from knowledge_graph import get_kg

log = logging.getLogger("shaaru.patterns")

def run_pattern_detection():
    print("[RSI] Pattern detection triggered")
    save_written = _find_save_patterns()
    skip_written = _find_skip_patterns()
    print(f"[OK] Pattern detection complete — {save_written + skip_written} patterns written")

def _find_save_patterns() -> int:
    if not get_kg().is_connected:
        return 0
        
    cypher = """
    MATCH (u:User)-[:SAVED]->(p:Product)
    WHERE u.monk_scale IS NOT NULL AND u.body_type IS NOT NULL
    RETURN u.monk_scale AS monk,
           u.body_type AS body_type,
           p.aesthetic AS aesthetic,
           count(*) AS save_count
    ORDER BY save_count DESC
    LIMIT 20
    """
    results = get_kg().query(cypher)
    written = 0
    now = datetime.now(timezone.utc).isoformat()
    
    for r in results:
        save_count = r.get("save_count", 0)
        if save_count >= 3:
            write_cypher = """
            MERGE (profile:StyleProfile {
                monk_scale: $monk, 
                body_type: $body_type
            })
            MERGE (a:Aesthetic {name: $aesthetic})
            MERGE (profile)-[r:RESPONDS_TO]->(a)
            SET r.save_rate = $save_count,
                r.last_updated = $now
            """
            get_kg().query(write_cypher, {
                "monk": r["monk"],
                "body_type": r["body_type"],
                "aesthetic": r["aesthetic"],
                "save_count": save_count,
                "now": now
            })
            written += 1
            
    log.info(f"[OK] _find_save_patterns wrote {written}")
    return written

def _find_skip_patterns() -> int:
    if not get_kg().is_connected:
        return 0
        
    cypher = """
    MATCH (u:User)-[:SKIPPED]->(p:Product)
    WHERE u.monk_scale IS NOT NULL AND u.body_type IS NOT NULL
    RETURN u.monk_scale AS monk,
           u.body_type AS body_type,
           p.aesthetic AS aesthetic,
           count(*) AS skip_count
    ORDER BY skip_count DESC
    LIMIT 20
    """
    results = get_kg().query(cypher)
    written = 0
    now = datetime.now(timezone.utc).isoformat()
    
    for r in results:
        skip_count = r.get("skip_count", 0)
        if skip_count >= 3:
            write_cypher = """
            MERGE (profile:StyleProfile {
                monk_scale: $monk, 
                body_type: $body_type
            })
            MERGE (a:Aesthetic {name: $aesthetic})
            MERGE (profile)-[r:SKIPPED]->(a)
            SET r.skip_rate = $skip_count,
                r.last_updated = $now
            """
            get_kg().query(write_cypher, {
                "monk": r["monk"],
                "body_type": r["body_type"],
                "aesthetic": r["aesthetic"],
                "skip_count": skip_count,
                "now": now
            })
            written += 1
            
    log.info(f"[OK] _find_skip_patterns wrote {written}")
    return written

def get_pattern_context(user_id: str) -> str:
    db = _get_db()
    if db is None or not get_kg().is_connected:
        return ""
        
    user = db["users"].find_one({"user_id": user_id})
    if not user:
        return ""
        
    # Get physical attributes, handle both formats
    visual = user.get("visual", {})
    monk = visual.get("monk_scale") or user.get("monk_scale")
    body_type = user.get("body_type")
    
    if not monk or not body_type:
        return ""
        
    cypher = """
    MATCH (p:StyleProfile {monk_scale: $monk, body_type: $body})-[r:RESPONDS_TO]->(a:Aesthetic)
    WHERE r.save_rate > 2
    RETURN a.name AS aesthetic, r.save_rate AS score
    ORDER BY score DESC LIMIT 5
    """
    results = get_kg().query(cypher, {"monk": monk, "body": body_type})
    if not results:
        return ""
        
    parts = [f"{r['aesthetic']} ({r['score']} saves)" for r in results]
    lines = [f"Profile patterns: [{', '.join(parts)}]"]
    
    # Check style evolution
    drift = detect_style_evolution(user_id, db)
    if drift:
        lines.append(
            f"⚠ TASTE DRIFT DETECTED: User evolved from "
            f"{drift['onboarding_aesthetic']} → "
            f"{drift['current_behavioral_aesthetic']}. "
            f"Adjust recommendations accordingly."
        )
    return "\n".join(lines)

def detect_style_evolution(user_id: str, db) -> dict | None:
    """
    Compares user's current behavioral signals against their onboarding aesthetic.
    Returns drift report if taste has evolved significantly.
    
    Called after every 10 sessions for a user.
    """
    try:
        from knowledge_graph import get_kg
        
        # Get onboarding aesthetic from MongoDB
        user = db["users"].find_one({"user_id": user_id})
        if not user:
            return None
            
        onboarding_aesthetic = (
            user.get("style_equation", {}).get("primary_aesthetic") or
            user.get("taste", {}).get("everyday", [None])[0]
        )
        if not onboarding_aesthetic:
            return None
        
        # Get top 3 aesthetics from recent behavioral signals in Neo4j
        results = get_kg().query("""
            MATCH (u:User {user_id: $user_id})-[r:SAVED]->(p:Product)
            WHERE p.aesthetic IS NOT NULL
            RETURN p.aesthetic AS aesthetic, count(r) AS save_count
            ORDER BY save_count DESC
            LIMIT 3
        """, {"user_id": user_id})
        
        if not results:
            return None
        
        top_behavioral_aesthetic = results[0]['aesthetic']
        
        # Check for drift
        drifted = top_behavioral_aesthetic.lower() != onboarding_aesthetic.lower()
        
        if drifted:
            drift_report = {
                "user_id": user_id,
                "onboarding_aesthetic": onboarding_aesthetic,
                "current_behavioral_aesthetic": top_behavioral_aesthetic,
                "all_recent": [r['aesthetic'] for r in results],
                "drift_detected": True,
                "detected_at": datetime.utcnow().isoformat()
            }
            
            # Save drift report to MongoDB
            db["style_evolution"].update_one(
                {"user_id": user_id},
                {"$set": drift_report},
                upsert=True
            )
            
            # Write evolution edge to Neo4j
            get_kg().query("""
                MATCH (u:User {user_id: $user_id})
                MERGE (a:Aesthetic {name: $new_aesthetic})
                MERGE (u)-[r:EVOLVED_TO]->(a)
                SET r.from_aesthetic = $old_aesthetic,
                    r.detected_at = $now
            """, {
                "user_id": user_id,
                "new_aesthetic": top_behavioral_aesthetic,
                "old_aesthetic": onboarding_aesthetic,
                "now": datetime.utcnow().isoformat()
            })
            
            print(f"[RSI] Style drift detected for {user_id}: "
                  f"{onboarding_aesthetic} → {top_behavioral_aesthetic}")
            return drift_report
        
        print(f"[RSI] No style drift for {user_id} — still {onboarding_aesthetic}")
        return None
        
    except Exception as e:
        print(f"[FAIL] detect_style_evolution: {e}")
        return None

if __name__ == "__main__":
    run_pattern_detection()
