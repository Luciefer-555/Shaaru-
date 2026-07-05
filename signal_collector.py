"""
signal_collector.py
SHAARU — RSI Signal Collection Layer

Captures user interaction signals and writes them to:
  1. MongoDB sessions collection (signals.saved/skipped/purchased arrays)
  2. Neo4j behavioral edges (User)-[:SAVED|SKIPPED|PURCHASED]->(Product)

Import: from signal_collector import collect_signal
"""

import logging
import threading
from datetime import datetime, timezone
from shaaru_brain import _get_db
from knowledge_graph import get_kg

log = logging.getLogger("shaaru.signals")

def _check_global_loop_trigger(db) -> None:
    """Fire pattern detection every 500 total interactions."""
    try:
        # Count total signal edges across all users in Neo4j
        from knowledge_graph import get_kg
        result = get_kg().query("""
            MATCH ()-[r:SAVED|SKIPPED|PURCHASED]->()
            RETURN count(r) as total
        """)
        total = result[0]['total'] if result else 0
        
        # Fire every 500 interactions
        if total > 0 and total % 500 == 0:
            print(f"[RSI] Global loop triggered at {total} interactions")
            from pattern_detector import run_pattern_detection
            import threading
            threading.Thread(
                target=run_pattern_detection, 
                daemon=True,
                name="rsi_global_loop"
            ).start()
        else:
            print(f"[RSI] Interaction count: {total} (next trigger at {((total // 500) + 1) * 500})")
    except Exception as e:
        print(f"[FAIL] Global loop check: {e}")

def collect_signal(user_id: str, signal_type: str, product_id: str, metadata: dict = None):
    if metadata is None:
        metadata = {}
        
    try:
        db = _get_db()
        if db is None:
            return

        now = datetime.now(timezone.utc)
        
        # 1. MongoDB Session Update
        session = db["sessions"].find_one(
            {"user_id": user_id, "ended_at": None},
            sort=[("started_at", -1)]
        )
        
        if session:
            db["sessions"].update_one(
                {"_id": session["_id"]},
                {"$push": {f"signals.{signal_type}": product_id}}
            )
        else:
            new_session = {
                "user_id": user_id,
                "started_at": now,
                "ended_at": None,
                "signals": {
                    "saved": [],
                    "skipped": [],
                    "purchased": []
                }
            }
            new_session["signals"][signal_type].append(product_id)
            db["sessions"].insert_one(new_session)

        # 2. Neo4j Edge
        if get_kg().is_connected:
            rel_type = signal_type.upper()
            if rel_type in ["SAVED", "SKIPPED", "PURCHASED"]:
                cypher = f"""
                MERGE (u:User {{user_id: $user_id}})
                MERGE (p:Product {{product_id: $product_id}})
                MERGE (u)-[r:{rel_type} {{type: $signal_type}}]->(p)
                SET r.timestamp = $now,
                    r.count = coalesce(r.count, 0) + 1
                """
                get_kg().query(cypher, {
                    "user_id": user_id,
                    "product_id": product_id,
                    "signal_type": signal_type,
                    "now": now.isoformat()
                })

        # 3. Trigger pattern detection every 500 interactions
        # Check interaction count: db["sessions"].count_documents({"user_id": {"$exists": True}})
        # if total signals across all sessions hits 500, trigger pattern detection:
        interaction_count = db["sessions"].count_documents({"user_id": {"$exists": True}})
        if interaction_count > 0 and interaction_count % 500 == 0:
            from pattern_detector import run_pattern_detection
            threading.Thread(target=run_pattern_detection, daemon=True).start()

        log.info(f"[OK] Signal: {signal_type} for {user_id} -> {product_id}")
        _check_global_loop_trigger(db)

    except Exception as e:
        log.error(f"[FAIL] Signal collection failed: {e}")

def get_user_signals(user_id: str, limit: int = 10) -> dict:
    try:
        db = _get_db()
        if db is None:
            return {"saved": [], "skipped": [], "purchased": []}
            
        sessions = list(db["sessions"].find(
            {"user_id": user_id},
            {"signals": 1}
        ).sort("started_at", -1).limit(limit))
        
        res = {"saved": [], "skipped": [], "purchased": []}
        for s in sessions:
            sig = s.get("signals", {})
            res["saved"].extend(sig.get("saved", []))
            res["skipped"].extend(sig.get("skipped", []))
            res["purchased"].extend(sig.get("purchased", []))
            
        return res
    except Exception as e:
        log.error(f"[FAIL] get_user_signals: {e}")
        return {"saved": [], "skipped": [], "purchased": []}
