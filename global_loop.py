import json
from datetime import datetime, timezone
from shaaru_brain import _get_db, nvidia_call, _get_client

def check_and_trigger_global_loop() -> bool:
    db = _get_db()
    if db is None:
        return False
        
    doc = db["rsi_counters"].find_one_and_update(
        {"_id": "interaction_count"},
        {"$inc": {"count": 1}},
        upsert=True,
        return_document=True
    )
    
    count = doc.get("count", 1)
    if count % 500 == 0:
        run_global_loop()
        return True
    return False

def run_global_loop() -> dict:
    try:
        from pattern_detector import detect_cross_user_patterns
        detect_cross_user_patterns()
    except Exception as e:
        print(f"[WARN] pattern_detector failed: {e}")
        
    try:
        from knowledge_graph import kg
        patterns_json = "[]"
        if kg and kg.is_connected:
            result = kg.query("MATCH (p:Pattern) WHERE p.confidence > 0.7 RETURN p LIMIT 20")
            patterns = [record["p"] for record in result]
            patterns_json = json.dumps(patterns, default=str)
    except Exception as e:
        patterns_json = "[]"
        print(f"[WARN] Neo4j query failed: {e}")

    prompt = f"""You are SHAARU's trend intelligence engine.
These cross-user behavioral patterns were detected this cycle:
{patterns_json}

Identify:
1. Emerging style micro-trends (min 3 users showing same behavior)
2. Product categories gaining momentum  
3. Aesthetic clusters forming

Return ONLY valid JSON:
{{
  "micro_trends": [{{"name": "", "signal_strength": 0.0, "user_count": 0, "description": ""}}],
  "momentum_categories": [""],
  "aesthetic_clusters": [{{"name": "", "defining_traits": [""]}}],
  "loop_timestamp": ""
}}"""

    try:
        client = _get_client()
        response_text = nvidia_call(
            client=client,
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
        result = json.loads(json_str)
        result["loop_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        db = _get_db()
        if db is not None:
            db["rsi_global_loops"].insert_one(result)
            print(f"[GLOBAL LOOP FIRED] at cycle.")
            print("[OK] Global loop finished successfully")
            return result
    except Exception as e:
        print(f"[FAIL] Global loop LLM processing failed: {e}")
    
    return {}
