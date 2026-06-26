import json
from datetime import datetime, timezone
from shaaru_brain import _get_db, nvidia_call, _get_client

def evaluate_riley_response(user_message: str, riley_response: str, user_profile: dict, context: dict) -> dict:
    prompt = f"""You are evaluating a fashion AI stylist's response quality.

User message: {user_message}
Riley's response: {riley_response}
User profile summary: {json.dumps(user_profile)}

Grade Riley on:
1. Style accuracy (did she recommend the right aesthetic for this user?) 
2. Voice consistency (warm, direct, bestie-coded — not generic)
3. Actionability (can the user act on this recommendation immediately?)
4. Personalization (did she reference the user's specific profile?)
5. Trend relevance (did she connect to what's current?)

Return ONLY valid JSON:
{{
  "scores": {{
    "style_accuracy": 0,
    "voice_consistency": 0,
    "actionability": 0,
    "personalization": 0,
    "trend_relevance": 0
  }},
  "overall": 0,
  "weakest_dimension": "",
  "improvement_note": "one specific thing Riley should do differently next time",
  "flag_for_review": false
}}"""

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
        
        doc = {
            "user_message": user_message,
            "riley_response": riley_response[:500],
            "scores": result.get("scores", {}),
            "overall": result.get("overall", 0),
            "improvement_note": result.get("improvement_note", ""),
            "flag_for_review": result.get("flag_for_review", False),
            "evaluated_at": datetime.now(timezone.utc)
        }
        
        db = _get_db()
        if db is not None:
            db["riley_evaluations"].insert_one(doc)
            
        return result
    except Exception as e:
        print(f"[FAIL] Riley evaluation failed: {e}")
        return {}

def get_riley_performance_summary() -> dict:
    db = _get_db()
    if db is None:
        return {}
        
    cursor = db["riley_evaluations"].find({}).sort("evaluated_at", -1).limit(100)
    evals = list(cursor)
    
    if not evals:
        return {
            "avg_overall": 0.0,
            "avg_by_dimension": {},
            "flagged_count": 0,
            "weakest_dimension": "",
            "sample_size": 0
        }
        
    total_overall = 0
    total_scores = {
        "style_accuracy": 0,
        "voice_consistency": 0,
        "actionability": 0,
        "personalization": 0,
        "trend_relevance": 0
    }
    flagged_count = 0
    
    for e in evals:
        total_overall += e.get("overall", 0)
        scores = e.get("scores", {})
        for k in total_scores.keys():
            total_scores[k] += scores.get(k, 0)
            
        if e.get("flag_for_review", False):
            flagged_count += 1
            
    n = len(evals)
    avg_by_dimension = {k: v / n for k, v in total_scores.items()}
    weakest = min(avg_by_dimension, key=avg_by_dimension.get) if avg_by_dimension else ""
    
    return {
        "avg_overall": total_overall / n,
        "avg_by_dimension": avg_by_dimension,
        "flagged_count": flagged_count,
        "weakest_dimension": weakest,
        "sample_size": n
    }
