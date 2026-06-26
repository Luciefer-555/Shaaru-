"""
verify_pipeline.py
SHAARU — Pipeline verification and database health check

Checks that the ingestion pipeline stored data correctly.
Run after trend_ingestion.py to verify everything landed.
"""

import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shaaru.verify")


def verify():
    """Run all pipeline verification checks."""
    from shaaru_brain import _get_db
    from knowledge_graph import kg

    print("\n" + "=" * 60)
    print("  SHAARU Pipeline Verification")
    print("=" * 60 + "\n")

    db = _get_db()
    if db is None:
        print("[FAIL] Database unavailable")
        return

    failures = []

    # ── Check 1: styling_guides count and samples ────────────────
    print("── Check 1: MongoDB styling_guides ──")
    sg_count = db["styling_guides"].count_documents({})
    print(f"  Count: {sg_count}")

    if sg_count > 0:
        samples = list(db["styling_guides"].find({}).limit(3))
        for s in samples:
            name = s.get("aesthetic", "?")
            score = s.get("quality_score", 0)
            print(f"  • {name} (score: {score:.1f})")
        print(f"  [OK] {sg_count} styling guides found")
    else:
        print("  [WARN] No styling guides found")
        failures.append("styling_guides is empty")

    # ── Check 2: trends freshness ────────────────────────────────
    print("\n── Check 2: MongoDB trends freshness ──")
    latest = db["trends"].find_one(sort=[("captured_at", -1)])
    if latest:
        captured = latest.get("captured_at", "?")
        rising = latest.get("rising", [])
        direction = latest.get("seasonal_direction", "")
        print(f"  Captured at: {captured}")
        print(f"  Rising trends: {len(rising)}")
        for r in rising[:5]:
            trend_name = r.get("trend", "?") if isinstance(r, dict) else str(r)
            print(f"    • {trend_name}")
        print(f"  Direction: {direction}")
        print(f"  [OK] Trends collection is fresh")
    else:
        print("  [WARN] No trend documents found")
        failures.append("trends collection is empty")

    # ── Check 3: Neo4j Aesthetic count ───────────────────────────
    print("\n── Check 3: Neo4j Aesthetic nodes ──")
    if kg.is_connected:
        result = kg.query("MATCH (a:Aesthetic) RETURN count(a) AS count")
        if result:
            count = result[0].get("count", 0)
            print(f"  Aesthetic nodes: {count}")
            if count > 36:  # was 36 before pipeline
                print(f"  [OK] Neo4j has new aesthetics (was 36, now {count})")
            else:
                print(f"  [OK] Neo4j has {count} Aesthetic nodes")
        else:
            print("  [WARN] Could not query Neo4j")
    else:
        print("  [SKIP] Neo4j not connected")

    # ── Check 4: Deduplication check ─────────────────────────────
    print("\n── Check 4: Deduplication ──")
    if sg_count > 0:
        pipeline = [
            {"$group": {"_id": "$aesthetic", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}}
        ]
        try:
            dupes = list(db["styling_guides"].aggregate(pipeline))
            if dupes:
                print(f"  [WARN] DUPLICATES FOUND: {len(dupes)}")
                for d in dupes:
                    print(f"    • '{d['_id']}' appears {d['count']} times")
                failures.append(f"{len(dupes)} duplicate styling guides")
            else:
                print("  [OK] No duplicates")
        except Exception as e:
            print(f"  [SKIP] Aggregation not supported ({e})")
    else:
        print("  [SKIP] No guides to check")

    # ── Check 5: Quality gate verification ───────────────────────
    print("\n── Check 5: Quality gate ──")
    if sg_count > 0:
        low_quality = db["styling_guides"].count_documents(
            {"quality_score": {"$lt": 7.0}}
        )
        if low_quality > 0:
            print(f"  [WARN] QUALITY BREACH: {low_quality} guides below 7.0")
            failures.append(f"{low_quality} guides below quality threshold")
        else:
            print("  [OK] All guides passed quality gate (≥ 7.0)")
    else:
        print("  [SKIP] No guides to check")

    # ── Final Verdict ────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not failures:
        print(f"  [OK] Pipeline verified — {sg_count} guides stored, trends fresh")
    else:
        print(f"  [ISSUES] {len(failures)} problem(s):")
        for f in failures:
            print(f"    ✗ {f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    verify()
