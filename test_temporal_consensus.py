#!/usr/bin/env python3
"""
test_temporal_consensus.py — Test Phase 2A TemporalConsensus extensions
-------------------------------------------------------------------------
Validates:
a) Persistent track_id assignment across scan cycles using IOU/spatial matching
b) EMA bbox smoothing (alpha=0.4) moving gradually toward new detections
c) Coasting state (item missing for exactly 1 cycle retains track_id & confidence)
d) Pruning state (item missing for 2+ cycles deleted; new detection gets fresh track_id)
"""

import sys
import json
from cv_engine import TemporalConsensus

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def run_test():
    print("=" * 90)
    print("PHASE 2A: TEMPORAL CONSENSUS & TRACK_ID / BBOX SMOOTHING TEST")
    print("=" * 90)
    
    tracker = TemporalConsensus(window_size=5)
    
    # ── Cycle 1: Initial Detection ──────────────────────────────────────────
    print("\n--- CYCLE 1: Initial Detections (2 items) ---")
    c1_items = [
        {"label": "black leather jacket", "category": "Leather", "bbox": {"x": 0.10, "y": 0.10, "w": 0.30, "h": 0.40}, "confidence": 0.85},
        {"label": "blue denim jeans", "category": "Denim", "bbox": {"x": 0.20, "y": 0.50, "w": 0.30, "h": 0.40}, "confidence": 0.90}
    ]
    res1 = tracker.update(c1_items)
    print(json.dumps(res1, indent=2))
    
    assert len(res1) == 2, "Expected 2 items in Cycle 1"
    track_id_A = res1[0]["track_id"]
    track_id_B = res1[1]["track_id"]
    assert track_id_A.startswith("track_") and track_id_B.startswith("track_"), "Track IDs must start with 'track_'"
    assert res1[0]["state"] == "new" and res1[1]["state"] == "new", "Initial state must be 'new'"
    assert res1[0]["bbox"]["x"] == 0.10, "Initial bbox should match raw bbox exactly"
    print("[PASS] Cycle 1: Assigned new track_ids and 'new' state.")

    # ── Cycle 2: Camera Movement (Shifted BBoxes) ───────────────────────────
    print("\n--- CYCLE 2: Camera Movement (Shifted bboxes by +0.05 in X) ---")
    c2_items = [
        {"label": "black leather jacket", "category": "Leather", "bbox": {"x": 0.15, "y": 0.12, "w": 0.30, "h": 0.40}, "confidence": 0.88},
        {"label": "blue denim jeans", "category": "Denim", "bbox": {"x": 0.25, "y": 0.52, "w": 0.30, "h": 0.40}, "confidence": 0.92}
    ]
    res2 = tracker.update(c2_items)
    print(json.dumps(res2, indent=2))
    
    assert len(res2) == 2, "Expected 2 items in Cycle 2"
    assert res2[0]["track_id"] == track_id_A and res2[1]["track_id"] == track_id_B, "Track IDs must persist!"
    assert res2[0]["state"] == "confirmed" and res2[1]["state"] == "confirmed", "State must transition to 'confirmed'"
    
    # Check EMA smoothing: alpha * new + (1-alpha) * old = 0.4 * 0.15 + 0.6 * 0.10 = 0.12
    expected_x_A = round(0.4 * 0.15 + 0.6 * 0.10, 4)
    assert res2[0]["bbox"]["x"] == expected_x_A, f"Expected smoothed x={expected_x_A}, got {res2[0]['bbox']['x']}"
    print(f"[PASS] Cycle 2: Track IDs persisted! EMA bbox smoothed x smoothly to {res2[0]['bbox']['x']} (not jumping to 0.15).")

    # ── Cycle 3: Coasting State (Item A missing for exactly 1 cycle) ────────
    print("\n--- CYCLE 3: Coasting State (Item A missing from vision scan) ---")
    c3_items = [
        {"label": "blue denim jeans", "category": "Denim", "bbox": {"x": 0.26, "y": 0.53, "w": 0.30, "h": 0.40}, "confidence": 0.90}
    ]
    res3 = tracker.update(c3_items)
    print(json.dumps(res3, indent=2))
    
    assert len(res3) == 2, "Expected 2 items in Cycle 3 (1 confirmed + 1 coasting)"
    coasting_item = next(i for i in res3 if i["track_id"] == track_id_A)
    assert coasting_item["state"] == "coasting", "Missing item must enter 'coasting' state"
    assert coasting_item["bbox"]["x"] == expected_x_A, "Coasting item must retain its last smoothed bbox"
    assert coasting_item["confidence"] == res2[0]["confidence"], "Coasting item must retain last confirmed confidence"
    print("[PASS] Cycle 3: Coasting works! Item A retained track_id, smoothed bbox, and confidence during 1-frame drop.")

    # ── Cycle 4: Recovery from Coasting (Item A reappears) ──────────────────
    print("\n--- CYCLE 4: Recovery from Coasting (Item A reappears) ---")
    c4_items = [
        {"label": "black leather jacket", "category": "Leather", "bbox": {"x": 0.14, "y": 0.11, "w": 0.30, "h": 0.40}, "confidence": 0.87},
        {"label": "blue denim jeans", "category": "Denim", "bbox": {"x": 0.26, "y": 0.53, "w": 0.30, "h": 0.40}, "confidence": 0.90}
    ]
    res4 = tracker.update(c4_items)
    print(json.dumps(res4, indent=2))
    
    recovered_A = next(i for i in res4 if i["track_id"] == track_id_A)
    assert recovered_A["state"] == "confirmed", "Recovered item must return to 'confirmed' state"
    print("[PASS] Cycle 4: Item A successfully recovered from coasting with same track_id!")

    # ── Cycle 5 & 6: Pruning (Item B missing for 2+ consecutive cycles) ─────
    print("\n--- CYCLE 5 & 6: Pruning (Item B missing for 2 consecutive cycles) ---")
    # Cycle 5: Item B misses cycle 1 -> enters coasting
    res5 = tracker.update([c4_items[0]])
    assert any(i["track_id"] == track_id_B and i["state"] == "coasting" for i in res5), "Item B should be coasting in Cycle 5"
    
    # Cycle 6: Item B misses cycle 2 -> pruned!
    res6 = tracker.update([c4_items[0]])
    print(json.dumps(res6, indent=2))
    assert not any(i["track_id"] == track_id_B for i in res6), "Item B should be pruned after 2 missed cycles!"
    print("[PASS] Cycle 5 & 6: Item B coasted for 1 cycle, then was pruned on the 2nd missed cycle.")

    # ── Cycle 7: New Detection in Old Area Gets Fresh Track ID ──────────────
    print("\n--- CYCLE 7: New Detection in Old Area Gets Fresh Track ID ---")
    c7_items = [
        c4_items[0],
        {"label": "blue denim trousers", "category": "Denim", "bbox": {"x": 0.25, "y": 0.52, "w": 0.30, "h": 0.40}, "confidence": 0.89}
    ]
    res7 = tracker.update(c7_items)
    print(json.dumps(res7, indent=2))
    
    new_B = next(i for i in res7 if i["label"] == "blue denim trousers")
    assert new_B["track_id"] != track_id_B, f"New detection must get a fresh track_id, not old {track_id_B}!"
    assert new_B["state"] == "new", "New detection state must be 'new'"
    print(f"[PASS] Cycle 7: Fresh track_id '{new_B['track_id']}' assigned to new detection in old area!")
    
    print("\n" + "=" * 90)
    print("ALL PHASE 2A TEMPORAL CONSENSUS TESTS PASSED PERFECTLY!")
    print("=" * 90)

if __name__ == "__main__":
    run_test()
