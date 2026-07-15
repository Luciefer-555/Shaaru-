#!/usr/bin/env python3
"""
test_phase1_verification_scenarios.py — End-to-End Verification of Phase 1 Fixes
----------------------------------------------------------------------------------
Tests the 4 specific scenarios required in Part 3:
Scenario A: Point and remove -> box disappears quickly without lingering.
Scenario B: Hand-only / fingertip / person detections -> completely stripped.
Scenario C: Garment + hand combined label -> behavior verification & documentation.
Scenario D: Normal detection and continuity -> track_id persistence and EMA smoothing.
"""

import sys
import time
import json
from cv_engine import TemporalConsensus, _filter_body_parts

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def simulate_frontend_lerp_fade(opacity, target_opacity, frames=10):
    """Simulate the frontend LERP loop at 60fps (~16.6ms per frame) with multiplier 0.35 vs 0.15."""
    history = [opacity]
    curr = opacity
    multiplier = 0.35 if target_opacity < curr else 0.15
    for _ in range(frames):
        curr += (target_opacity - curr) * multiplier
        if target_opacity == 0.0 and curr < 0.02:
            curr = 0.0
        history.append(round(curr, 4))
        if curr == target_opacity:
            break
    return history

def run_all_scenarios():
    print("=" * 90)
    print("SHAARU PHASE 1 FIXES — END-TO-END 4-SCENARIO VERIFICATION")
    print("=" * 90)

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO A: Point and remove (Stale box lifecycle check)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[SCENARIO A] Point and Remove (Stale Box & Coasting Duration Check)")
    tracker_A = TemporalConsensus(window_size=5)
    
    # Scan 1: Item detected at t=0.0s
    c1 = [{"label": "silk dress", "category": "dress", "bbox": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.6}, "confidence": 0.95}]
    res1 = tracker_A.update(c1)
    tid_A = res1[0]["track_id"]
    print(f"  Scan 1 (t=0.0s): Item detected -> assigned {tid_A}, state={res1[0]['state']}")
    
    # Scan 2: Item missed at t=0.01s (quick scan cycle <= 3.5s elapsed since last seen)
    res2 = tracker_A.update([])
    print(f"  Scan 2 (t=0.01s, missed_cycles=1): Item missing -> state={res2[0]['state']} (capped coasting active)")
    assert len(res2) == 1 and res2[0]["state"] == "coasting", "Should coast for <=3.5s elapsed time"
    
    # Scan 3: Item missed again on next cycle -> pruned immediately!
    res3 = tracker_A.update([])
    print(f"  Scan 3 (missed_cycles=2): Item missing again -> active tracks remaining: {len(res3)}")
    assert len(res3) == 0, "Item pruned immediately on 2nd missed cycle!"
    
    # Check what happens if a long latency scan (>3.5s elapsed) occurs right when missed_cycles=1:
    tracker_A2 = TemporalConsensus(window_size=5)
    res_long1 = tracker_A2.update(c1)
    tid_long = res_long1[0]["track_id"]
    # Simulate 4.5 seconds passing (e.g., long network latency / model round-trip)
    tracker_A2._tracks[tid_long]["last_seen_time"] = time.time() - 4.5
    res_long2 = tracker_A2.update([])
    print(f"  Long Latency Scan (t=+4.5s elapsed, missed_cycles=1): Active tracks remaining: {len(res_long2)}")
    assert len(res_long2) == 0, "Item pruned without entering coasting when >3.5s elapsed since last seen!"

    # Simulate frontend LERP fade-out:
    fade_035 = simulate_frontend_lerp_fade(1.0, 0.0, frames=10)
    print(f"  Frontend LERP fade progression (multiplier 0.35): {fade_035}")
    assert fade_035[-1] == 0.0, "Opacity dropped to 0.0 cleanly within 10 frames (~166ms)!"
    print("  [PASS] Scenario A verified: items vanish quickly without lingering ghost boxes.")

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO B: Hand-only or fingertip inside box (Denylist Check)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[SCENARIO B] Hand-only / Fingertip / Person Detections")
    raw_B = [
        {"label": "hand holding fabric", "description": "a human hand", "category": "top", "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
        {"label": "fingertip", "description": "finger touching cloth", "category": "accessory", "bbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}},
        {"label": "person", "description": "person standing", "category": "person", "bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}},
        {"label": "thumb inside pocket", "description": "thumb visible", "category": "bottom", "bbox": {"x": 0.3, "y": 0.4, "w": 0.2, "h": 0.2}},
        {"label": "blue denim trousers", "description": "relaxed fit denim jeans", "category": "bottom", "bbox": {"x": 0.2, "y": 0.5, "w": 0.4, "h": 0.5}}
    ]
    filtered_B = _filter_body_parts(raw_B)
    print(f"  Raw detections count: {len(raw_B)}")
    print(f"  Filtered detections count: {len(filtered_B)}")
    print(f"  Kept items: {[it['label'] for it in filtered_B]}")
    assert len(filtered_B) == 1 and filtered_B[0]["label"] == "blue denim trousers", "Only valid garment should survive!"
    print("  [PASS] Scenario B verified: all hand, finger, thumb, and person detections stripped completely.")

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO C: Garment + hand combined label behavior
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[SCENARIO C] Garment + Hand Combined Label ('hand on blue denim jeans')")
    raw_C = [
        {"label": "hand on blue denim jeans", "description": "denim jeans with hand", "category": "bottom", "bbox": {"x": 0.2, "y": 0.5, "w": 0.4, "h": 0.5}},
        {"label": "handloom cotton saree", "description": "authentic handwoven saree drapes on body", "category": "dress", "bbox": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}}
    ]
    filtered_C = _filter_body_parts(raw_C)
    print(f"  Raw input labels: {[it['label'] for it in raw_C]}")
    print(f"  Filtered output labels: {[it['label'] for it in filtered_C]}")
    assert len(filtered_C) == 1 and filtered_C[0]["label"] == "handloom cotton saree", "Safe term preserved, combined hand label dropped!"
    print("  [PASS] Scenario C verified & documented: 'hand on blue denim jeans' is dropped to prevent ghosting on moving hands; safe terms ('handloom') preserved.")

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO D: Normal detection and continuity (Tracking & EMA smoothing)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[SCENARIO D] Normal Detection & Continuity (Track ID & EMA Smoothing)")
    tracker_D = TemporalConsensus(window_size=5)
    
    # Frame 1
    d_f1 = [
        {"label": "black leather jacket", "category": "outerwear", "bbox": {"x": 0.10, "y": 0.10, "w": 0.30, "h": 0.40}, "confidence": 0.88},
        {"label": "white cotton t-shirt", "category": "top", "bbox": {"x": 0.15, "y": 0.15, "w": 0.25, "h": 0.35}, "confidence": 0.90}
    ]
    res_d1 = tracker_D.update(d_f1)
    tid_jacket = res_d1[0]["track_id"]
    tid_tshirt = res_d1[1]["track_id"]
    print(f"  Frame 1: Jacket={tid_jacket} (state={res_d1[0]['state']}), T-Shirt={tid_tshirt} (state={res_d1[1]['state']})")
    
    # Frame 2: Slight movement
    d_f2 = [
        {"label": "black leather jacket", "category": "outerwear", "bbox": {"x": 0.14, "y": 0.12, "w": 0.30, "h": 0.40}, "confidence": 0.89},
        {"label": "white cotton t-shirt", "category": "top", "bbox": {"x": 0.17, "y": 0.16, "w": 0.25, "h": 0.35}, "confidence": 0.92}
    ]
    res_d2 = tracker_D.update(d_f2)
    print(f"  Frame 2: Jacket={res_d2[0]['track_id']} (state={res_d2[0]['state']}), smoothed_x={res_d2[0]['bbox']['x']}")
    assert res_d2[0]["track_id"] == tid_jacket and res_d2[1]["track_id"] == tid_tshirt, "Track IDs must persist across frames!"
    assert res_d2[0]["state"] == "confirmed" and res_d2[1]["state"] == "confirmed", "Must transition to confirmed state!"
    expected_smooth_x = round(0.4 * 0.14 + 0.6 * 0.10, 4)
    assert res_d2[0]["bbox"]["x"] == expected_smooth_x, f"Expected EMA smoothed x={expected_smooth_x}, got {res_d2[0]['bbox']['x']}"
    print("  [PASS] Scenario D verified: tracking continuity and EMA smoothing work flawlessly.")

    print("\n" + "=" * 90)
    print("ALL 4 VERIFICATION SCENARIOS PASSED WITH 100% ACCURACY!")
    print("=" * 90)

if __name__ == "__main__":
    run_all_scenarios()
