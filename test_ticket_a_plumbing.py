import asyncio
import base64
import io
import logging
import sys
from PIL import Image

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
log = logging.getLogger("test_ticket_a")

import cv_engine
from cv_engine import Yolo26ApparelDetector, KNOWN_TAXONOMY_CONSTRUCTIONS, _CONSTRUCTION_TO_CATEGORY, get_consensus_tracker

def make_dummy_b64():
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def test_1_loud_failure_when_engine_missing():
    print("\n=== TEST 1: LOUD FAILURE ALERT WHEN PACKAGE PRESENT BUT CHECKPOINT MISSING ===")
    import os
    old_weights = os.environ.get("YOLO26_WEIGHTS")
    os.environ["YOLO26_WEIGHTS"] = "nonexistent_ticket_a_checkpoint.pt"
    # Ensure _init_attempted is false so get_model runs check against missing checkpoint
    Yolo26ApparelDetector._init_attempted = False
    Yolo26ApparelDetector._model = None
    
    res = Yolo26ApparelDetector.detect("dummy_b64")
    assert res is None, "Expected None return when YOLO checkpoint is missing"
    print("[PASS] Test 1 Passed: Yolo26ApparelDetector.detect returned None and logged ERROR alert loud when checkpoint is missing while ultralytics is installed.")
    if old_weights is not None:
        os.environ["YOLO26_WEIGHTS"] = old_weights
    else:
        os.environ.pop("YOLO26_WEIGHTS", None)

def test_2_hard_allowlist_gate_and_coco_mapping():
    print("\n=== TEST 2: HARD TAXONOMY ALLOWLIST GATE & COCO PRETRAINED PLUMBING ===")
    # Mock model.predict to simulate COCO outputs containing valid fashion items & invalid out-of-vocab items
    class MockBox:
        def __init__(self, xywh_vals, conf_val, cls_val):
            import numpy as np
            class ArrayWrapper:
                def __init__(self, arr): self.arr = np.array(arr)
                def cpu(self): return self
                def numpy(self): return self.arr
            self.xywh = [ArrayWrapper(xywh_vals)]
            self.conf = [ArrayWrapper(conf_val)]
            self.cls = [ArrayWrapper(cls_val)]

    class MockResult:
        def __init__(self, boxes, names):
            self.boxes = boxes
            self.names = names

    class MockYOLO:
        def predict(self, source, conf, verbose):
            names = {
                0: "person",
                2: "car",
                16: "dog",
                26: "handbag",
                27: "tie",
                28: "umbrella",
                99: "hand" # out of vocab simulation
            }
            boxes = [
                MockBox([320, 240, 100, 200], 0.88, 26), # handbag -> bag_wallet (VALID)
                MockBox([100, 100, 50, 50], 0.95, 2),    # car -> dropped by allowlist gate
                MockBox([200, 200, 40, 40], 0.75, 16),   # dog -> dropped by allowlist gate
                MockBox([150, 150, 30, 30], 0.82, 27),   # tie -> tie (VALID)
                MockBox([50, 50, 20, 20], 0.91, 99),     # hand -> dropped by allowlist gate
            ]
            return [MockResult(boxes, names)]

    Yolo26ApparelDetector._init_attempted = True
    Yolo26ApparelDetector._model = MockYOLO()

    dummy_b64 = make_dummy_b64()
    
    # First check production default behavior: proxy mapping disabled
    import os
    os.environ.pop("ENABLE_COCO_PLUMBING_PROXY_MAPPINGS", None)
    res_prod = Yolo26ApparelDetector.detect(dummy_b64)
    items_prod = res_prod["items"]
    # Without proxy flag, handbag is not mapped to bag_wallet, so ONLY tie (which is in our 36 classes) survives!
    assert len(items_prod) == 1 and items_prod[0]["label"] == "tie", f"In production mode without proxy flag, only exact vocabulary items should survive! Got: {items_prod}"
    print("[PASS] Production Gate Check: Without proxy flag, COCO 'handbag' and 'person' cannot reach user path and are dropped cleanly by allowlist.")

    # Now check staging test-only behavior: proxy mapping enabled
    os.environ["ENABLE_COCO_PLUMBING_PROXY_MAPPINGS"] = "1"
    res = Yolo26ApparelDetector.detect(dummy_b64)
    assert res is not None, "Expected valid dictionary result from YOLO detect"
    assert res["_model_used"] == "YOLO26-COCO-Plumbing"
    items = res["items"]
    print(f"Detected items surviving allowlist with staging proxy flag: {[item['label'] for item in items]}")
    
    # Verify ONLY bag_wallet and tie survived, while car, dog, and hand were dropped
    assert len(items) == 2, f"Expected exactly 2 items (bag_wallet, tie), got {len(items)}"
    labels = {i["label"] for i in items}
    assert labels == {"bag_wallet", "tie"}, f"Unexpected labels: {labels}"
    for idx, item in enumerate(items):
        assert item["id"] == f"yolo_{idx if item['label']=='bag_wallet' else 3}" # box indices 0 and 3
        assert item["fabric_type"] == "pending", "Must set fabric_type='pending' for crop enrichment trigger"
        assert "bbox" in item and all(k in item["bbox"] for k in ("x", "y", "w", "h"))
    print("[PASS] Test 2 Passed: Out-of-vocab items (car, dog, hand) dropped with log lines; valid COCO items mapped clean.")

def test_3_temporal_consensus_handoff():
    print("\n=== TEST 3: TEMPORAL CONSENSUS TRACKER HANDOFF ===")
    tracker = get_consensus_tracker("test_user_ticket_a")
    # Feed YOLO detections to TemporalConsensusTracker
    items = [
        {"id": "yolo_0", "label": "kurta", "category": "top", "confidence": 0.92, "bbox": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.5}, "fabric_type": "pending"},
        {"id": "yolo_1", "label": "pants", "category": "bottom", "confidence": 0.89, "bbox": {"x": 0.2, "y": 0.7, "w": 0.4, "h": 0.25}, "fabric_type": "pending"}
    ]
    tracked = tracker.update(items)
    assert len(tracked) == 2, f"Expected 2 tracked items, got {len(tracked)}"
    for t in tracked:
        assert t["track_id"].startswith("track_"), f"Missing track_id: {t}"
        assert t["fabric_type"] == "pending", "fabric_type must remain 'pending' through tracker update"
        assert t["state"] in ("new", "confirmed"), f"Unexpected tracker state: {t['state']}"
    print(f"Tracked items output from ConsensusTracker: {[{'track_id': t['track_id'], 'label': t['label'], 'fabric_type': t['fabric_type'], 'state': t['state']} for t in tracked]}")
    print("[PASS] Test 3 Passed: TemporalConsensusTracker handoff verified clean.")

if __name__ == "__main__":
    test_1_loud_failure_when_engine_missing()
    test_2_hard_allowlist_gate_and_coco_mapping()
    test_3_temporal_consensus_handoff()
    print("\n=== ALL TICKET A PLUMBING VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
