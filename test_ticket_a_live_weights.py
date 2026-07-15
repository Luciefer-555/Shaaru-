import os
import sys
import base64
import io
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
log = logging.getLogger("test_ticket_a_live")

from cv_engine import Yolo26ApparelDetector

def run_live_inference_test():
    print("\n=== TEST 4: LIVE END-TO-END INFERENCE WITH PRETRAINED COCO CHECKPOINT ===")
    # Use yolov8n.pt (~6MB) as our fast COCO checkpoint for testing the real PyTorch detection pipeline
    os.environ["YOLO26_WEIGHTS"] = "yolov8n.pt"
    Yolo26ApparelDetector._init_attempted = False
    Yolo26ApparelDetector._model = None

    # Create a realistic RGB image buffer (640x480)
    img = Image.new("RGB", (640, 480), color=(120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    dummy_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    print("Running Yolo26ApparelDetector.detect() with live PyTorch YOLO('yolov8n.pt') model...")
    res = Yolo26ApparelDetector.detect(dummy_b64)
    assert res is not None, "Expected valid result dict from live YOLO model"
    assert res.get("_model_used") == "YOLO26-COCO-Plumbing", f"Expected YOLO26-COCO-Plumbing, got {res.get('_model_used')}"
    assert isinstance(res.get("items"), list), "Expected list of items"
    print(f"[PASS] Live Inference Pass Successful! Model used: {res['_model_used']}, items detected: {len(res['items'])}")

if __name__ == "__main__":
    run_live_inference_test()
