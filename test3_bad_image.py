"""
test3_bad_image.py — Test 3: send a deliberately poor-quality image (solid gray, 
very small) to confirm frame_quality=poor and guidance returned, no items array.
"""
import base64, json, urllib.request, urllib.error
from PIL import Image
import io

# Generate a tiny, very dark 32x32 solid-color PNG — simulates bad lighting/blur
img = Image.new("RGB", (32, 32), color=(20, 20, 20))  # near-black
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

print(f"Bad image b64 length: {len(b64)} (tiny dark 32x32 PNG)")

payload = json.dumps({"image_b64": b64, "user_id": "test_user"}).encode()
req = urllib.request.Request(
    "http://localhost:8000/api/cv/scan",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode()
    print("=== TEST 3 BAD IMAGE SCAN RESPONSE ===")
    print(json.dumps(json.loads(body), indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()}")
except Exception as ex:
    print(f"ERROR: {ex}")
