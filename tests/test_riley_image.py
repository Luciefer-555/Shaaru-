"""
tests/test_riley_image.py
─────────────────────────────────────────────────────────────────────────────
Read-only test — no existing files are modified.

Sends a POST request to Riley's chat endpoint with:
  • The image at C:\\Users\\saipr\\Downloads\\600456562859419046.jpg encoded as base64
  • Message: "How do I wear a bandana like this? And where can I find one in India?"

Prints Riley's full raw response to stdout.

Endpoints probed (in priority order):
  1. POST http://localhost:8000/api/chat/message  (ChatMessageRequest — active Riley route)
  2. POST http://localhost:8000/api/chat           (ChatRequest with image_b64 — may not be wired)

Run with:
    python tests/test_riley_image.py
or via pytest (the module runs itself):
    pytest tests/test_riley_image.py -s
"""

import base64
import json
import pprint
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────
BASE_URL   = "http://localhost:8000"
IMAGE_PATH = Path(r"C:\Users\saipr\Downloads\600456562859419046.jpg")
USER_ID    = "test_user_riley_image"
MESSAGE    = "How do I wear a bandana like this? And where can I find one in India?"

# ── Helpers ───────────────────────────────────────────────────────────────

def _load_image_b64(path: Path) -> str:
    """Read image from disk and return a base64-encoded string."""
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Please make sure the file exists before running this test."
        )
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("utf-8")


def _post_json(url: str, payload: dict, timeout: int = 120) -> tuple[int, dict | str]:
    """
    POST *payload* as JSON to *url*.
    Returns (status_code, parsed_body).
    Body is returned as a dict when valid JSON, otherwise as a raw string.
    """
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status  = resp.status
            raw     = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except (ConnectionResetError, OSError):
            raw = f'{{"detail": "HTTP {exc.code} (body unreadable - connection reset)"}}'
    except urllib.error.URLError as exc:
        # Server offline / connection refused
        return -1, f"[CONNECTION ERROR] {exc.reason}"

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = raw

    return status, body


def _separator(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── Test logic ────────────────────────────────────────────────────────────

def run_riley_image_test() -> None:
    """Main test function — probes Riley's endpoints and prints raw responses."""

    # 1. Load and encode image
    _separator("Loading image")
    print(f"Image path : {IMAGE_PATH}")
    image_b64 = _load_image_b64(IMAGE_PATH)
    size_kb   = len(image_b64) * 3 / 4 / 1024  # approximate original size
    print(f"Base64 len : {len(image_b64):,} chars  (~{size_kb:.1f} KB original)")
    print(f"Preview    : {image_b64[:60]}...")

    # 2. ── Endpoint 1: POST /api/chat/message ────────────────────────────
    #    ChatMessageRequest: { user_id, message, history }
    #    This is the *active* Riley route (riley_think under the hood).
    #    image_b64 is NOT part of this schema — we include it as an extra
    #    field so the server can ignore it gracefully, while still sending
    #    the full message text which references the bandana style.
    _separator("Attempt 1 — POST /api/chat/message  (active Riley endpoint)")
    url1 = f"{BASE_URL}/api/chat/message"
    payload1 = {
        "user_id" : USER_ID,
        "message" : MESSAGE,
        "history" : [],
        # Extra hint field: server will ignore unknown fields (Pydantic default).
        # Included for completeness so the test payload is self-documenting.
        "image_b64": image_b64,
    }
    print(f"URL      : {url1}")
    print(f"Payload  : user_id={USER_ID!r}  message={MESSAGE!r}")
    print(f"           + image_b64 ({len(image_b64):,} chars)")
    print("\nPOSTing... (timeout 120 s)")

    status1, body1 = _post_json(url1, payload1)

    print(f"\n{'-'*72}")
    print(f"HTTP STATUS : {status1}")
    print(f"{'-'*72}")
    print("RAW RESPONSE:")
    if isinstance(body1, dict):
        pprint.pprint(body1, width=80, sort_dicts=False)
    else:
        print(body1)

    # 3. ── Endpoint 2: POST /api/chat ────────────────────────────────────
    #    ChatRequest: { user_id, message, image_b64, session_id }
    #    This model exists in api.py (line 121) but may not have an active route.
    #    We probe it anyway — Riley may be served on this path in some deployments.
    _separator("Attempt 2 — POST /api/chat  (ChatRequest with image_b64)")
    url2 = f"{BASE_URL}/api/chat"
    payload2 = {
        "user_id"   : USER_ID,
        "message"   : MESSAGE,
        "image_b64" : image_b64,
        "session_id": None,
    }
    print(f"URL      : {url2}")
    print(f"Payload  : user_id={USER_ID!r}  message={MESSAGE!r}")
    print(f"           + image_b64 ({len(image_b64):,} chars)")
    print("\nPOSTing... (timeout 120 s)")

    status2, body2 = _post_json(url2, payload2)

    print(f"\n{'-'*72}")
    print(f"HTTP STATUS : {status2}")
    print(f"{'-'*72}")
    print("RAW RESPONSE:")
    if isinstance(body2, dict):
        pprint.pprint(body2, width=80, sort_dicts=False)
    else:
        print(body2)

    # 4. Summary
    _separator("Summary")
    print(f"Endpoint 1  ({url1})  ->  HTTP {status1}")
    print(f"Endpoint 2  ({url2})  ->  HTTP {status2}")

    # Determine winner for pytest assertion (prefer whichever returned 200)
    if status1 == 200:
        reply = body1.get("reply") or body1.get("response") if isinstance(body1, dict) else body1
        print(f"\n[OK] Riley replied via /api/chat/message:\n  {reply}")
    elif status2 == 200:
        reply = body2.get("reply") or body2.get("response") if isinstance(body2, dict) else body2
        print(f"\n[OK] Riley replied via /api/chat:\n  {reply}")
    else:
        print("\n[!] Neither endpoint returned HTTP 200.")
        print("   Make sure the server is running:  uvicorn api:app --reload --port 8000")
        # Don't hard-fail so the test is still informative when server is offline
        return

    # 5. pytest-compatible assertion (optional — only if pytest is the runner)
    assert reply, "Riley's reply was empty — something went wrong upstream."
    print("\n[OK] Assertion passed -- Riley returned a non-empty reply.")


# ── Entry points ──────────────────────────────────────────────────────────

def test_riley_image_response() -> None:
    """pytest-compatible wrapper."""
    run_riley_image_test()


if __name__ == "__main__":
    try:
        run_riley_image_test()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(130)
