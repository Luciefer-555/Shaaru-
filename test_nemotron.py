import base64, json, sys, os
from cv_engine import _SCAN_PROMPT, _parse_scan_json
from shaaru_brain import _get_client

def test_nemotron(image_path: str):
    print(f"\n{'='*60}")
    print(f"IMAGE: {os.path.basename(image_path)}")
    print('='*60)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    client = _get_client()

    import time
    start = time.time()

    raw = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": _SCAN_PROMPT
                },
            ],
        }],
        temperature=0.1,
        max_tokens=4096,
        timeout=60.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    elapsed = time.time() - start
    msg = raw.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)

    print(f"Time          : {elapsed:.1f}s")
    print(f"content len   : {len(content)}")
    print(f"reasoning len : {len(reasoning) if reasoning else 0}")
    print(f"content (400) : {content[:400]}")
    if reasoning:
        print(f"reasoning(200): {reasoning[:200]}")

    parsed = _parse_scan_json(content)
    print(f"Parsed        : {'YES' if parsed else 'NO'}")
    if parsed:
        items = parsed.get("items", [])
        print(f"Items         : {len(items)}")
        for item in items:
            print(f"  [{item.get('category','?').upper():10}] {item.get('label','?')}")
            print(f"    color: {item.get('color','?')}")
            print(f"    desc : {item.get('description','?')}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            test_nemotron(p)
        except Exception as e:
            print(f"ERROR: {e}")
