import base64, json, sys, os

def test_combos(image_path: str):
    print(f"\n{'='*60}")
    print(f"IMAGE: {os.path.basename(image_path)}")
    print('='*60)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    from cv_engine import scan_frame
    from cv_router import _COMBO_PROMPT
    from cv_engine import _parse_scan_json
    from shaaru_brain import _get_client

    # Step 1: scan
    print("Scanning...")
    result = scan_frame(image_b64)
    items = result.get("items", [])
    print(f"Items found: {len(items)}")
    for item in items:
        print(f"  [{item.get('category','?').upper():10}] {item.get('label')}")

    if not items:
        print("No items -- skipping combos")
        return

    # Step 2: build combos
    print("\nGenerating combos...")
    items_block = "\n".join(
        f"  - id: {item.get('id','?')} | {item.get('label','?')} "
        f"({item.get('category','?')}, {item.get('color','?')})"
        for item in items
    )
    prompt = _COMBO_PROMPT.format(items_block=items_block)

    client = _get_client()
    raw = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.7,
    )
    content = raw.choices[0].message.content or ""
    parsed = _parse_scan_json(content)

    if not parsed or "combos" not in parsed:
        print(f"PARSE FAILED. Raw response (first 400 chars):")
        print(content[:400])
        return

    combos = parsed["combos"]
    print(f"Combos generated: {len(combos)}\n")

    for combo in combos:
        print(f"  COMBO: {combo.get('name','?').upper()}")
        print(f"  Vibe : {combo.get('vibe','?')}")
        used = combo.get('items_used', [])
        used_labels = [
            next((i.get('label') for i in items if i.get('id') == uid), uid)
            for uid in used
        ]
        print(f"  Uses : {', '.join(used_labels)}")
        print(f"  Directions: {combo.get('directions','')}")
        missing = combo.get('missing', [])
        if missing:
            print(f"  Find these:")
            for m in missing:
                print(f"    [{m.get('role','?').upper()}] {m.get('find','?')}")
        else:
            print(f"  Complete look - nothing missing")
        print()

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python test_combos.py image1.jpg image2.jpg ...")
        sys.exit(1)
    for p in paths:
        try:
            test_combos(p)
        except Exception as e:
            import traceback
            print(f"ERROR on {p}: {e}")
            traceback.print_exc()
