import base64, json, sys, os

def test_scan(image_path: str):
    print(f"\n{'='*60}")
    print(f"IMAGE: {os.path.basename(image_path)}")
    print('='*60)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    from cv_engine import scan_frame
    result = scan_frame(image_b64)

    print(f"Frame quality : {result.get('frame_quality')}")
    print(f"Scene lighting: {result.get('scene_lighting')}")
    print(f"Items found   : {len(result.get('items', []))}")
    if result.get('guidance'):
        print(f"Guidance      : {result.get('guidance')}")
    print()

    for item in result.get("items", []):
        bbox = item.get("bbox", {})
        bw = bbox.get("w", 0)
        bh = bbox.get("h", 0)
        bbox_ok = "OK" if (bw > 0 and bh > 0) else "ZERO BBOX"
        desc = item.get("description", "-")
        print(f"  [{item.get('category','?').upper():10}] {item.get('label','?')}")
        print(f"    description : {desc}")
        print(f"    color       : {item.get('color','-')}")
        print(f"    confidence  : {item.get('confidence', 0):.0%}")
        print(f"    bbox        : x={bbox.get('x',0):.2f} y={bbox.get('y',0):.2f} "
              f"w={bw:.2f} h={bh:.2f}  {bbox_ok}")
        print()

    if result.get("annotated_frame_b64"):
        out = f"annotated_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        with open(out, "wb") as f:
            f.write(base64.b64decode(result["annotated_frame_b64"]))
        print(f"  -> Saved: {out}")
    else:
        print("  -> No annotated frame returned")

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python test_cv.py image1.jpg image2.jpg ...")
        sys.exit(1)
    for p in paths:
        try:
            test_scan(p)
        except Exception as e:
            print(f"ERROR on {p}: {e}")
