import time, base64
from cv_engine import scan_frame

with open(r'C:\Users\saipr\Downloads\Bershka.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

t0 = time.time()
result = scan_frame(b64, run_combos=False)
t1 = time.time()

print(f'Total time: {round(t1-t0,2)}s')
print(f'Items: {len(result.get("items",[]))}')
for item in result.get('items', []):
    print(f"  {item['label']} | {item.get('color','?')} | conf: {item.get('confidence')}")
print(f'Scene: {result.get("scene_context", {}).get("scene_type")}')
print(f'Quality: {result.get("frame_quality")}')
