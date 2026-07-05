from cv_router import _COMBO_PROMPT
from cv_engine import _parse_scan_json
from shaaru_brain import _get_client
import json, time

items = [
  {"id": "item_1", "label": "oversized Breton stripe knit sweater", "category": "top", "color": "off-white with black horizontal stripes"},
  {"id": "item_2", "label": "black block-heel Chelsea boot", "category": "footwear", "color": "black"},
  {"id": "item_3", "label": "light blue pleated midi skirt", "category": "bottom", "color": "light blue"},
  {"id": "item_4", "label": "cream longline blazer", "category": "outerwear", "color": "cream"},
]

items_block = "\n".join(
  f"  - id: {i['id']} | {i['label']} ({i['category']}, {i['color']})"
  for i in items
)
prompt = _COMBO_PROMPT.format(items_block=items_block)
client = _get_client()
for attempt in range(1, 4):
    try:
        raw = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt < 3:
            print("Retrying in 10s...")
            time.sleep(10)
        else:
            raise
content = raw.choices[0].message.content or ""
parsed = _parse_scan_json(content)
print(json.dumps(parsed, indent=2))
