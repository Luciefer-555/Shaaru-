import json
data = json.load(open('pipeline/output/review/abhinav_mishra_batch_1.json', encoding='utf-8'))
passed = [p for p in data if p.get('quality_gate_passed')]
failed = [p for p in data if not p.get('quality_gate_passed')]
new_fabrics = set()
for p in data:
    for c in p.get('new_fabric_candidates', []):
        name = c.get('name') or c.get('fabric_name', '')
        if name: new_fabrics.add(name)
print(f'Total: {len(data)} | Passed: {len(passed)} | Failed: {len(failed)}')
print(f'Failure reasons:')
for p in failed: print(f'  {p["title"]}: {p.get("validation_failures")}')
print(f'New fabric candidates to seed: {new_fabrics}')
