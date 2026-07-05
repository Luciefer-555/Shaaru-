import urllib.request, json, sys

# Step 1: Check backend
try:
    urllib.request.urlopen('http://localhost:8000/docs', timeout=5)
    print('Backend:  RUNNING on :8000')
except Exception as e:
    print(f'Backend:  DOWN ({e})')

# Step 2: Check frontend
try:
    urllib.request.urlopen('http://localhost:3000', timeout=5)
    print('Frontend: RUNNING on :3000')
except Exception as e:
    print(f'Frontend: DOWN ({e})')

# Step 3: Get tunnel URLs
print()
try:
    raw = urllib.request.urlopen('http://127.0.0.1:4040/api/tunnels', timeout=5).read().decode()
    data = json.loads(raw)
    print(f'Tunnels:  {len(data["tunnels"])} active')
    for t in data['tunnels']:
        print(f'  {t["name"]:10s} -> {t["public_url"]}  (local: {t["config"]["addr"]})')
except Exception as e:
    print(f'Ngrok:    DOWN ({e})')

# Step 4: Touch test
print()
try:
    import urllib.parse
    body = json.dumps({
        'item_label': 'test item',
        'item_color': 'black',
        'item_category': 'top',
        'item_aesthetic': 'minimal',
        'all_items': ['test item'],
        'user_id': 'restart_test'
    }).encode()
    req = urllib.request.Request(
        'http://localhost:8000/api/cv/touch',
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    resp = urllib.request.urlopen(req, timeout=10)
    result = json.loads(resp.read().decode())
    print(f'Touch test: OK — {result}')
except Exception as e:
    print(f'Touch test: FAILED — {e}')
