import requests, time

print(f"Sending at {time.strftime('%H:%M:%S')}...")
resp = requests.post(
    'http://localhost:8000/api/cv/touch',
    json={
        'item_label': 'ribbed sweater',
        'item_color': 'off-white',
        'item_category': 'top',
        'item_aesthetic': 'minimalist',
        'all_items': ['ribbed sweater', 'black boots', 'wide-leg denim'],
        'user_id': 'fix_test'
    },
    timeout=45
)
print(f"Done at {time.strftime('%H:%M:%S')}")
print(resp.json())
