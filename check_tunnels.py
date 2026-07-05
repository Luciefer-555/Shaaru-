import urllib.request, json
raw = urllib.request.urlopen('http://127.0.0.1:4040/api/tunnels').read().decode()
data = json.loads(raw)
print(f"Total tunnels: {len(data['tunnels'])}")
print()
for t in data['tunnels']:
    print(f"  Name:       {t['name']}")
    print(f"  Public URL: {t['public_url']}")
    print(f"  Local:      {t['config']['addr']}")
    print(f"  Proto:      {t['proto']}")
    print()
