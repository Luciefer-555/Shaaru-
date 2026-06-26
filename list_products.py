import json
data = json.load(open('pipeline/output/review/abhinav_mishra_batch_1.json', encoding='utf-8'))
for i, p in enumerate(data, 1):
    print(f"{i}. {p['title']}")
