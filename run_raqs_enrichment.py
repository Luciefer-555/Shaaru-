import os, sys, json, asyncio, requests, re
from dotenv import load_dotenv
load_dotenv()
ROOT = os.getcwd()
sys.path.insert(0, ROOT)

def clean_html(html):
    return re.sub(r"<[^>]+>", " ", html or "").replace("&amp;", "&").strip()

async def run():
    from pipeline.db.db_loader import load_all_references
    from pipeline.extractors.tailor_engine import analyze_garment_deep
    r = requests.get("https://abhinavmishraofficial.com/products/raqs.json", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    p = r.json()["product"]
    product = {
        "id": str(p["id"]), "title": p["title"], "tags": p.get("tags", []),
        "raw_description": clean_html(p.get("body_html", "")),
        "body_html": p.get("body_html", ""), "variants": [],
        "images": [i["src"] for i in p.get("images", []) if i.get("src")],
        "source_url": "https://abhinavmishraofficial.com/products/raqs",
    }
    with open("pipeline/config/designers.json", encoding="utf-8") as f:
        designer = next(d for d in json.load(f) if d["id"] == "abhinav_mishra")
    with open("pipeline/config/quality_gates.json", encoding="utf-8") as f:
        gates = json.load(f)
    db_refs = load_all_references(os.getenv("MONGODB_URI"), os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    doc, cost = await analyze_garment_deep(product, designer, db_refs, gates)
    print("COST:", cost)
    print(json.dumps(doc, indent=2, default=str))

asyncio.run(run())
