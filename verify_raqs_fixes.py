import os, sys, json, asyncio, requests, re
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.getcwd())

USER_MESSAGE = (
    "I love this sherwani from Abhinav Mishra called RAQS - "
    "https://abhinavmishraofficial.com/products/raqs - "
    "the mirror work is insane. I want to get something exactly like this made "
    "for my best friend's wedding. Tell me everything - what fabric is this, "
    "what is this mirror work technique called, where do I source the materials "
    "in India, and give me a full tailor brief so I can actually get this made."
)

def clean_html(html):
    return re.sub(r"<[^>]+>", " ", html or "").replace("&amp;", "&").strip()

async def main():
    from shaaru_brain import answer
    from pipeline.db.db_loader import load_all_references
    from pipeline.extractors.tailor_engine import analyze_garment_deep

    print("=== FIX 1+2: shaaru_brain.answer() ===")
    result = await answer(USER_MESSAGE)
    p0 = (result.get("products") or [{}])[0]
    tb = result.get("tailor_brief", {})
    print("cache_hit:", result.get("instant_answer", {}).get("cache_hit"))
    print("match_type:", result.get("instant_answer", {}).get("match_type"))
    print("matched_on:", result.get("instant_answer", {}).get("matched_on"))
    print("products[0].title:", p0.get("title"))
    print("tailor_brief.base_fabric.name:", tb.get("base_fabric", {}).get("name"))

    print("\n=== FIX 3: analyze_garment_deep (no vision wait - empty image) ===")
    r = requests.get("https://abhinavmishraofficial.com/products/raqs.json", headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    p = r.json()["product"]
    product = {
        "id": str(p["id"]), "title": p["title"], "tags": p.get("tags", []),
        "raw_description": clean_html(p.get("body_html", "")),
        "variants": [], "images": [],
        "source_url": "https://abhinavmishraofficial.com/products/raqs",
    }
    with open("pipeline/config/designers.json", encoding="utf-8") as f:
        designer = next(d for d in json.load(f) if d["id"] == "abhinav_mishra")
    with open("pipeline/config/quality_gates.json", encoding="utf-8") as f:
        gates = json.load(f)
    db_refs = load_all_references(os.getenv("MONGODB_URI"), os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    doc, _ = await analyze_garment_deep(product, designer, db_refs, gates)
    techs = doc.get("techniques", {}).get("confirmed", [])
    print("techniques.confirmed:", techs)

    ok1 = (p0.get("title") or "").lower() == "raqs"
    ok2 = (tb.get("base_fabric", {}).get("name") or "").lower() == "velvet"
    ok3 = "mirror" in [t.lower() for t in techs] and "resham" in [t.lower() for t in techs]
    print("\n=== ASSERTIONS ===")
    print("products[0] is Raqs:", ok1)
    print("base_fabric.name is Velvet:", ok2)
    print("techniques has mirror + resham:", ok3)
    print("ALL PASS:", ok1 and ok2 and ok3)

asyncio.run(main())
