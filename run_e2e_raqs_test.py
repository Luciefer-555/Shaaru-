"""Full end-to-end test: shaaru_brain.answer() + RAQS enrichment + Neo4j references."""
import os
import sys
import json
import asyncio
import requests
import re
from dotenv import load_dotenv

load_dotenv()
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

USER_MESSAGE = (
    'I love this sherwani from Abhinav Mishra called RAQS — '
    'https://abhinavmishraofficial.com/products/raqs — '
    'the mirror work is insane. I want to get something exactly like this made '
    "for my best friend's wedding. Tell me everything — what fabric is this, "
    'what is this mirror work technique called, where do I source the materials '
    'in India, and give me a full tailor brief so I can actually get this made.'
)

RAQS_JSON_URL = "https://abhinavmishraofficial.com/products/raqs.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def clean_html(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "").replace("&amp;", "&").strip()


def fetch_raqs_product() -> dict:
    resp = requests.get(RAQS_JSON_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    product = resp.json()["product"]
    images = [img["src"] for img in product.get("images", []) if img.get("src")]
    variants = []
    for v in product.get("variants", []):
        variants.append({
            "color": v.get("option1") or "",
            "size": v.get("option2") or "",
            "price": float(v.get("price", 0)),
            "available": v.get("available", True),
        })
    return {
        "id": str(product["id"]),
        "title": product["title"],
        "handle": product.get("handle"),
        "product_type": product.get("product_type"),
        "tags": product.get("tags", []),
        "raw_description": clean_html(product.get("body_html", "")),
        "body_html": product.get("body_html", ""),
        "variants": variants,
        "images": images,
        "source_url": "https://abhinavmishraofficial.com/products/raqs",
    }


def load_designer_config(designer_id: str) -> dict:
    path = os.path.join(ROOT, "pipeline", "config", "designers.json")
    with open(path, encoding="utf-8") as f:
        designers = json.load(f)
    for d in designers:
        if d["id"] == designer_id:
            return d
    raise ValueError(f"Designer {designer_id} not found")


def load_gates() -> list:
    path = os.path.join(ROOT, "pipeline", "config", "quality_gates.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def query_neo4j_similar():
    from knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    if not kg.is_connected:
        return []
    cypher = """
        MATCH (p:Product)-[:HAS_TECHNIQUE]->(t:Technique)
        WHERE toLower(t.name) CONTAINS 'mirror' OR toLower(t.name) CONTAINS 'sheesha'
        RETURN p.title as title, p.image_url as image_url, p.source_url as source_url
        LIMIT 5
    """
    return kg.query(cypher)


def build_conversational_reply(result: dict, enrichment: dict | None) -> str:
    tb = result.get("tailor_brief") or {}
    sourcing = result.get("fabric_sourcing") or {}
    products = result.get("products") or []

    caption_text = ""
    techniques = []
    garment_title = "RAQS"
    fabric_info = ""
    price = ""

    if enrichment:
        cap = enrichment.get("caption") or {}
        caption_text = cap.get("text", "") if isinstance(cap, dict) else str(cap)
        tech = enrichment.get("techniques") or {}
        techniques = tech.get("confirmed", []) if isinstance(tech, dict) else tech
        garment_title = enrichment.get("title", garment_title)
        price = enrichment.get("price", "")
        raw_desc = enrichment.get("raw_description", "")
        if raw_desc and not caption_text:
            caption_text = raw_desc
        fab = enrichment.get("fabric") or {}
        if isinstance(fab, dict):
            fiber = (fab.get("fiber_type") or {}).get("value", "")
            color = (fab.get("primary_color") or {}).get("value", "")
            if fiber or color:
                fabric_info = f"{color} {fiber}".strip()
    elif products:
        p0 = products[0]
        cap = p0.get("caption", "")
        if isinstance(cap, dict):
            caption_text = cap.get("text", "")
        else:
            caption_text = str(cap) if cap else ""
        garment_title = p0.get("title", garment_title)

    if not techniques:
        techniques = result.get("graph_context", {}).get("techniques_detected", ["mirror work"])

    tech_name = techniques[0] if techniques else "mirror work (sheesha / abla bharat)"
    base_fabric = tb.get("base_fabric", {})
    fabric_name = base_fabric.get("name", "Raw Silk / Matka Silk with velvet jacket layer")
    markets = base_fabric.get("suggested_source", ["Chickpet", "Commercial Street"])

    sourcing_lines = []
    if isinstance(sourcing, dict):
        for city, data in sourcing.items():
            if isinstance(data, dict) and data.get("markets"):
                sourcing_lines.append(f"  - {city}: {', '.join(data['markets'][:4])}")

    lines = [
        "Okay bestie, RAQS is THE reference piece for your friend's wedding - let me break it down properly.",
        "",
        f"The piece: {garment_title}" + (f" ({price})" if price else "") + " by Abhinav Mishra.",
    ]
    if caption_text:
        lines.append(f"From the product listing: {caption_text[:400]}")
    if fabric_info:
        lines.append(f"Fabric-wise from vision analysis: {fabric_info}.")

    lines.extend([
        "",
        "Fabric breakdown: The RAQS set is a 3-piece - jacket and pants in velvet, kurta in special silk "
        "(Abhinav Mishra structured silk / dupion family). For replication, your tailor brief calls for "
        f"{fabric_name} as the base sherwani body at 220-280 GSM so mirror work sits flat.",
        "",
        "The mirror work technique: Sheesha embroidery (Abla Bharat) from Gujarat/Kutch tradition. "
        "Real foil-backed glass abla couched with resham silk thread - not plastic sequins.",
        "",
        "Where to source in India:",
    ])
    if sourcing_lines:
        lines.extend(sourcing_lines)
    else:
        lines.append(f"  - Bengaluru: {', '.join(markets)}")
        lines.append("  - Delhi: Chandni Chowk Kinari Bazaar, Lajpat Nagar")
        lines.append("  - Mumbai: Mangaldas Market, Bhuleshwar")

    if tb:
        lines.extend([
            "",
            "Tailor brief for your masterji:",
            f"  - Silhouette: {tb.get('silhouette_and_length', '')}",
            f"  - Base fabric: {base_fabric.get('name', '')} @ {base_fabric.get('weight', '')}",
            f"  - Lining: {tb.get('lining_recommendation', '')}",
            f"  - Embroidery: {tb.get('embroidery_technique_and_origin', tech_name)}",
            f"  - Placement: {tb.get('embroidery_placement', '')}",
            f"  - Mirror type: {tb.get('mirror_type', '')}",
            f"  - Collar: {tb.get('collar_construction', '')}",
            f"  - Closure: {tb.get('closure_type', '')}",
            f"  - Sleeves: {tb.get('sleeve_construction', '')}",
        ])
        verbatim = tb.get("tailor_instructions_verbatim")
        if verbatim:
            lines.extend(["", f'Hinglish brief: "{verbatim}"'])

    lines.extend([
        "",
        "Budget reality check: Abhinav original is about 2.51L. A skilled karigar replication runs 40K-1.2L "
        "depending on mirror density and city. Muslin trial first, then embroidery on final fabric.",
        "",
        "Want me to find sheesha specialists in your city? Tell me where you are.",
    ])
    return "\n".join(lines)


async def main():
    from shaaru_brain import answer
    from pipeline.db.db_loader import load_all_references
    from pipeline.extractors.tailor_engine import analyze_garment_deep

    print("=" * 80)
    print("STEP 1: shaaru_brain.answer()")
    print("=" * 80)

    result = await answer(USER_MESSAGE)
    enrichment = None

    cache_hit = result.get("instant_answer", {}).get("cache_hit", False)
    print(f"Cache hit: {cache_hit}")
    print(f"Needs enrichment: {result.get('needs_enrichment')}")
    print(f"Response speed: {result.get('response_speed')}")

    if result.get("needs_enrichment"):
        print("\n" + "=" * 80)
        print("STEP 2: Cache miss - fetch RAQS + analyze_garment_deep")
        print("=" * 80)

        product = fetch_raqs_product()
        print(f"Fetched product: {product['title']} (id={product['id']})")

        designer_config = load_designer_config("abhinav_mishra")
        gates = load_gates()
        db_refs = load_all_references(
            os.getenv("MONGODB_URI"),
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USER"),
            os.getenv("NEO4J_PASSWORD"),
        )

        enrichment, cost = await analyze_garment_deep(
            product, designer_config, db_refs, gates
        )
        print(f"Enrichment cost estimate: {cost}")
        result["enrichment"] = enrichment
        result["products"] = [enrichment]

    print("\n" + "=" * 80)
    print("STEP 3: FULL RAW OUTPUT DICT")
    print("=" * 80)
    print(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 80)
    print("STEP 4: TAILOR BRIEF (separate)")
    print("=" * 80)
    print(json.dumps(result.get("tailor_brief", {}), indent=2, default=str))

    print("\n" + "=" * 80)
    print("STEP 5: SIMILAR REFERENCES FROM NEO4J")
    print("=" * 80)
    refs = query_neo4j_similar()
    if refs:
        for i, r in enumerate(refs, 1):
            print(f"\n[{i}]")
            print(json.dumps(r, indent=2, default=str))
    else:
        print("(No results or Neo4j not connected)")

    print("\n" + "=" * 80)
    print("STEP 6: SHAARU CONVERSATIONAL REPLY")
    print("=" * 80)
    reply = build_conversational_reply(result, enrichment)
    print(reply)


if __name__ == "__main__":
    asyncio.run(main())
