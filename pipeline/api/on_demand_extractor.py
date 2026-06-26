import os
import sys
import json
import asyncio
import datetime
import requests
import urllib3
urllib3.disable_warnings()

# Add pipeline and project root directories to sys.path
pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pipeline_dir not in sys.path:
    sys.path.append(pipeline_dir)
root_dir = os.path.abspath(os.path.join(pipeline_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

kg = None
db_refs = None
gates = None

def _ensure_initialized():
    global kg, db_refs, gates
    if kg is None:
        try:
            from knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
        except Exception as e:
            print(f"[KG] Init error: {e}")
            kg = None
            
    if db_refs is None:
        from db.db_loader import load_all_references
        db_refs = load_all_references(
            os.getenv("MONGODB_URI"),
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USER"),
            os.getenv("NEO4J_PASSWORD")
        )
        
    if gates is None:
        from validators.quality_gate import load_quality_gates
        gates_path = os.path.join(pipeline_dir, "config", "quality_gates.json")
        gates = load_quality_gates(gates_path)

def get_designer_config(designer_id: str = None):
    config_path = os.path.join(pipeline_dir, "config", "designers.json")
    with open(config_path, "r", encoding="utf-8") as f:
        designers = json.load(f)
    if not designer_id:
        return designers[0]
    for d in designers:
        if d["id"] == designer_id or d["name"].lower() == designer_id.lower():
            return d
    return designers[0]

def save_to_review_folder(output: dict):
    output_dir = os.path.join(pipeline_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    source_id = output.get("source_id", "unknown")
    designer = output.get("designer", "designer").lower().replace(" ", "_")
    filepath = os.path.join(output_dir, f"{designer}_on_demand_{source_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([output], f, indent=2)

async def find_matching_product(query: str, designer_id: str = None):
    """
    Search Shopify for a product matching the user query.
    Uses tags and title matching.
    """
    config = get_designer_config(designer_id)
    store_url = config['url']
    base_url = store_url if store_url.startswith("http") else f"https://{store_url}"
    url = f"{base_url}/products.json?limit=250"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15, verify=False)
        products = resp.json().get("products", [])
    except Exception as e:
        print(f"Error fetching catalog from {url}: {e}")
        return None
    
    query_terms = query.lower().split()
    scored = []
    
    from scrapers.shopify_scraper import clean_html
    
    for p in products:
        score = 0
        searchable = (
            p.get("title", "") + " " +
            " ".join(p.get("tags", [])) + " " +
            p.get("body_html", "")
        ).lower()
        
        for term in query_terms:
            if term in searchable:
                score += 1
        
        if score > 0:
            desc = clean_html(p.get("body_html", ""))
            images = [img.get("src") for img in p.get("images", []) if img.get("src")]
            variants = []
            for v in p.get("variants", []):
                variants.append({
                    "color": v.get("option1") or "",
                    "size": v.get("option2") or "",
                    "price": float(v.get("price", 0)),
                    "available": v.get("available", True)
                })
            item_dict = {
                "id": str(p.get("id")),
                "title": p.get("title"),
                "handle": p.get("handle"),
                "product_type": p.get("product_type"),
                "tags": p.get("tags", []),
                "raw_description": desc,
                "variants": variants,
                "images": images,
                "created_at": p.get("created_at"),
                "source_url": f"{base_url}/products/{p.get('handle')}"
            }
            scored.append((score, item_dict))
    
    if not scored:
        return None
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

async def analyze_garment_deep(
    product: dict,
    designer_config: dict,
    db_refs: dict,
    gates: list
) -> tuple[dict, str]:
    from tailor_engine import analyze_garment_deep as engine_analyze
    from run_pipeline import get_image_base64
    
    image_b64 = ""
    if product.get("images"):
        image_b64 = get_image_base64(product["images"][0])
        
    product_page_text = f"Title: {product.get('title', '')}\n"
    product_page_text += f"Tags: {', '.join(product.get('tags', []))}\n"
    product_page_text += f"Description: {product.get('raw_description', '')}\n"

    price = ""
    if product.get("variants") and len(product["variants"]) > 0:
        price_val = product["variants"][0].get("price")
        if price_val: price = f"₹{price_val}"

    db_fabric_names = set(db_refs["fabrics"]["name_index"].keys())
    db_embellishment_ids = list(db_refs["embellishments"]["lookup"].keys())

    combined = await engine_analyze(
        image_b64=image_b64,
        product_page_text=product_page_text,
        db_fabric_ids=db_fabric_names,
        db_embellishment_ids=db_embellishment_ids,
        designer_config=designer_config,
        gates=gates,
        price=price
    )
    
    final_doc = {
        "source_id": product["id"],
        "designer": designer_config["name"],
        "platform": designer_config["platform"],
        "title": product["title"],
        "source_url": product["source_url"],
        "images": product["images"],
        "raw_description": product["raw_description"],
        "variants": product["variants"],
        "scraped_at": datetime.datetime.now().isoformat(),
        "reviewed": False,
        **combined
    }
    return final_doc, "~$0.02"

async def extract_on_demand(
    user_query: str,
    designer_id: str = None
) -> dict:
    """
    Triggered when user asks about a specific 
    aesthetic/technique/product.
    
    1. Search Neo4j for matching products already extracted
    2. If found with caption → return immediately (instant)
    3. If not found → find matching product from Shopify
    4. Run extraction pipeline on that ONE product
    5. Store in MongoDB + Neo4j
    6. Return to user
    """
    _ensure_initialized()
    
    # Step 1 — Check if already extracted
    if kg and kg.driver:
        try:
            with kg.driver.session() as session:
                existing = session.run("""
                    MATCH (p:Product)-[:HAS_TECHNIQUE]->(t:Technique)
                    WHERE toLower(t.name) CONTAINS toLower($query)
                    AND p.caption IS NOT NULL
                    RETURN p.title, p.source_url, p.caption,
                           p.designer, p.image_url
                    LIMIT 5
                """, query=user_query)
                
                results = [dict(r) for r in existing]
                if results:
                    return {
                        "source": "cached",
                        "products": results,
                        "extraction_needed": False
                    }
        except Exception as e:
            print(f"[KG] Cache check error: {e}")
    
    # Step 2 — Nothing cached, find and extract
    # Search Shopify for matching product
    matching_product = await find_matching_product(
        query=user_query,
        designer_id=designer_id
    )
    
    if not matching_product:
        return {"error": "no matching product found"}
    
    # Step 3 — Extract this one product
    output, cost = await analyze_garment_deep(
        product=matching_product,
        designer_config=get_designer_config(designer_id),
        db_refs=db_refs,
        gates=gates
    )
    
    # Step 4 — Store immediately
    if kg:
        try:
            kg.sync_product_document(output)
        except Exception as e:
            print(f"[KG] Sync error: {e}")
            
    try:
        from shaaru_brain import _get_db
        db = _get_db()
        if db:
            db['products'].update_one(
                {'source_id': output['source_id']},
                {'$set': output},
                upsert=True
            )
            print(f"[MongoDB] Synced on-demand product: {output.get('title')}")
    except Exception as e:
        print(f"[MongoDB] Upsert error: {e}")

    save_to_review_folder(output)
    
    # Step 5 — Return to user
    return {
        "source": "freshly_extracted",
        "product": output,
        "extraction_needed": False,
        "time_taken": "~2 minutes"
    }


class OnDemandExtractor:
    """Proactive wrapper for scheduler integration."""
    async def extract_for_trend(self, trend: dict, designer_id: str = None, count: int = 5):
        query = trend.get("trend_name", "")
        print(f"[Proactive] Extracting {count} items for trend '{query}' from {designer_id}...")
        res = await extract_on_demand(user_query=query, designer_id=designer_id)
        return res
