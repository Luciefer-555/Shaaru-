import os
import sys
import json
import asyncio
import requests
import base64
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from db.db_loader import load_all_references
from tailor_engine import analyze_garment_deep

def fetch_single_product(url):
    json_url = url + ".json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    resp = requests.get(json_url, headers=headers)
    if resp.status_code != 200:
        print(f"Failed to fetch {json_url}")
        return None
    data = resp.json()
    product = data.get("product")
    if not product:
        return None
        
    price = ""
    if product.get("variants") and len(product["variants"]) > 0:
        price_val = product["variants"][0].get("price")
        if price_val:
            price = f"₹{price_val}"

    return {
        "id": str(product["id"]),
        "title": product["title"],
        "source_url": url,
        "raw_description": product.get("body_html", ""),
        "price": price,
        "tags": product.get("tags", "").split(",") if isinstance(product.get("tags"), str) else product.get("tags", []),
        "images": [img["src"] for img in product.get("images", [])]
    }

def get_image_base64(image_url):
    resp = requests.get(image_url)
    if resp.status_code == 200:
        return base64.b64encode(resp.content).decode("utf-8")
    return ""

async def main():
    print("Loading DB references...")
    db_refs = load_all_references(
        os.getenv("MONGODB_URI"),
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )
    
    db_fabric_ids = set(db_refs["fabrics"]["lookup"].keys())
    db_embellishment_ids = list(db_refs["embellishments"]["lookup"].keys())
    
    url = "https://abhinavmishraofficial.com/products/raqs"
    print(f"Fetching product data from {url}")
    raw_p = fetch_single_product(url)
    
    if not raw_p:
        print("Could not fetch product.")
        return
        
    print(f"Fetching image from {raw_p['images'][0]}")
    image_b64 = get_image_base64(raw_p['images'][0])
    
    # Combine title, description, and tags for product_page_text
    product_page_text = f"Title: {raw_p['title']}\n"
    product_page_text += f"Tags: {', '.join(raw_p['tags'])}\n"
    product_page_text += f"Description: {raw_p['raw_description']}\n"
    
    print(f"Running Tailor Engine for {raw_p['title']}...")
    output = await analyze_garment_deep(
        image_b64=image_b64,
        product_page_text=product_page_text,
        db_fabric_ids=db_fabric_ids,
        db_embellishment_ids=db_embellishment_ids,
        price=raw_p['price']
    )
    
    print("\n--- EXTRACTED KNOWLEDGE DOCUMENT ---")
    print(json.dumps(output, indent=2))
    
    # Save output for viewing
    out_file = os.path.join(os.path.dirname(__file__), "output", "test_raqs_output.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
