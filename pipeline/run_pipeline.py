import argparse
import json
import os
import sys
import datetime
import traceback
import asyncio
import requests
import base64
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from scrapers.shopify_scraper import scrape_shopify
from scrapers.editorial_scraper import scrape_editorial
from scrapers.custom_scraper import scrape_custom
from tailor_engine import analyze_garment_deep
from deduplicator.dedup import deduplicate_batch
from db.schema import validate_product_document
from db.db_loader import load_all_references
from validators.quality_gate import load_quality_gates

def get_designer_config(source_id: str):
    config_path = os.path.join(os.path.dirname(__file__), "config", "designers.json")
    with open(config_path, "r") as f:
        designers = json.load(f)
    for d in designers:
        if d["id"] == source_id:
            return d
    return None

def get_image_base64(image_url):
    try:
        resp = requests.get(image_url, timeout=10)
        if resp.status_code == 200:
            image_data = resp.content
            # Resize image to prevent NVIDIA NIM 413 Payload Too Large
            try:
                img = Image.open(io.BytesIO(image_data))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                max_dim = 1024
                if img.width > max_dim or img.height > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                image_data = buf.getvalue()
            except Exception as resize_e:
                print(f"Warning: Failed to resize image, using original. Error: {resize_e}")

            return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        print(f"Failed to fetch image: {e}")
    return ""

def load_checkpoint(source_id: str):
    ckpt_path = os.path.join(os.path.dirname(__file__), "output", f"checkpoint_{source_id}.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            return json.load(f).get("processed_ids", [])
    return []

def save_checkpoint(source_id: str, processed_ids: list):
    ckpt_path = os.path.join(os.path.dirname(__file__), "output", f"checkpoint_{source_id}.json")
    with open(ckpt_path, "w") as f:
        json.dump({"processed_ids": processed_ids}, f)

def save_checkpoint_single(source: str, product_id: str):
    path = os.path.join(os.path.dirname(__file__), "output", f"checkpoint_{source}.json")
    existing = {"processed_ids": []}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    if "processed_ids" not in existing:
        existing["processed_ids"] = []
    if product_id not in existing["processed_ids"]:
        existing["processed_ids"].append(product_id)
        with open(path, "w") as f:
            json.dump(existing, f)


async def main():
    parser = argparse.ArgumentParser(description="Shaaru Fashion Knowledge Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Designer ID")
    parser.add_argument("--mode", type=str, choices=["product", "editorial"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit items to process")
    parser.add_argument("--product", type=str, help="Specific product ID or title to process")
    parser.add_argument("--balance-genders", action="store_true", help="Force 50/50 split of menswear and womenswear")
    parser.add_argument("--test-only", action="store_true", help="Run only the first product to test the quality gates")
    args = parser.parse_args()
    
    config = get_designer_config(args.source)
    if not config:
        print(f"Source {args.source} not found in designers.json")
        return
        
    print(f"Starting pipeline for {config['name']} ({args.mode} mode)")
    
    output_dir = os.path.join(os.path.dirname(__file__), "output", "review")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scrape
    products_scraped = []
    if args.mode == "product":
        men_count = 0
        women_count = 0
        target_half = args.limit // 2 if args.limit > 0 else 0
        
        def is_menswear(p):
            txt = (p.get("title", "") + " " + " ".join(p.get("tags", [])) + " " + str(p.get("product_type", ""))).lower()
            return any(w in txt for w in ["men", "sherwani", "kurta set", "groom", "bandi", "achkan", "menswear", "boy"])

        if config["platform"] == "shopify":
            for p in scrape_shopify(config["url"]):
                if args.balance_genders and args.limit > 0:
                    mens = is_menswear(p)
                    if mens and men_count >= target_half: continue
                    if not mens and women_count >= (args.limit - target_half): continue
                    if mens: men_count += 1
                    else: women_count += 1
                    
                products_scraped.append(p)
        else:
            for p in scrape_custom(config):
                if args.balance_genders and args.limit > 0:
                    mens = is_menswear(p)
                    if mens and men_count >= target_half: continue
                    if not mens and women_count >= (args.limit - target_half): continue
                    if mens: men_count += 1
                    else: women_count += 1
                    
                products_scraped.append(p)
            
    print(f"Scraped {len(products_scraped)} products.")
    
    print("Loading DB references...")
    db_refs = load_all_references(
        os.getenv("MONGODB_URI"),
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )
    
    db_fabric_names = set(db_refs["fabrics"]["name_index"].keys())
    db_embellishment_ids = list(db_refs["embellishments"]["lookup"].keys())
    
    print("Loading Quality Gates...")
    gates_path = os.path.join(os.path.dirname(__file__), "config", "quality_gates.json")
    gates = load_quality_gates(gates_path)
    
    processed_ids = load_checkpoint(args.source)
    print(f"Loaded checkpoint. Already processed {len(processed_ids)} items.")
    
    products_to_process = [p for p in products_scraped if p["id"] not in processed_ids]
    
    if args.product:
        products_to_process = [p for p in products_scraped if args.product.lower() in p["id"].lower() or args.product.lower() in p["title"].lower()]
        
    if args.limit > 0:
        products_to_process = products_to_process[:args.limit]
        
    print(f"Products left to process: {len(products_to_process)}")

    BATCH_SIZE = 25
    
    # Accumulators for the final complete summary
    complete_summary = {
        "total_processed": 0,
        "aesthetic_distribution": {},
        "new_fabric_candidates": [],
        "new_embellishment_candidates": [],
        "new_pattern_candidates": [],
        "needs_manual_review_count": 0,
        "total_cost": {}
    }

    # ─────────────────────────────────────────────────────────
    # MANDATORY: Single product test before batch
    # ─────────────────────────────────────────────────────────
    if products_to_process:
        print(f"Running mandatory single-product test for {config['id']}...")
        test_p = products_to_process[0]
        image_b64 = get_image_base64(test_p["images"][0]) if test_p.get("images") else ""
        product_page_text = f"Title: {test_p.get('title', '')}\nTags: {', '.join(test_p.get('tags', []))}\nDescription: {test_p.get('raw_description', '')}\n"
        price = f"₹{test_p['variants'][0].get('price')}" if test_p.get("variants") and len(test_p["variants"]) > 0 and test_p["variants"][0].get("price") else ""
        
        test_combined = await analyze_garment_deep(
            image_b64=image_b64,
            product_page_text=product_page_text,
            db_fabric_ids=db_fabric_names,
            db_embellishment_ids=db_embellishment_ids,
            designer_config=config,
            gates=gates,
            price=price
        )
        
        if not test_combined.get("quality_gate_passed"):
            print(f"[WARNING] Single product test failed for {config['id']}")
            print(f"Continuing batch run anyway, but quality may be degraded.")
            print(json.dumps(test_combined, indent=2))
        else:
            print(f"[PASSED] Single product test passed.\n")
        if args.test_only:
            print(f"--- TEST ONLY MODE ---")
            print(json.dumps(test_combined, indent=2))
            return

    batch_idx = 1
    for i in range(0, len(products_to_process), BATCH_SIZE):
        batch = products_to_process[i:i+BATCH_SIZE]
        print(f"\n--- Starting Batch {batch_idx} ({len(batch)} items) ---")
        
        batch_results = []
        batch_duplicates = []
        
        verification_report = {
            "batch_summary": {
                "total_processed": 0,
                "duplicates_flagged": 0,
                "needs_manual_review": 0
            },
            "aesthetic_distribution": {},
            "confidence_flags": [],
            "fabric_vocabulary_coverage": {
                "confirmed_only": 0,
                "vision_only_count": 0
            }
        }

        sem = asyncio.Semaphore(3)
        kg_instance = None
        try:
            import sys
            if "." not in sys.path: sys.path.append(".")
            from knowledge_graph import KnowledgeGraph
            kg_instance = KnowledgeGraph()
        except Exception as e:
            print(f"[KG] Init skipped in pipeline: {e}")

        async def process_item(raw_p):
            async with sem:
                print(f"Processing {raw_p['title']}...")
                
                image_b64 = ""
                if raw_p.get("images"):
                    image_b64 = get_image_base64(raw_p["images"][0])
                    
                product_page_text = f"Title: {raw_p.get('title', '')}\n"
                product_page_text += f"Tags: {', '.join(raw_p.get('tags', []))}\n"
                product_page_text += f"Description: {raw_p.get('raw_description', '')}\n"

                price = ""
                if raw_p.get("variants") and len(raw_p["variants"]) > 0:
                    price_val = raw_p["variants"][0].get("price")
                    if price_val: price = f"₹{price_val}"

                try:
                    combined = await analyze_garment_deep(
                        image_b64=image_b64,
                        product_page_text=product_page_text,
                        db_fabric_ids=db_fabric_names,
                        db_embellishment_ids=db_embellishment_ids,
                        designer_config=config,
                        gates=gates,
                        price=price
                    )
                except Exception as e:
                    print(f"Error processing {raw_p['id']}: {e}")
                    traceback.print_exc()
                    return None
                    
                final_doc = {
                    "source_id": raw_p["id"],
                    "designer": config["name"],
                    "platform": config["platform"],
                    "title": raw_p["title"],
                    "source_url": raw_p["source_url"],
                    "images": raw_p["images"],
                    "raw_description": raw_p["raw_description"],
                    "variants": raw_p["variants"],
                    "scraped_at": datetime.datetime.now().isoformat(),
                    "reviewed": False,
                    **combined
                }
                save_checkpoint_single(config["id"], raw_p["id"])
                item_file = os.path.join(output_dir, f"{config['id']}_item_{raw_p['id']}.json")
                with open(item_file, "w", encoding="utf-8") as f:
                    json.dump([final_doc], f, indent=2)

                if kg_instance:
                    try:
                        kg_instance.sync_product_document(final_doc)
                    except Exception as e:
                        print(f"[KG] Neo4j sync skipped: {e}")

                return raw_p["id"], final_doc

        tasks = [process_item(p) for p in batch]
        results = await asyncio.gather(*tasks)

        for res in results:
            if not res: continue
            raw_id, final_doc = res
            
            # Record summary data
            cat = final_doc.get("aesthetic_category", "Unknown")
            complete_summary["aesthetic_distribution"][cat] = complete_summary["aesthetic_distribution"].get(cat, 0) + 1
            verification_report["aesthetic_distribution"][cat] = verification_report["aesthetic_distribution"].get(cat, 0) + 1
            
            if final_doc.get("needs_manual_review"):
                complete_summary["needs_manual_review_count"] += 1
                verification_report["batch_summary"]["needs_manual_review"] += 1
                
            if final_doc.get("new_fabric_candidates"):
                complete_summary["new_fabric_candidates"].extend(final_doc["new_fabric_candidates"])
            if final_doc.get("new_embellishment_candidates"):
                complete_summary["new_embellishment_candidates"].extend(final_doc["new_embellishment_candidates"])
            if final_doc.get("new_pattern_candidates"):
                complete_summary["new_pattern_candidates"].extend(final_doc["new_pattern_candidates"])
                
            vocab = final_doc.get("fabric_vocabulary", {})
            if vocab.get("confirmed"): verification_report["fabric_vocabulary_coverage"]["confirmed_only"] += len(vocab["confirmed"])
            if vocab.get("vision_only"): verification_report["fabric_vocabulary_coverage"]["vision_only_count"] += len(vocab["vision_only"])

            if final_doc.get("confidence_notes"):
                verification_report["confidence_flags"].append({"id": raw_id, "notes": final_doc["confidence_notes"]})

            batch_results.append(final_doc)
            processed_ids.append(raw_id)

        # Dedup batch progressively
        existing = [] 
        unique_products, duplicates, emb_tokens = deduplicate_batch(batch_results, existing, config["id"])
        
        verification_report["batch_summary"]["total_processed"] = len(batch_results)
        verification_report["batch_summary"]["duplicates_flagged"] = len(duplicates)
        
        # Schema Validate
        final_output = []
        for doc in unique_products:
            if validate_product_document(doc):
                final_output.append(doc)
            else:
                print(f"Warning: {doc['source_id']} failed schema validation.")
                final_output.append(doc)
                    
        out_file = os.path.join(output_dir, f"{config['id']}_batch_{batch_idx}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)
                
        rep_file = os.path.join(os.path.dirname(__file__), "output", f"verification_report_{config['id']}_batch_{batch_idx}.json")
        with open(rep_file, "w", encoding="utf-8") as f:
            json.dump(verification_report, f, indent=2)
            
        complete_summary["total_processed"] += len(batch_results)
        batch_idx += 1
        
    # Write COMPLETE summary
    summary_file = os.path.join(os.path.dirname(__file__), "output", f"{config['id']}_COMPLETE_summary.json")
    with open(summary_file, "w") as f:
        json.dump(complete_summary, f, indent=2)
        
    print(f"Finished pipeline for {config['id']}. Processed {complete_summary['total_processed']} products.")

if __name__ == "__main__":
    asyncio.run(main())
