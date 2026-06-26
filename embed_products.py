"""
One-time script: embed all products with FashionCLIP and fix field names.
Run: python embed_products.py

Fixes:
  - product_name → name
  - pricing.price_inr → price (flat int)
  - adds image_embedding (512-dim vector)
  - sets embedding_source ("image" or "text:<fallback>")
"""
import logging
from shaaru_brain import _get_db
from fashion_clip_embedder import embed_image_url, embed_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_fallback_text(p: dict) -> str:
    parts = [
        p.get("name") or p.get("product_name", ""),
        p.get("aesthetic", ""),
        p.get("color", ""),
        p.get("category", ""),
        p.get("silhouette", ""),
    ]
    return " ".join(x for x in parts if x).strip()


def run():
    db = _get_db()
    if db is None:
        logger.error("MongoDB connection failed")
        return

    products = list(db["products"].find({}))
    logger.info(f"Processing {len(products)} products")
    success = skipped = failed = 0

    for p in products:
        pid = p["_id"]
        updates: dict = {}

        # Fix field names
        if "product_name" in p and "name" not in p:
            updates["name"] = p["product_name"]

        pricing = p.get("pricing", {})
        if isinstance(pricing, dict) and "price" not in p:
            updates["price"] = pricing.get("price_inr")

        # Skip embedding if already done
        if p.get("image_embedding"):
            if updates:
                db["products"].update_one({"_id": pid}, {"$set": updates})
            skipped += 1
            continue

        # Embed
        image_url = p.get("image_url")
        if image_url:
            vector = embed_image_url(image_url)
            source = "image"
        else:
            fallback = _build_fallback_text(p)
            vector = embed_text(fallback)
            source = f"text:{fallback}"

        if vector:
            updates["image_embedding"] = vector
            updates["embedding_source"] = source
            success += 1
            logger.info(f"  ✓ {updates.get('name') or p.get('product_name')} ({source[:40]})")
        else:
            failed += 1
            logger.warning(f"  ✗ Failed: {pid}")

        if updates:
            db["products"].update_one({"_id": pid}, {"$set": updates})

    logger.info(f"\nDone — {success} embedded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    run()
