"""
tag_product_gender.py — Backfill gender tags on existing products.

One-shot script that tags products in products_seed.json and MongoDB
with gender based on brand defaults.
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.gender_tag")

BRAND_GENDER_DEFAULTS = {
    "Holy Headen": "unisex",
    "Farak": "unisex",
    "Breakkin Even": "unisex",
    "Hermyne": "female",
    "Aiv Atelier": "female",
    "Mahima Mahajan": "female",
    "Rare Rabbit": "male",
    "Nicobar": "unisex",
    "Meluku": "unisex",
    "Kisah": "male",
    "October Jaipur": "male",
    "Zara": "unisex",
    "H&M": "unisex",
    "UNIQLO India": "unisex",
}

SEED_PATH = os.path.join(os.path.dirname(__file__), "products_seed.json")


def tag_products_in_json():
    """Tag products in products_seed.json with gender field."""
    if not os.path.exists(SEED_PATH):
        log.warning(f"[GENDER] {SEED_PATH} not found")
        return

    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            products = json.load(f)
    except Exception as e:
        log.error(f"[GENDER] Error reading seed file: {e}")
        return

    tagged = 0
    for product in products:
        if not product.get("gender"):
            brand = product.get("brand", "")
            product["gender"] = BRAND_GENDER_DEFAULTS.get(brand, "unisex")
            tagged += 1

    try:
        with open(SEED_PATH, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        log.info(f"[GENDER] Tagged {tagged} products in products_seed.json")
    except Exception as e:
        log.error(f"[GENDER] Error writing seed file: {e}")


def tag_products_in_mongo():
    """Tag products in MongoDB with gender field."""
    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("MONGODB_DB", "shaaru")
        db = MongoClient(uri, serverSelectionTimeoutMS=3000)[db_name]

        tagged = 0
        for product in db["products"].find({"gender": {"$exists": False}}):
            brand = product.get("brand", "")
            gender = BRAND_GENDER_DEFAULTS.get(brand, "unisex")
            db["products"].update_one(
                {"_id": product["_id"]},
                {"$set": {"gender": gender}},
            )
            tagged += 1

        log.info(f"[GENDER] Tagged {tagged} products in MongoDB")
    except Exception as e:
        log.error(f"[GENDER] MongoDB tagging failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Tagging products in JSON...")
    tag_products_in_json()
    print("Tagging products in MongoDB...")
    tag_products_in_mongo()
    print("Done.")
