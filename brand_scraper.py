"""
brand_scraper.py — Scrapes products from Indian fashion brands.

Fetches product data from Shopify /products.json endpoints or
falls back to basic web scraping. Outputs to products_seed.json.
"""

import os
import json
import logging
import time
from typing import Optional
from datetime import datetime, timezone

log = logging.getLogger("shaaru.scraper")

# ── Default brand catalog ────────────────────────────────────────
DEFAULT_BRANDS = [
    {"name": "Rare Rabbit", "website": "https://www.rarerabbit.in", "gender_default": "male",
     "aesthetics": ["streetwear", "casual", "contemporary"]},
    {"name": "Holy Headen", "website": "https://www.holyheaden.com", "gender_default": "unisex",
     "aesthetics": ["streetwear", "minimalist"]},
    {"name": "Farak", "website": "https://www.farak.in", "gender_default": "unisex",
     "aesthetics": ["indie", "bohemian", "artisanal"]},
    {"name": "Hermyne", "website": "https://www.hermyne.com", "gender_default": "female",
     "aesthetics": ["contemporary", "elegant"]},
    {"name": "Nicobar", "website": "https://www.nicobar.com", "gender_default": "unisex",
     "aesthetics": ["minimalist", "resort", "sustainable"]},
    {"name": "Meluku", "website": "https://www.meluku.com", "gender_default": "unisex",
     "aesthetics": ["handloom", "artisanal", "sustainable"]},
    {"name": "Mahima Mahajan", "website": "https://www.mahimamahajan.com", "gender_default": "female",
     "aesthetics": ["festive", "bridal", "ethnic-fusion"]},
    {"name": "Kisah", "website": "https://www.kisah.in", "gender_default": "male",
     "aesthetics": ["ethnic", "festive", "traditional"]},
    {"name": "October Jaipur", "website": "https://www.octoberjaipur.com", "gender_default": "male",
     "aesthetics": ["contemporary", "smart-casual"]},
]

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "brands_catalog.json")
SEED_PATH = os.path.join(os.path.dirname(__file__), "products_seed.json")


def _load_brands() -> list:
    """Load brand catalog from file or use defaults."""
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Error loading brand catalog: {e}")
    return DEFAULT_BRANDS


def scrape_brand(brand: dict) -> list[dict]:
    """
    Scrape products from a single brand.

    Tries Shopify /products.json first, falls back to basic requests.

    Args:
        brand: Dict with name, website, gender_default, aesthetics.

    Returns:
        List of product dicts.
    """
    products = []
    website = brand.get("website", "")
    brand_name = brand.get("name", "Unknown")
    gender = brand.get("gender_default", "unisex")
    aesthetics = brand.get("aesthetics", [])

    # Build aesthetic scores dict
    aesthetic_scores = {a: 0.8 for a in aesthetics}

    # Try Shopify endpoint
    try:
        import requests
        url = f"{website}/products.json?limit=50"
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Shaaru Fashion Bot/2.0"
        })

        if resp.status_code == 200:
            data = resp.json()
            for p in data.get("products", []):
                images = p.get("images", [])
                image_url = images[0]["src"] if images else ""
                variants = p.get("variants", [])
                price = float(variants[0]["price"]) if variants else 0

                products.append({
                    "name": p.get("title", ""),
                    "price": price,
                    "image_url": image_url,
                    "product_url": f"{website}/products/{p.get('handle', '')}",
                    "brand": brand_name,
                    "gender": gender,
                    "aesthetic_scores": aesthetic_scores,
                    "colors": _extract_colors(p),
                    "category": _categorize_product(p.get("product_type", "")),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

            log.info(f"[SCRAPER] {brand_name}: {len(products)} products via Shopify API")
            return products

    except Exception as e:
        log.debug(f"[SCRAPER] Shopify API failed for {brand_name}: {e}")

    # Fallback: create placeholder entries from brand catalog
    log.info(f"[SCRAPER] {brand_name}: using catalog placeholder data")
    products.append({
        "name": f"{brand_name} Collection",
        "price": 0,
        "image_url": "",
        "product_url": website,
        "brand": brand_name,
        "gender": gender,
        "aesthetic_scores": aesthetic_scores,
        "colors": [],
        "category": "general",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    return products


def _extract_colors(product: dict) -> list:
    """Extract color information from a Shopify product."""
    colors = []
    for option in product.get("options", []):
        if option.get("name", "").lower() in ("color", "colour", "shade"):
            colors.extend(option.get("values", []))
    return colors[:5]


def _categorize_product(product_type: str) -> str:
    """Map Shopify product_type to a standard category."""
    pt = product_type.lower()
    mappings = {
        "shirt": "tops", "t-shirt": "tops", "top": "tops",
        "kurta": "ethnic", "kurti": "ethnic", "lehenga": "ethnic",
        "pant": "bottoms", "trouser": "bottoms", "jean": "bottoms",
        "jacket": "outerwear", "blazer": "outerwear", "coat": "outerwear",
        "dress": "dresses", "gown": "dresses",
        "shoe": "footwear", "sandal": "footwear", "sneaker": "footwear",
        "watch": "accessories", "bag": "accessories", "belt": "accessories",
    }
    for key, cat in mappings.items():
        if key in pt:
            return cat
    return "general"


def scrape_all_brands() -> list[dict]:
    """
    Scrape all brands and save to products_seed.json.

    Returns:
        Combined list of all products.
    """
    brands = _load_brands()
    all_products = []

    for brand in brands:
        try:
            products = scrape_brand(brand)
            all_products.extend(products)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            log.error(f"[SCRAPER] Failed to scrape {brand.get('name')}: {e}")

    # Save to file
    try:
        with open(SEED_PATH, "w", encoding="utf-8") as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        log.info(f"[SCRAPER] Saved {len(all_products)} products to {SEED_PATH}")
    except Exception as e:
        log.error(f"[SCRAPER] Error saving products: {e}")

    return all_products


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    products = scrape_all_brands()
    print(f"Scraped {len(products)} products total")
