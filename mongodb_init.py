"""
mongodb_init.py
SHAARU — MongoDB RSI + LUM Brain Initialization
Builds all 6 collections, indexes, and seeds product + brand data.
Run from: C:\\Users\\saipr\\Downloads\\Shaaru
Run: python mongodb_init.py
"""

import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid

load_dotenv()

# ── ENV ──────────────────────────────────────────────────────────────────────
MONGO_URI = (
    os.getenv("MONGODB_URI")
    or os.getenv("MONGO_URI")
    or os.getenv("MONGODB_URL")
)
DB_NAME = os.getenv("MONGODB_DB_NAME", "shaaru")

if not MONGO_URI:
    print("[FAIL] No MongoDB URI found.")
    print("       Add MONGODB_URI=<your atlas uri> to your .env file")
    sys.exit(1)

NOW = datetime.now(timezone.utc)


# ── LOOKUP MAPS ───────────────────────────────────────────────────────────────

CATEGORY_SILHOUETTE = {
    "Dresses":  "dress",
    "Co-ords":  "co-ord set",
    "Tops":     "draped top",
    "Blazers":  "structured blazer",
    "Shirts":   "relaxed shirt",
    "Kurtas":   "straight kurta",
    "Suits":    "tailored suit",
    "Gowns":    "floor-length gown",
    "Lehengas": "lehenga flare",
    "Sarees":   "draped saree",
    "Jewelry":  "accessory",
    "T-Shirts": "oversized tee",
}

AESTHETIC_BODY_MAP = {
    "Minimalist":       ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Quiet Luxury":     ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Editorial":        ["hourglass", "inverted_triangle", "rectangle"],
    "Streetwear":       ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Bohemian":         ["apple", "hourglass", "pear"],
    "Resort Wear":      ["hourglass", "pear", "rectangle"],
    "Ethnic":           ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Festive":          ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Y2K":              ["hourglass", "inverted_triangle", "rectangle"],
    "Feminine":         ["hourglass", "pear"],
    "Old Money":        ["hourglass", "inverted_triangle", "rectangle"],
    "Cottagecore":      ["apple", "hourglass", "pear"],
    "Avant-garde":      ["inverted_triangle", "rectangle"],
    "High Fashion":     ["inverted_triangle", "rectangle"],
    "Couture":          ["hourglass", "inverted_triangle"],
    "Indo-western":     ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Casual":           ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Everyday":         ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Indie":            ["hourglass", "pear", "rectangle"],
    "Traditional":      ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
    "Formal":           ["hourglass", "inverted_triangle", "rectangle"],
}

COLOR_MONK_MAP = {
    "Ivory":          ["M1", "M2", "M3", "M4"],
    "Ecru":           ["M1", "M2", "M3", "M4"],
    "Champagne":      ["M1", "M2", "M3", "M4", "M5"],
    "Off White":      ["M1", "M2", "M3", "M4"],
    "Natural Beige":  ["M3", "M4", "M5", "M6"],
    "Warm Sand":      ["M3", "M4", "M5", "M6"],
    "Black":          ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Navy Blue":      ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Midnight Blue":  ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Deep Maroon":    ["M4", "M5", "M6", "M7", "M8"],
    "Sage Green":     ["M2", "M3", "M4", "M5", "M6"],
    "Sage":           ["M2", "M3", "M4", "M5", "M6"],
    "Forest Green":   ["M4", "M5", "M6", "M7", "M8"],
    "Gold":           ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Deep Gold":      ["M4", "M5", "M6", "M7", "M8"],
    "Pearl White":    ["M1", "M2", "M3", "M4", "M5"],
    "Silver":         ["M1", "M2", "M3", "M4", "M5", "M6"],
    "Terracotta":     ["M4", "M5", "M6", "M7", "M8"],
    "Rust":           ["M4", "M5", "M6", "M7", "M8"],
    "Multi":          ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Sunset Multi":   ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Denim Blue":     ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Rose Gold":      ["M2", "M3", "M4", "M5", "M6"],
    "Teal":           ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Ivory Gold":     ["M2", "M3", "M4", "M5", "M6"],
    "Haldi Yellow":   ["M4", "M5", "M6", "M7", "M8"],
    "Cream":          ["M2", "M3", "M4", "M5"],
    "Dusty Rose":     ["M2", "M3", "M4", "M5", "M6"],
    "Indigo":         ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Indigo Block":   ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Vibrant Indigo": ["M3", "M4", "M5", "M6", "M7", "M8"],
    "Royal Blue":     ["M2", "M3", "M4", "M5", "M6", "M7"],
    "Slate Blue":     ["M2", "M3", "M4", "M5", "M6"],
    "Powder Blue":    ["M1", "M2", "M3", "M4", "M5"],
    "Charcoal":       ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    "Warm Sand":      ["M3", "M4", "M5", "M6"],
}

FABRIC_BY_BRAND = {
    "Studio Picante":       "premium resort fabric",
    "Osé Studios":          "structured woven",
    "Amossh":               "metal alloy",
    "Label Sugar":          "Indian cotton",
    "Kalpraag":             "pure viscose",
    "Jatin Malik Couture":  "handcrafted couture",
    "Sarab Khanijou":       "raw silk",
    "Devnaagri":            "handwoven artisan fabric",
    "Kisah":                "silk blend",
    "Nicobar":              "organic cotton",
    "Meluku":               "linen cotton",
    "Nishorama":            "cotton",
    "Gopi Vaid":            "embroidered fabric",
    "Vvani Vats":           "silk",
    "Gaurav Gupta Couture": "couture fabric",
    "Jiwya":                "plant-based linen",
    "October Jaipur":       "cotton block-print",
}

AESTHETIC_OCCASION_MAP = {
    "Minimalist":   ["brunch", "college", "work"],
    "Quiet Luxury": ["brunch", "formal", "work"],
    "Editorial":    ["editorial", "events"],
    "Streetwear":   ["casual", "college", "nights_out"],
    "Bohemian":     ["brunch", "casual", "travel"],
    "Resort Wear":  ["brunch", "casual", "travel"],
    "Ethnic":       ["festive", "pooja", "weddings"],
    "Festive":      ["festive", "sangeet", "weddings"],
    "Y2K":          ["college", "nights_out"],
    "Feminine":     ["brunch", "casual", "dates"],
    "Old Money":    ["events", "formal", "work"],
    "Cottagecore":  ["brunch", "casual"],
    "Avant-garde":  ["editorial", "events"],
    "High Fashion": ["editorial", "events"],
    "Couture":      ["events", "gala", "weddings"],
    "Indo-western": ["casual", "festive", "weddings"],
    "Casual":       ["casual", "college"],
    "Everyday":     ["casual", "college", "work"],
    "Indie":        ["casual", "college", "concerts"],
    "Traditional":  ["festive", "pooja", "weddings"],
    "Formal":       ["events", "formal", "work"],
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_price(price_str: str) -> int:
    """Convert '₹14,500' → 14500"""
    try:
        return int(re.sub(r"[₹,\s]", "", str(price_str)))
    except Exception:
        return 0


def strip_json_comments(text: str) -> str:
    """Strip // line comments from JSON-like text."""
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def get_body_compatibility(aesthetics: list) -> list:
    """Union of body types across all aesthetics, deduplicated and sorted."""
    body_types: set = set()
    for aesthetic in aesthetics:
        body_types.update(
            AESTHETIC_BODY_MAP.get(
                aesthetic,
                ["apple", "hourglass", "inverted_triangle", "pear", "rectangle"],
            )
        )
    return sorted(body_types)


def infer_occasions(aesthetics: list) -> list:
    """Union of occasions across all aesthetics."""
    occasions: set = set()
    for aesthetic in aesthetics:
        occasions.update(AESTHETIC_OCCASION_MAP.get(aesthetic, ["casual"]))
    return sorted(occasions)


def transform_product(raw: dict) -> dict | None:
    """Transform raw product entry into full MongoDB RSI schema."""
    try:
        color      = raw.get("color", "")
        aesthetics = raw.get("aesthetic", [])
        category   = raw.get("category", "")
        brand      = raw.get("brand", "")

        discount_raw = raw.get("discount_percent", "0%").replace("%", "")
        discount     = float(discount_raw) if discount_raw else 0.0

        return {
            "product_name": raw.get("product_name", ""),
            "brand":        brand,
            "category":     category,
            "gender":       raw.get("gender", "unisex"),
            "discover_tab": raw.get("discover_tab", "EVERYDAY"),

            "aesthetic":  aesthetics,
            "silhouette": CATEGORY_SILHOUETTE.get(category, "standard"),
            "color":      color,
            "fabric":     FABRIC_BY_BRAND.get(brand, "mixed fabric"),

            "description": raw.get("description", ""),
            "image_url":   raw.get("image_url", ""),
            "product_url": raw.get("product_url", ""),

            "pricing": {
                "price_inr":          parse_price(raw.get("price_inr", "0")),
                "original_price_inr": parse_price(raw.get("original_price", raw.get("price_inr", "0"))),
                "discount_percent":   discount,
            },

            "compatibility": {
                "body_types":   get_body_compatibility(aesthetics),
                "monk_scales":  COLOR_MONK_MAP.get(
                    color,
                    ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
                ),
                "occasions":    infer_occasions(aesthetics),
            },

            "availability": {
                "in_stock":              raw.get("in_stock", True),
                "last_checked":          NOW,
                "check_frequency_hours": 6,
            },

            "performance": {
                "save_rate":             0.0,
                "purchase_rate":         0.0,
                "skip_rate":             0.0,
                "total_recommendations": 0,
            },

            "embedding":   [],          # populated by vector pipeline
            "created_at":  NOW,
            "updated_at":  NOW,
        }
    except Exception as e:
        print(f"  [FAIL] transform_product '{raw.get('product_name', '?')}': {e}")
        return None


# ── COLLECTION DEFINITIONS ────────────────────────────────────────────────────

COLLECTION_INDEXES = {
    "users": [
        [("user_id", ASCENDING)],
        [("meta.tier", ASCENDING)],
        [("meta.last_active", DESCENDING)],
    ],
    "products": [
        [("brand", ASCENDING)],
        [("availability.in_stock", ASCENDING)],
        [("pricing.price_inr", ASCENDING)],
        [("aesthetic", ASCENDING)],
        [("gender", ASCENDING)],
        [("discover_tab", ASCENDING)],
        [("compatibility.body_types", ASCENDING)],
        [("compatibility.monk_scales", ASCENDING)],
    ],
    "styling_guides": [
        [("quality_score", DESCENDING)],
        [("aesthetic", ASCENDING)],
        [("body_compatibility", ASCENDING)],
    ],
    "sessions": [
        [("user_id", ASCENDING)],
        [("started_at", DESCENDING)],
    ],
    "briefs": [
        [("user_id", ASCENDING)],
        [("expires_at", ASCENDING)],
    ],
    "trends": [
        [("captured_at", DESCENDING)],
        [("expires_at", ASCENDING)],
    ],
    "brands": [
        [("name", ASCENDING)],
        [("type", ASCENDING)],
    ],
}


# ── STEPS ─────────────────────────────────────────────────────────────────────

def step_collections(db) -> bool:
    """Create all collections and indexes."""
    try:
        for name, index_list in COLLECTION_INDEXES.items():
            try:
                db.create_collection(name)
                print(f"  [OK] Created → {name}")
            except CollectionInvalid:
                print(f"  [OK] Exists  → {name}")

            for index_spec in index_list:
                try:
                    db[name].create_index(index_spec)
                except Exception as e:
                    print(f"  [FAIL] Index {name}.{index_spec}: {e}")

        return True
    except Exception as e:
        print(f"  [FAIL] Collections step: {e}")
        return False


def step_seed_products(db) -> int:
    """Seed products from new_brand_entries.json."""
    entries_path = Path("new_brand_entries.json")

    if not entries_path.exists():
        print("  [FAIL] new_brand_entries.json not found — copy it to the Shaaru directory")
        return 0

    try:
        raw_text  = entries_path.read_text(encoding="utf-8")
        clean     = strip_json_comments(raw_text)
        raw_data  = json.loads(clean)
        raw_list  = raw_data.get("new_products", [])
    except Exception as e:
        print(f"  [FAIL] Parse new_brand_entries.json: {e}")
        return 0

    transformed = [t for raw in raw_list if (t := transform_product(raw)) is not None]

    if not transformed:
        print("  [FAIL] No products transformed")
        return 0

    col            = db["products"]
    existing_names = {p["product_name"] for p in col.find({}, {"product_name": 1})}
    new_products   = [p for p in transformed if p["product_name"] not in existing_names]

    if not new_products:
        print(f"  [OK] All {len(transformed)} products already seeded — skipping")
        return 0

    try:
        result = col.insert_many(new_products)
        print(f"  [OK] Seeded {len(result.inserted_ids)} products ({len(existing_names)} already existed)")
        return len(result.inserted_ids)
    except Exception as e:
        print(f"  [FAIL] Insert products: {e}")
        return 0


def step_seed_brands(db) -> int:
    """Seed indie brand catalog from new_brands_catalog.json."""
    catalog_path = Path("new_brands_catalog.json")

    if not catalog_path.exists():
        print("  [FAIL] new_brands_catalog.json not found — copy it to the Shaaru directory")
        return 0

    try:
        raw_data     = json.loads(catalog_path.read_text(encoding="utf-8"))
        indie_brands = raw_data.get("indie_brands", [])
    except Exception as e:
        print(f"  [FAIL] Parse new_brands_catalog.json: {e}")
        return 0

    col            = db["brands"]
    existing_names = {b["name"] for b in col.find({}, {"name": 1})}
    new_brands     = []

    for brand in indie_brands:
        if brand["name"] not in existing_names:
            brand["created_at"]    = NOW
            brand["neo4j_synced"]  = False
            new_brands.append(brand)

    if not new_brands:
        print(f"  [OK] All {len(indie_brands)} brands already seeded — skipping")
        return 0

    try:
        result = col.insert_many(new_brands)
        print(f"  [OK] Seeded {len(result.inserted_ids)} indie brands")
        return len(result.inserted_ids)
    except Exception as e:
        print(f"  [FAIL] Insert brands: {e}")
        return 0


def step_seed_demo_user(db) -> bool:
    """Seed the demo user that mirrors the Neo4j demo node."""
    col = db["users"]

    if col.find_one({"user_id": "demo_user_001"}):
        print("  [OK] Demo user already exists — skipping")
        return True

    demo_user = {
        "user_id":    "demo_user_001",
        "name":       "Riya",
        "created_at": NOW,

        "visual": {
            "monk_scale":       "M4",
            "undertone":        "warm",
            "face_shape":       "oval",
            "hair_color":       "dark brown",
            "eye_color":        "dark brown",
            "photo_url":        "",
            "confidence_score": 0.95,
        },

        "physical": {
            "height_cm": 163,
            "body_type": "pear",
        },

        "taste": {
            "everyday":      ["Casual", "Minimalist"],
            "cozy":          ["Cottagecore", "Bohemian"],
            "fashion_week":  ["Editorial", "Avant-garde"],
            "dream_outfit":  ["Quiet Luxury", "Old Money"],
            "color_palette": ["earth tones", "neutrals", "warm whites"],
            "occasion":      ["brunch", "college", "nights_out"],
            "style_icon":    "Janhvi Kapoor",
        },

        "style_equation": {
            "primary_aesthetic":   "Quiet Luxury",
            "secondary_aesthetic": "Minimalist",
            "profile_hash":        "M4_oval_pear_163_warm_QuiteLuxury",
            "generated_at":        NOW,
        },

        "meta": {
            "tier":                "free",
            "onboarding_complete": True,
            "photo_verified":      False,
            "sessions_count":      0,
            "last_active":         NOW,
        },
    }

    try:
        col.insert_one(demo_user)
        print("  [OK] Demo user seeded → Riya | M4 | oval | pear | Quiet Luxury")
        return True
    except Exception as e:
        print(f"  [FAIL] Demo user: {e}")
        return False


def step_seed_trends(db) -> bool:
    """Seed initial trend context so Riley always has something to read."""
    col = db["trends"]

    if col.find_one():
        print("  [OK] Trend document exists — skipping")
        return True

    trend_doc = {
        "captured_at":        NOW,
        "expires_at":         NOW + timedelta(hours=6),
        "source":             ["seed"],

        "rising": [
            {
                "trend":               "sheer layering",
                "silhouettes":         ["fluid", "layered"],
                "compatible_profiles": ["M1-M8"],
                "indian_relevance":    8.5,
            },
            {
                "trend":               "quiet luxury minimalism",
                "silhouettes":         ["relaxed fit", "straight cut"],
                "compatible_profiles": ["M1-M8"],
                "indian_relevance":    8.0,
            },
            {
                "trend":               "indo-western fusion",
                "silhouettes":         ["draped", "kurta with western bottom"],
                "compatible_profiles": ["M1-M8"],
                "indian_relevance":    9.5,
            },
            {
                "trend":               "desi craft revival",
                "silhouettes":         ["block print", "hand embroidered"],
                "compatible_profiles": ["M1-M8"],
                "indian_relevance":    9.8,
            },
        ],

        "declining":          ["neon accents", "oversized logos", "fast fashion basics"],
        "seasonal_direction": "Light fabrics, earthy tones, Indian craft aesthetics dominating. Indo-western fusion at peak.",
        "seed":               True,
    }

    try:
        col.insert_one(trend_doc)
        print("  [OK] Initial trend context seeded")
        return True
    except Exception as e:
        print(f"  [FAIL] Trend seed: {e}")
        return False


def print_summary(db) -> None:
    """Print final collection document counts."""
    print("\n── SUMMARY ──────────────────────────────────────────────")
    collections = ["users", "products", "brands", "styling_guides", "sessions", "briefs", "trends"]
    for name in collections:
        try:
            count = db[name].count_documents({})
            status = "✓" if count > 0 else "○"
            print(f"  {status}  {name:<20} {count:>4} documents")
        except Exception:
            print(f"  ✗  {name:<20} not found")

    print("\n── ATLAS VECTOR SEARCH — Manual Step Required ───────────")
    print("  Go to: Atlas UI → your cluster → Search → Create Index")
    print()
    print("  Index 1:")
    print("    Collection : shaaru.products")
    print("    Field      : embedding")
    print("    Dimensions : 1536")
    print("    Similarity : cosine")
    print()
    print("  Index 2:")
    print("    Collection : shaaru.styling_guides")
    print("    Field      : embedding")
    print("    Dimensions : 1536")
    print("    Similarity : cosine")
    print("─────────────────────────────────────────────────────────")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   SHAARU — MongoDB RSI + LUM Brain Init      ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Connect
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        db = client[DB_NAME]
        print(f"[OK] Connected → {DB_NAME}\n")
    except Exception as e:
        print(f"[FAIL] MongoDB connection: {e}")
        sys.exit(1)

    # Step 1 — Collections + Indexes
    print("── STEP 1: Collections + Indexes ──────────────────────")
    step_collections(db)

    # Step 2 — Products
    print("\n── STEP 2: Product Catalog ─────────────────────────────")
    step_seed_products(db)

    # Step 3 — Indie Brands
    print("\n── STEP 3: Indie Brand Catalog ─────────────────────────")
    step_seed_brands(db)

    # Step 4 — Demo User
    print("\n── STEP 4: Demo User ───────────────────────────────────")
    step_seed_demo_user(db)

    # Step 5 — Trend Context
    print("\n── STEP 5: Trend Context ───────────────────────────────")
    step_seed_trends(db)

    # Summary
    print()
    print_summary(db)

    client.close()
    print("\n[OK] Riley's brain is initialized. Ready to wire into shaaru_brain.py\n")


if __name__ == "__main__":
    main()
