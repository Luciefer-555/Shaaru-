"""
Watches designer Shopify stores for new collections.
Products added in last 30 days = trend signal.
Designer pushes many new items = new collection = trend.
"""

import os
import json
import requests
import urllib3
urllib3.disable_warnings()
from datetime import datetime, timedelta


def load_designers() -> list:
    path = "pipeline/config/designers.json"
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "designers.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scan_new_arrivals() -> list:
    """
    Checks all designer Shopify stores for recent additions.
    Returns trend signals with actual product data.
    """
    designers = load_designers()
    signals = []
    cutoff = datetime.utcnow() - timedelta(days=30)
    
    for designer in designers:
        if not designer.get("active"):
            continue
        if designer.get("platform") != "shopify":
            continue
        
        url_base = designer.get("url", "")
        if not url_base:
            continue
        
        try:
            response = requests.get(
                f"https://{url_base}/products.json?limit=250",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
                verify=False
            )
            products = response.json().get("products", [])
        except Exception:
            continue
        
        recent = []
        for p in products:
            created_str = p.get("created_at", "")
            if not created_str:
                continue
            try:
                created = datetime.fromisoformat(
                    created_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if created > cutoff:
                    recent.append(p)
            except Exception:
                continue
        
        if len(recent) >= 5:
            techniques = _extract_techniques_from_products(recent)
            signals.append({
                "signal_type": "new_collection",
                "designer_id": designer["id"],
                "designer_name": designer["name"],
                "aesthetic_hint": designer.get("aesthetic_hint", ""),
                "new_product_count": len(recent),
                "recent_products": recent[:10],
                "techniques_detected": techniques,
                "confidence": min(0.5 + (len(recent) * 0.02), 0.9)
            })
    
    return signals


def _extract_techniques_from_products(products: list) -> list:
    """
    Extracts technique keywords from product descriptions.
    """
    TECHNIQUE_KEYWORDS = [
        "mirror", "sheesha", "zardozi", "resham",
        "kantha", "block print", "natural dye",
        "gota", "dabka", "chikankari", "phulkari",
        "sequin", "crystal", "zari", "bandhani",
        "leheriya", "ikat", "jamdani"
    ]
    
    found = set()
    for product in products:
        text = (
            product.get("title", "") + " " +
            product.get("body_html", "") + " " +
            " ".join(product.get("tags", []))
        ).lower()
        
        for keyword in TECHNIQUE_KEYWORDS:
            if keyword in text:
                found.add(keyword)
    
    return list(found)
