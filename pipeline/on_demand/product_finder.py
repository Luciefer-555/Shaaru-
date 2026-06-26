"""
When cache misses — find the best matching product
from a designer's Shopify catalog to extract.
"""

import requests
import json
import os
import urllib3
urllib3.disable_warnings()
from typing import Optional


def load_designers() -> list:
    path = "pipeline/config/designers.json"
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "config", "designers.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_best_matching_product(
    query: str,
    techniques: list = None,
    aesthetic: str = None,
    designer_id: str = None
) -> Optional[tuple[dict, dict]]:
    """
    Searches designer Shopify catalogs for best match.
    Returns (product, designer_config) or None.
    """
    designers = load_designers()
    
    if designer_id:
        designer = next(
            (d for d in designers if d["id"] == designer_id),
            None
        )
        if designer:
            product = _search_shopify(designer, query, techniques)
            if product:
                return product, designer
    
    aesthetic_designer_map = {
        "mirror maximalism":  ["abhinav_mishra"],
        "heritage couture":   ["sabyasachi"],
        "handloom minimal":   ["raw_mango", "anavila"],
        "folk maximalist":    ["torani", "pero"],
        "artisan craft":      ["injiri"],
        "graphic pop indian": ["house_of_masaba"],
        "avant-garde":        ["rimzim_dadu"],
        "festive occasion":   ["anita_dongre"],
    }
    
    if aesthetic:
        for aesthetic_key, designer_ids in aesthetic_designer_map.items():
            if aesthetic_key in aesthetic.lower():
                for did in designer_ids:
                    designer = next(
                        (d for d in designers if d["id"] == did),
                        None
                    )
                    if not designer:
                        continue
                    product = _search_shopify(
                        designer, query, techniques
                    )
                    if product:
                        return product, designer
    
    for designer in designers:
        if not designer.get("active"):
            continue
        product = _search_shopify(designer, query, techniques)
        if product:
            return product, designer
    
    return None


def _search_shopify(
    designer_config: dict,
    query: str,
    techniques: list = None
) -> Optional[dict]:
    """
    Searches one designer's Shopify for best match.
    Scores products by relevance and returns the best one.
    """
    url_base = designer_config.get("url", "")
    if not url_base:
        return None
    
    try:
        response = requests.get(
            f"https://{url_base}/products.json?limit=250",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
            verify=False
        )
        products = response.json().get("products", [])
    except Exception:
        return None
    
    query_terms = query.lower().split()
    technique_terms = [t.lower() for t in (techniques or [])]
    
    scored = []
    for product in products:
        score = 0
        searchable = (
            product.get("title", "") + " " +
            product.get("body_html", "") + " " +
            " ".join(product.get("tags", []))
        ).lower()
        
        for term in query_terms:
            if term in searchable:
                score += 2
        
        for term in technique_terms:
            if term in searchable:
                score += 3
        
        if score > 0:
            product["source_url"] = (
                f"https://{url_base}/products/"
                f"{product.get('handle', '')}"
            )
            scored.append((score, product))
    
    if not scored:
        return None
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
