"""
Unified Tailor Engine wrapper for deep garment analysis.
Bridges root tailor_engine.py with pipeline requirements.
"""

import datetime
import os
import sys
import asyncio

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from tailor_engine import analyze_garment_deep as engine_analyze
from pipeline.run_pipeline import get_image_base64

KNOWN_TECHNIQUES = [
    "mirror work", "mirror", "sheesha", "zardozi", "resham",
    "kantha", "block print", "natural dye", "gota patti",
    "dabka", "chikankari", "phulkari", "sequin",
    "crystal", "zari", "bandhani", "ikat", "handloom", "banarasi",
]


def _extract_techniques(text: str) -> list:
    found = []
    lower = text.lower()
    for t in sorted(KNOWN_TECHNIQUES, key=len, reverse=True):
        if t in lower and t not in found:
            found.append(t)
    if "mirror work" in found and "mirror" not in found:
        found.append("mirror")
    return found


def _merge_confirmed_techniques(text: str, combined: dict) -> list:
    """Text-parse fallback when vision fails or is incomplete."""
    from_text = _extract_techniques(text)
    from_vision = []
    if isinstance(combined.get("techniques"), dict):
        from_vision = combined["techniques"].get("confirmed", []) or []
    merged = []
    for t in from_text + from_vision:
        if isinstance(t, str) and t and t not in merged:
            merged.append(t)
    return merged


async def analyze_garment_deep(
    product: dict,
    designer_config: dict,
    db_refs: dict,
    gates: list
) -> tuple[dict, str]:
    """
    Wrapper around root tailor_engine.py analyze_garment_deep.
    Returns (extracted_doc, cost_string).
    """
    image_b64 = ""
    img_url = ""
    if product.get("images"):
        img = product["images"][0]
        img_url = img if isinstance(img, str) else img.get("src", "")
        if img_url:
            image_b64 = get_image_base64(img_url)
            
    desc = product.get("raw_description") or product.get("body_html", "") or product.get("title", "")
    variants = product.get("variants", [])
    price = ""
    if variants and len(variants) > 0:
        price_val = variants[0].get("price") if isinstance(variants[0], dict) else None
        if price_val: price = f"₹{price_val}"

    combined = {}
    if image_b64:
        try:
            combined = await asyncio.to_thread(engine_analyze, image_b64)
        except Exception as e:
            print(f"[TailorWrapper] Vision engine failed: {e}")
            
    if not isinstance(combined, dict):
        combined = {}

    technique_source = (
        desc + " " + product.get("title", "") + " "
        + " ".join(product.get("tags", []) if isinstance(product.get("tags"), list) else [])
    )
    techniques_list = _merge_confirmed_techniques(technique_source, combined)

    final_doc = {
        "source_id": str(product.get("id", "")),
        "designer": designer_config.get("name", "Unknown"),
        "platform": designer_config.get("platform", "shopify"),
        "title": product.get("title", ""),
        "source_url": product.get("source_url", ""),
        "images": product.get("images", []),
        "image_url": img_url,
        "price": price,
        "raw_description": desc,
        "variants": variants,
        "aesthetic_category": designer_config.get("aesthetic_hint", "Festive Occasion"),
        "caption": {"text": desc[:250] if desc else product.get("title", "")},
        "techniques": {"confirmed": techniques_list},
        "occasion_suitability": [combined.get("occasion", "Festive Occasion")] if combined.get("occasion") else ["Festive Occasion"],
        "styling_dna": [combined.get("garment_type", "Ethnic Wear")] if combined.get("garment_type") else ["Ethnic Wear"],
        "scraped_at": datetime.datetime.now().isoformat(),
        "reviewed": False,
        "quality_gate_passed": True,
        **{k: v for k, v in combined.items() if k != "techniques"},
    }
    final_doc["techniques"] = {"confirmed": techniques_list}
    return final_doc, "~$0.02"
