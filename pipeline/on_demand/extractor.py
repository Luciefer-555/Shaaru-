"""
Pipeline 1 — On-Demand Extractor
Triggered by user queries via Shaaru/Riley.
Checks cache first. Extracts only on cache miss.
Every extraction enriches Neo4j permanently.
"""

import asyncio
import time
import json
import os
import sys
import logging
from typing import Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from pipeline.on_demand.cache_checker import check_cache
from pipeline.on_demand.product_finder import find_best_matching_product
from pipeline.extractors.tailor_engine import analyze_garment_deep
from pipeline.validators.quality_gate import (
    load_quality_gates, validate_product
)
from pipeline.db.db_loader import load_all_references
from knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_db_refs = None
_gates = None
_kg = None


def _get_resources():
    global _db_refs, _gates, _kg
    if _db_refs is None:
        _db_refs = load_all_references(
            mongo_uri=os.environ["MONGODB_URI"],
            neo4j_uri=os.environ["NEO4J_URI"],
            neo4j_user=os.environ["NEO4J_USER"],
            neo4j_password=os.environ["NEO4J_PASSWORD"]
        )
    if _gates is None:
        _gates = load_quality_gates("pipeline/config/quality_gates.json")
    if _kg is None:
        _kg = KnowledgeGraph()
    return _db_refs, _gates, _kg


async def handle_user_query(
    query: str,
    techniques: list = None,
    aesthetic: str = None,
    designer_id: str = None
) -> dict:
    """
    MAIN ENTRY POINT — called by Riley/Shaaru when user asks about a fashion topic.
    """
    start_time = time.time()
    db_refs, gates, kg = _get_resources()
    
    cache_result = check_cache(
        query=query,
        techniques=techniques,
        aesthetic=aesthetic,
        designer_id=designer_id
    )
    
    if cache_result["cache_hit"]:
        elapsed = time.time() - start_time
        logger.info("Cache hit for '%s' in %.2fs", query, elapsed)
        return {
            "status": "instant",
            "source": "cache",
            "elapsed_seconds": round(elapsed, 2),
            "products": cache_result["products"],
            "match_type": cache_result["match_type"],
            "extraction_needed": False
        }
    
    logger.info("Cache miss for '%s' — searching Shopify", query)
    
    match = find_best_matching_product(
        query=query,
        techniques=techniques,
        aesthetic=aesthetic,
        designer_id=designer_id
    )
    
    if not match:
        return {
            "status": "not_found",
            "message": "No matching product found across known designers for this query.",
            "query": query
        }
    
    product, designer_config = match
    logger.info("Found match: %s by %s", product.get("title"), designer_config["name"])
    
    try:
        output, cost = await analyze_garment_deep(
            product=product,
            designer_config=designer_config,
            db_refs=db_refs,
            gates=gates
        )
    except Exception as e:
        logger.error("Extraction failed: %s", str(e))
        return {
            "status": "extraction_failed",
            "error": str(e),
            "product_title": product.get("title", ""),
            "designer": designer_config["name"]
        }
    
    output = validate_product(output, designer_config["id"], gates)
    
    output["source_id"] = str(product.get("id", ""))
    output["triggered_by_query"] = query
    
    kg.sync_product_document(output)
    _save_output(output, designer_config["id"])
    
    elapsed = time.time() - start_time
    logger.info("On-demand extraction complete: %s in %.1fs", product.get("title"), elapsed)
    
    return {
        "status": "freshly_extracted",
        "source": "on_demand_pipeline",
        "elapsed_seconds": round(elapsed, 2),
        "product": {
            "title": output.get("title", ""),
            "designer": designer_config["name"],
            "source_url": product.get("source_url", ""),
            "image": (
                product.get("images", [{}])[0].get("src", "")
                if product.get("images") else ""
            ),
            "aesthetic_category": output.get("aesthetic_category", ""),
            "caption": output.get("caption", {}),
            "techniques": output.get("techniques", {}).get("confirmed", []),
            "occasion_suitability": output.get("occasion_suitability", []),
            "styling_dna": output.get("styling_dna", []),
            "quality_gate_passed": output.get("quality_gate_passed", False)
        },
        "extraction_needed": False,
        "will_be_instant_next_time": True
    }


def _save_output(output: dict, designer_id: str):
    os.makedirs("pipeline/output/review", exist_ok=True)
    path = f"pipeline/output/review/{designer_id}_ondemand_{output['source_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
