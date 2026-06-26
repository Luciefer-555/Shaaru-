# pipeline/db/fabric_db_lookup.py

from rapidfuzz import process, fuzz

def enrich_fabrics(observed_fabrics: list, db_refs: dict) -> list:
    """
    For each fabric the vision model identified,
    pull the full fabric_intelligence document.
    Adds drape_score, structure_score, embellishment_compatibility etc.
    """
    enriched = []
    name_index = db_refs["fabrics"]["name_index"]
    lookup = db_refs["fabrics"]["lookup"]
    
    for fabric in observed_fabrics:
        fabric_id = fabric.get("fabric_id", "")
        
        # Direct match first
        if fabric_id in lookup:
            fabric["db_data"] = lookup[fabric_id]
            fabric["db_matched"] = True
            enriched.append(fabric)
            continue
        
        # Fuzzy match against common_names
        match_result = process.extractOne(
            fabric_id.lower(),
            name_index.keys(),
            scorer=fuzz.WRatio
        )
        
        if match_result:
            match, score, _ = match_result
        else:
            match, score = None, 0
        
        if score >= 85:
            matched_id = name_index[match]
            fabric["db_data"] = lookup[matched_id]
            fabric["db_matched"] = True
            fabric["fuzzy_match_score"] = score
            fabric["matched_to"] = matched_id
        else:
            # No match — flag for DB addition
            fabric["db_matched"] = False
            fabric["new_fabric_candidate"] = True
            fabric["db_data"] = None
        
        enriched.append(fabric)
    
    return enriched


def get_embellishment_compatibility(fabric_id: str, technique_id: str, db_refs: dict) -> dict:
    """
    Checks if a technique is compatible with a fabric
    using embellishment_compatibility field in fabric_intelligence.
    Powers the pairs_with logic.
    """
    fabric_doc = db_refs["fabrics"]["lookup"].get(fabric_id)
    if not fabric_doc:
        return {"compatible": "unknown", "reason": "fabric not in DB"}
    
    compat = fabric_doc.get("embellishment_compatibility", {})
    
    # Map technique_id to compatibility category
    technique_category_map = {
        "zardozi_thread_gold": "embroidery",
        "sheesha_mirror": "embroidery",
        "resham_thread": "embroidery",
        "sequin_dori": "sequins",
        "swarovski_crystal": "heavy_crystal_pearl"
    }
    
    category = technique_category_map.get(technique_id, "embroidery")
    compatibility_level = compat.get(category, "unknown")
    reason = compat.get("reason", "")
    
    return {
        "compatible": compatibility_level,
        "category": category,
        "reason": reason
    }
