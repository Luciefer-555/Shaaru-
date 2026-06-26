# Schema validation logic for MongoDB
# This enforces the structure before insertion

REQUIRED_FIELDS = [
    "source_id", "designer", "platform", "title", "source_url", 
    "images", "raw_description", "variants", "fabric_vocabulary", 
    "techniques", "embroidery_density", "drape_behavior", "surface_texture",
    "weight_estimate", "silhouette", "color_palette", "occasion_suitability",
    "aesthetic_category", "aesthetic_justification",
    "region_of_craft", "price_tier", "caption", "styling_observations",
    "designer_notes", "confidence_notes", "dedup_hash", "scraped_at", "reviewed"
]

def validate_product_document(doc: dict) -> bool:
    """
    Validates that a product document matches the exact expected MongoDB schema.
    Returns True if valid, False otherwise.
    """
    if "collection_name" not in doc:
        doc["collection_name"] = None

    for field in REQUIRED_FIELDS:
        if field not in doc:
            print(f"Validation Error: Missing required field '{field}'")
            return False
            
    # Check fabric_vocabulary structure
    fv = doc.get("fabric_vocabulary", {})
    for subfield in ["confirmed", "vision_only", "text_only"]:
        if subfield not in fv:
            print(f"Validation Error: Missing '{subfield}' in fabric_vocabulary")
            return False
            
    # Check techniques structure
    tech = doc.get("techniques", {})
    for subfield in ["confirmed", "vision_only", "text_only"]:
        if subfield not in tech:
            print(f"Validation Error: Missing '{subfield}' in techniques")
            return False
            
    # Check caption structure (Now the Complete Knowledge Document)
    cap = doc.get("caption")
    if cap is not None:
        expected_blocks = [
            "fabric_and_craft", 
            "weave_and_construction_knowledge", 
            "body_type_compatibility", 
            "colour_intelligence",
            "occasion_mapping",
            "styling_context", 
            "cultural_significance",
            "styling_dna"
        ]
        for subfield in expected_blocks:
            if subfield not in cap:
                print(f"Validation Error: Missing '{subfield}' in knowledge document (caption)")
                return False
                
    if not isinstance(doc.get("reviewed"), bool):
        print("Validation Error: 'reviewed' must be a boolean")
        return False
        
    return True
