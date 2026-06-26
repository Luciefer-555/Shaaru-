import json
from config.models import get_client, MODELS
from rapidfuzz import process, fuzz

ANCHOR_PROMPT = """You are a fashion data parser. 
Extract the fundamental ground truth facts from this product description.
Return a valid JSON object ONLY.

Product Text:
{text}

Output format:
{{
  "fabric_map": {{"<component name>": "<fabric name>"}}, 
  "techniques": ["<technique 1>", "<technique 2>"],
  "color_name": "<color name, or null>",
  "color_hex": "<color hex if provided, or null>"
}}
"""

def parse_product_metadata(product_page_text: str) -> dict:
    """
    Parses the product page text to establish ground truth.
    Uses a fast LLM call to extract fabrics, techniques, and color.
    """
    client = get_client(MODELS["text_extractor"]["provider"])
    prompt = ANCHOR_PROMPT.format(text=product_page_text)
    
    try:
        response = client.chat.completions.create(
            model=MODELS["text_extractor"]["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        gt = json.loads(content.strip())
        return gt
    except Exception as e:
        print(f"Error parsing product metadata: {e}")
        return {
            "fabric_map": {},
            "techniques": [],
            "color_name": None,
            "color_hex": None
        }

def build_anchor_block(gt: dict, db_fabric_ids: set) -> str:
    """
    Builds the ground truth string to be injected into the prompt.
    Also populates gt['unknown_fabrics'] with any brand-confirmed fabrics 
    that do not match our database.
    """
    anchor_lines = []
    anchor_lines.append("GROUND TRUTH ANCHOR (Treat this as absolute truth from the brand):")
    
    gt["unknown_fabrics"] = []
    
    if gt.get("fabric_map"):
        anchor_lines.append("Confirmed Fabrics by Component:")
        for comp, fab in gt["fabric_map"].items():
            anchor_lines.append(f"  - {comp}: {fab}")
            
            # Check if this fabric exists in our DB via fuzzy match
            # If not, add to unknown_fabrics
            match_result = process.extractOne(
                fab.lower(),
                db_fabric_ids,
                scorer=fuzz.WRatio
            )
            
            if match_result:
                match, score, _ = match_result
            else:
                match, score = None, 0
                
            if score < 85:
                gt["unknown_fabrics"].append(fab)
    else:
        anchor_lines.append("Confirmed Fabrics: None specified")
        
    if gt.get("techniques"):
        anchor_lines.append("Confirmed Techniques: " + ", ".join(gt["techniques"]))
        
    if gt.get("color_name"):
        anchor_lines.append(f"Confirmed Color: {gt['color_name']}")
        
    return "\n".join(anchor_lines)
