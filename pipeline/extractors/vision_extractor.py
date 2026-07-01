import json
import time
from config.models import get_client

def build_vision_prompt(db_refs: dict) -> str:
    fabric_list = "\n".join([
        f"  - {f}" for f in db_refs["fabrics"]["injection_list"]
    ])
    
    embellishment_list = "\n".join([
        f"  - {e}" for e in db_refs["embellishments"]["injection_list"]
    ])
    
    construction_list = "\n".join([
        f"  - {c}" for c in db_refs["constructions"]["injection_list"]
    ])

    return f"""
You are a master Indian fashion expert and textile scholar with 20 years 
of experience identifying fabrics, embroidery techniques, motifs, 
and silhouette construction for Indian menswear and womenswear — 
purely from visual inspection.

═══════════════════════════════════════════════════
CORE RULE: NULL IS ALWAYS BETTER THAN WRONG
If you cannot identify something with certainty from the image,
return null for that field and write your uncertainty in 
confidence_notes. Never guess. Never assume.
A confident wrong answer corrupts the entire knowledge base.
═══════════════════════════════════════════════════

DATABASE REFERENCE — USE THESE EXACT IDs WHERE POSSIBLE:

You have access to a curated fabric database. When you identify a fabric,
return the exact fabric_id from this list. If you observe a fabric
NOT in this list, describe it precisely and flag it as 
"new_fabric_candidate" — do not guess the closest match.

KNOWN FABRICS (use fabric_id exactly as written):
{fabric_list}

KNOWN EMBELLISHMENTS/TECHNIQUES (use embellishment_id exactly):
{embellishment_list}

KNOWN GARMENT CONSTRUCTIONS (use garment_id exactly):
{construction_list}

═══════════════════════════════════════════════════
IDENTIFICATION RULES:

FABRIC CONFIRMATION TESTS — visual evidence required:
  velvet          → dense pile surface, depth of colour, soft matte sheen
  organza         → crisp stiff mesh, NOT fluid — must see mesh structure
  net             → visible hexagonal or diamond mesh
  chanderi_silk   → very fine, semi-transparent, soft natural drape  
  raw_silk_dupion → irregular slubs visible, crisp, medium weight
  chiffon         → ultra-light, transparent, complete fluidity
  georgette       → crinkled surface, fluid, lightweight
  katan silk      → smooth, heavier drape, no sheen variation

  IF tag says "Special Silk" and you cannot confirm type visually:
  → return "silk_unconfirmed" and flag in confidence_notes
  IF you see satin-like sheen on inner layer:
  → return "satin_base_unconfirmed" — do not guess silk type

EMBELLISHMENT CONFIRMATION TESTS:
  sheesha/mirror  → small circular mirrors sewn onto surface
                    ALSO NOTE: scatter pattern —
                    (constellation / jaal / border / yoke / buta)
  zardozi         → heavy metal wire coiled in relief — 3D raised effect
  zari_flat       → flat gold thread, woven or stitched flat
  resham          → coloured silk thread, smooth flat stitch, matte finish
  dabka           → coiled metal wire flat (lighter than zardozi, flatter)
  gota_patti      → flat metallic ribbon folded into petal/leaf shapes
  mukaish         → tiny metal chips scattered like dust
  sequin_dori     → sequins on thread in visible lines
  kantha_stitch   → running stitch visible on both sides, uneven charm
  chikankari      → fine white thread, shadow work, pastel/white base

  IMPORTANT — DO NOT CONFUSE:
  zari ≠ resham (zari = metallic, resham = silk thread)
  zardozi ≠ dabka (zardozi = 3D raised, dabka = flat coil)
  mirror work ≠ sequins (mirrors = circular glass, sequins = flat plastic/metal)

GENDER — MANDATORY FIRST:
  menswear  → sherwani, achkan, bandhgala, kurta-pyjama, 
              jodhpuri, nehru jacket
  womenswear → lehenga, saree, anarkali, sharara, ghagra, 
               dupatta set

SILHOUETTE — COMPONENT LEVEL:
  Name every component separately, with length and structure:
  "3-piece achkan set: open-front embroidered jacket (knee length) 
   + inner kurta + straight pants"
  NEVER just "sherwani set" or "lehenga set" — always break down.

MOTIF IDENTIFICATION:
  jaal           → continuous all-over lattice
  buta           → individual large motif scattered
  boota          → small motif densely scattered
  paisley/kairi  → teardrop curved
  peacock        → bird form or feather abstraction
  floral_vine    → continuous vine with flowers
  geometric      → angular repeating, no organic curves
  constellation  → organic scattered with no fixed repeat
  temple_border  → repeating arch at hem

═══════════════════════════════════════════════════
Return ONLY a valid JSON object.
Start your response with {{ and end with }}
No markdown. No backticks. No text before or after.
═══════════════════════════════════════════════════

{{
  "gender": "",
  "garment_type": "",
  "matched_construction_id": "",
  "silhouette": "",
  "components": [],
  "observed_fabrics": [
    {{
      "fabric_id": "",
      "location_on_garment": "",
      "confidence": "confirmed/probable/uncertain",
      "is_new_candidate": false,
      "new_candidate_description": ""
    }}
  ],
  "observed_techniques": [
    {{
      "embellishment_id": "",
      "location_on_garment": "",
      "confidence": "confirmed/probable/uncertain",
      "is_new_candidate": false,
      "new_candidate_description": "",
      "mirror_scatter_pattern": ""
    }}
  ],
  "embroidery_density": "",
  "motif_vocabulary": {{
    "primary_motif": "",
    "secondary_motifs": [],
    "motif_style": "",
    "motif_placement": ""
  }},
  "color_palette": [],
  "color_harmony": {{
    "temperature": "",
    "family": "",
    "contrast_level": "",
    "metallic_accent": ""
  }},
  "surface_texture": "",
  "drape_behavior": "",
  "visual_weight": "",
  "weight_estimate": "",
  "outfit_completeness": {{
    "is_set": false,
    "includes": [],
    "missing_for_complete_look": []
  }},
  "styling_observations": "",
  "occasion_suitability": [],
  "styling_dna": [],
  "confidence_notes": ""
}}
"""


def _merge_vision_results(results: list) -> dict:
    if not results:
        return {}
    if len(results) == 1:
        return results[0]
        
    merged = dict(results[0])
    list_keys = ["observed_fabrics", "observed_techniques", "color_palette", "occasion_suitability", "styling_dna"]
    for k in list_keys:
        items = []
        seen = set()
        for r in results:
            val = r.get(k, [])
            if isinstance(val, list):
                for item in val:
                    key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item).lower()
                    if key not in seen:
                        seen.add(key)
                        items.append(item)
        merged[k] = items

    str_keys = ["surface_texture", "drape_behavior", "visual_weight", "weight_estimate", "styling_observations", "confidence_notes"]
    for k in str_keys:
        best = ""
        for r in results:
            val = str(r.get(k, "")).strip()
            if len(val) > len(best):
                best = val
        merged[k] = best

    return merged


def _extract_single_url(client, url: str, prompt_text: str, model_config: dict):
    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": url}}
    ]
    kwargs = {
        "model": model_config["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1500,
        "temperature": 0.1
    }
    if model_config.get("json_mode"):
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(2):
        try:
            response = client.chat.completions.create(**kwargs)
            usage = response.usage
            tokens = {"input": usage.prompt_tokens, "output": usage.completion_tokens}
            raw_text = response.choices[0].message.content or ""
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0]
            data = json.loads(raw_text.strip())
            return data, tokens
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                return None, {"input": 0, "output": 0}
    return None, {"input": 0, "output": 0}


def extract_vision(image_urls: list, model_config: dict, db_refs: dict):
    """
    Extracts structured vision data from up to 3 image URLs using the specified model.
    Runs individual vision calls per image angle and reconciles/merges results.
    """
    client = get_client(model_config["provider"])
    prompt_text = build_vision_prompt(db_refs)
    
    # Cap at 3 images max to avoid token explosion
    target_urls = image_urls[:3] if image_urls else []
    if not target_urls:
        return None, {"input": 0, "output": 0}

    results = []
    total_tokens = {"input": 0, "output": 0}
    for url in target_urls:
        data, tokens = _extract_single_url(client, url, prompt_text, model_config)
        total_tokens["input"] += tokens.get("input", 0)
        total_tokens["output"] += tokens.get("output", 0)
        if data and isinstance(data, dict):
            results.append(data)

    if not results:
        return None, total_tokens

    merged_data = _merge_vision_results(results)
    return merged_data, total_tokens
