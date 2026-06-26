import json
from config.models import get_client, MODELS
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from extractors.vision_extractor import extract_vision
from extractors.text_extractor import extract_text
from db.fabric_db_lookup import enrich_fabrics

CLASSIFIER_PROMPT = """You must classify this garment into exactly ONE of the following aesthetic categories. 
Read all descriptions carefully before choosing. 
If it genuinely does not fit any category, use "Unique" and explain why.

CATEGORIES:
1. Heritage Couture        — Sabyasachi-esque, zardozi, antique palettes, Bengal/Mughal heritage
2. Handloom Minimal        — Raw Mango / Anavila style, linen-forward, quiet luxury, muted tones
3. Folk Maximalist         — Torani / Pero style, heavy folk embroidery, playful, rural craft narratives
4. Bollywood Glam          — Sequin-heavy, net fabric, lehenga drama, red carpet energy
5. Contemporary Indie      — Bodice / Payal Khandwala style, clean cuts, conceptual, anti-bridal
6. Artisan Craft           — Injiri style, block print, natural dye, slow fashion, regional story-first
7. Festive Occasion        — Broad appeal, zari and silk, suitable for family weddings, not strictly bridal
8. Avant-Garde             — Rimzim Dadu / Gaurav Gupta style, sculptural, experimental textiles
9. Graphic / Pop Indian    — House of Masaba style, bold graphic prints, Gen Z palette, pop motifs
10. Mirror Maximalism      — Abhinav Mishra style, sheesha-forward, pastel bridal, festive opulence
11. Unique                 — Does not fit above — explain in one sentence why

Aesthetic Hint: {aesthetic_hint}
(Note: Use this hint only as a soft prior to break ties; it must not override the evidence).

Product Data:
{combined_data}

Return ONLY a valid JSON object:
{{
  "aesthetic_category": "Category Name",
  "aesthetic_justification": "1-line justification"
}}"""

CAPTION_PROMPT = """You are building the complete knowledge document for SHAARU,
an AI Indian fashion assistant. This is not a product listing.
This is a deep knowledge entry that Shaaru will use to:

  1. Identify garments from images
  2. Match pieces to people, occasions, body types, skin tones
  3. Explain craft and cultural context to customers
  4. Recommend complementary pieces
  5. Understand where this piece sits in Indian fashion history

You have received the following inputs:
  - Vision extraction output (fabrics, patterns, techniques)
  - Text extraction output (from product listing)
  - DB enrichment (from fabric_intelligence + garment_construction)
  - Neo4j context (aesthetics, occasions, body types)
  - Designer context (aesthetic_hint, region, collection)

═══════════════════════════════════════════════════════════════
WRITING RULES — ALL NON-NEGOTIABLE:
  - Natural flowing prose only — no bullets inside text sections
  - Never use: stunning, gorgeous, beautiful, perfect, elegant,
    exquisite, breathtaking, dreamy, luxurious, timeless
  - Never reference JSON keys, field names, or data terms
  - Specific facts only — no vague generalisations
  - If you cannot fill a section with real knowledge → null
  - Menswear stays menswear throughout — never crossover language
  - Null with a note beats a vague sentence every time
═══════════════════════════════════════════════════════════════

Generate the following complete knowledge document:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 1 — FABRIC & MATERIAL KNOWLEDGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fabric_and_craft (4-5 sentences):
Name every fabric and technique with full precision.
Explain WHY each fabric was chosen for this silhouette —
what does velvet on a jacket achieve that raw silk would not?
What does mirror work on velvet do to light differently than
mirror work on net?
For patterns: explain if it is woven INTO the fabric (jacquard,
ikat, brocade) or applied ONTO the surface (print, embroidery).
Name the weave structure where identifiable.
Explain the hand feel and drape behaviour the customer will experience.

pattern_intelligence (3-4 sentences):
If a surface pattern exists:
  - Name the pattern type precisely (pinstripe, jaal, buta, 
    ikat, bandhani, block print floral, etc.)
  - Identify the print method (block, screen, digital, resist)
  - Describe the scale and placement
  - Explain the cultural or design origin of this pattern
  - Note how the pattern interacts with any embellishment
If no surface pattern → explain what the absence of pattern 
achieves aesthetically.

weave_and_construction_knowledge (2-3 sentences):
Explain the garment construction — how is it actually built?
Reference construction sequence where known from DB.
Explain what the construction achieves for the wearer —
why this silhouette serves this occasion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 2 — BODY & FIT INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

body_type_compatibility:
Based on silhouette, construction, and fabric weight — which body 
types does this garment work best for and why?
Be specific about WHY — not just which.

  works_best_for: []
    → explain the visual logic for each
    → e.g. "A-line lehenga flare creates hip-to-waist proportion 
       that benefits rectangle and inverted triangle frames"

  works_with_consideration: []
    → body types that can wear it but need styling adjustment
    → explain exactly what the adjustment is

  avoid_if: []
    → be honest — some silhouettes genuinely don't serve 
      some body types
    → explain why without being negative

fit_notes (2-3 sentences):
How does this fabric behave on the body?
Use drape_score and structure_score from fabric DB:
  high drape + low structure → clings and moves with body
  low drape + high structure → holds shape away from body
What does the customer need to know before ordering?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 3 — SKIN TONE & COLOUR INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

colour_intelligence:
  palette: []
    → specific colour names, not just "beige" — 
      "warm champagne beige with silver-gold mirror reflection"

  skin_tone_guide:
    works_beautifully_for: []
      → list undertones: warm/cool/neutral + depth: fair/medium/deep
      → explain WHY — e.g. "champagne reflects warmth upward 
         into medium-warm skin tones"

    works_with_consideration: []
      → explain what to pair to make it work

    avoid_if: []
      → honest guidance — e.g. "very pale champagne 
         can wash out fair cool-toned skin without 
         contrast jewellery at the neckline"

  season_wearability:
    → which Indian seasons/months does this work for?
    → reference fabric seasonal data from DB
    → e.g. "velvet base makes this best suited for 
       November-February wedding season in North India"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 4 — OCCASION & STYLING INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

occasion_mapping:
  primary_occasion: ""
    → the ONE occasion this was designed for

  suitable_occasions: []
    → all occasions this works for with brief logic

  not_suitable_for: []
    → be honest about what this doesn't work for
    → e.g. "too heavy for mehendi in summer months"

  time_of_day: ""
    → daytime / evening / night only
    → explain why (fabric reflects light differently,
      embellishment reads better at night, etc.)

styling_context (3-4 sentences):
Be occasion-specific and accessory-specific.
Name jewellery type, not just family —
kundan satlada vs polki choker vs chandbali jhumkas.
For menswear: specify footwear type, pocket square,
whether maala serves or overwhelms.
What NOT to do is as important as what to do.
Include one specific DO NOT for this garment.

pairs_with:
  jewellery_type: ""
  jewellery_metal_tone: ""
  jewellery_avoid: ""
  footwear: ""
  bag: ""
  blouse_if_saree: ""
  dupatta_if_missing: ""
  menswear_accessories: ""
  maala_recommendation: ""
  what_not_to_do: ""

styling_alternatives (2-3 sentences):
What other aesthetic or silhouette approach could achieve
a similar effect? This gives Shaaru options if this exact
piece isn't right for the customer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 5 — CULTURAL & CRAFT SIGNIFICANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cultural_significance (5-6 sentences):
This is the most important block. Teach Shaaru something real.

Answer ALL of these:
  → Which community or region produces this craft/textile?
  → How old is this tradition and what was its original purpose?
  → What makes this technique technically difficult or rare?
  → What does this craft mean within Indian wedding or 
    celebration culture specifically?
  → Why is this designer's use of it notable?
  → What is happening to this craft today — 
    is it thriving, endangered, reviving?
  → What should Shaaru say when a customer asks 
    "tell me about this piece"?

craft_community: ""
  → Specific community name if known
  → e.g. "Rabari mirror workers of Kutch, Gujarat"
  → or "Banarasi karigar families of Varanasi"

gi_status: ""
  → GI tagged or not, and what that means for authenticity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 6 — TREND & MARKET INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

trend_position: ""
  → "classic"   — has been relevant 10+ years, will remain
  → "current"   — actively trending right now
  → "emerging"  — gaining momentum, not mainstream yet
  → "heritage"  — intentionally archival, not trend-driven
  → "declining" — was popular, now less so

designer_context (2-3 sentences):
How does this piece represent this designer's signature?
Is this a core piece or an experimental departure?
What collection does it belong to and what was
the collection's stated inspiration?

price_intelligence:
  price_tier: ""
  value_assessment: ""
    → for this construction + technique + fabric combination,
      is this price justified? explain specifically
    → e.g. "at ₹2,51,000 for all-over hand-applied sheesha
      on velvet across 3 components with 40+ labor hours,
      this sits at fair luxury pricing for Indian couture"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK 7 — SHAARU MATCHING SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This block powers Shaaru's matching engine.
These are the structured signals she uses to find 
similar pieces and make recommendations.

styling_dna: []
  → exactly 4-6 keywords that define this piece's 
    aesthetic identity
  → used for nearest-neighbor matching
  → e.g. ["mirror-maximalist", "tonal-menswear",
           "festive-groom", "pastel-opulent",
           "velvet-base", "constellation-scatter"]

aesthetic_category: ""
  → from the hardcoded 11-category taxonomy

aesthetic_justification: ""
  → one sentence explaining the classification

matching_fabrics: []
  → fabric_ids from DB that would pair well with this piece
  → e.g. a jacket pairs with: raw_silk_dupion (inner kurta),
    chanderi_silk (pocket square), velvet (pants match)

matching_aesthetics: []
  → other aesthetic categories this could bridge to
  → e.g. Mirror Maximalism bridges to Heritage Couture 
    and Festive Occasion

not_matches: []
  → aesthetic categories this would clash with
  → explain why briefly

keyword_tags: []
  → broad searchable tags for retrieval
  → e.g. ["menswear", "sherwani", "velvet", "mirror-work",
           "beige", "sangeet", "groom", "3-piece",
           "abhinav-mishra", "mirror-maximalism",
           "pastel", "festive", "winter-wedding"]

new_fabric_candidates: []
  → fabrics observed that are NOT in the DB yet
  → these should be flagged for manual addition

new_embellishment_candidates: []
  → techniques observed that are NOT in DB yet

new_pattern_candidates: []
  → surface patterns not in DB yet

═══════════════════════════════════════════════════════════════
Product data to work from:
{combined_enriched_product_data}
═══════════════════════════════════════════════════════════════

Return ONLY valid JSON with this exact structure:

{{
  "fabric_and_craft": "",
  "pattern_intelligence": "",
  "weave_and_construction_knowledge": "",
  "body_type_compatibility": {{
    "works_best_for": [],
    "works_with_consideration": [],
    "avoid_if": []
  }},
  "fit_notes": "",
  "colour_intelligence": {{
    "palette": [],
    "skin_tone_guide": {{
      "works_beautifully_for": [],
      "works_with_consideration": [],
      "avoid_if": []
    }},
    "season_wearability": ""
  }},
  "occasion_mapping": {{
    "primary_occasion": "",
    "suitable_occasions": [],
    "not_suitable_for": [],
    "time_of_day": ""
  }},
  "styling_context": "",
  "pairs_with": {{
    "jewellery_type": "",
    "jewellery_metal_tone": "",
    "jewellery_avoid": "",
    "footwear": "",
    "bag": "",
    "blouse_if_saree": "",
    "dupatta_if_missing": "",
    "menswear_accessories": "",
    "maala_recommendation": "",
    "what_not_to_do": ""
  }},
  "styling_alternatives": "",
  "cultural_significance": "",
  "craft_community": "",
  "gi_status": "",
  "trend_position": "",
  "designer_context": "",
  "price_intelligence": {{
    "price_tier": "luxury",
    "value_assessment": ""
  }},
  "styling_dna": [],
  "aesthetic_category": "",
  "aesthetic_justification": "",
  "matching_fabrics": [],
  "matching_aesthetics": [],
  "not_matches": [],
  "keyword_tags": [],
  "new_fabric_candidates": [],
  "new_embellishment_candidates": [],
  "new_pattern_candidates": []
}}
"""

def _call_llm(prompt: str, model_config: dict):
    client = get_client(model_config["provider"])
    
    kwargs = {
        "model": model_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    if model_config.get("json_mode"):
        kwargs["response_format"] = {"type": "json_object"}
        
    tokens = {"input": 0, "output": 0}
    try:
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        tokens = {"input": usage.prompt_tokens, "output": usage.completion_tokens}
        
        raw_text = response.choices[0].message.content
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0]
        
        return json.loads(raw_text.strip()), tokens
    except Exception as e:
        return {}, tokens

def reconcile_vision(v_pri, v_sec):
    reconciled = {}
    needs_review = False
    confidence_notes = []
    
    if not v_pri and not v_sec:
        return None, True, ["Both vision models failed"]
        
    if v_pri and not v_sec:
        v_sec = {}
    if v_sec and not v_pri:
        v_pri = {}
        
    for key in set(list(v_pri.keys()) + list(v_sec.keys())):
        val_p = v_pri.get(key)
        val_s = v_sec.get(key)
        
        def _norm(v):
            if isinstance(v, str): return v.lower().strip()
            if isinstance(v, list): return sorted([str(x).lower().strip() for x in v])
            return v
            
        norm_p = _norm(val_p)
        norm_s = _norm(val_s)
        
        # Simple string/list normalize
        if norm_p == norm_s:
            reconciled[key] = val_p
        elif val_p and not val_s:
            reconciled[key] = val_p
        elif val_s and not val_p:
            reconciled[key] = val_s
        else:
            # Different values. 
            # If lists, we might merge or see if one is subset. For now, flag conflict.
            # In a real NLP approach we'd compare length for specificity.
            if isinstance(val_p, str) and isinstance(val_s, str):
                if len(val_p) > len(val_s):
                    reconciled[key] = val_p
                else:
                    reconciled[key] = val_s
            else:
                reconciled[key] = val_p # Default to primary
            
            # They differ and both exist -> Conflict
            needs_review = True
            confidence_notes.append(f"Conflict on {key}: Primary '{val_p}' vs Secondary '{val_s}'")
            
    return reconciled, needs_review, confidence_notes

def process_product_pipeline(product: dict, aesthetic_hint: str, db_refs: dict):
    """
    Runs vision (primary & secondary) and text extraction in parallel.
    Reconciles, then runs classifier, then caption generator.
    """
    results = {}
    cost_log = {
        "vision_primary_tokens": {"input": 0, "output": 0},
        "vision_secondary_tokens": {"input": 0, "output": 0},
        "text_extractor_tokens": {"input": 0, "output": 0},
        "classifier_tokens": {"input": 0, "output": 0},
        "caption_tokens": {"input": 0, "output": 0},
        "total_api_calls": 0,
        "conflicts_flagged": 0
    }
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_v_pri = executor.submit(extract_vision, product.get("images", []), MODELS["vision_primary"], db_refs)
        f_v_sec = executor.submit(extract_vision, product.get("images", []), MODELS["vision_secondary"], db_refs)
        f_txt = executor.submit(extract_text, product.get("title", ""), product.get("raw_description", ""), product.get("tags", []), MODELS["text_extractor"])
        
        v_pri, cost_log["vision_primary_tokens"] = f_v_pri.result()
        v_sec, cost_log["vision_secondary_tokens"] = f_v_sec.result()
        txt_out, cost_log["text_extractor_tokens"] = f_txt.result()
        
    cost_log["total_api_calls"] += 3
    
    # 1. Reconcile vision
    reconciled_vision, needs_review, confidence_notes = reconcile_vision(v_pri, v_sec)
    
    # If any vision model failed, mark needs_manual_review: true
    if v_pri is None or v_sec is None:
        needs_review = True
        confidence_notes.append("One or both vision models failed to return data.")
    
    if needs_review:
        cost_log["conflicts_flagged"] += 1
        
    # DB Enrichment for Fabrics
    observed_fabrics = reconciled_vision.get("observed_fabrics", []) if reconciled_vision else []
    enriched_fabrics = enrich_fabrics(observed_fabrics, db_refs) if observed_fabrics else []
    
    # 2. Merge Text
    fabric_vocab = {
        "confirmed": txt_out.get("confirmed_fabrics", []) if txt_out else [],
        "vision_only": enriched_fabrics,
        "text_only": []
    }
    techniques = {
        "confirmed": txt_out.get("confirmed_techniques", []) if txt_out else [],
        "vision_only": reconciled_vision.get("observed_techniques", []) if reconciled_vision else [],
        "text_only": []
    }
    
    combined_data = {
        "fabric_vocabulary": fabric_vocab,
        "techniques": techniques,
        "embroidery_density": reconciled_vision.get("embroidery_density", "") if reconciled_vision else "",
        "drape_behavior": reconciled_vision.get("drape_behavior", "") if reconciled_vision else "",
        "surface_texture": reconciled_vision.get("surface_texture", "") if reconciled_vision else "",
        "weight_estimate": reconciled_vision.get("weight_estimate", "") if reconciled_vision else "",
        "silhouette": reconciled_vision.get("silhouette", "") if reconciled_vision else "",
        "color_palette": reconciled_vision.get("color_palette", []) if reconciled_vision else [],
        "color_harmony": reconciled_vision.get("color_harmony", {}) if reconciled_vision else {},
        "visual_weight": reconciled_vision.get("visual_weight", "") if reconciled_vision else "",
        "motif_vocabulary": reconciled_vision.get("motif_vocabulary", {}) if reconciled_vision else {},
        "components": reconciled_vision.get("components", []) if reconciled_vision else [],
        "gender": reconciled_vision.get("gender", "") if reconciled_vision else "",
        "garment_type": reconciled_vision.get("garment_type", "") if reconciled_vision else "",
        "matched_construction_id": reconciled_vision.get("matched_construction_id", "") if reconciled_vision else "",
        "outfit_completeness": reconciled_vision.get("outfit_completeness", {}) if reconciled_vision else {},
        "styling_dna": reconciled_vision.get("styling_dna", []) if reconciled_vision else [],
        "occasion_suitability": reconciled_vision.get("occasion_suitability", []) if reconciled_vision else [],
        "styling_observations": reconciled_vision.get("styling_observations", "") if reconciled_vision else "",
        "collection_name": txt_out.get("collection_name", "") if txt_out else "",
        "designer_notes": txt_out.get("designer_notes", "") if txt_out else "",
        "region_of_craft": txt_out.get("region_of_craft", "") if txt_out else "",
        "price_tier": txt_out.get("price_tier", "") if txt_out else "",
        "confidence_notes": " | ".join(confidence_notes)
    }
    
    # 3. Classify
    class_prompt = CLASSIFIER_PROMPT.format(aesthetic_hint=aesthetic_hint, combined_data=json.dumps(combined_data, indent=2))
    class_out, cost_log["classifier_tokens"] = _call_llm(class_prompt, MODELS["classifier"])
    cost_log["total_api_calls"] += 1
    
    if class_out:
        combined_data["aesthetic_category"] = class_out.get("aesthetic_category", "")
        combined_data["aesthetic_justification"] = class_out.get("aesthetic_justification", "")
        
    # 4. Generate Caption
    if txt_out and reconciled_vision:
        cap_prompt = CAPTION_PROMPT.format(combined_enriched_product_data=json.dumps(combined_data, indent=2))
        cap_out, cost_log["caption_tokens"] = _call_llm(cap_prompt, MODELS["caption"])
        cost_log["total_api_calls"] += 1
        combined_data["caption"] = cap_out
    else:
        combined_data["caption"] = None
        needs_review = True
        
    combined_data["needs_manual_review"] = needs_review
    
    return combined_data, cost_log
