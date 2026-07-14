"""
cv_engine.py — SHAARU Computer Vision Pipeline

Two core functions:

  scan_frame(image_b64)
    — Detects all visible garments/accessories in a frame.
    — Returns bbox coordinates (0.0–1.0 fractions), confidence, guidance.

  analyze_item(image_b64, item_label, user_profile)
    — Deep-dives one detected item: garment analysis, fabric intelligence,
      and profile compatibility cross-reference.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from shaaru_brain import _get_client

log = logging.getLogger("shaaru.cv")

# ─────────────────────────────────────────────────────────────────
#  Model constants
# ─────────────────────────────────────────────────────────────────
_MODEL_VISION_90B = "meta/llama-3.2-90b-vision-instruct"
_MODEL_VISION_11B = "meta/llama-3.2-11b-vision-instruct"
_MODEL_NEMOTRON   = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

_CAT_COLORS: dict[str, str] = {
    "top":       "#39FF14",
    "bottom":    "#E040FB",
    "outerwear": "#FF6D00",
    "footwear":  "#00E5FF",
    "dress":     "#FF4081",
    "set":       "#FF4081",
    "accessory": "#FFD700",
}
_CAT_COLOR_DEFAULT = "#A855F7"

_SCAN_PROMPT = """Return only the JSON object. No markdown, no explanation.
CONTEXT: Professional B2B Commercial Product Photography Analysis of Apparel Merchandise.
You are an AI computer vision specialist indexing inanimate clothing items and textile merchandise.
SAFETY INSTRUCTION: Do NOT evaluate, describe, profile, or analyze any human body parts, skin tone, physical subjects, or age. Focus strictly and exclusively on inanimate apparel items, fabric construction, and colorways.

Analyze this image and detect every visible garment or accessory item.
For each item, return a JSON object in this exact structure:

{
  "items": [
    {
      "id": "item_1",
      "label": "<specific fashion name>",
      "description": "<one precise sentence about fabric and construction>",
      "fabric_reason": "<step 1: terse structured phrase (5-10 words max), e.g. 'diagonal twill, matte, stiff drape'>",
      "fabric_type": "<step 2: exact term from Tier 2 specific fabrics below based on visual cues, or 'uncertain'>",
      "category": "top|bottom|outerwear|footwear|accessory|dress|set",
      "color": "<precise color name, not just 'blue' — say 'indigo' or 'slate grey' or 'off-white'>",
      "aesthetic": "maximalist|minimalist|streetwear|editorial|fusion|formal|traditional|resort|workwear",
      "bbox": { "x": 0.15, "y": 0.22, "w": 0.30, "h": 0.48 },
      "confidence": 0.87
    }
  ],
  "scene_lighting": "warm_indoor|cool_indoor|natural_daylight|golden_hour|overcast|harsh_flash|poor",
  "frame_quality": "good|poor"
}

CRITICAL RULES for the label field:
- Never use generic names. Use specific fashion nomenclature aligned with Fashionpedia and Indian ethnic apparel ontology categories (shirt_blouse, top_t_shirt_sweatshirt, sweater, cardigan, jacket, vest, pants, shorts, skirt, coat, dress, jumpsuit, cape, glasses, hat, tie, glove, watch, belt, shoe, bag_wallet, scarf, saree, lehenga_set, kurta, salwar_kameez_set, sharara_set, anarkali_dress, dupatta, co_ord_set).
- Examples of what NOT to write vs what TO write:
  BAD: "white shirt" → GOOD: "mandarin collar poplin shirt_blouse"
  BAD: "blue jeans" → GOOD: "wide-leg indigo cargo denim pants"
  BAD: "black jacket" → GOOD: "double-breasted wool blazer jacket"
  BAD: "white t-shirt" → GOOD: "boxy ribbed henley top_t_shirt_sweatshirt"
  BAD: "dress" → GOOD: "asymmetric draped midi slip dress"
  BAD: "shoes" → GOOD: "chunky lug-sole leather derby shoe"
  BAD: "bag" → GOOD: "structured top-handle trapeze bag_wallet"
  BAD: "kurta" → GOOD: "straight-hem embroidered kurta with thread work"
  BAD: "saree" → GOOD: "kanjeevaram silk saree with gold zari border"
  BAD: "wrap dress" → GOOD: "belted satin faux-wrap midi dress"
  BAD: "co-ord set" → GOOD: "tailored linen blazer and pleated trouser co_ord_set"
  BAD: "skirt" → GOOD: "high-waisted sunray pleated midi skirt"
  BAD: "blouse" → GOOD: "puff-sleeve silk organza pussy-bow shirt_blouse"
- For Indian garments, be equally specific — name the exact construction silhouette (e.g. lehenga_set, salwar_kameez_set, sharara_set, anarkali_dress, dupatta), fabric, and embellishment if visible
- Describe collar type, sleeve style, fabric texture, and structural details.

CRITICAL RULES for the description field:
- One sentence only, max 20 words
- Must mention: silhouette, fabric/texture if readable, and one 
  standout detail
- Examples:
  "Relaxed cotton henley with ribbed placket and dropped shoulders, 
   reads clean and off-duty."
  "Wide-leg cargo denim with oversized patch pockets and raw hem, 
   strong Gen Z street energy."
  "Fitted mandarin-collar poplin in crisp off-white with hidden 
   placket — formal-adjacent but minimal."
  "Fitted wrap dress in fluid crepe with self-tie waist, 
   reads effortless and body-conscious."
  "Pleated linen co-ord set with relaxed crop blazer and wide 
   trouser, refined summer tailoring."

CRITICAL RULES for the fabric_type field (HIERARCHICAL TAXONOMY):
You MUST reason in two steps:
Step 1: In "fabric_reason", provide a TERSE STRUCTURED PHRASE (5-10 words max) noting surface texture, weave, drape, and sheen (e.g. "diagonal twill, matte, stiff drape" or "fine smooth knit, soft drape, matte"). Do NOT write full sentences or prose.
Step 2: In "fabric_type", select ONE specific fabric term from the grounded Tier 2 vocabulary below that matches your observed visual cues (or return "uncertain"). Do NOT use freeform text outside this list:

1. Woven Plain / Crisp (Visual cues: visible plain weave, matte finish, breathable medium structure, crisp folds):
   - poplin: Crisp, tightly woven, smooth surface with subtle sheen (classic dress shirts).
   - linen: Lightweight, breathable weave with visible natural slubs and slight wrinkling/texture.
   - chambray: Plain weave with white weft and colored warp, resembling lightweight denim but smoother.
   - khadi cotton: Handspun, handwoven cotton with rustic natural texture and breathable body.
   - handloom cotton: Traditional artisan weave with soft tactile body and organic texture.

2. Woven Twill / Denim (Visual cues: distinct diagonal ribbing/twill lines, sturdy architectural structure, matte or utility finish):
   - denim: Sturdy cotton twill with diagonal ribbing and visible weave structure (jeans, jackets).
   - twill: Distinct diagonal rib weave pattern, softer and more drapeable than canvas.
   - canvas: Heavy, rugged plain-weave fabric with coarse matte texture (utility wear, bags).
   - corduroy: Distinct vertical raised wales/cords with velvety texture.
   - gabardine: Tightly woven twill with a smooth face and diagonal rib on reverse (suiting, trench coats).

3. Knit Ribbed / Cable (Visual cues: raised vertical ridges or braided loops, high horizontal stretch, textured matte):
   - ribbed knit: Distinct vertical raised rows/ribs, stretchy and textured (turtlenecks, cuffs, sweaters).
   - cable knit: Chunky, textured knit with braided or twisting rope-like patterns.
   - waffle knit: Three-dimensional grid/honeycomb texture (thermal wear, sweaters).

4. Knit Jersey / Fine (Visual cues: smooth fine gauge, fluid drape that clings/follows body, high stretch, subtle soft sheen):
   - jersey knit: Smooth, fine gauge knit with high stretch and soft drape (t-shirts, casual wear).
   - interlock: Double-knit construction, thicker and smoother than jersey with identical face and back.
   - athletic mesh: Breathable open-hole knit structure with technical athletic stretch.

5. Sheen / Fluid Drape (Visual cues: specular light highlights, gravity-hugging fluid drape, smooth lustrous surface):
   - silk satin: High specular luster on face, fluid flowing drape that highlights body contours.
   - crepe de chine: Lightweight silk or synthetic with subtle pebbled texture, matte sheen, and fluid drape.
   - charmeuse: Lightweight satin weave with reflective sheen and extreme drapeability.
   - georgette: Sheer or semi-sheer lightweight fabric with grainy crepe surface and flowing drape.

6. Sheer / Crisp / Open Weave (Visual cues: semi-transparent or open mesh, stiff sculptural volume or airy lightness):
   - organza: Sheer, lightweight fabric with stiff, sculptural volume and subtle shimmer.
   - chanderi silk: Traditional Indian sheer fabric with crisp lightweight body and subtle sheen.
   - chiffon: Very lightweight, sheer, plain-woven fabric with soft, floating drape and slight crepe texture.
   - net: Open mesh or tulle construction with structured or soft volume.
   - mulmul: Ultra-soft, lightweight Indian cotton muslin with breathable airy drape.

7. Structured / Jacquard / Brocade (Visual cues: woven-in metallic/patterned texture, stiff architectural drape, rich surface dimension):
   - raw silk dupion: Crisp, structured silk with irregular slubs and distinctive tactile sheen (luxury ethnic/formal wear).
   - banarasi brocade: Rich Indian silk weave with elaborate gold/silver zari metallic patterns and stiff drape.
   - jacquard: Intricate woven-in patterns with dimensional texture and structured body.
   - ikat: Traditional resist-dyed weave with characteristic blurred geometric patterns and medium structure.

8. Leather / Coated / Suede (Visual cues: consistent surface sheen, visible grain/pores or nap, rigid architectural drape):
   - genuine leather: Rich natural grain, smooth sheen, structured drape with natural creasing.
   - faux/PU leather: Uniform artificial grain, high gloss or synthetic sheen, rigid plastic-like drape.
   - suede: Soft, napped/velvety matte surface texture without sheen.
   - patent leather: High-gloss mirror-like waterproof finish with rigid structure.

9. Heavy / Wool / Felted (Visual cues: dense, matte, napped/fuzzy or felted surface, insulating volume):
   - fine wool: Smooth, high-grade wool with refined drape and soft matte finish (luxury suiting/coats).
   - wool-blend: Textured fabric showing wool warmth combined with synthetic structure or smoothness.
   - tweed: Coarse, textured wool weave with speckled or herringbone color patterns.
   - cashmere: Ultra-soft, fine luxury wool with delicate fuzzy nap and lightweight warmth.
   - fleece: Deep, soft synthetic pile with plush matte texture.

10. Embroidered / Craft Work (Visual cues: surface embellishments, thread work, or openwork lace over a base fabric):
    - chikankari: Delicate white-on-white or pastel Indian shadow thread embroidery on sheer/light base.
    - zardozi: Elaborate heavy metallic thread embroidery with sequins/beads on velvet or silk.
    - kantha: Distinctive running-stitch embroidery creating quilted texture across cotton or silk.
    - phulkari: Vibrant geometric floral embroidery covering the fabric surface in silk floss.
    - chantilly lace: Delicate openwork floral net lace with scalloped edges.

EXPLICIT UNCERTAINTY OPTION (IMPORTANT - DO NOT FORCE GUESSES):
- If you cannot 100% definitively distinguish the exact fabric weave or material (for example, distinguishing between linen vs poplin, denim vs canvas, genuine vs faux leather, or if the weave is subtle/unclear), you MUST return "fabric_type": "uncertain" and explain what you see in "fabric_reason".
- Guessing a fabric from the list when you are not 100% certain is a MAJOR FAILURE and WORSE than saying "uncertain".
- When in doubt between two or more fabrics, ALWAYS return "fabric_type": "uncertain"!

CRITICAL RULES for the confidence field:
- Estimate per-item based on visibility and how clearly visual anchor criteria were observed:
- 0.85-0.95: weave/texture, drape, and light behavior clearly visible and match Tier 2 anchor criteria unambiguously.
- 0.65-0.84: macro-category is clear from drape/sheen, but specific weave/fiber is inferred or item is partially occluded.
- 0.40-0.64: texture unreadable (distance, motion blur, low light, heavy occlusion) -> return "uncertain" for fabric_type!
- Never return the same value for all items in a frame

CRITICAL RULES for the bbox field:
- Values MUST be normalized fractions between 0.0 and 1.0 relative to image dimensions
- x, y = top-left corner as fraction of image width/height
- w, h = item width/height as fraction of image width/height
- NEVER use pixel coordinates — always fractions between 0.0 and 1.0
- A shirt covering left half, top two-thirds of frame =
  bbox: { "x": 0.0, "y": 0.0, "w": 0.50, "h": 0.67 }

CRITICAL RULES for duplicates:
- Each physical item must appear ONCE only — even if visible from
  multiple angles or displayed as a pair (two shoes on a shelf =
  ONE footwear entry, not two)
- If two items have identical labels, merge them into one entry

CRITICAL RULES for frame_quality and image texture:
- Do NOT reject an image or mark frame_quality as "poor" due to micro-texture, contrast grain, or sharpness artifacts. These are NORMAL CLAHE contrast-enhancement and white-balance preprocessing artifacts designed to reveal fabric weave details.
- ONLY mark frame_quality as "poor" (and return empty items[]) under extreme, unusable conditions:
  1. Extreme motion blur where garment boundaries and silhouettes are completely indistinguishable.
  2. Extreme darkness or overexposure where zero color or surface detail can be discerned.
  3. The garment occupies less than 10% of the total frame area (too far away).
- Under all normal lighting and clear silhouettes, you MUST return "frame_quality": "good" and detect all items.
- If frame_quality is genuinely poor per the strict criteria above: return 
{ "items": [], "frame_quality": "poor", 
  "guidance": "<one specific instruction for the user to fix lighting, distance, or blur>" }
"""


_FAST_SCAN_PROMPT = """Return only the JSON object. No markdown, no explanation.
CONTEXT: Real-Time B2B Commercial Product Detection for Apparel Merchandise.
You are an AI computer vision detector indexing inanimate clothing items and textile merchandise.
SAFETY INSTRUCTION: Do NOT evaluate, describe, profile, or analyze any human body parts, skin tone, physical subjects, or age. Focus strictly and exclusively on inanimate apparel items and colorways.

Analyze this image and detect every visible garment or accessory item.
For each item, return a JSON object in this exact structure:

{
  "items": [
    {
      "id": "item_1",
      "label": "<specific fashion name aligned with apparel construction taxonomy, e.g. 'mandarin collar poplin shirt_blouse', 'wide-leg indigo cargo denim pants', 'belted satin slip dress', 'tailored linen co_ord_set', or 'kanjeevaram silk saree'>",
      "category": "top|bottom|outerwear|footwear|accessory|dress|set",
      "color": "<precise color name, e.g. 'indigo' or 'slate grey' or 'off-white'>",
      "aesthetic": "maximalist|minimalist|streetwear|editorial|fusion|formal|traditional|resort|workwear",
      "bbox": { "x": 0.15, "y": 0.22, "w": 0.30, "h": 0.48 },
      "confidence": 0.88
    }
  ],
  "scene_lighting": "warm_indoor|cool_indoor|natural_daylight|golden_hour|overcast|harsh_flash|poor",
  "frame_quality": "good|poor"
}

RULES:
1. "category" MUST be one of: top, bottom, outerwear, footwear, accessory, dress, set.
2. "label" must use specific fashion nomenclature aligned with construction categories (e.g., 'double-breasted wool blazer jacket' not just 'jacket', or 'sunray pleated midi skirt' not just 'skirt').
3. "bbox" coordinates (x, y, w, h) must be normalized fractions (0.0 to 1.0) of image width/height.
4. Do NOT output fabric_type or fabric_reason here — keep the response fast and light.
5. Return ONLY JSON inside ```json ... ``` or raw JSON, no explanatory text.
"""


def _repair_json_with_llm(malformed_text: str) -> Optional[dict]:
    """Second pass JSON repair using fast LLM text model."""
    try:
        client = _get_client()
        prompt = (
            "Fix and complete this malformed/truncated JSON object so it parses validly. "
            "Return ONLY valid JSON, no explanation.\n\n"
            f"{malformed_text[:2500]}"
        )
        raw = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
            timeout=12.0,
        )
        content = raw.choices[0].message.content or ""
        clean = content.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(clean[start:end])
        return json.loads(clean)
    except Exception as e:
        log.warning(f"[CV] LLM JSON repair failed: {e}")
        return None


def _normalize_parsed_fabrics(data: Optional[dict]) -> Optional[dict]:
    """Map predicted fabric names to canonical MongoDB/Neo4j IDs to prevent downstream query drift."""
    if not data or not isinstance(data, dict):
        return data

    FABRIC_ID_MAP = {
        # Woven Plain / Crisp
        "poplin": "cotton_poplin",
        "linen": "linen_lightweight",
        "chambray": "cotton_chambray",
        "khadi": "khadi_cotton",
        "khadi cotton": "khadi_cotton",
        "cambric": "cotton_cambric",
        "handloom cotton": "handloom_cotton",
        # Woven Twill / Denim
        "denim": "denim_heavy",
        "twill": "cotton_twill",
        "gabardine": "gabardine_wool",
        "canvas": "canvas_cotton",
        "corduroy": "corduroy_cotton",
        # Knit Ribbed / Cable
        "ribbed knit": "cotton_rib_knit",
        "rib knit": "cotton_rib_knit",
        "cable knit": "cable_knit_fabric",
        "waffle knit": "thermal_waffle_knit",
        "thermal waffle": "thermal_waffle_knit",
        # Knit Jersey / Fine
        "jersey": "jersey_cotton",
        "jersey knit": "jersey_cotton",
        "interlock": "cotton_interlock",
        "athletic mesh": "athletic_mesh",
        "mesh": "athletic_mesh",
        # Sheen / Fluid Drape
        "silk satin": "satin_silk",
        "satin": "satin_silk",
        "charmeuse": "satin_silk",
        "crepe": "crepe_de_chine",
        "crepe de chine": "crepe_de_chine",
        "georgette": "pure_georgette",
        "pure georgette": "pure_georgette",
        # Sheer / Crisp / Open Weave
        "organza": "organza_silk",
        "organza silk": "organza_silk",
        "silk organza": "organza_silk",
        "chanderi": "chanderi_silk",
        "chanderi silk": "chanderi_silk",
        "chiffon": "chiffon_polyester",
        "net": "net_nylon",
        "tulle": "soft_tulle",
        "mulmul": "mulmul_cotton",
        "mulmul cotton": "mulmul_cotton",
        # Structured / Jacquard / Brocade
        "raw silk": "raw_silk_dupion",
        "raw silk dupion": "raw_silk_dupion",
        "dupioni": "raw_silk_dupion",
        "banarasi": "banarasi_silk",
        "banarasi brocade": "banarasi_silk",
        "banarasi silk": "banarasi_silk",
        "brocade": "zari_brocade",
        "zari brocade": "zari_brocade",
        "jacquard": "jacquard_silk",
        "jacquard silk": "jacquard_silk",
        "ikat": "ikat",
        # Leather / Coated / Suede
        "genuine leather": "genuine_leather",
        "faux leather": "faux_leather_pu",
        "faux/pu leather": "faux_leather_pu",
        "pu leather": "faux_leather_pu",
        "suede": "faux_suede",
        "patent leather": "leather_patent",
        # Heavy / Wool / Felted
        "wool": "fine_merino_wool",
        "fine wool": "fine_wool",
        "tweed": "tweed_wool",
        "cashmere": "pashmina_cashmere",
        "pashmina": "pashmina_cashmere",
        "fleece": "fleece_polyester",
        "wool-blend": "wool_blend_suiting",
        "wool blend": "wool_blend_suiting",
        # Embroidered / Craft Work
        "chikankari": "chikankari_georgette",
        "zardozi": "zardozi_embroidered",
        "kantha": "kantha_stitched_cotton",
        "phulkari": "phulkari_embroidered",
        "lace": "lace_chantilly",
        "chantilly lace": "lace_chantilly",
    }

    if "items" in data and isinstance(data["items"], list):
        for item in data["items"]:
            if isinstance(item, dict):
                raw_fab = str(item.get("fabric_type", "")).strip().lower()
                if raw_fab and raw_fab != "uncertain":
                    if raw_fab in FABRIC_ID_MAP:
                        item["fabric_type"] = FABRIC_ID_MAP[raw_fab]
                    else:
                        for key, val in FABRIC_ID_MAP.items():
                            if key in raw_fab or raw_fab in val.replace("_", " "):
                                item["fabric_type"] = val
                                break
    return data


def _parse_scan_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from a model response string.
    Falls back to partial recovery and LLM repair pass for truncated responses."""
    if not text:
        return None
    text = text.strip()

    # Direct parse
    try:
        return _normalize_parsed_fabrics(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```" in text:
        for part in text.split("```"):
            clean = part.strip().lstrip("json").strip()
            try:
                return _normalize_parsed_fabrics(json.loads(clean))
            except json.JSONDecodeError:
                continue

    # Brace extraction
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return _normalize_parsed_fabrics(json.loads(text[start:end]))
        except json.JSONDecodeError:
            pass

    # Partial recovery — JSON was truncated mid-stream
    # Extract only the complete item objects before truncation
    import re
    items_start = text.find('"items"')
    if items_start != -1:
        try:
            item_pattern = re.compile(
                r'\{[^{}]*"id"\s*:\s*"[^"]*"[^{}]*\}',
                re.DOTALL
            )
            found = item_pattern.findall(text)
            if found:
                items = []
                for raw_item in found:
                    try:
                        items.append(json.loads(raw_item))
                    except json.JSONDecodeError:
                        continue
                if items:
                    log.warning(
                        f"[CV] Partial JSON recovery: salvaged {len(items)} items "
                        f"from truncated response"
                    )
                    return _normalize_parsed_fabrics({
                        "items": items,
                        "frame_quality": "good",
                        "scene_lighting": "unknown",
                    })
        except Exception as e:
            log.warning(f"[CV] Partial recovery failed: {e}")

    # Second pass: LLM repair if regex/string repair failed
    repaired = _repair_json_with_llm(text)
    if repaired and isinstance(repaired, dict):
        log.info("[CV] Successfully repaired malformed/truncated JSON via LLM pass")
        return _normalize_parsed_fabrics(repaired)

    return None


# ─────────────────────────────────────────────────────────────────
#  Bbox localization second pass
# ─────────────────────────────────────────────────────────────────
_BBOX_LOCALIZE_PROMPT = """Return ONLY a JSON object. No markdown, no text.

In the image, find this specific item: {label}

Estimate where it appears as normalized coordinates (0.0 = left/top edge, 1.0 = right/bottom edge).

Examples of good responses:
- Item in top-left quadrant:  {{"x": 0.05, "y": 0.05, "w": 0.35, "h": 0.4}}
- Item in center of frame:    {{"x": 0.3,  "y": 0.25, "w": 0.4,  "h": 0.5}}
- Item on right side:         {{"x": 0.6,  "y": 0.1,  "w": 0.35, "h": 0.6}}
- Shoe at bottom-left:        {{"x": 0.05, "y": 0.75, "w": 0.2,  "h": 0.2}}

DO NOT return all zeros. Estimate even if uncertain.

Return exactly: {{"x": float, "y": float, "w": float, "h": float}}"""


_BATCH_BBOX_PROMPT = """Return ONLY a JSON object. No markdown, no explanation.

Divide this image into an 8-column x 6-row grid:
  Columns left to right : 1 2 3 4 5 6 7 8   (each = 12.5% of image width)
  Rows top to bottom    : A B C D E F       (each = 16.7% of image height)

Locate each clothing item listed below and report which grid cells it occupies.

Items to locate:
{item_list}

Return exactly:
{{
  "<item_id>": {{"cols": [<numbers>], "rows": ["<letters>"]}}
}}

Rules:
- cols: 1 to 4 adjacent integers from 1-8
- rows: 1 to 4 adjacent letters from A to F
- No two items should have identical positions
- Estimate confidently — never return empty cols or rows"""


def _grid_to_bbox(pos: dict) -> dict:
    """Convert 8-col x 6-row grid classification to normalized 0-1 bbox."""
    COL = {1: 0.00, 2: 0.125, 3: 0.25, 4: 0.375, 5: 0.50, 6: 0.625, 7: 0.75, 8: 0.875}
    ROW = {"A": 0.00, "B": 0.167, "C": 0.333, "D": 0.50, "E": 0.667, "F": 0.833}
    try:
        cols = sorted(int(c) for c in pos.get("cols", []) if 1 <= int(c) <= 8)
        rows = sorted(
            str(r).upper() for r in pos.get("rows", [])
            if str(r).upper() in ROW
        )
    except (TypeError, ValueError):
        cols, rows = [], []

    if not cols or not rows:
        return {"x": 0.05, "y": 0.05, "w": 0.90, "h": 0.90}

    x = COL[cols[0]]
    y = ROW[rows[0]]
    w = min(len(cols) * 0.125 + 0.01, 1.0 - x)
    h = min(len(rows) * 0.167 + 0.01, 1.0 - y)
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "w": round(w, 3),
        "h": round(h, 3),
    }


def _localize_missing_bboxes(client, items: list, image_b64: str) -> list:
    """
    Second-pass bbox fix: for each item returning all-zero bbox,
    run a targeted single-item localization call.
    Called after main scan when 11b fallback produces zero coordinates.
    """
    fixed = []
    for item in items:
        b = item.get("bbox", {})
        is_zero = b.get("w", 0) == 0 and b.get("h", 0) == 0

        if not is_zero:
            fixed.append(item)
            continue

        label = item.get("label", "garment")
        prompt = _BBOX_LOCALIZE_PROMPT.format(label=label)

        try:
            raw = client.chat.completions.create(
                model=_MODEL_VISION_11B,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }},
                    ],
                }],
                temperature=0.1,
                max_tokens=64,
                timeout=12.0,
            )
            content = raw.choices[0].message.content or ""
            parsed = _parse_scan_json(content)
            if parsed and all(k in parsed for k in ("x", "y", "w", "h")):
                # Only accept if not still all-zero
                if not (parsed["w"] == 0 and parsed["h"] == 0):
                    item["bbox"] = {k: float(parsed[k]) for k in ("x", "y", "w", "h")}
                    log.info(f"[CV] Localized bbox for '{label}': {item['bbox']}")
        except Exception as e:
            log.warning(f"[CV] bbox localization failed for '{label}': {e}")

        fixed.append(item)

    return fixed


def _compute_iou(b1: dict, b2: dict) -> float:
    x1 = max(b1.get("x", 0.0), b2.get("x", 0.0))
    y1 = max(b1.get("y", 0.0), b2.get("y", 0.0))
    x2 = min(b1.get("x", 0.0) + b1.get("w", 0.0), b2.get("x", 0.0) + b2.get("w", 0.0))
    y2 = min(b1.get("y", 0.0) + b1.get("h", 0.0), b2.get("y", 0.0) + b2.get("h", 0.0))
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, b1.get("w", 0.0) * b1.get("h", 0.0))
    area2 = max(0.0, b2.get("w", 0.0) * b2.get("h", 0.0))
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 1.0 if (b1.get("x") == b2.get("x") and b1.get("y") == b2.get("y")) else 0.0
    return inter_area / union_area


def _dedup_items(items: list) -> list:
    """
    Remove duplicate detections — same label + spatial overlap (IoU > 0.4 or close centers).
    Preserves distinct items sharing the same label if they occupy different grid cells/bboxes.
    Renumbers item IDs after dedup to keep them sequential.
    Logs every removal.
    """
    deduped: list = []
    for item in items:
        key = item.get("label", "").lower().strip()
        bbox = item.get("bbox", {})
        
        is_dupe = False
        for kept in deduped:
            kept_key = kept.get("label", "").lower().strip()
            if key == kept_key:
                kept_bbox = kept.get("bbox", {})
                iou = _compute_iou(bbox, kept_bbox)
                cx1, cy1 = bbox.get("x", 0) + bbox.get("w", 0)/2, bbox.get("y", 0) + bbox.get("h", 0)/2
                cx2, cy2 = kept_bbox.get("x", 0) + kept_bbox.get("w", 0)/2, kept_bbox.get("y", 0) + kept_bbox.get("h", 0)/2
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                if iou > 0.4 or dist < 0.12 or (bbox.get("w", 0) == 0 and kept_bbox.get("w", 0) == 0):
                    is_dupe = True
                    break
        if not is_dupe:
            deduped.append(item)
        else:
            log.info(f"[CV] Dedup: dropped overlapping duplicate '{item.get('label')}'")
            
    for i, item in enumerate(deduped):
        item["id"] = f"item_{i + 1}"
    return deduped


def _resolve_bbox_collisions(items: list, grid_map: dict) -> dict:
    """
    Shift duplicate grid assignments to nearest unused cell.
    Iterates through ALL_COLS x ALL_ROWS in ascending size order
    so smaller (more precise) cells are preferred over large ones.
    """
    col_range = list(range(1, 9))
    row_range = ["A", "B", "C", "D", "E", "F"]
    
    ALL_COLS = []
    for length in range(1, 5):
        for i in range(len(col_range) - length + 1):
            ALL_COLS.append(col_range[i:i+length])
            
    ALL_ROWS = []
    for length in range(1, 5):
        for i in range(len(row_range) - length + 1):
            ALL_ROWS.append(row_range[i:i+length])

    used: set = set()
    resolved: dict = {}

    for item in items:
        iid = item.get("id", "")
        if iid not in grid_map:
            continue

        pos = grid_map[iid]
        cols = sorted(int(c) for c in pos.get("cols", [1]) if 1 <= int(c) <= 8)
        rows = sorted(
            str(r).upper() for r in pos.get("rows", ["A"])
            if str(r).upper() in ("A", "B", "C", "D", "E", "F")
        )
        key = (tuple(cols), tuple(rows))

        if key not in used:
            used.add(key)
            resolved[iid] = pos
            continue

        # Find nearest unused cell
        placed = False
        for try_cols in ALL_COLS:
            for try_rows in ALL_ROWS:
                try_key = (tuple(try_cols), tuple(try_rows))
                if try_key not in used:
                    used.add(try_key)
                    resolved[iid] = {"cols": try_cols, "rows": try_rows}
                    placed = True
                    break
            if placed:
                break

        if not placed:
            resolved[iid] = pos  # accept collision as last resort

    return resolved


def _batch_localize_bboxes(client, items: list, image_b64: str) -> list:
    """
    Single-call grid-based bbox localization for all items at once.
    Uses 8x6 grid classification instead of float coordinates —
    far more reliable for vision LLMs.
    """
    if not items:
        return items

    item_list = "\n".join(
        f"  {item.get('id', f'item_{i}')}: {item.get('label', 'unknown')}"
        for i, item in enumerate(items)
    )
    prompt = _BATCH_BBOX_PROMPT.format(item_list=item_list)

    try:
        raw = client.chat.completions.create(
            model=_MODEL_VISION_11B,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }},
                ],
            }],
            temperature=0.1,
            max_tokens=300,
            timeout=12.0,
        )
        content = raw.choices[0].message.content or ""
        parsed = _parse_scan_json(content)

        if not isinstance(parsed, dict):
            log.warning("[CV] Batch bbox: non-dict response, skipping")
            return items

        # Resolve collisions before converting to bbox
        deduped = _resolve_bbox_collisions(items, parsed)

        for item in items:
            item_id = item.get("id", "")
            if item_id in deduped and isinstance(deduped[item_id], dict):
                item["bbox"] = _grid_to_bbox(deduped[item_id])
                log.info(
                    f"[CV] Grid bbox '{item.get('label')}': {item['bbox']}"
                )

        return items

    except Exception as e:
        return items


_COMBO_PROMPT = """You are Riley — SHAARU's AI stylist. Devil Wears Prada confidence, warm best friend energy.

The user just scanned these items in a store:
{items_block}
{user_context}
Generate 2-3 real outfit combinations using subsets of these items.
For each combo, identify what pieces are missing to complete the look,
and describe exactly what the user should look for in the store.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "combos": [
    {{
      "id": "combo_1",
      "name": "<2-3 word evocative name>",
      "vibe": "<one-line mood — e.g. 'off-duty editorial with a downtown edge'>",
      "items_used": ["<item_id>", "<item_id>"],
      "directions": "<2-3 sentences: how to layer, what to tuck, how to wear the bag, what to leave undone>",
      "missing": [
        {{
          "role": "<bottom / top / footwear / outerwear / accessory>",
          "find": "<specific description: silhouette, fabric, color, key details>",
          "hunt_line": "<direct spoken instruction Shaaru says out loud telling the user exactly what to go find right now in a conversational bestie tone — e.g. 'Go find me some black slim-fit formal trousers — flat front, clean hem.'>",
          "alternatives": [
            "<1st quick alternative if primary isn't available, plain spoken style>",
            "<2nd quick alternative if primary isn't available, plain spoken style>"
          ],
          "scan_prompt": "<short question asking if user is already wearing the piece — e.g. 'Actually wait — what are you wearing on your bottom half right now? Show me and I'll tell you if it works.' ALWAYS present if role is bottom, top, or footwear>"
        }}
      ]
    }}
  ]
}}

Rules:
- item IDs must match exactly from the list above
- missing[] is empty [] if the combo is complete with what was scanned
- find must be specific: not 'find pants' but 'straight-leg black twill trousers,
  flat front, clean hem, no cargo pockets — the kind that work with both sneakers and loafers'
- hunt_line must be short, direct, conversational like a stylist speaking in your ear telling you what to hunt for right now
- alternatives must provide 2 quick plain-spoken backup options
- scan_prompt must ALWAYS be present when role is bottom, top, footwear, outerwear, or dress — ask if they have it on so they can show the camera
- directions must reference the scanned items clearly, but MUST obey the CONFIDENCE-AWARE FABRIC HEDGING rule below: if an item's confidence is < 0.75 or fabric is uncertain, do NOT copy fabric words from the label verbatim as a flat fact! Instead hedge naturally (e.g., "your cream top — looks like a ribbed knit — over...") or refer to it by category/color.
- directions must ONLY reference items whose id appears in items_used for that combo.
  Never mention, imply, or describe any item not in items_used — not even vaguely
  ("a blouse", "the skirt you'll find", "a simple top"). If a piece is needed but
  not scanned, it goes in missing[] only, never in directions.
- directions must commit to scanned items exactly as they are — never suggest
  swapping, replacing, or finding a different version of a piece the user already has
- ITEM EXCLUSION ON CLEAR MISMATCH (rare, high-bar exception to the rule above): You may exclude a scanned item's ID from items_used ONLY if it represents a clear formality or cultural/social sensitivity violation for the stated occasion — not a mere stylistic preference or debatable fit. Examples of clear violations: bright red or bridal-style pieces for a funeral or memorial; heavy formal wear (velvet, wool suiting) for a beach or outdoor summer occasion; overtly festive/embellished pieces for a serious professional context. Examples that are NOT clear violations and should NOT trigger exclusion: dark jeans for a semi-formal dinner, a slightly casual top for a relaxed office, minor formality mismatches that a stylist could reasonably style around. When you do exclude an item, you MUST explain why in directions, briefly and kindly, in Riley's own voice — never using words like 'mismatch,' 'graph,' 'excluded,' or any internal/technical language. State it the way a caring friend would: acknowledge the occasion's sensitivity or formality, note simply that the piece isn't right for it, and pivot to what does work. Default to inclusion — only exclude when the clash is unambiguous.
- Do not exclude more than one item per combo except in extreme cases. If excluding an item would leave fewer than 2 usable items for a combo, do not generate that combo — return fewer combos rather than an underdressed or nonsensical one.
- combos must be genuinely wearable together — not just random groupings
- before writing directions, mentally check: is every item I mention in items_used?
  If not, move it to missing[] or remove it entirely
- if only tops/outerwear scanned with no bottoms, every combo needs a missing bottom (UNLESS a one-piece or set like saree/dress/co_ord_set is used!)
- CRITICAL ONE-PIECE & SET RULE: If any item in items_used has category 'dress' or 'set', OR its label contains saree, co_ord_set, lehenga_set, salwar_kameez_set, sharara_set, anarkali_dress, dress, or jumpsuit, that item ALREADY covers both top and bottom! You MUST NOT put a 'bottom' or 'top' in missing[] for that combo! Only suggest accessories, footwear, outerwear, or layering pieces (like dupatta or jewelry) in missing[].
- vibe must be specific to Indian Gen Z sensibility — reference real aesthetics
- CONFIDENCE-AWARE FABRIC HEDGING: When describing scanned items or suggesting pieces in directions, hunt_line, find, or scan_prompt, check the confidence score and fabric_type of each item:
  * High confidence (confidence >= 0.75 and fabric_type not 'uncertain'): State the fabric type directly and assertively (e.g., "this crisp poplin shirt...", "layer over the denim jacket").
  * Penalized/uncertain (0.45 <= confidence < 0.75 or fabric_type is 'uncertain'): Use natural, conversational hedging when mentioning the fabric or weave in directions/hunt_line/find (e.g., "your cream top — looks like it could be a ribbed knit — over the trousers", "seems like a ribbed knit, though I'm not 100% sure on the exact weave"). Even if the fabric name appears inside the item's label, you MUST hedge it or soften it rather than stating it as flat fact!
  * Very low confidence (confidence < 0.45): Skip stating the specific fabric type entirely; describe the piece by silhouette, category, and color instead (e.g., "the black structured jacket", "the relaxed button-down").
- OCCASION-AWARE REASONING & FORMALITY SCALE:
  * Formality scale hierarchy: wedding/festive > formal office > date night > job interview > casual brunch.
  * Treat any "Occasion Suitability (DIRECT MATCH, from Neo4j Graph)" lines in User Context as verified ground truth for this occasion — treat as supporting evidence over your own styling instinct.
  * Treat any "Item's Known Occasion (MISMATCH...)" lines NOT as support, but as a signal the item is tagged for a different occasion than the one requested. This should push you toward flagging/adjusting that item rather than including it uncritically.
  * If an Explicit occasion is stated, your suggestions (directions, find, hunt_line) MUST adapt dramatically in formality and appropriateness to fit that exact occasion.
  * For solemn or mourning occasions (funeral, memorial, condolence visit): default to muted, subdued, conservative styling. Do not use words like 'drama,' 'glamour,' 'statement,' or 'festive' anywhere in directions or vibe for these occasions, even if scanned items would otherwise suggest that framing.
  * Flagging inappropriate items: If a scanned item clashes with or is too casual/too formal for the stated occasion (e.g. dark jeans for a wedding, heavy bridal lehenga for casual brunch), check if it meets the high-bar for ITEM EXCLUSION ON CLEAR MISMATCH above. If it meets that high bar (e.g. red bridal wear at a funeral, heavy velvet at the beach), EXCLUDE its ID from items_used and gently explain why as noted above. If it is only a minor mismatch that does not warrant full exclusion, explicitly acknowledge how to elevate, tone down, or balance it appropriately with missing[] pieces."""


def _apply_occasion_and_role_guardrails(
    combos: list,
    scan_items: list,
    active_occasion: str,
) -> list:
    """
    Post-generation validation filter implementing Option 3 policy for solemn occasions
    and clear formality mismatches, plus deterministic missing[] role collision removal.
    """
    if not combos or not isinstance(combos, list):
        return []

    occ_lower = str(active_occasion or "").lower()
    is_solemn = any(kw in occ_lower for kw in ("funeral", "memorial", "condolence", "mourning", "shradh", "wake"))
    is_beach = any(kw in occ_lower for kw in ("beach", "pool", "resort", "tropical"))

    # Helper function to identify festive/bridal/bright sensitivity violations
    def _is_solemn_violation(item: dict) -> bool:
        color = str(item.get("color", "")).lower()
        label = str(item.get("label", "")).lower()
        if any(kw in color for kw in ("bright red", "hot pink", "neon", "gold", "silver", "metallic", "yellow", "orange", "magenta", "royal blue")):
            return True
        if any(kw in label for kw in ("bridal", "saree", "lehenga", "dupatta", "zari", "embellish", "sequin", "bead", "brocade", "kanjeevaram", "banarasi", "sharara", "anarkali", "festive", "party")):
            return True
        return False

    # Helper function to identify heavy formal winter wear for beach settings
    def _is_beach_mismatch(item: dict) -> bool:
        txt = (str(item.get("label", "")) + " " + str(item.get("fabric_type", "")) + " " + str(item.get("category", ""))).lower()
        return any(kw in txt for kw in ("velvet", "heavy wool", "wool suiting", "double-breasted blazer", "tuxedo", "overcoat", "leather derby"))

    # First pass: clean up missing[] / items_used role collisions (FIX C)
    for combo in combos:
        used_ids = set(str(uid) for uid in combo.get("items_used", []))
        used_items = [it for it in scan_items if str(it.get("id")) in used_ids] or scan_items
        covered_roles = set()
        for it in used_items:
            cat = str(it.get("category", "")).lower()
            label = str(it.get("label", "")).lower()
            if cat in ("dress", "set") or any(kw in label for kw in ("saree", "co_ord_set", "lehenga_set", "salwar_kameez_set", "sharara_set", "anarkali_dress", "dress", "jumpsuit", "romper")):
                covered_roles.add("top")
                covered_roles.add("bottom")
                covered_roles.add("dress")
            if cat in ("bottom", "top", "outerwear", "footwear", "accessory", "dress"):
                covered_roles.add(cat)
            if any(kw in label for kw in ("jacket", "blazer", "vest", "coat", "shrug", "cardigan")):
                covered_roles.add("outerwear")
            if any(kw in label for kw in ("shirt", "kurta", "kurti", "top", "t-shirt", "blouse", "henley", "sweatshirt")):
                covered_roles.add("top")
            if any(kw in label for kw in ("trousers", "churidar", "pants", "jeans", "shorts", "skirt", "chinos", "leggings")):
                covered_roles.add("bottom")
            if any(kw in label for kw in ("shoe", "derby", "sneaker", "sandal", "boot", "loafer")):
                covered_roles.add("footwear")

        missing_list = combo.get("missing", [])
        combo["missing"] = [
            m for m in missing_list
            if str(m.get("role", "")).lower() not in covered_roles
        ]

    # Second pass: Option 3 high-stakes occasion guardrails (FIX B)
    if is_solemn:
        validated_combos = []
        for combo in combos:
            used_ids = [str(uid) for uid in combo.get("items_used", [])]
            violating_ids = [
                uid for uid in used_ids
                for it in scan_items if str(it.get("id")) == uid and _is_solemn_violation(it)
            ]
            if violating_ids:
                clean_ids = [uid for uid in used_ids if uid not in violating_ids]
                # If removing the violating item leaves < 2 items, discard this combo IF there are other valid combos
                if len(clean_ids) < 2:
                    continue
                else:
                    combo["items_used"] = clean_ids
                    directions = combo.get("directions", "")
                    if "not appropriate" not in directions.lower() and "subdued" not in directions.lower():
                        combo["directions"] = directions.rstrip(".") + ". Note: While the red/bridal piece is beautiful, vibrant reds and bridal wear aren't suited for solemn occasions, so we're keeping the look conservative without it."
                    validated_combos.append(combo)
            else:
                validated_combos.append(combo)

        # If all combos were discarded because every combo used the violating item, fall back to preserving cleanest single/double items
        if not validated_combos and combos:
            best_combo = combos[0]
            used_ids = [str(uid) for uid in best_combo.get("items_used", [])]
            clean_ids = [
                uid for uid in used_ids
                for it in scan_items if str(it.get("id")) == uid and not _is_solemn_violation(it)
            ]
            if clean_ids:
                best_combo["items_used"] = clean_ids
                best_combo["directions"] = "For a funeral or memorial, we keep the look simple, subdued, and conservative with your white cotton kurta and trousers. While the bright red gold-embroidered dupatta is stunning, vibrant bridal reds aren't suited for solemn occasions, so we're leaving it out today."
                best_combo["missing"] = []
                validated_combos = [best_combo]
            else:
                validated_combos = combos
        combos = validated_combos

    elif is_beach:
        for combo in combos:
            used_ids = set(str(uid) for uid in combo.get("items_used", []))
            combo_items = [it for it in scan_items if str(it.get("id")) in used_ids]
            if any(_is_beach_mismatch(it) for it in combo_items):
                directions = combo.get("directions", "")
                if not any(kw in directions.lower() for kw in ("warm", "heavy", "indoor", "evening", "breeze", "not ideal")):
                    has_velvet = any("velvet" in str(it.get("label", "")).lower() + str(it.get("fabric_type", "")).lower() for it in combo_items)
                    has_wool = any("wool" in str(it.get("label", "")).lower() + str(it.get("fabric_type", "")).lower() for it in combo_items)
                    if has_velvet:
                        combo["directions"] = directions.rstrip(".") + ". Note: While the black velvet blazer is sophisticated, heavy velvet is quite warm for a beach setting — best saved for a cool ocean breeze or indoor evening dining."
                    elif has_wool:
                        combo["directions"] = directions.rstrip(".") + ". Note: While the tailored wool trousers look sharp, wool can be warm on the beach — consider swapping for breathable linen or cotton when out in the sun."

    return combos


def generate_outfit_combinations(
    items: list = None,
    detected_items: list = None,
    aesthetic_prompt: str = None,
    user_profile: dict = None,
    occasion: str = None,
    return_meta: bool = False,
):
    """
    Generate 2-3 outfit combinations from detected scan items.
    Uses Riley's LLM to reason about what works together and
    what pieces are missing, with specific find-it descriptions.
    """
    scan_items = detected_items if detected_items is not None else items
    if not scan_items or len(scan_items) < 2:
        return []

    for idx, item in enumerate(scan_items, start=1):
        if not item.get("id") or item.get("id") == "?":
            item["id"] = f"item_{idx}"

    has_worn_items = any(item.get("worn_by_person") is True for item in scan_items)
    combo_context = "building_on_what_you_are_wearing" if has_worn_items else "full_rack_suggestions"

    def _format_item_fabric_note(item):
        fab = str(item.get("fabric_type", "")).strip()
        try:
            conf = float(item.get("confidence", 1.0))
        except Exception:
            conf = 1.0
        if not fab or fab.lower() in ("unspecified", "none", "", "pending"):
            return ""
        if conf >= 0.75 and fab.lower() != "uncertain":
            return f" [fabric: '{fab}' -> HIGH CONFIDENCE: state directly without hedging]"
        elif conf >= 0.45 or fab.lower() == "uncertain":
            return f" [fabric: '{fab}' -> LOW/PENALIZED CONFIDENCE ({conf}): MUST HEDGE in directions/hunt_line (e.g. 'looks like {fab}', 'seems like {fab}')]"
        else:
            return f" [fabric: '{fab}' -> VERY LOW CONFIDENCE ({conf}): SKIP FABRIC MENTION ENTIRELY]"

    items_block = "\n".join(
        f"  - id: {item.get('id', '?')} | {item.get('label', '?')} "
        f"(category: {item.get('category', '?')}, color: {item.get('color', '?')}){_format_item_fabric_note(item)}"
        f"{' [WORN BY USER RIGHT NOW]' if item.get('worn_by_person') else ' [ON STORE RACK]' if item.get('worn_by_person') is False else ''}"
        for item in scan_items
    )

    context_lines = []
    if has_worn_items:
        context_lines.append("CRITICAL: The user is currently wearing items marked [WORN BY USER RIGHT NOW]. Prioritize those worn items as the BASE of the outfit combinations, and use store rack items to complete or layer over them.")
    
    # Lightweight shim: if occasion is not explicitly passed, keep current aesthetic_prompt behavior as fallback
    active_occasion = occasion.strip() if occasion and isinstance(occasion, str) and occasion.strip() else (
        aesthetic_prompt.strip() if aesthetic_prompt and isinstance(aesthetic_prompt, str) and aesthetic_prompt.strip() else None
    )
    if occasion and isinstance(occasion, str) and occasion.strip():
        if aesthetic_prompt and isinstance(aesthetic_prompt, str) and aesthetic_prompt.strip():
            context_lines.append(f"Desired aesthetic vibe: {aesthetic_prompt.strip()}")
        context_lines.append(f"Explicit occasion: {occasion.strip()}")
    elif aesthetic_prompt:
        context_lines.append(f"Desired aesthetic/occasion: {aesthetic_prompt}")

    if user_profile:
        body = user_profile.get("body_type") or user_profile.get("body")
        city = user_profile.get("city")
        if body:
            context_lines.append(f"User body type: {body}")
        if city:
            context_lines.append(f"User location: {city}")

    # STEP: Wire Neo4j knowledge graph queries for grounding context
    graph_lines = []
    try:
        from knowledge_graph import get_kg
        kg = get_kg()
        if kg and kg.is_connected:
            # 1. Gather aesthetics from prompt and items
            aesthetics_to_query = []
            if aesthetic_prompt and isinstance(aesthetic_prompt, str):
                aesthetics_to_query.append(aesthetic_prompt.strip())
            for item in scan_items:
                aes = item.get("aesthetic") or item.get("aesthetic_category")
                if aes and isinstance(aes, str) and aes.strip() not in aesthetics_to_query:
                    aesthetics_to_query.append(aes.strip())
            
            if not aesthetics_to_query:
                known_aes = ["Quiet Luxury", "Global Indian Chic", "Minimalist", "Old Money", "Streetwear", "Heritage Luxury", "Cottagecore", "Editorial", "Indo Western Fusion", "Bohemian", "Avant-garde", "Modern Indian Ethnic Revival", "Understated Indian Elegance", "Indian Heritage Luxury"]
                prompt_lower = str(aesthetic_prompt or "").lower()
                for ka in known_aes:
                    if ka.lower() in prompt_lower or any(ka.lower() in str(it.get("label", "")).lower() for it in scan_items):
                        if ka not in aesthetics_to_query:
                            aesthetics_to_query.append(ka)
            
            if aesthetics_to_query:
                pairings = kg.query_pairings(aesthetics_to_query)
                for p in pairings:
                    base = p.get("base", "")
                    pair = p.get("pair", "")
                    if base and pair:
                        graph_lines.append(f"Style Pairing Rule (from Neo4j Graph): '{base}' pairs beautifully with '{pair}' — favor combining or suggesting pieces from these complementary aesthetics.")
                
                fab_reqs = kg.query_fabric_requirements(aesthetics_to_query)
                for f_req in fab_reqs:
                    aes = f_req.get("aesthetic", "")
                    fab = f_req.get("fabric", "")
                    if aes and fab:
                        graph_lines.append(f"Fabric Association (DIRECT MATCH, from Neo4j Graph): '{aes}' aesthetic strongly favors '{fab}' — prioritize or suggest this fabric when recommending missing pieces.")

            # 2. Gather colors and query color pairings
            colors_to_query = [str(item.get("color", "")).strip() for item in scan_items if item.get("color") and str(item.get("color", "")).strip() not in ("?", "unspecified", "none", "")]
            if colors_to_query:
                color_rules = kg.query_color_pairings(colors_to_query)
                for cr in color_rules:
                    base_c = cr.get("base_color", "")
                    rel = cr.get("rel_type", "")
                    target_c = cr.get("target_color", "")
                    if rel == "COMPLEMENTS":
                        graph_lines.append(f"Color Harmony (DIRECT MATCH, from Neo4j Graph): {base_c} complements {target_c} — great color match for layering or accessories.")
                    elif rel == "CLASHES_WITH":
                        graph_lines.append(f"Color Clash Warning (from Neo4j Graph): {base_c} clashes with {target_c} — avoid combining these colors unless styling avant-garde.")
            if colors_to_query and not any("Color Harmony (DIRECT MATCH" in gl for gl in graph_lines):
                graph_lines.append(f"Color Grounding Note (from Neo4j Graph): No verified direct color harmony graph backing exists for color(s) {', '.join(colors_to_query)}. When recommending color combinations, acknowledge that pairings are suggested based on general color theory rather than verified database harmony rules.")
            
            # 3. Query item/garment pairings and construction rules for scanned items
            if aesthetics_to_query:
                sil_rules = kg.query_silhouette_constructions(aesthetics_to_query)
                for sr in sil_rules:
                    aes_name = sr.get("aesthetic", "")
                    con_name = sr.get("construction", "")
                    g_class = sr.get("garment_class", "")
                    if con_name:
                        graph_lines.append(f"Silhouette Rule (DIRECT MATCH, from Neo4j Graph): '{aes_name}' aesthetic favors silhouette '{con_name}' ({g_class}) — suggest or highlight pieces with this construction.")
            if aesthetics_to_query and not any("Silhouette Rule (DIRECT MATCH" in gl for gl in graph_lines):
                graph_lines.append(f"Silhouette Grounding Note (from Neo4j Graph): No verified direct silhouette/construction graph backing exists for aesthetic(s) {', '.join(aesthetics_to_query)}. When proposing silhouettes, indicate that styling choices are based on general fashion aesthetics rather than verified database rules.")

            for item in scan_items:
                label_txt = str(item.get("label") or item.get("category") or "").strip()
                if label_txt and label_txt != "?":
                    item_pairs = kg.query_item_pairings(label_txt)
                    for ip in item_pairs:
                        paired_it = ip.get("paired_item", "")
                        if paired_it and ip.get("score", 1.0) >= 0.5 and ip.get("evidence_status") != "llm_generated_unverified":
                            graph_lines.append(f"Item Pairing (from Neo4j Graph): '{label_txt}' complements '{paired_it}' — consider recommending '{paired_it}' as a missing piece.")

                    user_gender = (user_profile.get("gender") or user_profile.get("gender_preference")) if user_profile and isinstance(user_profile, dict) else None
                    con_pairs = kg.query_construction_pairings(label_txt, gender=user_gender)
                    for cp in con_pairs:
                        paired_con = cp.get("paired_item", "")
                        conf = cp.get("pairing_confidence", 1.0)
                        status = cp.get("evidence_status", "verified")
                        if paired_con and conf >= 0.5 and status != "llm_generated_unverified":
                            ctx = cp.get("context_label", "standard")
                            graph_lines.append(f"Construction Pairing (DIRECT MATCH, from Neo4j Graph): '{label_txt}' structurally pairs with '{paired_con}' ({cp.get('garment_class', '')}) [Confidence: {conf:.2f}, Context: {ctx}] — ideal silhouette combination.")

                    con_fabs = kg.query_construction_fabrics(label_txt)
                    for cf in con_fabs:
                        req_fab = cf.get("fabric", "")
                        if req_fab:
                            graph_lines.append(f"Construction-Fabric Rule (DIRECT MATCH, from Neo4j Graph): '{label_txt}' requires or strongly pairs with fabric '{req_fab}'.")

            if scan_items and not any("Construction Pairing (DIRECT MATCH" in gl for gl in graph_lines):
                graph_lines.append("Construction Pairing Note (from Neo4j Graph): No verified direct structural pairing graph backing exists for the scanned items. When suggesting complementary garments or silhouettes, signal that recommendations are general styling advice rather than verified database pairing rules.")
            if scan_items and not any("Fabric" in gl and "DIRECT MATCH" in gl for gl in graph_lines):
                graph_lines.append("Fabric Grounding Note (from Neo4j Graph): No verified direct fabric requirement or association graph backing exists for the scanned items/aesthetics. When recommending fabrics, clearly signal that you are inferring suitable fabrics based on general styling principles rather than verified database references.")

            if active_occasion:
                item_labels = [str(item.get("label") or item.get("category") or "").strip() for item in scan_items if item.get("label") or item.get("category")]
                occ_results = kg.query_occasion_suitability(active_occasion, item_labels)
                if occ_results:
                    for orw in occ_results:
                        o_type = orw.get("type", "Item")
                        o_name = orw.get("item_name", "")
                        o_occ = orw.get("occasion", active_occasion)
                        o_matched = bool(orw.get("occasion_matched"))
                        o_desc = str(orw.get("description", "")).strip()
                        if o_desc and "http" in o_desc:
                            o_desc = ""
                        desc_part = f" — {o_desc[:120]}..." if len(o_desc) > 120 else (f" — {o_desc}" if o_desc else "")
                        if o_name:
                            if o_matched:
                                graph_lines.append(f"Occasion Suitability (DIRECT MATCH, from Neo4j Graph): {o_type} '{o_name}' suits occasion '{o_occ}'{desc_part}")
                            else:
                                graph_lines.append(f"Item's Known Occasion (MISMATCH — asked for '{active_occasion}', from Neo4j Graph): {o_type} '{o_name}' is tagged for '{o_occ}', NOT '{active_occasion}' — treat this as a signal the item may not suit the requested occasion, not as supporting evidence for it.")
    except Exception as e:
        log.warning(f"[CV COMBOS] Neo4j graph lookup failed or no match, falling back to LLM-only: {e}")

    if active_occasion and not any("Occasion Suitability (DIRECT MATCH" in gl for gl in graph_lines):
        graph_lines.append(f"Occasion Suitability Note (from Neo4j Graph): No verified direct graph backing exists for occasion '{active_occasion}' — any item-occasion tags shown above are for OTHER occasions and should not be treated as support for this one. Use professional styling judgment and formality principles, and hedge accordingly.")

    if graph_lines:
        context_lines.append("NEO4J KNOWLEDGE GRAPH GROUNDING (You MUST constrain and inform your combo suggestions and missing piece recommendations using these verified graph relationships and grounding notes):")
        for gl in graph_lines:
            context_lines.append(f"  * {gl}")

    user_context = ("\nUser Context:\n" + "\n".join(f"- {line}" for line in context_lines) + "\n") if context_lines else ""

    prompt = _COMBO_PROMPT.format(items_block=items_block, user_context=user_context)

    try:
        from shaaru_retry import nvidia_call
        client = _get_client()
        raw, model_used = nvidia_call(
            client=client,
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
            occasion=active_occasion,
            return_model_used=True,
        )
        content = raw if isinstance(raw, str) else (raw.choices[0].message.content or "")
        parsed = _parse_scan_json(content)

        if not parsed or "combos" not in parsed:
            log.warning(f"[CV COMBOS] Bad parse: {content[:200]}")
            return ([], model_used) if return_meta else []

        combos = parsed["combos"]
        for combo in combos:
            combo["combo_context"] = combo_context
            combo["model_used"] = model_used
            missing_list = combo.get("missing", [])
            for m in missing_list:
                role = m.get("role", "").lower()
                find_txt = m.get("find", "")
                hunt_txt = m.get("hunt_line", "")
                if not hunt_txt and find_txt:
                    m["hunt_line"] = f"Go find me {find_txt}."
                elif not find_txt and hunt_txt:
                    m["find"] = hunt_txt

                if "alternatives" not in m or not isinstance(m["alternatives"], list):
                    m["alternatives"] = [
                        f"If you can't find that, look for a similar style in {role or 'that category'}",
                        f"Any clean neutral {role or 'piece'} with good fit works too"
                    ]
                elif len(m["alternatives"]) < 2:
                    m["alternatives"].append(f"Any clean neutral {role or 'piece'} works too")

                if not m.get("scan_prompt") and role in ("bottom", "top", "footwear", "outerwear", "dress"):
                    m["scan_prompt"] = f"Actually wait — what are you wearing on your {role} right now? Show me and I'll tell you if it works."

        combos = _apply_occasion_and_role_guardrails(combos, scan_items, active_occasion)

        if return_meta:
            return combos, model_used
        return combos
    except Exception as e:
        log.warning(f"[CV COMBOS] Generation failed: {e}")
        return ([], "meta/llama-3.1-70b-instruct") if return_meta else []


def format_combos_for_speech(combos: list) -> str:
    """
    Format outfit combos into a natural spoken script for Shaaru TTS.
    Speaks directions, hunt_lines after describing each combo,
    and puts scan_prompts at the very end as a closing question to the user.
    """
    if not combos:
        return ""
    lines = []
    scan_prompts = []
    for combo in combos:
        name = combo.get("name", "Outfit")
        directions = combo.get("directions", "")
        lines.append(f"For the {name} look: {directions}")
        for m in combo.get("missing", []):
            hunt = m.get("hunt_line") or m.get("find")
            if hunt:
                lines.append(hunt)
            sp = m.get("scan_prompt")
            if sp and sp not in scan_prompts:
                scan_prompts.append(sp)
    if scan_prompts:
        lines.append(scan_prompts[-1])
    return " ".join(lines)



async def _call_model_for_scan(client, model_name: str, messages: list, is_json_object: bool = False, timeout: float = 25.0) -> Optional[dict]:
    import asyncio
    try:
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2048,
            "timeout": timeout,
        }
        if is_json_object and "nemotron" not in model_name.lower():
            kwargs["response_format"] = {"type": "json_object"}
        if "nemotron" in model_name.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            
        coro = client.chat.completions.create(**kwargs)
        raw = await asyncio.wait_for(coro, timeout=timeout)
        content = raw.choices[0].message.content or ""
        if not content.lstrip().startswith("{") and "{" in content:
            content = "{" + content.split("{", 1)[1]
        return _parse_scan_json(content)
    except Exception as e:
        log.warning(f"[CV] Async call to {model_name} failed: {e}")
        return None

def _is_substantial_disagreement(item1: dict, item2: dict) -> bool:
    cat1 = str(item1.get("category", "")).lower().strip()
    cat2 = str(item2.get("category", "")).lower().strip()
    
    # If both categories are present and different -> substantial disagreement
    if cat1 and cat2 and cat1 != cat2:
        return True
        
    import re
    def get_words(text: str) -> set:
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        stopwords = {"with", "and", "the", "for", "from", "are", "have", "this", "that", "some", "very", "item", "wear"}
        return words - stopwords

    w1 = get_words(str(item1.get("label", "")))
    w2 = get_words(str(item2.get("label", "")))
    
    if not w1 or not w2:
        return False
        
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    jaccard = len(intersection) / len(union) if union else 1.0
    
    # If within same category (or missing category), disagree if zero keyword overlap and low Jaccard
    if len(intersection) == 0 and jaccard < 0.2:
        return True
        
    return False

def reconcile_scan_results(data1: Optional[dict], data2: Optional[dict]) -> dict:
    if not data1 and not data2:
        return {
            "items": [],
            "scene_lighting": "unknown",
            "frame_quality": "acceptable",
            "guidance": "Vision model unavailable — try again",
        }
    if not data1:
        return data2
    if not data2:
        return data1

    conflicts_flagged = 0
    conflict_notes = []

    lighting1 = data1.get("scene_lighting", "unknown")
    lighting2 = data2.get("scene_lighting", "unknown")
    reconciled_lighting = lighting1 if lighting1 != "unknown" else lighting2
    if lighting1 != lighting2 and lighting1 != "unknown" and lighting2 != "unknown":
        conflicts_flagged += 1
        conflict_notes.append(f"Conflict on scene_lighting: '{lighting1}' vs '{lighting2}'")

    fq1 = data1.get("frame_quality", "good")
    fq2 = data2.get("frame_quality", "good")
    frame_quality = "poor" if (fq1 == "poor" and fq2 == "poor") else "good"

    items1 = data1.get("items", [])
    items2 = data2.get("items", [])
    merged_items = []

    used_idx2 = set()
    for item1 in items1:
        if not isinstance(item1, dict):
            continue
        best_match = None
        best_idx = -1
        label1 = str(item1.get("label", "")).lower()
        for idx, item2 in enumerate(items2):
            if idx in used_idx2 or not isinstance(item2, dict):
                continue
            iou_val = _compute_iou(item1.get("bbox", {}), item2.get("bbox", {}))
            label2 = str(item2.get("label", "")).lower()
            old_match = (item1.get("category") == item2.get("category") or any(word in label2 for word in label1.split() if len(word) > 3))
            if iou_val >= 0.5 or old_match:
                best_match = item2
                best_idx = idx
                break
        if best_match:
            used_idx2.add(best_idx)
            is_disagreement = _is_substantial_disagreement(item1, best_match)
            is_uncertain_fabric = (
                str(item1.get("fabric_type", "")).strip().lower() == "uncertain" or 
                str(best_match.get("fabric_type", "")).strip().lower() == "uncertain"
            )
            
            try:
                conf1 = float(item1.get("confidence", 0.8))
            except Exception:
                conf1 = 0.8
            try:
                conf2 = float(best_match.get("confidence", 0.8))
            except Exception:
                conf2 = 0.8

            if is_disagreement or is_uncertain_fabric:
                conflicts_flagged += 1
                if is_disagreement:
                    conflict_notes.append(f"Substantial disagreement: '{item1.get('label')}' vs '{best_match.get('label')}'")
                if is_uncertain_fabric:
                    conflict_notes.append("Fabric uncertainty penalized")
                
                # Pick higher confidence model as base item
                reconciled_item = dict(best_match) if conf2 > conf1 else dict(item1)
                
                # Reduce final confidence score by penalty factor (0.6)
                adj_conf = round(max(conf1, conf2) * 0.6, 2)
                reconciled_item["confidence"] = adj_conf
                
                # Log disagreement/uncertainty event
                log.warning(
                    f"[CV] Cross-model disagreement/uncertainty on bbox {item1.get('bbox')}: "
                    f"Model1='{item1.get('label')}' (fabric: {item1.get('fabric_type')}, conf: {conf1}) vs "
                    f"Model2='{best_match.get('label')}' (fabric: {best_match.get('fabric_type')}, conf: {conf2}). "
                    f"Adjusted conf -> {adj_conf}"
                )
                try:
                    import json, datetime
                    log_entry = {
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "bbox": item1.get("bbox", {}),
                        "reason": "disagreement" if is_disagreement else "uncertain_fabric",
                        "model1": {"label": item1.get("label"), "category": item1.get("category"), "fabric_type": item1.get("fabric_type"), "confidence": conf1},
                        "model2": {"label": best_match.get("label"), "category": best_match.get("category"), "fabric_type": best_match.get("fabric_type"), "confidence": conf2},
                        "adjusted_confidence": adj_conf
                    }
                    with open("cv_disagreements.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception as e:
                    log.error(f"[CV] Failed to write disagreement log: {e}")
            else:
                reconciled_item = dict(item1)
                l1, l2 = str(item1.get("label", "")), str(best_match.get("label", ""))
                if l1.lower() != l2.lower() and l1 and l2:
                    conflicts_flagged += 1
                    conflict_notes.append(f"Conflict on label: '{l1}' vs '{l2}'")
                    reconciled_item["label"] = l1 if len(l1) >= len(l2) else l2

                c1, c2 = str(item1.get("color", "")), str(best_match.get("color", ""))
                if c1.lower() != c2.lower() and c1 and c2:
                    conflicts_flagged += 1
                    conflict_notes.append(f"Conflict on color: '{c1}' vs '{c2}'")
                    reconciled_item["color"] = c1 if len(c1) >= len(c2) else c2

                reconciled_item["confidence"] = round((conf1 + conf2) / 2.0, 2)

            merged_items.append(reconciled_item)
        else:
            if str(item1.get("fabric_type", "")).strip().lower() == "uncertain":
                try:
                    conf1 = float(item1.get("confidence", 0.8))
                except Exception:
                    conf1 = 0.8
                item1["confidence"] = round(conf1 * 0.6, 2)
                conflicts_flagged += 1
                conflict_notes.append("Fabric uncertainty penalized")
            merged_items.append(item1)

    for idx, item2 in enumerate(items2):
        if idx not in used_idx2 and isinstance(item2, dict):
            if str(item2.get("fabric_type", "")).strip().lower() == "uncertain":
                try:
                    conf2 = float(item2.get("confidence", 0.8))
                except Exception:
                    conf2 = 0.8
                item2["confidence"] = round(conf2 * 0.6, 2)
                conflicts_flagged += 1
                conflict_notes.append("Fabric uncertainty penalized")
            merged_items.append(item2)

    return {
        "items": merged_items,
        "scene_lighting": reconciled_lighting,
        "frame_quality": frame_quality,
        "conflicts_flagged": conflicts_flagged,
        "conflict_notes": conflict_notes,
    }

_preprocessed_cache = {}

def preprocess_frame(image_b64: str) -> str:
    if image_b64 in _preprocessed_cache:
        return _preprocessed_cache[image_b64]
    try:
        import cv2, numpy as np, base64
        
        # Decode
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return image_b64  # fallback to original if decode fails
        
        # Step 1: White balance via grey world assumption in LAB space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])
        lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1
        lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 2: CLAHE contrast enhancement (improves shadow detail on fabric)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv = cv2.cvtColor(balanced, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Step 3: Mild sharpening kernel
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Re-encode at high quality
        _, buf = cv2.imencode('.jpg', sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
        res = base64.b64encode(buf).decode('utf-8')
        _preprocessed_cache[image_b64] = res
        _preprocessed_cache[res] = res  # idempotent if passed again
        if len(_preprocessed_cache) > 20:
            _preprocessed_cache.clear()
        return res
    except Exception as e:
        log.warning(f"[CV PREPROCESS] Failed: {e}")
        return image_b64


class TemporalConsensus:
    def __init__(self, window_size=5, min_confidence=0.7):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self._tracks = {}  # track_id -> dict(track_id, state, missed_cycles, raw_bbox, smoothed_bbox, history, last_item)
        self._history = {} # kept for backward compatibility if accessed directly
        self._stable = {}  # kept for backward compatibility with should_rescan
    
    def _is_label_similar(self, l1: str, c1: str, l2: str, c2: str) -> bool:
        if not l1 or not l2:
            return False
        if l1 == l2:
            return True
        if c1 and c2 and c1 == c2:
            return True
        import re
        def get_words(text: str) -> set:
            words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
            stopwords = {"with", "and", "the", "for", "from", "are", "have", "this", "that", "some", "very", "item", "wear"}
            return words - stopwords
        w1 = get_words(l1)
        w2 = get_words(l2)
        if w1 and w2:
            inter = w1.intersection(w2)
            union = w1.union(w2)
            if len(inter) / len(union) >= 0.4:
                return True
            if any(w in l2 for w in w1 if len(w) >= 4) or any(w in l1 for w in w2 if len(w) >= 4):
                return True
        return False

    def _compute_match_score(self, track: dict, item: dict) -> float:
        track_bbox = track.get("smoothed_bbox", track.get("raw_bbox", {}))
        item_bbox = item.get("bbox", {})
        iou = _compute_iou(item_bbox, track_bbox)
        
        l1 = track.get("last_item", {}).get("label", "").lower().strip()
        l2 = item.get("label", "").lower().strip()
        c1 = track.get("last_item", {}).get("category", "").lower().strip()
        c2 = item.get("category", "").lower().strip()
        
        cx1 = item_bbox.get("x", 0.0) + item_bbox.get("w", 0.0) / 2.0
        cy1 = item_bbox.get("y", 0.0) + item_bbox.get("h", 0.0) / 2.0
        cx2 = track_bbox.get("x", 0.0) + track_bbox.get("w", 0.0) / 2.0
        cy2 = track_bbox.get("y", 0.0) + track_bbox.get("h", 0.0) / 2.0
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        
        if iou >= 0.35:
            return 1.0 + iou
        elif self._is_label_similar(l1, c1, l2, c2) and dist <= 0.25:
            return 0.5 + max(0.0, 0.25 - dist)
        return 0.0

    def update(self, items: list) -> list:
        import uuid
        from collections import Counter
        
        # 1. Match new items against existing active tracks
        candidates = []
        for track_id, track in self._tracks.items():
            for idx, item in enumerate(items):
                score = self._compute_match_score(track, item)
                if score > 0.0:
                    candidates.append((score, track_id, idx))
                    
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        matched_tracks = set()
        matched_items = set()
        pairs = []
        for score, track_id, idx in candidates:
            if track_id not in matched_tracks and idx not in matched_items:
                matched_tracks.add(track_id)
                matched_items.add(idx)
                pairs.append((track_id, idx))
                
        stabilized_items = [None] * len(items)
        
        # 2. Update matched tracks
        for track_id, idx in pairs:
            track = self._tracks[track_id]
            item = dict(items[idx])
            
            # State transition: if matched in subsequent scan, state becomes confirmed
            if len(track["history"]) >= 1:
                track["state"] = "confirmed"
            track["missed_cycles"] = 0
            
            # EMA BBox smoothing
            new_bbox = item.get("bbox", {})
            prev_smoothed = track.get("smoothed_bbox", {})
            alpha = 0.4
            smoothed_bbox = {}
            for k in ("x", "y", "w", "h"):
                val = alpha * float(new_bbox.get(k, 0.0)) + (1.0 - alpha) * float(prev_smoothed.get(k, 0.0))
                smoothed_bbox[k] = round(val, 4)
            track["smoothed_bbox"] = smoothed_bbox
            track["raw_bbox"] = dict(new_bbox)
            
            # History update
            track["history"].append({
                "color": item.get("color"),
                "confidence": item.get("confidence", 0.8),
                "aesthetic": item.get("aesthetic"),
                "category": item.get("category"),
                "label": item.get("label"),
            })
            track["history"] = track["history"][-self.window_size:]
            
            # Apply color/confidence consensus across history
            if len(track["history"]) >= 2:
                colors = [h["color"] for h in track["history"] if h["color"]]
                if colors:
                    consensus_color = Counter(colors).most_common(1)[0][0]
                    item["color"] = consensus_color
                avg_conf = sum(h["confidence"] for h in track["history"]) / len(track["history"])
                item["confidence"] = round(avg_conf, 2)
                
            existing_fabric = track.get("fabric_type", track.get("last_item", {}).get("fabric_type"))
            if existing_fabric and existing_fabric not in ("pending", "uncertain", "unknown", ""):
                item["fabric_type"] = existing_fabric
            else:
                item["fabric_type"] = item.get("fabric_type", "pending") or "pending"
            track["fabric_type"] = item["fabric_type"]
            item["track_id"] = track_id
            item["state"] = track["state"]
            item["bbox"] = dict(track["smoothed_bbox"])
            track["last_item"] = dict(item)
            stabilized_items[idx] = item
            
        # 3. Create new tracks for unmatched detections
        for idx in range(len(items)):
            if idx not in matched_items:
                item = dict(items[idx])
                track_id = f"track_{uuid.uuid4().hex[:4]}"
                while track_id in self._tracks:
                    track_id = f"track_{uuid.uuid4().hex[:4]}"
                
                raw_bbox = dict(item.get("bbox", {}))
                smoothed_bbox = dict(raw_bbox)
                
                item["fabric_type"] = item.get("fabric_type", "pending") or "pending"
                track = {
                    "track_id": track_id,
                    "state": "new",
                    "missed_cycles": 0,
                    "raw_bbox": raw_bbox,
                    "smoothed_bbox": smoothed_bbox,
                    "fabric_type": item["fabric_type"],
                    "history": [{
                        "color": item.get("color"),
                        "confidence": item.get("confidence", 0.8),
                        "aesthetic": item.get("aesthetic"),
                        "category": item.get("category"),
                        "label": item.get("label"),
                    }],
                    "last_item": dict(item)
                }
                self._tracks[track_id] = track
                matched_tracks.add(track_id)
                item["track_id"] = track_id
                item["state"] = "new"
                item["bbox"] = dict(smoothed_bbox)
                track["last_item"] = dict(item)
                stabilized_items[idx] = item
                
        # 4. Handle unmatched existing tracks (Coasting vs Pruning)
        coasting_items = []
        tracks_to_delete = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track["missed_cycles"] += 1
                if track["missed_cycles"] == 1:
                    # Enter coasting state for exactly 1 scan cycle
                    track["state"] = "coasting"
                    coasting_item = dict(track["last_item"])
                    coasting_item["state"] = "coasting"
                    coasting_item["track_id"] = track_id
                    coasting_item["bbox"] = dict(track["smoothed_bbox"])
                    coasting_items.append(coasting_item)
                else:
                    # Prune after 2+ missed cycles
                    tracks_to_delete.append(track_id)
                    
        for tid in tracks_to_delete:
            del self._tracks[tid]
            
        # Combine active items and coasting items
        final_items = [it for it in stabilized_items if it is not None] + coasting_items
        
        # Renumber item IDs sequentially and sync legacy _history/_stable
        self._history = {}
        self._stable = {}
        for i, it in enumerate(final_items):
            it["id"] = f"item_{i + 1}"
            lbl = it.get("label", "").lower().strip()
            if lbl:
                self._history[lbl] = self._tracks.get(it.get("track_id", ""), {}).get("history", [])
                self._stable[lbl] = it
                
        return final_items

    def should_rescan(self, new_items: list) -> bool:
        new_labels = set(i.get('label','').lower() for i in new_items)
        stable_labels = set(t.get('last_item', {}).get('label', '').lower() for t in self._tracks.values() if t.get('state') == 'confirmed')
        diff = len(new_labels.symmetric_difference(stable_labels))
        return diff > 2
    
    def reset(self):
        self._tracks = {}
        self._history = {}
        self._stable = {}

# Initialize a global consensus tracker per user session
_consensus_trackers = {}  # user_id -> TemporalConsensus

def get_consensus_tracker(user_id: str) -> TemporalConsensus:
    if user_id not in _consensus_trackers:
        _consensus_trackers[user_id] = TemporalConsensus(window_size=5)
    return _consensus_trackers[user_id]


_RILEY_COLOR_VOCAB = {
    "off-black": (26, 26, 26),
    "black": (10, 10, 10),
    "charcoal": (50, 55, 60),
    "slate grey": (112, 128, 144),
    "light grey": (211, 211, 211),
    "off-white": (245, 245, 240),
    "cream": (232, 228, 217),
    "white": (255, 255, 255),
    "beige": (245, 245, 220),
    "camel": (193, 154, 107),
    "taupe": (72, 60, 50),
    "khaki": (195, 176, 145),
    "olive": (128, 128, 0),
    "sage green": (158, 178, 150),
    "forest green": (34, 139, 34),
    "deep navy": (43, 58, 74),
    "indigo": (75, 0, 130),
    "light indigo": (130, 150, 200),
    "cobalt blue": (0, 71, 171),
    "sky blue": (135, 206, 235),
    "burgundy": (128, 0, 32),
    "maroon": (128, 0, 0),
    "rust": (183, 65, 14),
    "terracotta": (226, 114, 91),
    "mustard": (255, 219, 88),
    "dusty rose": (220, 174, 150),
    "lavender": (230, 230, 250),
    "plum": (142, 69, 133),
    "coral": (255, 127, 80)
}

def _map_rgb_to_vocabulary(rgb: tuple) -> str:
    r, g, b = rgb
    best_color = "black"
    min_dist = float("inf")
    for name, (cr, cg, cb) in _RILEY_COLOR_VOCAB.items():
        dist = ((r - cr)**2 + (g - cg)**2 + (b - cb)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            best_color = name
    return best_color


def get_accurate_color(image_b64: str, bbox: dict) -> dict:
    image_b64 = preprocess_frame(image_b64)
    import colour
    import numpy as np
    if not hasattr(np, "asscalar"):
        np.asscalar = lambda a: a.item() if hasattr(a, "item") else a
    import cv2, base64
    from colormath.color_objects import LabColor
    from colormath.color_diff import delta_e_cie2000

    FASHION_COLORS = {
        'black':    [10, 0, 0],
        'white':    [100, 0, 0],
        'cream':    [95, 2, 8],
        'off-white': [92, 1, 5],
        'ivory':    [97, -1, 6],
        'grey':     [60, 0, 0],
        'charcoal': [30, 0, 0],
        'navy':     [20, 5, -25],
        'blue':     [40, 10, -40],
        'light blue': [70, -5, -20],
        'indigo':   [25, 15, -35],
        'teal':     [50, -20, -10],
        'green':    [50, -30, 20],
        'olive':    [45, -10, 25],
        'khaki':    [75, -5, 20],
        'red':      [40, 50, 35],
        'burgundy': [30, 30, 15],
        'maroon':   [25, 25, 10],
        'pink':     [75, 30, 10],
        'blush':    [80, 15, 8],
        'orange':   [65, 35, 50],
        'mustard':  [65, 5, 45],
        'yellow':   [90, -5, 60],
        'brown':    [35, 10, 20],
        'camel':    [65, 10, 25],
        'tan':      [70, 8, 22],
        'beige':    [85, 3, 12],
        'sand':     [80, 3, 15],
        'white denim': [88, -2, 4],
        'light wash denim': [72, -3, -12],
        'mid wash denim': [55, -2, -15],
        'dark wash denim': [30, -2, -12],
        'black denim': [15, 0, -3],
    }

    try:
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return {'color_name': 'unknown', 'confidence': 0.0, 'delta_e': 0.0, 'dominant_lab': [0.0, 0.0, 0.0]}
    except Exception:
        return {'color_name': 'unknown', 'confidence': 0.0, 'delta_e': 0.0, 'dominant_lab': [0.0, 0.0, 0.0]}

    h, w = img.shape[:2]
    if not isinstance(bbox, dict) or 'w' not in bbox or 'h' not in bbox:
        return {'color_name': 'unknown', 'confidence': 0.0, 'delta_e': 0.0, 'dominant_lab': [0.0, 0.0, 0.0]}

    x1 = max(0, int(bbox.get('x', 0) * w))
    y1 = max(0, int(bbox.get('y', 0) * h))
    x2 = min(w, int((bbox.get('x', 0) + bbox.get('w', 1)) * w))
    y2 = min(h, int((bbox.get('y', 0) + bbox.get('h', 1)) * h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] <= 2 or crop.shape[1] <= 2:
        return {'color_name': 'unknown', 'confidence': 0.0, 'delta_e': 0.0, 'dominant_lab': [0.0, 0.0, 0.0]}

    best_match = 'unknown'
    best_delta = float('inf')
    best_lab = [0.0, 0.0, 0.0]

    try:
        import colorgram
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        extracted = colorgram.extract(pil_img, 4)
        for c in extracted:
            if c.proportion < 0.20 and c != extracted[0]:
                continue
            r, g, b = c.rgb.r / 255.0, c.rgb.g / 255.0, c.rgb.b / 255.0
            lab_arr = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.array([[[r, g, b]]], dtype=np.float32)))[0, 0]
            cand_lab = LabColor(float(lab_arr[0]), float(lab_arr[1]), float(lab_arr[2]))

            for cname, cvals in FASHION_COLORS.items():
                ref_lab = LabColor(float(cvals[0]), float(cvals[1]), float(cvals[2]))
                d = delta_e_cie2000(cand_lab, ref_lab)
                if d < best_delta:
                    best_delta = d
                    best_match = cname
                    best_lab = [round(float(v), 2) for v in lab_arr]
    except Exception:
        rgb_float = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb_float))
        pixels = lab.reshape(-1, 3).astype(np.float32)
        if len(pixels) > 5000:
            idx = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[idx]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, min(3, len(pixels)), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant_lab = centers[np.argmax(counts)]
        cand_lab = LabColor(float(dominant_lab[0]), float(dominant_lab[1]), float(dominant_lab[2]))
        for cname, cvals in FASHION_COLORS.items():
            ref_lab = LabColor(float(cvals[0]), float(cvals[1]), float(cvals[2]))
            d = delta_e_cie2000(cand_lab, ref_lab)
            if d < best_delta:
                best_delta = d
                best_match = cname
                best_lab = [round(float(v), 2) for v in dominant_lab]

    confidence = max(0.0, min(1.0, 1.0 - (best_delta / 30.0)))
    return {
        'color_name': best_match,
        'confidence': round(confidence, 2),
        'delta_e': round(float(best_delta), 2),
        'dominant_lab': best_lab
    }
def segment_garments(
    image_b64: str,
    detected_items: list
) -> list:
    # FastSAM / GrabCut removed for cloud deployment.
    # Colour-science color extraction still runs via
    # get_accurate_color() which is called in the scan loop.
    return detected_items


def detect_scene_context(image_b64: str, client=None) -> dict:
    if client is None:
        client = _get_client()
    
    prompt = """Analyze this image and answer ONLY in this exact JSON format:
{
  "has_person": true/false,
  "person_in_foreground": true/false,
  "rack_items_visible": true/false,
  "scene_type": "person_wearing" | "store_rack" | "mixed" | "single_item",
  "foreground_focus": "person" | "rack" | "item_closeup"
}

Rules:
- has_person: true only if a human body is clearly visible
- person_in_foreground: true only if the person is the main subject, not background
- scene_type person_wearing: person is wearing the clothes being analyzed
- scene_type store_rack: clothes hanging on racks, no person wearing them
- scene_type mixed: both a person and rack items visible
- scene_type single_item: one item held up or displayed close range
No other text. JSON only."""

    try:
        resp = client.chat.completions.create(
            model='meta/llama-3.2-11b-vision-instruct',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {
                        'url': f'data:image/jpeg;base64,{image_b64}'
                    }},
                    {'type': 'text', 'text': prompt}
                ]
            }],
            max_tokens=150,
            temperature=0.1
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        log.warning(f"[SCENE CONTEXT] Failed: {e}")
        return {
            'has_person': False,
            'person_in_foreground': False,
            'rack_items_visible': True,
            'scene_type': 'store_rack',
            'foreground_focus': 'rack'
        }


async def _detect_scene_context_async(image_b64: str, async_client) -> dict:
    prompt = """Analyze this image and answer ONLY in this exact JSON format:
{
  "has_person": true/false,
  "person_in_foreground": true/false,
  "rack_items_visible": true/false,
  "scene_type": "person_wearing" | "store_rack" | "mixed" | "single_item",
  "foreground_focus": "person" | "rack" | "item_closeup"
}

Rules:
- has_person: true only if a human body is clearly visible
- person_in_foreground: true only if the person is the main subject, not background
- scene_type person_wearing: person is wearing the clothes being analyzed
- scene_type store_rack: clothes hanging on racks, no person wearing them
- scene_type mixed: both a person and rack items visible
- scene_type single_item: one item held up or displayed close range
No other text. JSON only."""

    try:
        resp = await async_client.chat.completions.create(
            model='meta/llama-3.2-11b-vision-instruct',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {
                        'url': f'data:image/jpeg;base64,{image_b64}'
                    }},
                    {'type': 'text', 'text': prompt}
                ]
            }],
            max_tokens=150,
            temperature=0.1
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        log.warning(f"[SCENE CONTEXT ASYNC] Failed: {e}")
        return {
            'has_person': False,
            'person_in_foreground': False,
            'rack_items_visible': True,
            'scene_type': 'store_rack',
            'foreground_focus': 'rack'
        }


def enrich_items_with_scene_context(items: list, scene_context: dict) -> list:
    scene_type = scene_context.get("scene_type", "store_rack")
    person_fg = scene_context.get("person_in_foreground", False)
    
    enriched = []
    for item in items:
        bbox = item.get("bbox", {})
        y = bbox.get("y", 0.5)
        
        if scene_type == "person_wearing":
            item = {**item, "worn_by_person": True}
        elif scene_type == "store_rack":
            item = {**item, "worn_by_person": False}
        elif scene_type == "mixed":
            if person_fg and (y < 0.75):
                item = {**item, "worn_by_person": True}
            else:
                item = {**item, "worn_by_person": False}
        else:
            item = {**item, "worn_by_person": False}
            
        enriched.append(item)
    return enriched


async def call_with_timeout(coro, seconds=12):
    import asyncio
    try:
        return await asyncio.wait_for(coro, seconds)
    except asyncio.TimeoutError:
        return None


async def scan_frame_async(image_b64: str, user_id: str = "default", _apply_consensus: bool = True, run_combos: bool = False) -> dict:
    import os
    import asyncio
    from openai import AsyncOpenAI

    image_b64 = preprocess_frame(image_b64)

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    base_url = "https://integrate.api.nvidia.com/v1"
    # Client-level timeout must be HIGHER than the largest per-call timeout (35s)
    # otherwise the client-level cap silently kills calls before the per-call timeout fires.
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)

    nemotron_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
                },
                {"type": "text", "text": _SCAN_PROMPT},
            ],
        }
    ]
    llama_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SCAN_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        },
        {"role": "assistant", "content": "{"}
    ]

    import time

    # ── [TIMING] Preprocess already done in scan_frame — log it here only in async path
    async def run_dual_models():
        # Fast Path: Llama-3.2-11B-Vision with _FAST_SCAN_PROMPT (~6-10s)
        # Note: 90B and Nemotron-4-340B removed from synchronous live scan path.
        # 90B is now used for Asynchronous Fabric Enrichment on cropped item regions.
        fast_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _FAST_SCAN_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
            {"role": "assistant", "content": "{"}
        ]
        res_fast = await call_with_timeout(_call_model_for_scan(async_client, _MODEL_VISION_11B, fast_messages, True, timeout=12.0), 12.0)
        if isinstance(res_fast, dict):
            res_fast["_model_used"] = "Llama-3.2-11B-Fast"
            for item in res_fast.get("items", []):
                if isinstance(item, dict) and not item.get("fabric_type"):
                    item["fabric_type"] = "pending"
        else:
            res_fast = {"items": [], "frame_quality": "acceptable", "scene_lighting": "unknown", "_model_used": "11B-Fast-Failed"}
        return None, res_fast

    async def timed_dual():
        t_d = time.time()
        res = await run_dual_models()
        print(f"[TIMING] Dual vision: {round(time.time()-t_d,2)}s")
        return res

    async def timed_scene():
        t_s = time.time()
        # scene context runs IN PARALLEL with dual models — not before them
        res = await call_with_timeout(_detect_scene_context_async(image_b64, async_client), 15.0)
        print(f"[TIMING] Scene context: {round(time.time()-t_s,2)}s")
        return res

    # ── Run dual vision models + scene context in TRUE parallel ──
    (res1, res2), scene_context_raw = await asyncio.gather(
        timed_dual(),
        timed_scene()
    )

    # ── Scene context: never let a failure degrade quality ────────
    if not scene_context_raw or not isinstance(scene_context_raw, dict) or "scene_type" not in scene_context_raw:
        scene_context = {
            'has_person': False,
            'person_in_foreground': False,
            'rack_items_visible': True,
            'scene_type': 'store_rack',
            'foreground_focus': 'rack'
        }
        scene_context_failed = True
    else:
        scene_context = scene_context_raw
        scene_context_failed = False

    data = reconcile_scan_results(res1, res2)
    m1 = (res1 or {}).get("_model_used") if isinstance(res1, dict) else None
    m2 = (res2 or {}).get("_model_used") if isinstance(res2, dict) else None
    if m1 and m2 and m1 != m2 and "items" in (res1 or {}) and "items" in (res2 or {}):
        data["_model_used"] = f"Dual ({m1} + {m2})"
    else:
        data["_model_used"] = m1 or m2 or "Unknown"
    data["scene_context"] = scene_context

    # Normalize frame_quality — always set to a valid string.
    # Catches: None (model didn't return it), 'poor' (blocked scan), or missing key.
    # scene_context failure also forces 'acceptable' regardless of model output.
    current_fq = data.get("frame_quality")
    if scene_context_failed or not current_fq or current_fq == "poor":
        data["frame_quality"] = "acceptable"

    items = enrich_items_with_scene_context(data.get("items", []), scene_context)

    # ── Segmentation (includes color extraction per item) ─────────
    t_segment = time.time()
    items = segment_garments(image_b64, items)
    print(f"[TIMING] Segmentation+color: {round(time.time()-t_segment,2)}s")
    # NOTE: segment_garments already calls get_accurate_color internally.
    # DO NOT call get_accurate_color again here — that was the double-extraction bug.

    data["items"] = items
    if _apply_consensus:
        tracker = get_consensus_tracker(user_id or "default")
        data["items"] = tracker.update(items)
    try:
        await async_client.close()
    except Exception:
        pass
    return data


def scan_frame(image_b64: str, user_id: str = "default", run_combos: bool = False) -> dict:
    """
    Detect all visible garments/accessories in a single image frame.
    Calls two vision models concurrently and reconciles outputs.
    """
    import time
    t_preprocess = time.time()
    image_b64 = preprocess_frame(image_b64)
    print(f"[TIMING] Preprocess: {round(time.time()-t_preprocess,2)}s")
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            data = pool.submit(lambda: asyncio.run(scan_frame_async(image_b64, user_id, _apply_consensus=False, run_combos=run_combos))).result()
    else:
        data = asyncio.run(scan_frame_async(image_b64, user_id, _apply_consensus=False, run_combos=run_combos))

    if not data:
        return {
            "items": [],
            "scene_lighting": "unknown",
            "frame_quality": "acceptable",
            "guidance": "Vision model unavailable — try again",
        }

    # ── Normalise and validate ──────────────────────────────────
    items = data.get("items", [])
    frame_quality = data.get("frame_quality", "good")

    # ── Batch grid-based bbox localization first ─────────────────
    # Run before dedup so distinct items sharing a label (e.g. two black t-shirts)
    # can be distinguished by their spatial grid locations / bounding boxes.
    t_bbox = time.time()
    if items and any(not i.get("bbox") for i in items):
        log.info(f"[CV] Running batch grid localization for items missing bboxes")
        client = _get_client()
        items = _batch_localize_bboxes(client, items, image_b64)
        items = segment_garments(image_b64, items)
        data["items"] = items
        print(f"[TIMING] BBox: {round(time.time()-t_bbox,2)}s")
    else:
        print(f"[TIMING] BBox: 0.00s (all bboxes present)")

    # ── Dedup first, then guard ──────────────────────────────────
    items = _dedup_items(items)
    data["items"] = items

    label_counts: dict[str, int] = {}
    for item in items:
        k = f"{item.get('category','')}::{item.get('label','').lower()}"
        label_counts[k] = label_counts.get(k, 0) + 1
    max_dupes = max(label_counts.values(), default=0)
    if max_dupes >= 3:
        log.warning(
            f"[CV] Structural hallucination post-dedup: one label appears "
            f"{max_dupes}x across {len(items)} items. Continuing without wiping."
        )
        if "guidance" not in data:
            data["guidance"] = "Move closer to the items if details are unclear."

    # Never block a scan because of a quality flag — always default to 'acceptable'.
    # Also catches None (model omitted the key) via falsy check.
    if not frame_quality or frame_quality == "poor" or not data.get("frame_quality") or data.get("frame_quality") == "poor":
        log.info("[CV] Frame quality missing/poor, defaulting to acceptable without blocking scan")
        data["frame_quality"] = "acceptable"
        frame_quality = "acceptable"
        if "guidance" not in data:
            data["guidance"] = "move closer to the rack"

    # ── Per-item guidance only — NO color re-extraction here ────
    # segment_garments() already called get_accurate_color for each item.
    # Calling it again here was causing ~5-8s of duplicate work.
    for item in items:
        conf = item.get("confidence", 1.0)
        if conf < 0.6 and "guidance" not in item:
            item["guidance"] = "move_closer"

    # ── Proactive outfit combinations for live mall scenario ─────
    wearable_cats = ("top", "bottom", "outerwear", "footwear", "dress", "set")
    wearable_items = [i for i in items if i.get("category", "").lower() in wearable_cats]
    import concurrent.futures
    combos_future = None
    pool = None
    if run_combos and len(wearable_items) >= 2:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        log.info(f"[CV] Proactively generating outfit combinations for {len(wearable_items)} wearable items")
        combos_future = pool.submit(generate_outfit_combinations, items)

    # ── Supervision Annotation ──────────────────────────────────
    try:
        import supervision as sv
        import numpy as np
        from PIL import Image
        import io
        import base64

        # Decode image to get dimensions and array
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        img_w, img_h = pil_img.size

        # Normalize pixel-coordinate bboxes to 0-1 fractions
        # Nemotron sometimes returns pixel coords despite prompt instructions
        for item in items:
            b = item.get("bbox", {})
            if any(v > 1.5 for v in [
                b.get("x", 0), b.get("y", 0),
                b.get("w", 0), b.get("h", 0)
            ]):
                item["bbox"] = {
                    "x": round(min(b.get("x", 0) / img_w, 1.0), 3),
                    "y": round(min(b.get("y", 0) / img_h, 1.0), 3),
                    "w": round(min(b.get("w", 0) / img_w, 1.0), 3),
                    "h": round(min(b.get("h", 0) / img_h, 1.0), 3),
                }
                log.info(f"[CV] Normalized pixel bbox for '{item.get('label')}'")
        scene = np.array(pil_img)

        if items:
            annotated = scene.copy()
            import cv2
            all_xyxy = []
            for item in items:
                b = item.get("bbox", {})
                x, y = b.get("x", 0.0), b.get("y", 0.0)
                w, h = b.get("w", 0.0), b.get("h", 0.0)
                x1, y1 = int(x * img_w), int(y * img_h)
                x2, y2 = int((x + w) * img_w), int((y + h) * img_h)
                all_xyxy.append([x1, y1, x2, y2])

                # Thin corner-bracket L-marks in #39FF14 green (RGB: 57, 255, 20)
                green = (57, 255, 20)
                c_len = max(int(min(w * img_w, h * img_h) * 0.18), 12)
                # Top-Left
                cv2.line(annotated, (x1, y1), (x1 + c_len, y1), green, 2)
                cv2.line(annotated, (x1, y1), (x1, y1 + c_len), green, 2)
                # Top-Right
                cv2.line(annotated, (x2, y1), (x2 - c_len, y1), green, 2)
                cv2.line(annotated, (x2, y1), (x2, y1 + c_len), green, 2)
                # Bottom-Left
                cv2.line(annotated, (x1, y2), (x1 + c_len, y2), green, 2)
                cv2.line(annotated, (x1, y2), (x1, y2 - c_len), green, 2)
                # Bottom-Right
                cv2.line(annotated, (x2, y2), (x2 - c_len, y2), green, 2)
                cv2.line(annotated, (x2, y2), (x2, y2 - c_len), green, 2)

            pil_annotated = Image.fromarray(annotated)
            buffer = io.BytesIO()
            pil_annotated.save(buffer, format="PNG")
            data["annotated_frame_b64"] = base64.b64encode(buffer.getvalue()).decode()
            data["pixel_boxes"] = [
                {"id": item.get("id"), "xyxy": [int(v) for v in box]}
                for item, box in zip(items, all_xyxy)
            ]
        else:
            data["annotated_frame_b64"] = ""
            data["pixel_boxes"] = []
            
    except Exception as e:
        log.error(f"[CV] Supervision annotation failed: {e}")
        data["annotated_frame_b64"] = ""
        data["pixel_boxes"] = []

    if combos_future and pool:
        try:
            data["combos"] = combos_future.result(timeout=10.0)
        except Exception as e:
            log.warning(f"[CV] Proactive combo generation failed/timed out: {e}")
            data["combos"] = []
        finally:
            pool.shutdown(wait=False)
    else:
        data["combos"] = []
        if pool:
            pool.shutdown(wait=False)

    tracker = get_consensus_tracker(user_id or 'default')
    items = tracker.update(items)
    data["items"] = items
    return data


# ─────────────────────────────────────────────────────────────────
#  Compatibility helper
# ─────────────────────────────────────────────────────────────────

_COMPAT_PROMPT_TMPL = """You are a fashion stylist AI.
Given these facts about a garment and a user's style profile, write ONE sentence 
(plain English, no markdown) explaining whether the garment is compatible with 
the user and why.

Garment type: {garment_type}
Garment aesthetic: {aesthetic}
Primary color: {color}

User profile:
  Monk skin tone scale: {monk_scale}
  Body type: {body_type}
  Primary aesthetic: {primary_aesthetic}

Is this garment compatible with this user? Answer with ONLY a JSON object:
{{ "compatible": true_or_false, "reason": "one sentence" }}
No markdown. No extra text."""


def _assess_compatibility(garment_analysis: dict, user_profile: dict) -> dict:
    """
    Use the text LLM to assess profile compatibility in one sentence.
    Returns {"compatible": bool, "reason": str}.
    """
    from shaaru_brain import _get_client
    from shaaru_retry import nvidia_call

    garment_type = garment_analysis.get("garment_type", "garment")
    # Pull primary color safely from nested structure
    fabric_info = garment_analysis.get("fabric", {})
    if isinstance(fabric_info, dict):
        primary_color_obj = fabric_info.get("primary_color", {})
        if isinstance(primary_color_obj, dict):
            color = primary_color_obj.get("value", "unknown")
        else:
            color = str(primary_color_obj) if primary_color_obj else "unknown"
    else:
        color = "unknown"

    # Aesthetic from silhouette block or top-level
    aesthetic = garment_analysis.get("occasion", "")
    if not aesthetic:
        sil = garment_analysis.get("silhouette", {})
        aesthetic = sil.get("overall_shape", {}).get("value", "unknown") if isinstance(sil, dict) else "unknown"

    monk_scale = user_profile.get("monk_scale", "unknown")
    body_type = user_profile.get("body_type", "unknown")
    primary_aesthetic = user_profile.get("primary_aesthetic", "unknown")

    prompt = _COMPAT_PROMPT_TMPL.format(
        garment_type=garment_type,
        aesthetic=aesthetic,
        color=color,
        monk_scale=monk_scale,
        body_type=body_type,
        primary_aesthetic=primary_aesthetic,
    )

    try:
        client = _get_client()
        response = nvidia_call(
            client=client,
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3,
        )
        parsed = _parse_scan_json(response)
        if parsed and "compatible" in parsed and "reason" in parsed:
            return {
                "compatible": bool(parsed["compatible"]),
                "reason": str(parsed["reason"]),
            }
    except Exception as e:
        log.warning(f"[CV] Compatibility assessment failed: {e}")

    # Safe fallback
    return {
        "compatible": True,
        "reason": "Could not assess compatibility automatically — review manually.",
    }


# ─────────────────────────────────────────────────────────────────
#  analyze_item
# ─────────────────────────────────────────────────────────────────

def analyze_item(image_b64: str, item_label: str, user_profile: dict) -> dict:
    """
    Deep-dive analysis of one detected garment item.

    Steps:
      1. analyze_garment_deep() — full construction/fabric/silhouette breakdown
      2. query_fashion_intelligence() — fabric sourcing, measurements, construction DB
      3. Cross-reference result against user_profile
      4. Return combined dict

    Args:
        image_b64:    Base64-encoded image of the garment.
        item_label:   Human-readable label returned by scan_frame (e.g. "oversized white shirt").
        user_profile: Dict with keys monk_scale, body_type, primary_aesthetic.

    Returns:
        dict with keys: item_label, garment_analysis, fabric_intelligence,
        profile_compatibility, tailor_available.
    """
    from tailor_engine import analyze_garment_deep, query_fashion_intelligence

    # ── Step 1: Deep garment analysis ──────────────────────────
    log.info(f"[CV] analyze_item: running analyze_garment_deep for '{item_label}'")
    try:
        garment_analysis = analyze_garment_deep(image_b64)
    except Exception as e:
        log.error(f"[CV] analyze_garment_deep failed: {e}")
        garment_analysis = {}

    # ── Step 2: Fashion intelligence DB lookup ─────────────────
    garment_type = garment_analysis.get("garment_type", item_label)
    fabric_block = garment_analysis.get("fabric", {})
    fabric_need = ""
    if isinstance(fabric_block, dict):
        fiber_obj = fabric_block.get("fiber_type", {})
        if isinstance(fiber_obj, dict):
            fabric_need = fiber_obj.get("value", "")
        else:
            fabric_need = str(fiber_obj) if fiber_obj else ""

    # Derive fit from silhouette block
    sil_block = garment_analysis.get("silhouette", {})
    fit = ""
    if isinstance(sil_block, dict):
        fit_obj = sil_block.get("fit", {})
        fit = fit_obj.get("value", "") if isinstance(fit_obj, dict) else str(fit_obj)

    # Use sensible defaults for city / height so the call always works
    city = user_profile.get("city", "bengaluru")
    height_ft = user_profile.get("height_ft", 5.4)

    log.info(f"[CV] analyze_item: querying fashion intelligence for garment_type='{garment_type}'")
    try:
        fabric_intelligence = query_fashion_intelligence(
            garment_type=garment_type,
            fabric_need=fabric_need,
            city=city,
            height_ft=height_ft,
            fit=fit,
        )
    except Exception as e:
        log.error(f"[CV] query_fashion_intelligence failed: {e}")
        fabric_intelligence = {}

    # ── Step 3: Profile compatibility ─────────────────────────
    profile_compatibility = _assess_compatibility(garment_analysis, user_profile)

    log.info(
        f"[CV] analyze_item complete: garment='{garment_type}' "
        f"compatible={profile_compatibility.get('compatible')}"
    )

    return {
        "item_label": item_label,
        "garment_analysis": garment_analysis,
        "fabric_intelligence": fabric_intelligence,
        "profile_compatibility": profile_compatibility,
        "tailor_available": True,
    }


async def enrich_item_fabric_async(
    image_b64: str,
    bbox: dict,
    label: str,
    category: str,
    track_id: str,
    user_id: str = "default"
) -> dict:
    """
    Asynchronously enrich the fabric_type of a detected item from its cropped bounding box.
    Runs hierarchical _SCAN_PROMPT on 90B (35s timeout) -> 11B fallback (15s timeout).
    Updates ConsensusTracker so subsequent synchronous scans preserve the enriched fabric.
    """
    import base64, io
    from PIL import Image
    try:
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        w_img, h_img = pil_img.size
        
        bx = float(bbox.get("x", 0.0)) * w_img
        by = float(bbox.get("y", 0.0)) * h_img
        bw = float(bbox.get("w", 1.0)) * w_img
        bh = float(bbox.get("h", 1.0)) * h_img
        
        pad_w = int(bw * 0.08)
        pad_h = int(bh * 0.08)
        crop_box = (
            max(0, int(bx - pad_w)),
            max(0, int(by - pad_h)),
            min(w_img, int(bx + bw + pad_w)),
            min(h_img, int(by + bh + pad_h))
        )
        cropped = pil_img.crop(crop_box)
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=85)
        crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        log.warning(f"[CV ENRICH FABRIC] Crop failed for {track_id}, using full image: {e}")
        crop_b64 = image_b64

    enrich_prompt = f"""Return only the JSON object. No markdown, no explanation.
CONTEXT: High-Precision Textile & Fabric Classification of a Cropped Garment.
Item label: {label} ({category})

Analyze this cropped closeup image of the garment and determine its exact fabric construction using the HIERARCHICAL TAXONOMY rules below.
You MUST reason in two steps:
Step 1: In "fabric_reason", provide a TERSE STRUCTURED PHRASE (5-10 words max) noting surface texture, weave, drape, and sheen (e.g. "diagonal twill, matte, stiff drape" or "fine smooth knit, soft drape, matte"). Do NOT write full sentences or prose.
Step 2: In "fabric_type", select ONE specific fabric term from the grounded Tier 2 vocabulary below that matches your observed visual cues (or return "uncertain"). Do NOT use freeform text outside this list:

1. Woven Plain / Crisp: poplin, linen, chambray, khadi cotton, handloom cotton
2. Woven Twill / Denim: denim, twill, canvas, corduroy, gabardine
3. Knit Ribbed / Cable: ribbed knit, cable knit, waffle knit
4. Knit Jersey / Fine: jersey knit, interlock, athletic mesh
5. Sheen / Fluid Drape: silk satin, crepe de chine, charmeuse, georgette
6. Sheer / Crisp / Open Weave: organza, chanderi silk, chiffon, net, mulmul
7. Structured / Jacquard / Brocade: raw silk dupion, banarasi brocade, jacquard, ikat
8. Leather / Coated / Suede: genuine leather, faux/PU leather, suede, patent leather
9. Heavy / Wool / Felted: fine wool, wool-blend, tweed, cashmere, fleece
10. Embroidered / Craft Work: chikankari, zardozi, kantha, phulkari, chantilly lace

Return ONLY a JSON object exactly matching this structure:
{{
  "fabric_reason": "<terse structured phrase, e.g. 'diagonal twill, matte, stiff drape'>",
  "fabric_type": "<exact Tier 2 fabric term from the list above, or 'uncertain'>",
  "confidence": 0.88
}}
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": enrich_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
            ],
        },
        {"role": "assistant", "content": "{"}
    ]
    
    async_client = _get_async_client()
    res = await call_with_timeout(_call_model_for_scan(async_client, _MODEL_VISION_90B, messages, True, timeout=35.0), 35.0)
    if not res or not res.get("fabric_type"):
        log.info(f"[CV ENRICH FABRIC] 90B timed out/failed on {track_id}, falling back to 11B")
        res = await call_with_timeout(_call_model_for_scan(async_client, _MODEL_VISION_11B, messages, True, timeout=15.0), 15.0)
        
    fabric_type = res.get("fabric_type", "uncertain") if isinstance(res, dict) else "uncertain"
    fabric_reason = res.get("fabric_reason", "") if isinstance(res, dict) else ""
    confidence = res.get("confidence", 0.8) if isinstance(res, dict) else 0.8
    
    tracker = get_consensus_tracker(user_id or "default")
    if track_id in tracker._tracks:
        track = tracker._tracks[track_id]
        track["fabric_type"] = fabric_type
        if "last_item" in track and isinstance(track["last_item"], dict):
            track["last_item"]["fabric_type"] = fabric_type
            track["last_item"]["fabric_reason"] = fabric_reason
            
    log.info(f"[CV ENRICH FABRIC] track_id={track_id} -> fabric_type='{fabric_type}' ({fabric_reason})")
    return {
        "track_id": track_id,
        "fabric_type": fabric_type,
        "fabric_reason": fabric_reason,
        "confidence": confidence,
    }


# ------------------------------------------------------------------
#  targeted_scan_frame -- voice-triggered occluded item detection
# ------------------------------------------------------------------

_TARGETED_SCAN_PROMPT = """You are a precise garment detection assistant. The user is pointing out a SPECIFIC item they can see but may not have been detected yet.

Target description: {target_description}

Currently already detected items (ignore these, DO NOT return them):
{current_labels_block}

Inspect this camera frame carefully for an item matching the target description.
The item may be partially visible behind or beneath another garment, in the background, or folded over a rack.

Return ONLY a JSON object in this exact format (no markdown, no explanation):
{{
  "found": true,
  "item": {{
    "label": "<short descriptive name>",
    "category": "<top|bottom|outerwear|footwear|accessory|dress|set>",
    "color": "<primary color>",
    "confidence": 0.72,
    "bbox": {{"x": 0.12, "y": 0.35, "w": 0.30, "h": 0.45}},
    "fabric_type": "pending",
    "description": "<one sentence describing what you see>",
    "aesthetic": "<casual|formal|streetwear|etc>"
  }}
}}

If the described item is genuinely NOT visible anywhere in the frame:
{{ "found": false, "item": null }}

Rules:
- bbox must be normalized 0.0-1.0 fractions of the image dimensions
- DO NOT return an item already in the current detected list
- DO NOT hallucinate: only return found=true if you can actually locate something resembling the target
- confidence must reflect genuine visual certainty, not optimism
- Only return ONE item (the best match to the description)"""


async def _targeted_scan_async(image_b64: str, target_description: str, current_labels: list) -> dict:
    """Run a targeted single-item search via 11B vision model."""
    import os
    from openai import AsyncOpenAI
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    async_client = AsyncOpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1", timeout=18.0, max_retries=0)
    labels_block = "\n".join(f"  - {lbl}" for lbl in current_labels) if current_labels else "  (none yet)"
    prompt = _TARGETED_SCAN_PROMPT.format(target_description=target_description, current_labels_block=labels_block)
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]},
        {"role": "assistant", "content": "{"},
    ]
    try:
        res = await call_with_timeout(
            _call_model_for_scan(async_client, _MODEL_VISION_11B, messages, True, timeout=15.0),
            15.0,
        )
        return res or {}
    except Exception as e:
        log.warning(f"[CV TARGETED] Async call failed: {e}")
        return {}
    finally:
        try:
            await async_client.close()
        except Exception:
            pass


def targeted_scan_frame(image_b64: str, target_description: str, current_labels: list, user_id: str = "default") -> dict:
    """
    Search a camera frame for a specific garment described by the user.

    Called when voice input signals the user is pointing out an item the
    regular scan has not detected (occluded, partially visible, etc.).

    Returns:
        {"found": bool, "item": dict|None, "source": "voice_targeted_scan"}
    """
    import asyncio
    image_b64 = preprocess_frame(image_b64)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            raw = pool.submit(lambda: asyncio.run(_targeted_scan_async(image_b64, target_description, current_labels))).result()
    else:
        raw = asyncio.run(_targeted_scan_async(image_b64, target_description, current_labels))

    found = bool(raw.get("found"))
    item = raw.get("item")
    if found and isinstance(item, dict):
        bbox = item.get("bbox", {})
        if not bbox or bbox.get("w", 0) < 0.03 or bbox.get("h", 0) < 0.03:
            log.warning("[CV TARGETED] Item found but bbox too small/missing -- discarding")
            return {"found": False, "item": None, "source": "voice_targeted_scan"}
        item.setdefault("fabric_type", "pending")
        item.setdefault("aesthetic", "casual")
        item.setdefault("description", target_description)
        item.setdefault("state", "voice_triggered")
        import uuid
        item["track_id"] = f"voice_{uuid.uuid4().hex[:6]}"
        item["id"] = item["track_id"]
        log.info(f"[CV TARGETED] Found '{item.get('label')}' for description='{target_description}' conf={item.get('confidence', '?')} bbox={bbox}")
        return {"found": True, "item": item, "source": "voice_targeted_scan"}
    log.info(f"[CV TARGETED] Not found for description='{target_description}'")
    return {"found": False, "item": None, "source": "voice_targeted_scan"}

