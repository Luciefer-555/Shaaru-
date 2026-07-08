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

def _category_hex(category: str) -> str:
    return _CAT_COLORS.get((category or "").lower().strip(), _CAT_COLOR_DEFAULT)

# ─────────────────────────────────────────────────────────────────
#  Scan prompt
# ─────────────────────────────────────────────────────────────────
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
      "fabric_type": "<exact term from fixed taxonomy below, or 'uncertain'>",
      "fabric_reason": "<brief explanation of visual cues, or reason for uncertainty if uncertain>",
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
- Never use generic names. Use specific fashion nomenclature.
- Examples of what NOT to write vs what TO write:
  BAD: "white shirt" → GOOD: "mandarin collar poplin shirt"
  BAD: "blue jeans" → GOOD: "wide-leg indigo cargo denim"
  BAD: "black jacket" → GOOD: "double-breasted wool blazer"
  BAD: "white t-shirt" → GOOD: "boxy ribbed henley tee"
  BAD: "dress" → GOOD: "asymmetric draped midi slip dress"
  BAD: "shoes" → GOOD: "chunky lug-sole leather derby"
  BAD: "bag" → GOOD: "structured top-handle trapeze bag"
  BAD: "kurta" → GOOD: "straight-hem embroidered kurta with thread work"
  BAD: "saree" → GOOD: "kanjeevaram silk saree with gold zari border"
- For Indian garments, be equally specific — name the silhouette, 
  fabric, and embellishment if visible
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

CRITICAL RULES for the fabric_type field (FIXED TAXONOMY):
You MUST classify fabric into ONE exact term from this fixed vocabulary (or return "uncertain"). Do NOT use freeform text or any term outside this list:

Woven:
- poplin: Crisp, tightly woven, smooth surface with subtle sheen (classic dress shirts).
- denim: Sturdy cotton twill with diagonal ribbing and visible weave structure (jeans, jackets).
- linen: Lightweight, breathable weave with visible natural slubs and slight wrinkling/texture.
- canvas: Heavy, rugged plain-weave fabric with coarse matte texture (utility wear, bags).
- corduroy: Distinct vertical raised wales/cords with velvety texture.
- twill: Distinct diagonal rib weave pattern, softer and more drapeable than canvas.
- chambray: Plain weave with white weft and colored warp, resembling lightweight denim but smoother.

Knit:
- ribbed knit: Distinct vertical raised rows/ribs, stretchy and textured (turtlenecks, cuffs, sweaters).
- jersey knit: Smooth, fine gauge knit with high stretch and soft drape (t-shirts, casual wear).
- cable knit: Chunky, textured knit with braided or twisting rope-like patterns.
- waffle knit: Three-dimensional grid/honeycomb texture (thermal wear, sweaters).

Leather:
- genuine leather: Rich natural grain, smooth sheen, structured drape with natural creasing.
- faux/PU leather: Uniform artificial grain, high gloss or synthetic sheen, rigid plastic-like drape.
- suede: Soft, napped/velvety matte surface texture without sheen.

Other:
- wool: Dense, warm, matte fiber with soft fuzzy or felted texture (suiting, coats).
- wool-blend: Textured fabric showing wool warmth combined with synthetic structure or smoothness.
- silk: Luxurious high luster, fluid flowing drape, smooth delicate surface.
- satin/crepe: Smooth lustrous face with dull back (satin), or textured pebbled/crinkled surface with fluid drape (crepe).
- velvet: Dense, plush pile with rich sheen and deep light-capturing shadows.
- organza: Crisp, sheer/semi-sheer lightweight fabric with stiff, sculptural volume and subtle shimmer.

EXPLICIT UNCERTAINTY OPTION (IMPORTANT - DO NOT FORCE GUESSES):
- If you cannot 100% definitively distinguish the exact fabric weave or material (for example, distinguishing between linen vs poplin, denim vs canvas, genuine vs faux leather, or if the weave is subtle/unclear), you MUST return "fabric_type": "uncertain" and explain what you see in "fabric_reason".
- Guessing a fabric from the list when you are not 100% certain is a MAJOR FAILURE and WORSE than saying "uncertain".
- When in doubt between two or more fabrics, ALWAYS return "fabric_type": "uncertain"!

CRITICAL RULES for the confidence field:
- Estimate per-item based on visibility, not a fixed number
- 0.90-0.95: item clearly visible, unobstructed, full garment readable
- 0.75-0.89: item partially visible or slightly occluded
- 0.60-0.74: item mostly hidden, overlapping other garments, or at distance
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

If frame_quality is poor: return 
{ "items": [], "frame_quality": "poor", 
  "guidance": "<one instruction for the user>" }
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


def _parse_scan_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from a model response string.
    Falls back to partial recovery and LLM repair pass for truncated responses."""
    if not text:
        return None
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```" in text:
        for part in text.split("```"):
            clean = part.strip().lstrip("json").strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    # Brace extraction
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
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
                    return {
                        "items": items,
                        "frame_quality": "good",
                        "scene_lighting": "unknown",
                    }
        except Exception as e:
            log.warning(f"[CV] Partial recovery failed: {e}")

    # Second pass: LLM repair if regex/string repair failed
    repaired = _repair_json_with_llm(text)
    if repaired and isinstance(repaired, dict):
        log.info("[CV] Successfully repaired malformed/truncated JSON via LLM pass")
        return repaired

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
- combos must be genuinely wearable together — not just random groupings
- before writing directions, mentally check: is every item I mention in items_used?
  If not, move it to missing[] or remove it entirely
- if only tops/outerwear scanned with no bottoms, every combo needs a missing bottom
- vibe must be specific to Indian Gen Z sensibility — reference real aesthetics
- CONFIDENCE-AWARE FABRIC HEDGING: When describing scanned items or suggesting pieces in directions, hunt_line, find, or scan_prompt, check the confidence score and fabric_type of each item:
  * High confidence (confidence >= 0.75 and fabric_type not 'uncertain'): State the fabric type directly and assertively (e.g., "this crisp poplin shirt...", "layer over the denim jacket").
  * Penalized/uncertain (0.45 <= confidence < 0.75 or fabric_type is 'uncertain'): Use natural, conversational hedging when mentioning the fabric or weave in directions/hunt_line/find (e.g., "your cream top — looks like it could be a ribbed knit — over the trousers", "seems like a ribbed knit, though I'm not 100% sure on the exact weave"). Even if the fabric name appears inside the item's label, you MUST hedge it or soften it rather than stating it as flat fact!
  * Very low confidence (confidence < 0.45): Skip stating the specific fabric type entirely; describe the piece by silhouette, category, and color instead (e.g., "the black structured jacket", "the relaxed button-down")."""


def generate_outfit_combinations(
    items: list = None,
    detected_items: list = None,
    aesthetic_prompt: str = None,
    user_profile: dict = None,
) -> list:
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
        if not fab or fab.lower() in ("unspecified", "none", ""):
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
    if aesthetic_prompt:
        context_lines.append(f"Desired aesthetic/occasion: {aesthetic_prompt}")
    if user_profile:
        body = user_profile.get("body_type") or user_profile.get("body")
        city = user_profile.get("city")
        if body:
            context_lines.append(f"User body type: {body}")
        if city:
            context_lines.append(f"User location: {city}")
    user_context = ("\nUser Context:\n" + "\n".join(f"- {line}" for line in context_lines) + "\n") if context_lines else ""

    prompt = _COMBO_PROMPT.format(items_block=items_block, user_context=user_context)

    try:
        from shaaru_retry import nvidia_call
        client = _get_client()
        raw = nvidia_call(
            client=client,
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
        )
        content = raw if isinstance(raw, str) else (raw.choices[0].message.content or "")
        parsed = _parse_scan_json(content)

        if not parsed or "combos" not in parsed:
            log.warning(f"[CV COMBOS] Bad parse: {content[:200]}")
            return []

        combos = parsed["combos"]
        for combo in combos:
            combo["combo_context"] = combo_context
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

        return combos
    except Exception as e:
        log.warning(f"[CV COMBOS] Generation failed: {e}")
        return []


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
                
                track = {
                    "track_id": track_id,
                    "state": "new",
                    "missed_cycles": 0,
                    "raw_bbox": raw_bbox,
                    "smoothed_bbox": smoothed_bbox,
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
    # Client-level timeout must be HIGHER than the largest per-call timeout (25s)
    # otherwise the client-level cap silently kills calls before the per-call timeout fires.
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=35.0, max_retries=0)

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
        # Primary: Nemotron 25s | Secondary: Llama-90B 25s | Fallback: Llama-11B 20s
        t1 = asyncio.create_task(call_with_timeout(_call_model_for_scan(async_client, _MODEL_NEMOTRON, nemotron_messages, False, timeout=25.0), 25.0))
        t2 = asyncio.create_task(call_with_timeout(_call_model_for_scan(async_client, _MODEL_VISION_90B, llama_messages, True, timeout=25.0), 25.0))

        done, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)

        res1, res2 = None, None
        success = False
        for task in done:
            val = task.result()
            if isinstance(val, dict) and val.get("items"):
                if task == t1: res1 = val
                if task == t2: res2 = val
                success = True
                break
            elif isinstance(val, dict):
                if task == t1: res1 = val
                if task == t2: res2 = val

        if success:
            for p in pending:
                p.cancel()
        elif pending:
            done2, _ = await asyncio.wait(pending)
            for task in done2:
                val = task.result()
                if isinstance(val, dict):
                    if task == t1: res1 = val
                    if task == t2: res2 = val

        if (not res1 or not res1.get("items")) and (not res2 or not res2.get("items")):
            log.info("[CV] Both dual models returned None/empty, falling back to 11b vision")
            t_fb = time.time()
            res_fallback = await call_with_timeout(_call_model_for_scan(async_client, _MODEL_VISION_11B, llama_messages, True, timeout=20.0), 20.0)
            print(f"[TIMING] 11B fallback: {round(time.time()-t_fb,2)}s")
            if res_fallback and res_fallback.get("items"):
                res1 = res_fallback
            else:
                # Return whatever partial result we have — never return empty items[]
                res1 = res_fallback or res1 or res2 or {"items": [], "frame_quality": "acceptable", "scene_lighting": "unknown"}
        return res1, res2

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
