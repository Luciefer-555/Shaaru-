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


def _parse_scan_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from a model response string.
    Falls back to partial recovery for truncated responses."""
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
    if items_start == -1:
        return None
    try:
        # Find all complete item objects using regex
        item_pattern = re.compile(
            r'\{[^{}]*"id"\s*:\s*"[^"]*"[^{}]*\}',
            re.DOTALL
        )
        found = item_pattern.findall(text)
        if not found:
            return None
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

Divide this image into a 4-column x 3-row grid:
  Columns left to right : 1  2  3  4   (each = 25% of image width)
  Rows top to bottom    : A  B  C       (each = 33% of image height)

Locate each clothing item listed below and report which grid cells it occupies.

Items to locate:
{item_list}

Return exactly:
{{
  "<item_id>": {{"cols": [<numbers>], "rows": ["<letters>"]}}
}}

Rules:
- cols: 1 to 3 adjacent integers from 1-4
- rows: 1 to 3 adjacent letters from A, B, C
- No two items should have identical positions
- Items on a hanging rack: torso/upper body usually in rows A-B, trousers/shoes in rows B-C
- Estimate confidently — never return empty cols or rows

Example for 3 items on a rack:
{{
  "item_1": {{"cols": [1, 2], "rows": ["A", "B"]}},
  "item_2": {{"cols": [3], "rows": ["A", "B"]}},
  "item_3": {{"cols": [2], "rows": ["B", "C"]}}
}}"""


def _grid_to_bbox(pos: dict) -> dict:
    """Convert 4-col x 3-row grid classification to normalized 0-1 bbox."""
    COL = {1: 0.00, 2: 0.25, 3: 0.50, 4: 0.75}
    ROW = {"A": 0.00, "B": 0.33, "C": 0.67}
    try:
        cols = sorted(int(c) for c in pos.get("cols", []) if 1 <= int(c) <= 4)
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
    w = min(len(cols) * 0.25 + 0.02, 1.0 - x)
    h = min(len(rows) * 0.33 + 0.02, 1.0 - y)
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "w": round(w, 2),
        "h": round(h, 2),
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
                timeout=30.0,
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


def _dedup_items(items: list) -> list:
    """
    Remove duplicate detections — same label = same item detected twice.
    Renumbers item IDs after dedup to keep them sequential.
    Logs every removal.
    """
    seen: dict[str, int] = {}   # label_key -> first index kept
    deduped: list = []
    for item in items:
        key = item.get("label", "").lower().strip()
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(item)
        else:
            log.info(f"[CV] Dedup: dropped duplicate '{item.get('label')}'")
    # Renumber IDs
    for i, item in enumerate(deduped):
        item["id"] = f"item_{i + 1}"
    return deduped


def _resolve_bbox_collisions(items: list, grid_map: dict) -> dict:
    """
    Shift duplicate grid assignments to nearest unused cell.
    Iterates through ALL_COLS x ALL_ROWS in ascending size order
    so smaller (more precise) cells are preferred over large ones.
    """
    ALL_COLS = [
        [1],[2],[3],[4],
        [1,2],[2,3],[3,4],
        [1,2,3],[2,3,4],
        [1,2,3,4],
    ]
    ALL_ROWS = [
        ["A"],["B"],["C"],
        ["A","B"],["B","C"],
        ["A","B","C"],
    ]

    used: set = set()
    resolved: dict = {}

    for item in items:
        iid = item.get("id", "")
        if iid not in grid_map:
            continue

        pos = grid_map[iid]
        cols = sorted(int(c) for c in pos.get("cols", [1]) if 1 <= int(c) <= 4)
        rows = sorted(
            str(r).upper() for r in pos.get("rows", ["A"])
            if str(r).upper() in ("A", "B", "C")
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
    Uses 4x3 grid classification instead of float coordinates —
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
            timeout=45.0,
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
        log.warning(f"[CV] Batch bbox localization failed: {e}")
        return items


async def _call_model_for_scan(client, model_name: str, messages: list, is_json_object: bool = False) -> Optional[dict]:
    import asyncio
    try:
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
            "timeout": 45.0,
        }
        if is_json_object and "nemotron" not in model_name.lower():
            kwargs["response_format"] = {"type": "json_object"}
        if "nemotron" in model_name.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            
        coro = client.chat.completions.create(**kwargs)
        raw = await asyncio.wait_for(coro, timeout=50.0)
        content = raw.choices[0].message.content or ""
        if not content.lstrip().startswith("{") and "{" in content:
            content = "{" + content.split("{", 1)[1]
        return _parse_scan_json(content)
    except Exception as e:
        log.warning(f"[CV] Async call to {model_name} failed: {e}")
        return None

def reconcile_scan_results(data1: Optional[dict], data2: Optional[dict]) -> dict:
    if not data1 and not data2:
        return {
            "items": [],
            "scene_lighting": "unknown",
            "frame_quality": "poor",
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
    frame_quality = "poor" if fq1 == "poor" or fq2 == "poor" else "good"

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
            label2 = str(item2.get("label", "")).lower()
            if item1.get("category") == item2.get("category") or any(word in label2 for word in label1.split() if len(word) > 3):
                best_match = item2
                best_idx = idx
                break
        if best_match:
            used_idx2.add(best_idx)
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

            try:
                conf1 = float(item1.get("confidence", 0.8))
                conf2 = float(best_match.get("confidence", 0.8))
                reconciled_item["confidence"] = round((conf1 + conf2) / 2.0, 2)
            except Exception:
                pass
            merged_items.append(reconciled_item)
        else:
            merged_items.append(item1)

    for idx, item2 in enumerate(items2):
        if idx not in used_idx2 and isinstance(item2, dict):
            merged_items.append(item2)

    return {
        "items": merged_items,
        "scene_lighting": reconciled_lighting,
        "frame_quality": frame_quality,
        "conflicts_flagged": conflicts_flagged,
        "conflict_notes": conflict_notes,
    }

async def scan_frame_async(image_b64: str) -> dict:
    import os
    import asyncio
    from openai import AsyncOpenAI

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    base_url = "https://integrate.api.nvidia.com/v1"
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

    task1 = _call_model_for_scan(async_client, _MODEL_NEMOTRON, nemotron_messages, False)
    task2 = _call_model_for_scan(async_client, _MODEL_VISION_90B, llama_messages, True)

    results = await asyncio.gather(task1, task2, return_exceptions=True)
    res1 = results[0] if isinstance(results[0], dict) else None
    res2 = results[1] if isinstance(results[1], dict) else None

    if not res1 and not res2:
        log.info("[CV] Both dual models returned None, falling back to 11b vision")
        res1 = await _call_model_for_scan(async_client, _MODEL_VISION_11B, llama_messages, True)

    data = reconcile_scan_results(res1, res2)
    return data


def scan_frame(image_b64: str) -> dict:
    """
    Detect all visible garments/accessories in a single image frame.
    Calls two vision models concurrently and reconciles outputs.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            data = pool.submit(lambda: asyncio.run(scan_frame_async(image_b64))).result()
    else:
        data = asyncio.run(scan_frame_async(image_b64))

    if not data:
        return {
            "items": [],
            "scene_lighting": "unknown",
            "frame_quality": "poor",
            "guidance": "Vision model unavailable — try again",
        }

    # ── Normalise and validate ──────────────────────────────────
    items = data.get("items", [])
    frame_quality = data.get("frame_quality", "good")

    # ── Dedup first, then guard ──────────────────────────────────
    # Dedup removes same-label pairs (two shoes on shelf = one entry).
    # Guard only fires after dedup — 3+ identical labels post-dedup
    # means the model is genuinely stuck in a loop, not just
    # counting both shoes separately.
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
            f"{max_dupes}x across {len(items)} items. Falling back."
        )
        # Don't return poor — let 90b fallback handle it if Nemotron looped
        items = []
        data["items"] = []
        data["frame_quality"] = "poor"
        data["guidance"] = "Move closer to the items — Riley couldn't read this frame clearly."
    # ── End guard ────────────────────────────────────────────────

    # If frame is poor, strip items per spec
    if frame_quality == "poor":
        data["items"] = []
        if "guidance" not in data:
            data["guidance"] = "move closer to the rack"
        return data

    # Inject per-item guidance for low-confidence items
    for item in items:
        conf = item.get("confidence", 1.0)
        if conf < 0.6 and "guidance" not in item:
            item["guidance"] = "move_closer"  # safe default

    # ── Batch grid-based bbox localization ──────────────────────
    # Always runs — LLaMA vision can't reliably return float coords,
    # so we classify into a 4x3 grid and convert to bbox in code.
    if items:
        log.info(f"[CV] Running batch grid localization for {len(items)} items")
        items = _batch_localize_bboxes(client, items, image_b64)
        data["items"] = items

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
            all_xyxy = []
            for item in items:
                b = item.get("bbox", {})
                x, y = b.get("x", 0.0), b.get("y", 0.0)
                w, h = b.get("w", 0.0), b.get("h", 0.0)
                xyxy = np.array([[
                    x * img_w, y * img_h,
                    (x + w) * img_w, (y + h) * img_h
                ]])
                conf = np.array([item.get("confidence", 1.0)])
                det = sv.Detections(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=np.zeros(1, dtype=int)
                )
                item_hex = _category_hex(item.get("category", ""))
                item_color = sv.Color.from_hex(item_hex)
                annotated = sv.RoundBoxAnnotator(
                    color=item_color, thickness=2
                ).annotate(scene=annotated, detections=det)
                annotated = sv.LabelAnnotator(
                    color=item_color,
                    text_color=sv.Color.from_hex("#000000"),
                    text_scale=0.45,
                    text_thickness=1,
                    text_padding=5,
                    border_radius=4,
                ).annotate(
                    scene=annotated,
                    detections=det,
                    labels=[f"{item.get('label', 'Unknown')} · {item.get('color', '')}"]
                )
                all_xyxy.append(xyxy[0].tolist())

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
