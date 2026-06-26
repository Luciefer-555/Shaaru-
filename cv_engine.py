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

log = logging.getLogger("shaaru.cv")

# ─────────────────────────────────────────────────────────────────
#  Model constants
# ─────────────────────────────────────────────────────────────────
_MODEL_VISION_90B = "meta/llama-3.2-90b-vision-instruct"
_MODEL_VISION_11B = "meta/llama-3.2-11b-vision-instruct"

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
You are a professional fashion analyst with deep knowledge of 
garment construction, fabric, and fashion nomenclature.

Analyze this image and detect every visible garment or accessory.
For each item, return a JSON object in this exact structure:

{
  "items": [
    {
      "id": "item_1",
      "label": "<specific fashion name>",
      "description": "<one precise sentence>",
      "category": "top|bottom|outerwear|footwear|accessory|dress|set",
      "color": "<precise color name, not just 'blue' — say 'indigo' or 'slate grey' or 'off-white'>",
      "aesthetic": "maximalist|minimalist|streetwear|editorial|fusion|formal|traditional|resort|workwear",
      "bbox": { "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0 },
      "confidence": 0.0
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
- If uncertain of exact construction, describe what is visible: 
  collar type, sleeve type, hem length, visible fabric texture, 
  notable detail

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

If frame_quality is poor: return 
{ "items": [], "frame_quality": "poor", 
  "guidance": "<one instruction for the user>" }
"""


def _parse_scan_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from a model response string."""
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
    return None


# ─────────────────────────────────────────────────────────────────
#  Bbox localization second pass
# ─────────────────────────────────────────────────────────────────
_BBOX_LOCALIZE_PROMPT = """Return ONLY a JSON object. No markdown, no explanation, no preamble.

Locate the following garment in the image: {label}

Return its bounding box as fractional coordinates (0.0 to 1.0):
{{"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}}

x,y = top-left corner of the item, w,h = width and height as fractions of image size."""


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


def scan_frame(image_b64: str) -> dict:
    """
    Detect all visible garments/accessories in a single image frame.

    Args:
        image_b64: Base64-encoded JPEG/PNG image string.

    Returns:
        dict with keys: items, scene_lighting, frame_quality,
        and optional top-level guidance.
    """
    from shaaru_brain import _get_client  # reuse shared client

    client = _get_client()

    messages = [
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
        {
            "role": "assistant",
            "content": "{"
        }
    ]

    response_text: Optional[str] = None

    # ── Primary model: 90b ──────────────────────────────────────
    used_fix = "none"
    try:
        try:
            raw = client.chat.completions.create(
                model=_MODEL_VISION_90B,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                timeout=90.0,
                response_format={"type": "json_object"}
            )
            used_fix = "response_format"
        except Exception as api_err:
            log.warning(f"[CV] 90b json_object failed, retrying without: {api_err}")
            raw = client.chat.completions.create(
                model=_MODEL_VISION_90B,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                timeout=90.0,
            )
            used_fix = "prefill_fallback"
            
        content = raw.choices[0].message.content
        response_text = "{" + content if not content.lstrip().startswith("{") else content
        log.info(f"[CV] scan_frame 90b response length={len(response_text or '')}. Fix: {used_fix}")
    except Exception as e:
        log.warning(f"[CV] 90b vision failed: {e}")

    data = _parse_scan_json(response_text) if response_text else None

    # ── Fallback: 11b ───────────────────────────────────────────
    if not data:
        log.info("[CV] Falling back to 11b vision for scan_frame")
        try:
            try:
                raw2 = client.chat.completions.create(
                    model=_MODEL_VISION_11B,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    timeout=90.0,
                    response_format={"type": "json_object"}
                )
                used_fix = "response_format"
            except Exception as api_err:
                log.warning(f"[CV] 11b json_object failed, retrying without: {api_err}")
                raw2 = client.chat.completions.create(
                    model=_MODEL_VISION_11B,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                    timeout=90.0,
                )
                used_fix = "prefill_fallback"
                
            content2 = raw2.choices[0].message.content
            response_text = "{" + content2 if not content2.lstrip().startswith("{") else content2
            data = _parse_scan_json(response_text)
            log.info(f"[CV] scan_frame 11b response length={len(response_text or '')}. Fix: {used_fix}")
        except Exception as e2:
            log.error(f"[CV] 11b vision also failed: {e2}")

    # ── Complete parse failure ──────────────────────────────────
    if not data:
        log.error(f"[CV] Both vision models failed to return parsable JSON for scan_frame. Raw 90b: {response_text}")
        return {
            "items": [],
            "scene_lighting": "unknown",
            "frame_quality": "poor",
            "guidance": "Vision model unavailable — try again",
        }

    # ── Normalise and validate ──────────────────────────────────
    items = data.get("items", [])
    frame_quality = data.get("frame_quality", "good")

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

    # ── Second-pass bbox fix for zero coordinates ───────────────
    zero_count = sum(
        1 for item in items
        if item.get("bbox", {}).get("w", 0) == 0 and item.get("bbox", {}).get("h", 0) == 0
    )
    if zero_count > 0:
        log.info(f"[CV] {zero_count}/{len(items)} items have zero bboxes — running localization pass")
        items = _localize_missing_bboxes(client, items, image_b64)
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
