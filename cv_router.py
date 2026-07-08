"""
cv_router.py — SHAARU Computer Vision API Routes

Endpoints (mounted at /api/cv):
  POST /api/cv/scan    — detect all garments in a frame
  POST /api/cv/analyze — deep-dive one detected item
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from cv_engine import scan_frame, analyze_item, generate_outfit_combinations

import time

log = logging.getLogger("shaaru.cv.router")

router = APIRouter(prefix="/api/cv", tags=["cv"])

_last_scan_cache = {}  # user_id -> {items, timestamp}


# ── Request Models ────────────────────────────────────────────────

class ScanRequest(BaseModel):
    image_b64: str
    user_id: str

class AnalyzeRequest(BaseModel):
    image_b64: str
    item_id: str
    item_label: str
    user_id: str

class StyleCombosRequest(BaseModel):
    items: list
    user_id: str
    enable_tts: Optional[bool] = False
    aesthetic_prompt: Optional[str] = None
    user_profile: Optional[dict] = None

class CorrectionRequest(BaseModel):
    track_id: str
    original: dict
    corrected: dict
    confidence: Optional[float] = 0.0
    user_id: Optional[str] = "default"

class EnrichFabricRequest(BaseModel):
    image_b64: str
    track_id: str
    bbox: dict
    label: str
    category: str
    user_id: Optional[str] = "default"


# ══════════════════════════════════════════════════════════════════
#  POST /api/cv/scan
# ══════════════════════════════════════════════════════════════════

@router.post("/scan")
async def cv_scan(req: ScanRequest):
    """
    Detect all visible garments/accessories in a frame.

    Returns scan_frame result: items with bboxes, confidences,
    scene lighting, frame quality, and optional guidance.
    Graceful degradation — never raises HTTPException.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        user_key = req.user_id or "default"
        now = time.time()
        if user_key in _last_scan_cache:
            cached = _last_scan_cache[user_key]
            if now - cached["timestamp"] < 8.0:
                log.info(f"[CV SCAN] Returning cached scan for user={user_key} (< 8s since last scan)")
                return cached["result"]

        result = scan_frame(req.image_b64, user_id=req.user_id, run_combos=False)
        item_count = len(result.get("items", []))
        print(
            f"[CV SCAN] {ts} | user={req.user_id} | "
            f"items={item_count} | quality={result.get('frame_quality', 'unknown')}"
        )
        if "error" not in result and result.get("frame_quality") != "poor":
            _last_scan_cache[user_key] = {
                "items": result.get("items", []),
                "result": result,
                "timestamp": now,
            }
        try:
            from shaaru_brain import _get_db
            db = _get_db()
            if db is not None and "error" not in result:
                items_list = [item.get("label", "garment") for item in result.get("items", []) if isinstance(item, dict)]
                db["scanned_items"].insert_one({
                    "user_id": req.user_id,
                    "type": "scan",
                    "timestamp": ts,
                    "created_at": datetime.now(timezone.utc),
                    "items": items_list,
                    "combos": result.get("combos", []),
                    "summary": f"Scanned items: {', '.join(items_list)}" if items_list else "Scanned frame with no items detected."
                })
        except Exception as write_err:
            log.warning(f"[CV SCAN] Failed to persist scan result: {write_err}")
        return result
    except Exception as e:
        log.error(f"[CV SCAN] Exception for user={req.user_id}: {e}")
        return {
            "error": str(e),
            "items": [],
            "guidance": "try again",
        }


# ══════════════════════════════════════════════════════════════════
#  POST /api/cv/analyze
# ══════════════════════════════════════════════════════════════════

@router.post("/analyze")
async def cv_analyze(req: AnalyzeRequest):
    """
    Deep-dive analysis of one detected garment item.

    Fetches user profile from MongoDB, then calls analyze_item().
    Returns garment_analysis + fabric_intelligence + profile_compatibility.
    Graceful degradation — never raises HTTPException.
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── Fetch user profile ────────────────────────────────────
    user_profile: dict = {}
    try:
        from shaaru_brain import _get_db
        db = _get_db()
        if db is not None:
            user_doc = db["users"].find_one({"user_id": req.user_id}) or {}
            face_data = user_doc.get("face_data", {})
            physical = user_doc.get("physical", {})
            style_eq = user_doc.get("style_equation", {})

            user_profile = {
                "monk_scale": face_data.get("monk_scale"),
                "body_type": physical.get("body_type"),
                "primary_aesthetic": style_eq.get("primary_aesthetic"),
                # optional extras used by query_fashion_intelligence
                "city": user_doc.get("meta", {}).get("city", "bengaluru"),
                "height_ft": (
                    user_doc.get("physical", {}).get("height_cm", 163) / 30.48
                ),
            }
    except Exception as e:
        log.warning(f"[CV ANALYZE] Profile fetch failed for user={req.user_id}: {e}")

    # ── Run analysis ──────────────────────────────────────────
    try:
        result = analyze_item(req.image_b64, req.item_label, user_profile)
        compat = result.get("profile_compatibility", {})
        print(
            f"[CV ANALYZE] {ts} | user={req.user_id} | "
            f"item='{req.item_label}' | "
            f"compatible={compat.get('compatible')} | "
            f"reason='{compat.get('reason', '')[:60]}'"
        )
        try:
            from shaaru_brain import _get_db
            db = _get_db()
            if db is not None and "error" not in result:
                garment = result.get("garment_analysis", {})
                fabric = result.get("fabric_intelligence", {})
                g_type = garment.get("garment_type", req.item_label) if isinstance(garment, dict) else req.item_label
                color = garment.get("color_palette", {}).get("dominant", "") if isinstance(garment, dict) else ""
                fab_name = fabric.get("recommended_fabric", {}).get("name", "") if isinstance(fabric, dict) else ""
                summary_parts = [p for p in [color, fab_name, g_type] if p]
                summary_str = f"Analyzed item '{req.item_label}': {' '.join(summary_parts)}" if summary_parts else f"Analyzed item '{req.item_label}'"
                db["scanned_items"].insert_one({
                    "user_id": req.user_id,
                    "type": "analyze",
                    "item_label": req.item_label,
                    "timestamp": ts,
                    "created_at": datetime.now(timezone.utc),
                    "garment_analysis": garment,
                    "fabric_intelligence": fabric,
                    "summary": summary_str
                })
        except Exception as write_err:
            log.warning(f"[CV ANALYZE] Failed to persist analysis result: {write_err}")
        return result
    except Exception as e:
        log.error(f"[CV ANALYZE] Exception for user={req.user_id}: {e}")
        return {
            "error": str(e),
            "item_label": req.item_label,
            "garment_analysis": {},
            "fabric_intelligence": {},
            "profile_compatibility": {
                "compatible": None,
                "reason": "Analysis unavailable",
            },
            "tailor_available": False,
        }


# ══════════════════════════════════════════════════════════════════
#  POST /api/cv/style-combos
# ══════════════════════════════════════════════════════════════════

@router.post("/style-combos")
async def cv_style_combos(req: StyleCombosRequest):
    """
    Generate 2-3 outfit combinations from detected scan items.
    Uses Riley's LLM to reason about what works together and
    what pieces are missing, with specific find-it descriptions.
    Formats active hunt directives and scan prompts for voice TTS.
    """
    if not req.items:
        return {"combos": []}
    combos = generate_outfit_combinations(
        detected_items=req.items,
        aesthetic_prompt=req.aesthetic_prompt,
        user_profile=req.user_profile,
    )
    res = {"combos": combos}
    try:
        from cv_engine import format_combos_for_speech
        spoken_text = format_combos_for_speech(combos)
        res["spoken_text"] = spoken_text
        if req.enable_tts and spoken_text:
            from voice_router import generate_tts_audio
            audio_b64 = generate_tts_audio(spoken_text)
            if audio_b64:
                res["audio_base64"] = audio_b64
                res["audio_format"] = "mp3"
    except Exception as e:
        log.warning(f"[CV COMBOS] Voice formatting failed: {e}")
    return res


@router.post("/touch")
async def cv_touch(request: Request):
    try:
        body = await request.json()
        item_label = body.get("item_label", "")
        item_color = body.get("item_color", "")
        item_category = body.get("item_category", "")
        item_aesthetic = body.get("item_aesthetic", "")
        item_fabric = body.get("item_fabric", body.get("fabric_type", ""))
        try:
            item_conf = float(body.get("item_confidence", body.get("confidence", 1.0)))
        except Exception:
            item_conf = 1.0
        all_items = body.get("all_items", [])
        user_id = body.get("user_id", "default")
        
        if not item_label:
            return {"comment": None}
        
        # Build context of other items in scene
        others = [i for i in all_items 
                  if i != item_label]
        others_str = (", ".join(others[:4]) 
                     if others else "nothing else visible")
        
        # Generate Shaaru's natural comment via Riley
        from riley_brain import riley_think
        
        fabric_hint = f", fabric: '{item_fabric}' (conf: {item_conf})" if item_fabric else ""
        if item_fabric:
            if item_conf >= 0.75 and str(item_fabric).strip().lower() != "uncertain":
                hedging_rule = f" Because fabric confidence is high ({item_conf}), state the fabric '{item_fabric}' directly and assertively without hedging."
            elif item_conf >= 0.45 or str(item_fabric).strip().lower() == "uncertain":
                hedging_rule = f" Because fabric confidence is low/penalized ({item_conf}), you MUST use conversational hedging when mentioning '{item_fabric}' (e.g. 'looks like it could be {item_fabric}', 'seems like {item_fabric}'). Never state it as flat fact!"
            else:
                hedging_rule = " Because fabric confidence is very low (< 0.45), skip mentioning the specific fabric type entirely; describe by silhouette/color instead."
        else:
            hedging_rule = ""

        prompt = (
            f"[SHAARU LIVE CAMERA — user just touched "
            f"or pointed at: {item_color} {item_label} "
            f"({item_category}, {item_aesthetic} vibe{fabric_hint}). "
            f"Other items visible in scene: {others_str}. "
            f"Give ONE natural warm reaction in Shaaru's "
            f"voice — like a knowledgeable bestie noticing "
            f"what they picked up. Comment on this specific "
            f"item and casually suggest one thing that would "
            f"pair well with it from the visible items or "
            f"something to look for. Max 20 words.{hedging_rule} "
            f"No hashtags, no bullet points, just talk.]"
        )
        
        result = riley_think(
            user_message=prompt,
            user_id=user_id
        )
        
        comment = result.get("reply", "")
        
        # Trim to under 25 words just in case
        words = comment.split()
        if len(words) > 25:
            comment = " ".join(words[:25]) + "."
        
        # Log touch event to MongoDB
        try:
            try:
                from database import get_db
                db = get_db()
            except ImportError:
                from shaaru_brain import _get_db
                db = _get_db()
            if db is not None:
                db["touch_events"].insert_one({
                    "user_id": user_id,
                    "item": item_label,
                    "color": item_color,
                    "comment": comment,
                    "timestamp": __import__("time").time()
                })
        except Exception:
            pass  # non-blocking
        
        return {"comment": comment, "item": item_label}
    
    except Exception as e:
        print(f"[CV TOUCH] Error: {e}")
        return {"comment": None, "error": str(e)}


@router.post("/correct")
async def cv_correct(req: CorrectionRequest):
    """
    Log user tap-to-correct feedback for future fine-tuning/eval.
    Writes to cv_corrections.jsonl in JSONL format.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        import json
        log_entry = {
            "timestamp": ts,
            "track_id": req.track_id,
            "user_id": req.user_id,
            "original": req.original,
            "corrected": req.corrected,
            "confidence": req.confidence,
        }
        with open("cv_corrections.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        log.info(f"[CV CORRECT] Logged correction for track_id={req.track_id}: {req.original} -> {req.corrected}")
        try:
            try:
                from database import get_db
                db = get_db()
            except ImportError:
                from shaaru_brain import _get_db
                db = _get_db()
            if db is not None:
                db["cv_corrections"].insert_one(log_entry)
        except Exception:
            pass
        return {"status": "success", "logged": True}
    except Exception as e:
        log.error(f"[CV CORRECT] Failed to write correction log: {e}")
        return {"status": "error", "error": str(e)}


@router.post("/enrich-fabric")
async def cv_enrich_fabric(req: EnrichFabricRequest):
    """
    Asynchronously enrich the fabric_type of a detected item from its cropped bounding box.
    Runs hierarchical _SCAN_PROMPT on 90B (35s timeout) -> 11B fallback (15s timeout).
    Updates ConsensusTracker so subsequent synchronous scans preserve the enriched fabric.
    """
    try:
        from cv_engine import enrich_item_fabric_async
        res = await enrich_item_fabric_async(
            image_b64=req.image_b64,
            bbox=req.bbox,
            label=req.label,
            category=req.category,
            track_id=req.track_id,
            user_id=req.user_id or "default"
        )
        return {
            "track_id": req.track_id,
            "fabric_type": res.get("fabric_type", "uncertain"),
            "fabric_reason": res.get("fabric_reason", ""),
            "confidence": res.get("confidence", 0.8),
            "status": "success"
        }
    except Exception as e:
        log.warning(f"[CV ENRICH FABRIC] Error for track_id={req.track_id}: {e}")
        return {
            "track_id": req.track_id,
            "fabric_type": "uncertain",
            "fabric_reason": "async enrichment failed",
            "confidence": 0.5,
            "status": "error",
            "error": str(e)
        }
