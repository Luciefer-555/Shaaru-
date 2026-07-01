"""
cv_router.py — SHAARU Computer Vision API Routes

Endpoints (mounted at /api/cv):
  POST /api/cv/scan    — detect all garments in a frame
  POST /api/cv/analyze — deep-dive one detected item
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from cv_engine import scan_frame, analyze_item, generate_outfit_combinations

log = logging.getLogger("shaaru.cv.router")

router = APIRouter(prefix="/api/cv", tags=["cv"])


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
        result = scan_frame(req.image_b64)
        item_count = len(result.get("items", []))
        print(
            f"[CV SCAN] {ts} | user={req.user_id} | "
            f"items={item_count} | quality={result.get('frame_quality', 'unknown')}"
        )
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
    """
    if not req.items:
        return {"combos": []}
    combos = generate_outfit_combinations(req.items)
    return {"combos": combos}
