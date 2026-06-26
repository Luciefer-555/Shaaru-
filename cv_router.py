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

from cv_engine import scan_frame, analyze_item

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
