"""
comfort_profile.py — User comfort/style profile management.

Reads and writes user profiles (face shape, skin tone, wardrobe,
style preferences) from MongoDB.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.profile")

# ── MongoDB connection (lazy) ────────────────────────────────────
_db = None


def _get_db():
    """Lazy-init MongoDB connection."""
    global _db
    if _db is None:
        try:
            from pymongo import MongoClient
            uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGODB_DB", "shaaru")
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            _db = client[db_name]
            log.info(f"[PROFILE] Connected to MongoDB: {db_name}")
        except Exception as e:
            log.error(f"[PROFILE] MongoDB connection failed: {e}")
            return None
    return _db


def get_profile(user_id: str) -> dict:
    """
    Fetch the user's comfort/style profile from MongoDB.

    Returns a dict with keys:
        user_id, face_shape, skin_tone, skin_tone_label, monk_scale,
        hair_type, eye_color, body_type, preferred_styles[], avoided_styles[],
        preferred_colors[], avoided_colors[], fit_preference,
        occasion_needs[], location, climate, budget_range,
        wardrobe_items[], wardrobe_gaps[], pronouns, adventure_score

    Returns empty dict if user not found or DB unavailable.
    """
    db = _get_db()
    if db is None:
        return {}

    try:
        # Check comfort_profiles collection first
        profile = db["comfort_profiles"].find_one({"user_id": user_id})
        if not profile:
            # Fallback: check users collection
            profile = db["users"].find_one({"user_id": user_id})

        if not profile:
            log.info(f"[PROFILE] No profile found for {user_id}")
            return {}

        # Remove MongoDB internal _id field
        profile.pop("_id", None)

        # Ensure all expected fields exist with defaults
        defaults = {
            "user_id": user_id,
            "face_shape": None,
            "skin_tone": None,
            "skin_tone_label": None,
            "monk_scale": None,
            "hair_type": None,
            "eye_color": None,
            "body_type": None,
            "preferred_styles": [],
            "avoided_styles": [],
            "preferred_colors": [],
            "avoided_colors": [],
            "fit_preference": None,
            "occasion_needs": [],
            "location": None,
            "climate": None,
            "budget_range": None,
            "wardrobe_items": [],
            "wardrobe_gaps": [],
            "pronouns": None,
            "adventure_score": None,
        }

        for key, default in defaults.items():
            if key not in profile:
                profile[key] = default

        return profile

    except Exception as e:
        log.error(f"[PROFILE] Error fetching profile for {user_id}: {e}")
        return {}


def save_profile(user_id: str, profile_data: dict) -> bool:
    """
    Save or update a user's comfort profile in MongoDB.

    Args:
        user_id:      The user's unique identifier.
        profile_data: Dict of profile fields to upsert.

    Returns:
        True if save succeeded, False otherwise.
    """
    db = _get_db()
    if db is None:
        return False

    try:
        profile_data["user_id"] = user_id
        db["comfort_profiles"].update_one(
            {"user_id": user_id},
            {"$set": profile_data},
            upsert=True,
        )
        log.info(f"[PROFILE] Saved profile for {user_id}")
        return True
    except Exception as e:
        log.error(f"[PROFILE] Error saving profile for {user_id}: {e}")
        return False


def update_wardrobe(user_id: str, items: list[str], gaps: list[str] = None) -> bool:
    """Update wardrobe items and gaps for a user."""
    db = _get_db()
    if db is None:
        return False

    try:
        update = {"$set": {"wardrobe_items": items}}
        if gaps is not None:
            update["$set"]["wardrobe_gaps"] = gaps

        db["comfort_profiles"].update_one(
            {"user_id": user_id},
            update,
            upsert=True,
        )
        return True
    except Exception as e:
        log.error(f"[PROFILE] Error updating wardrobe for {user_id}: {e}")
        return False


def save_face_analysis(user_id: str, face_data: dict) -> bool:
    """
    Save face analysis results into the user's comfort profile.

    Args:
        user_id:   The user's unique identifier.
        face_data: Dict with face_shape, skin_tone, skin_tone_label,
                   monk_scale, hair_type, eye_color, body_type, etc.
    """
    return save_profile(user_id, face_data)
