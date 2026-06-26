"""
Taste vector management.
- derive_initial_style_dna: converts existing taste/style_equation → numeric scores
- update_taste_vector: called after every save/skip/brief interaction
- migrate_all_users: one-time migration script
"""
import logging
from datetime import datetime, timezone
from shaaru_brain import _get_db

logger = logging.getLogger(__name__)

AESTHETIC_KEYS = [
    "quiet_luxury", "global_indian_chic", "minimalist", "maximalist_bridal",
    "cottagecore", "streetwear", "bohemian", "editorial", "avant_garde",
    "old_money", "indo_western_fusion", "heritage_luxury",
]

INTERACTION_WEIGHTS = {
    "save":             +0.08,
    "skip":             -0.04,
    "brief_generated":  +0.12,
    "quiz_answer":      +0.15,
}

# Map raw string values from taste fields → AESTHETIC_KEYS
_TASTE_ALIAS: dict[str, str] = {
    "quiet luxury":         "quiet_luxury",
    "old money":            "old_money",
    "global indian chic":   "global_indian_chic",
    "minimalist":           "minimalist",
    "maximalist bridal":    "maximalist_bridal",
    "cottagecore":          "cottagecore",
    "streetwear":           "streetwear",
    "bohemian":             "bohemian",
    "editorial":            "editorial",
    "avant-garde":          "avant_garde",
    "indo western fusion":  "indo_western_fusion",
    "heritage luxury":      "heritage_luxury",
}


def _normalize_key(raw: str) -> str | None:
    return _TASTE_ALIAS.get(raw.lower().strip())


def derive_initial_style_dna(user: dict) -> dict[str, float]:
    """
    Build first-pass style_dna from existing profile fields.
    Primary aesthetic → 0.85, secondary → 0.65, other taste mentions → 0.35
    """
    dna: dict[str, float] = {k: 0.0 for k in AESTHETIC_KEYS}

    style_eq = user.get("style_equation", {})
    primary_key = _normalize_key(style_eq.get("primary_aesthetic", ""))
    secondary_key = _normalize_key(style_eq.get("secondary_aesthetic", ""))

    if primary_key:
        dna[primary_key] = 0.85
    if secondary_key:
        dna[secondary_key] = 0.65

    taste = user.get("taste", {})
    all_mentions = (
        taste.get("everyday", []) +
        taste.get("cozy", []) +
        taste.get("fashion_week", []) +
        taste.get("dream_outfit", [])
    )
    for raw in all_mentions:
        k = _normalize_key(raw)
        if k and dna.get(k, 0.0) == 0.0:
            dna[k] = 0.35

    return dna


def update_taste_vector(
    user_id: str,
    interaction_type: str,
    aesthetic_signals: list[str],
    fabric_signals: list[str] | None = None,
) -> bool:
    """
    Update style_dna + fabric_preferences after a user interaction.

    Args:
        user_id:            MongoDB user_id string
        interaction_type:   'save' | 'skip' | 'brief_generated' | 'quiz_answer'
        aesthetic_signals:  aesthetic keys that were expressed (e.g. ['quiet_luxury'])
        fabric_signals:     optional fabric names (e.g. ['silk', 'linen'])

    Returns:
        True on success.
    """
    db = _get_db()
    if db is None:
        return False

    weight = INTERACTION_WEIGHTS.get(interaction_type, 0.05)

    try:
        user = db["users"].find_one({"user_id": user_id})
        if not user:
            logger.error(f"User not found: {user_id}")
            return False

        style_dna: dict[str, float] = user.get("style_dna", {k: 0.0 for k in AESTHETIC_KEYS})
        fabric_prefs: dict[str, float] = user.get("fabric_preferences", {})

        for raw in aesthetic_signals:
            k = _normalize_key(raw) or raw.lower().replace(" ", "_")
            current = style_dna.get(k, 0.0)
            style_dna[k] = round(max(0.0, min(1.0, current + weight)), 4)

        if fabric_signals:
            for fabric in fabric_signals:
                k = fabric.lower().replace(" ", "_")
                current = fabric_prefs.get(k, 0.5)
                fabric_prefs[k] = round(max(0.0, min(1.0, current + weight)), 4)

        db["users"].update_one(
            {"user_id": user_id},
            {"$set": {
                "style_dna": style_dna,
                "fabric_preferences": fabric_prefs,
                "meta.last_active": datetime.now(timezone.utc),
            }}
        )
        logger.info(f"Taste vector updated — {user_id} | {interaction_type} | {aesthetic_signals}")
        return True

    except Exception as e:
        logger.error(f"update_taste_vector failed for {user_id}: {e}")
        return False


def migrate_all_users():
    """One-time migration: adds style_dna to users who don't have it."""
    db = _get_db()
    if db is None:
        return
    users = list(db["users"].find({"style_dna": {"$exists": False}}))
    logger.info(f"Migrating {len(users)} users")
    for user in users:
        dna = derive_initial_style_dna(user)
        db["users"].update_one(
            {"_id": user["_id"]},
            {"$set": {
                "style_dna": dna,
                "fabric_preferences": {},
                "silhouette_preferences": {},
                "avoid": [],
            }}
        )
        logger.info(f"  ✓ {user.get('user_id')} → {dna}")
    logger.info("Migration complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_all_users()
