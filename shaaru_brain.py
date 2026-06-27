"""
shaaru_brain.py — Core AI engine for SHAARU, the personal fashion stylist.

Routes text and vision queries through NVIDIA-hosted LLMs,
builds user-specific context from comfort profiles, wardrobe data,
and fashion knowledge, then generates responses in Shaaru's voice.
"""

import os
import re
import time
import base64
import logging
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
load_dotenv()
log = logging.getLogger("shaaru.brain")

try:
    from pipeline.on_demand.cache_checker import check_cache
except ImportError:
    check_cache = None

# ── NVIDIA client (lazy init) ────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── MongoDB connection (lazy init) ───────────────────────────────
MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
DB_NAME   = os.getenv("MONGODB_DB_NAME", "shaaru")

_mongo_client = None
_mongo_db     = None

def _get_db():
    global _mongo_client, _mongo_db
    if _mongo_db is not None:
        return _mongo_db
    from pymongo import MongoClient
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "shaaru")
    _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _mongo_db = _mongo_client[db_name]
    return _mongo_db

def _seed_mock_db(db):
    from datetime import timezone
    now = datetime.now(timezone.utc)
    db["users"].insert_one({
        "user_id": "demo_user_001",
        "name": "Riya",
        "visual": {"monk_scale":"M4","undertone":"warm","face_shape":"oval","hair_color":"dark brown","eye_color":"dark brown"},
        "physical": {"height_cm":163,"body_type":"pear"},
        "taste": {"everyday":["Casual","Minimalist"],"cozy":["Cottagecore"],"fashion_week":["Editorial"],"dream_outfit":["Quiet Luxury"],"color_palette":["earth tones","neutrals"],"occasion":["college","brunch"],"style_icon":"Janhvi Kapoor"},
        "style_equation": {"primary_aesthetic":"Quiet Luxury","secondary_aesthetic":"Minimalist"},
        "meta": {"tier":"free","onboarding_complete":True},
    })
    db["trends"].insert_one({
        "captured_at": now,
        "rising": [{"trend":"indo-western fusion"},{"trend":"quiet luxury minimalism"}],
        "seasonal_direction": "Light fabrics, earthy tones, Indian craft aesthetics.",
    })
    try:
        db["products"].insert_many([
            {
                "product_name": "Nicobar Linen Blouse",
                "brand": "Nicobar",
                "category": "top",
                "aesthetic": "Quiet Luxury",
                "color": "Beige",
                "silhouette": "Flowy",
                "pricing": {"price_inr": 2499},
                "availability": {"in_stock": True},
                "compatibility": {
                    "monk_scales": "M4",
                    "body_types": "pear",
                    "occasions": ["casual", "brunch"]
                }
            },
            {
                "product_name": "Levi's Straight Leg Jeans",
                "brand": "Levi's",
                "category": "bottom",
                "aesthetic": "Minimalist",
                "color": "Light Wash",
                "silhouette": "Straight",
                "pricing": {"price_inr": 3499},
                "availability": {"in_stock": True},
                "compatibility": {
                    "monk_scales": "M4",
                    "body_types": "pear",
                    "occasions": ["casual", "mall"]
                }
            }
        ])
    except Exception as e:
        log.warning(f"Failed to seed products into mock db: {e}")
    log.info("[DB] Mock DB seeded with demo_user_001 and mock products")

_client = None

def _get_client() -> OpenAI:
    """Lazy-init the NVIDIA OpenAI client on first use."""
    global _client
    if _client is None:
        api_key = os.getenv("NVIDIA_API_KEY")
        _client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=120.0, max_retries=0)
    return _client

# ── Model roster (confirmed working 2026-06-13) ─────────────────
MODEL_TEXT       = "meta/llama-3.1-70b-instruct"
MODEL_VISION_11B = "meta/llama-3.2-11b-vision-instruct"
MODEL_VISION_90B = "meta/llama-3.2-90b-vision-instruct"
MODEL_COMPLEX    = "nvidia/llama-3.1-nemotron-70b-instruct"

# ── Retry wrapper import ─────────────────────────────────────────
try:
    from shaaru_retry import nvidia_call
except (ImportError, SyntaxError, ValueError):
    log.warning("shaaru_retry not found or corrupted — using direct API calls (no retry)")
    def nvidia_call(client, model, messages, max_tokens=1024,
                    temperature=0.7, **kwargs):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return resp.choices[0].message.content

# ── Profile import ───────────────────────────────────────────────
try:
    from comfort_profile import get_profile
except (ImportError, SyntaxError, ValueError):
    log.warning("comfort_profile not found or corrupted — profiles will be empty")
    def get_profile(user_id: str) -> dict:
        return {}

# ── Knowledge graph (optional) ───────────────────────────────────
try:
    from knowledge_graph import kg
    HAS_KG = True
except ImportError:
    kg = None
    HAS_KG = False
    log.info("knowledge_graph not available — running without Neo4j")


# ══════════════════════════════════════════════════════════════════
#  PERSONALITY SYSTEM PROMPT — 25 Sacred Rules
# ══════════════════════════════════════════════════════════════════

PERSONALITY_LAW = """You are Shaaru — a direct, confident, fashion-expert best friend.

RULES (never break these):
1. Your name is Shaaru. Never say "I am an AI", "As an AI assistant", or similar.
2. Voice: direct, fashion-expert, casual bestie energy.
3. Mix Hindi/English naturally — "Kya hai bestie", "haan", "acha", "bilkul".
4. Never give generic advice — always specific to THIS user's profile.
5. Reference the user's face shape, skin tone, and body type naturally.
6. You know their wardrobe — reference specific gaps when relevant.
7. Never say "I cannot", "I am unable", "As an AI assistant".
8. If something fails, say: "send that again, something dropped on my end".
9. Use ★ and ✨ sparingly for emphasis — no emoji overload.
10. Playfair Display italic is your visual voice — write like that sounds.
11. Opening greeting should reference the user's physical profile data.
12. Never recommend what the user already owns.
13. When SPECIFIC PRODUCTS appear in your context under "SPECIFIC PRODUCTS FROM OUR CATALOG" — you MUST name them by exact product name and brand. Say "the Nicobar Linen Overshirt" not just "a linen shirt". Never mention prices or INR amounts.
14. Know the Indian fashion landscape — indie brands, price points in INR.
15. Treat the user like a close friend, not a customer.
16. Keep responses short unless the user asks for detail.
17. Never list more than 3 things at once.
18. Push back if the user's idea won't work for their body type.
19. Celebrate good choices — "yes, exactly right".
20. Remember context within the conversation.
21. When the user shares a garment image — consider tailor intent detection.
22. Detect TAILOR_INTENT when user says: make, build, stitch, tailor, recreate, construct, sew.
23. Fashion focus only — deflect non-fashion topics warmly.
24. Know seasonal context — reference weather, occasions, festivals.
25. End conversations with a specific next action for the user.
26. For tailor briefs specifically (when TAILOR_INTENT is detected), structure the breakdown as:
   - Silhouette & length
   - Base fabric (name, weight, suggested source)
   - Lining recommendation
   - Embroidery technique name + regional origin
   - Embroidery placement description (front panel, collar, hem etc)
   - Mirror type (sheesha glass vs sequin vs synthetic)
   - Collar construction (mandarin/band collar height)
   - Closure type (button placket, hooks)
   - Sleeve construction
   - What to tell the tailor verbatim in Hindi/Urdu if possible
"""

# ── LUM constants ────────────────────────────────────────────────
TOKEN_CEILING      = 700   # hard brief ceiling (chars ÷ 4)
BRIEF_TTL_MINUTES  = 30    # MongoDB brief TTL

# ── Tailor intent keywords ───────────────────────────────────────
TAILOR_KEYWORDS = {
    "make", "build", "stitch", "tailor", "recreate",
    "construct", "sew", "custom", "alter", "replicate",
}

# ── Fashion focus items for detection ────────────────────────────
FASHION_ITEMS = [
    "kurta", "kurti", "saree", "sari", "lehenga", "sherwani", "churidar",
    "salwar", "palazzo", "dupatta", "jacket", "blazer", "shirt", "jeans",
    "trousers", "pants", "skirt", "dress", "top", "blouse", "hoodie",
    "sweater", "cardigan", "coat", "suit", "chinos", "shorts", "lehnga",
    "anarkali", "sharara", "gharara", "dhoti", "lungi", "nehru jacket",
    "bandhgala", "jodhpuri", "crop top", "tank top", "t-shirt", "tee",
    "polo", "henley", "sneakers", "heels", "sandals", "boots", "loafers",
    "juttis", "kolhapuris", "watch", "bag", "clutch", "earrings",
    "necklace", "bracelet", "ring", "sunglasses", "scarf", "stole",
    "belt", "cap", "hat", "ethnic wear", "indo-western", "western wear",
]

# ── Technical fashion terms (trigger complex model) ──────────────
COMPLEX_TERMS = [
    "color theory", "undertone", "warm tone", "cool tone", "neutral tone",
    "capsule wardrobe", "silhouette", "draping", "proportion", "layering",
    "monochrome", "analogous", "complementary", "contrast ratio",
    "fabric weight", "GSM", "thread count", "weave", "pattern mixing",
    "body proportion", "golden ratio", "visual weight", "hem length",
]


# ══════════════════════════════════════════════════════════════════
#  CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════

# ── Token helpers ────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def _enforce_ceiling(zone1: str, zone2: str, zone3: str, trend: str) -> str:
    """
    Stack zones and enforce 700-token hard ceiling.
    Zone 1 is never cut. Zones 2/3 truncate if budget is tight.
    Trend injects into remaining space.
    """
    result    = zone1
    remaining = TOKEN_CEILING - _count_tokens(zone1)

    for zone in [zone2, zone3, trend]:
        if zone and remaining > 50:
            chunk  = zone[: remaining * 4]
            result += "\n\n" + chunk
            remaining -= _count_tokens(chunk)

    return result


# ── Zone builders ────────────────────────────────────────────────

def _build_zone1(user: dict) -> str:
    """Zone 1 — Core identity (never compressed). ~200 tokens."""
    try:
        visual   = user.get("visual",          {})
        physical = user.get("physical",        {})
        eq       = user.get("style_equation",  {})

        lines = [
            f"Name: {user.get('name', 'unknown')}",
            f"Monk scale: {visual.get('monk_scale', 'unknown')}",
            f"Undertone: {visual.get('undertone', 'unknown')}",
            f"Face shape: {visual.get('face_shape', 'unknown')}",
            f"Hair: {visual.get('hair_color', 'unknown')}",
            f"Eyes: {visual.get('eye_color', 'unknown')}",
            f"Height: {physical.get('height_cm', 'unknown')}cm",
            f"Body type: {physical.get('body_type', 'unknown')}",
            f"Primary aesthetic: {eq.get('primary_aesthetic', 'unknown')}",
            f"Secondary aesthetic: {eq.get('secondary_aesthetic', 'unknown')}",
        ]
        return "CORE IDENTITY:\n" + "\n".join(lines)
    except Exception as e:
        log.warning(f"Zone 1 build failed: {e}")
        return "CORE IDENTITY: unavailable"


def _build_zone2(user: dict) -> str:
    """Zone 2 — Taste profile + Neo4j behavioral edges. ~300 tokens."""
    try:
        taste = user.get("taste", {})
        lines = []

        if taste.get("everyday"):
            lines.append(f"Everyday style: {', '.join(taste['everyday'])}")
        if taste.get("cozy"):
            lines.append(f"Cozy style: {', '.join(taste['cozy'])}")
        if taste.get("fashion_week"):
            lines.append(f"Aspirational: {', '.join(taste['fashion_week'])}")
        if taste.get("color_palette"):
            lines.append(f"Color palette: {', '.join(taste['color_palette'])}")
        if taste.get("occasion"):
            lines.append(f"Dresses for: {', '.join(taste['occasion'])}")
        if taste.get("style_icon"):
            lines.append(f"Style icon: {taste['style_icon']}")

        # Pull top behavioral edges from Neo4j
        if HAS_KG and kg:
            try:
                sessions_count = user.get("meta", {}).get("sessions_count", 0)
                if sessions_count > 0:
                    edges = _query_behavioral_edges(user.get("user_id", ""))
                    if edges:
                        lines.append("Behavioral signals: " + "; ".join(edges[:10]))
            except Exception as e:
                log.debug(f"Neo4j behavioral edge query failed: {e}")

        return "TASTE PROFILE:\n" + "\n".join(lines) if lines else ""
    except Exception as e:
        log.warning(f"Zone 2 build failed: {e}")
        return ""


def _build_zone3(user_id: str) -> str:
    """Zone 3 — Rolling 3-session signal window. ~200 tokens."""
    try:
        db       = _get_db()
        sessions = list(
            db["sessions"]
            .find({"user_id": user_id}, {"signals": 1})
            .sort("started_at", -1)
            .limit(3)
        )
        if not sessions:
            return ""

        lines = []
        for s in sessions:
            sig = s.get("signals", {})
            if sig.get("saved"):
                lines.append(f"Saved: {', '.join(str(x) for x in sig['saved'][:3])}")
            if sig.get("skipped"):
                lines.append(f"Skipped: {', '.join(str(x) for x in sig['skipped'][:3])}")
            if sig.get("purchased"):
                lines.append(f"Bought: {', '.join(str(x) for x in sig['purchased'][:2])}")

        from pattern_detector import get_pattern_context
        pattern_ctx = get_pattern_context(user_id)
        if pattern_ctx:
            lines.append(pattern_ctx)

        return "RECENT SIGNALS:\n" + "\n".join(lines) if lines else ""
    except Exception as e:
        log.debug(f"Zone 3 build failed (no sessions yet — normal for new users): {e}")
        return ""


def _get_trend_context() -> str:
    """Pull latest trend snapshot from MongoDB trends collection."""
    try:
        db  = _get_db()
        doc = db["trends"].find_one(sort=[("captured_at", -1)])
        if not doc:
            return ""

        rising   = doc.get("rising", [])
        seasonal = doc.get("seasonal_direction", "")
        lines    = []

        if rising:
            trend_names = [t.get("trend", "") for t in rising[:3] if t.get("trend")]
            if trend_names:
                lines.append(f"Rising trends: {', '.join(trend_names)}")
        if seasonal:
            lines.append(f"Seasonal direction: {seasonal}")

        return "LIVE TRENDS:\n" + "\n".join(lines) if lines else ""
    except Exception as e:
        log.debug(f"Trend context fetch failed: {e}")
        return ""


def _query_behavioral_edges(user_id: str) -> list[str]:
    """
    Pull top weighted behavioral edges for this user from Neo4j.
    Returns list of readable signal strings.
    Falls back to empty list if user has no graph history yet.
    """
    if not (HAS_KG and kg):
        return []
    try:
        # Use raw driver query if available
        driver = getattr(kg, "driver", None)
        if not driver:
            return []

        with driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {user_id: $uid})-[r:SAVED|PURCHASED|SKIPPED]->(p)
                RETURN type(r) AS action, p.name AS item, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT 10
                """,
                uid=user_id,
            )
            edges = []
            for record in result:
                action = record.get("action", "interacted with")
                item   = record.get("item", "unknown")
                edges.append(f"{action.lower()} {item}")
            return edges
    except Exception as e:
        log.debug(f"Behavioral edge query: {e}")
        return []


# ── Brief cache (MongoDB) ─────────────────────────────────────────

def _get_cached_brief(user_id: str) -> Optional[str]:
    """Pull compiled brief from MongoDB briefs collection."""
    try:
        db  = _get_db()
        doc = db["briefs"].find_one(
            {"user_id": user_id},
            sort=[("generated_at", -1)],
        )
        if not doc:
            return None
        expires_at = doc.get("expires_at")
        if expires_at and expires_at < datetime.now(timezone.utc):
            return None  # expired
        return doc.get("compiled_brief")
    except Exception as e:
        log.debug(f"Brief cache read failed: {e}")
        return None


def _save_brief(user_id: str, brief: str) -> None:
    """Write compiled brief to MongoDB briefs collection."""
    try:
        db  = _get_db()
        now = datetime.now(timezone.utc)
        db["briefs"].insert_one({
            "user_id":        user_id,
            "generated_at":   now,
            "expires_at":     now + timedelta(minutes=BRIEF_TTL_MINUTES),
            "compiled_brief": brief,
            "token_count":    _count_tokens(brief),
        })
    except Exception as e:
        log.debug(f"Brief cache write failed: {e}")


def _compress_brief(brief: str, max_tokens: int = 700) -> str:
    """Hard ceiling on brief length. Keeps core identity, compresses rest."""
    # Approximate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    if len(brief) <= max_chars:
        return brief
    
    # Split into lines, keep most important ones
    lines = brief.split('\n')
    
    # Always keep these sections (core identity — never compress)
    protected_keywords = [
        'monk_scale', 'face_shape', 'body_type', 'undertone',
        'primary_aesthetic', 'You are', 'RULES', 'PHYSICAL'
    ]
    
    protected = []
    compressible = []
    
    for line in lines:
        if any(kw.lower() in line.lower() for kw in protected_keywords):
            protected.append(line)
        else:
            compressible.append(line)
    
    # Build brief: protected first, then fill remaining space
    result = '\n'.join(protected)
    remaining = max_chars - len(result)
    
    if remaining > 0:
        filler = '\n'.join(compressible)[:remaining]
        result = result + '\n' + filler
    
    print(f"[OK] Brief compressed: {len(brief)} → {len(result)} chars")
    return result

# ── Main context builder (3-zone LUM) ────────────────────────────

def build_riley_context(user_id: str, location: Optional[str] = None) -> str:
    """
    Build Riley's 3-zone user brief from MongoDB + Neo4j.
    Hard ceiling: 700 tokens.

    Zone 1 — Core identity     (MongoDB users, never cut)
    Zone 2 — Taste + behavior  (MongoDB taste + Neo4j edges)
    Zone 3 — Recent signals    (MongoDB sessions, rolling 3)
    Trend  — Live context      (MongoDB trends)
    """
    try:
        db   = _get_db()
        user = db["users"].find_one({"user_id": user_id})
    except Exception as e:
        log.error(f"MongoDB user fetch failed: {e}")
        user = None

    # No MongoDB profile — fall back to comfort_profile (legacy)
    if not user:
        log.warning(f"No MongoDB profile for {user_id} — falling back to comfort_profile")
        try:
            legacy = get_profile(user_id)
            if legacy:
                return _legacy_context(legacy, location)
        except Exception:
            pass
        return (
            "No profile found yet. "
            "Ask the user about their style to get started."
        )

    # Build 3 zones
    zone1 = _build_zone1(user)
    zone2 = _build_zone2(user)
    zone3 = _build_zone3(user_id)
    trend = _get_trend_context()

    # Location injection (override if passed)
    loc = location or user.get("meta", {}).get("location", "")
    if loc:
        zone1 += f"\nLocation: {loc}"

    # Enforce ceiling and return
    products = _get_product_recommendations(user_id, user)
    brief    = _enforce_ceiling(zone1, zone2, zone3, trend)
    
    from riley_evaluator import get_evaluation_context
    eval_ctx = get_evaluation_context(user_id)
    if eval_ctx:
        brief += f"\n{eval_ctx}\n"

    if products:
        brief += "\n\n" + products
    log.info(f"[LUM] Brief built for {user_id} — {_count_tokens(brief)} tokens")
    return _compress_brief(brief)


def _legacy_context(profile: dict, location: Optional[str] = None) -> str:
    """
    Fallback context builder using comfort_profile schema.
    Only used when no MongoDB profile exists.
    """
    sections = []

    phys = []
    for key, label in [
        ("face_shape",      "Face shape"),
        ("skin_tone_label", "Skin tone"),
        ("monk_scale",      "Monk scale"),
        ("body_type",       "Body type"),
        ("hair_type",       "Hair"),
        ("eye_color",       "Eyes"),
    ]:
        if profile.get(key):
            phys.append(f"{label}: {profile[key]}")
    if phys:
        sections.append("PHYSICAL PROFILE:\n" + "\n".join(phys))

    style = []
    if profile.get("preferred_styles"):
        style.append(f"Loves: {', '.join(profile['preferred_styles'])}")
    if profile.get("preferred_colors"):
        style.append(f"Colors: {', '.join(profile['preferred_colors'])}")
    if style:
        sections.append("STYLE DNA:\n" + "\n".join(style))

    return "\n\n".join(sections) if sections else "Minimal profile."


# ══════════════════════════════════════════════════════════════════
#  FOCUS ITEM DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_focus_item(message: str) -> Optional[str]:
    """
    Extract the primary fashion item being discussed from a message.
    Returns the item name or None.
    """
    msg_lower = message.lower()
    # Check longest items first to match "nehru jacket" before "jacket"
    sorted_items = sorted(FASHION_ITEMS, key=len, reverse=True)
    for item in sorted_items:
        if item in msg_lower:
            log.info(f"[BRAIN] Detected focus item: {item}")
            return item
    return None


# ══════════════════════════════════════════════════════════════════
#  TAILOR INTENT DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_tailor_intent(message: str, has_image: bool = False) -> bool:
    """
    Detect if the user wants to make/tailor/recreate a garment.
    Returns True if tailor intent keywords are found.
    """
    msg_lower = message.lower()
    words = set(re.findall(r'\b\w+\b', msg_lower))
    return bool(words & TAILOR_KEYWORDS)


# ══════════════════════════════════════════════════════════════════
#  MODEL SELECTION
# ══════════════════════════════════════════════════════════════════

def _select_model(message: str, has_image: bool) -> tuple[str, int]:
    """
    Choose the right model and max_tokens based on the query.
    Returns (model_name, max_tokens).
    """
    if has_image:
        return MODEL_VISION_11B, 1024

    word_count = len(message.split())
    msg_lower = message.lower()

    # Complex query detection
    is_complex = (
        word_count > 50
        or any(term in msg_lower for term in COMPLEX_TERMS)
    )

    if is_complex:
        return MODEL_TEXT, 2048
    else:
        return MODEL_TEXT, 1024


def _log_session(
    user_id: str,
    user_message: str,
    riley_response: str,
    focus_item: Optional[str],
    tailor_intent: bool,
) -> None:
    from datetime import timezone
    try:
        db  = _get_db()
        now = datetime.now(timezone.utc)
        db["sessions"].insert_one({
            "user_id":    user_id,
            "started_at": now,
            "ended_at":   now,
            "messages": [
                {"role": "user",  "content": user_message,  "timestamp": now},
                {"role": "riley", "content": riley_response, "timestamp": now},
            ],
            "signals": {
                "saved":     [],
                "skipped":   [],
                "purchased": [],
                "focus_item":    focus_item or "",
                "tailor_intent": tailor_intent,
                "aesthetic_drift": False,
            },
        })
        db["users"].update_one(
            {"user_id": user_id},
            {"$inc": {"meta.sessions_count": 1},
             "$set": {"meta.last_active": now}},
        )
        log.info(f"[SESSION] Logged for {user_id}")
    except Exception as e:
        log.debug(f"Session log write failed: {e}")

def _get_product_recommendations(
    user_id: str,
    user: dict,
    focus_item: Optional[str] = None,
    occasion: Optional[str] = None,
    limit: int = 3,
) -> str:
    try:
        db = _get_db()
        visual   = user.get("visual",   {})
        physical = user.get("physical", {})
        taste    = user.get("taste",    {})
        eq       = user.get("style_equation", {})

        monk       = visual.get("monk_scale",  "")
        body_type  = physical.get("body_type", "")
        aesthetic  = eq.get("primary_aesthetic", "")
        occasions  = taste.get("occasion", [])

        query: dict = {"availability.in_stock": True}

        if monk:
            query["compatibility.monk_scales"] = monk
        if body_type:
            query["compatibility.body_types"] = body_type
        if aesthetic:
            query["aesthetic"] = aesthetic
        if occasion:
            query["compatibility.occasions"] = occasion
        elif occasions:
            query["compatibility.occasions"] = {"$in": occasions}
        if focus_item:
            query["category"] = {
                "$regex": focus_item,
                "$options": "i"
            }

        products = list(
            db["products"]
            .find(query, {
                "product_name": 1,
                "brand": 1,
                "pricing.price_inr": 1,
                "color": 1,
                "silhouette": 1,
                "product_url": 1,
            })
            .limit(limit)
        )

        if not products:
            products = list(
                db["products"]
                .find(
                    {"availability.in_stock": True},
                    {"product_name":1,"brand":1,"pricing.price_inr":1,"color":1,"silhouette":1}
                )
                .limit(limit)
            )

        if not products:
            return ""

        lines = ["SPECIFIC PRODUCTS FROM OUR CATALOG (recommend these by exact name and brand):"]
        for p in products:
            price = p.get("pricing", {}).get("price_inr", 0)
            lines.append(
                f"- {p['product_name']} by {p['brand']} "
                f"| {p.get('color','')} | {p.get('silhouette','')}"
            )
        return "\n".join(lines)

    except Exception as e:
        log.debug(f"Product recommendation query failed: {e}")
        return ""

# ══════════════════════════════════════════════════════════════════
#  MAIN CHAT FUNCTION
# ══════════════════════════════════════════════════════════════════

def chat_with_riley(
    user_id: str,
    message: str,
    history: list[dict],
    image_b64: Optional[str] = None,
) -> str:
    """
    Main entry point for Shaaru chat. Builds context, selects model,
    handles vision, detects tailor intent, and returns response.

    Args:
        user_id:   The user's unique identifier.
        message:   The user's current message text.
        history:   List of prior messages [{role, content}, ...].
        image_b64: Optional base64-encoded image string.

    Returns:
        The assistant's response string.
        Includes "TAILOR_INTENT_DETECTED" signal when applicable.
    """
    has_image = bool(image_b64)

    # ── Build context ────────────────────────────────────────────
    try:
        context = build_riley_context(user_id)
    except Exception as e:
        log.error(f"Context build failed: {e}")
        context = "Profile unavailable — give general fashion advice."

    # ── Detect focus item for KG enrichment ──────────────────────
    focus_item = detect_focus_item(message)
    focus_context = ""
    if focus_item and HAS_KG and kg:
        try:
            log.info(f"[BRAIN] Detected focus item: {focus_item}. Querying Neo4j...")
            if hasattr(kg, "query_item_pairings"):
                pairings = kg.query_item_pairings(focus_item)
                if pairings:
                    pairs_str = ", ".join(
                        f"{p.get('paired_item', '?')}" for p in pairings[:5]
                    )
                    focus_context = (
                        f"\n\nFor {focus_item}, trending pairings: {pairs_str}"
                    )
        except Exception as e:
            log.warning(f"KG focus query failed: {e}")

    # ── Check tailor intent ──────────────────────────────────────
    tailor_intent = detect_tailor_intent(message, has_image)
    tailor_signal = ""
    if tailor_intent and has_image:
        tailor_signal = "\n\nTAILOR_INTENT_DETECTED"
        log.info(f"[BRAIN] Tailor intent detected for {user_id}")

    # ── Select model ─────────────────────────────────────────────
    model, max_tokens = _select_model(message, has_image)

    # ── Build messages array ─────────────────────────────────────
    system_msg = (
        f"{PERSONALITY_LAW}\n\n"
        f"USER CONTEXT:\n{context}"
        f"{focus_context}"
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add conversation history (keep last 10 turns for context window)
    if history:
        messages.extend(history[-10:])

    # ── Build user message (text or vision) ──────────────────────
    if has_image:
        user_content = [
            {"type": "text", "text": message or "What do you think of this?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                },
            },
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": message})

    # ── Call the model ───────────────────────────────────────────
    try:
        response = nvidia_call(
            client=_get_client(),
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
    except Exception as e:
        log.error(f"Primary model call failed ({model}): {e}")
        # Vision fallback: 11b → 90b
        if has_image and model == MODEL_VISION_11B:
            log.info("[BRAIN] Falling back to 90b vision model...")
            try:
                response = nvidia_call(
                    client=_get_client(),
                    model=MODEL_VISION_90B,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
            except Exception as e2:
                log.error(f"Vision fallback also failed: {e2}")
                return (
                    "Yaar, send that again — something dropped on my end. "
                    "The image didn't come through clearly."
                )
        else:
            return "Send that again, something dropped on my end ✨"

    # ── RSI signal logging ───────────────────────────────────────
    signal_type = "chat_message"
    if tailor_intent:
        signal_type = "tailor_intent"
    elif focus_item:
        signal_type = f"focus_{focus_item}"
    log.info(f"[RSI] Signal: {signal_type} for {user_id}")

    try:
        _log_session(user_id, message, response or "", focus_item, tailor_intent)
    except Exception as e:
        log.warning(f"Session log failed (non-blocking): {e}")

    return (response or "").strip() + tailor_signal


# ══════════════════════════════════════════════════════════════════
#  CACHED CHAT (in-memory context cache)
# ══════════════════════════════════════════════════════════════════

# Simple TTL cache for context strings
_context_cache: dict[str, tuple[float, str]] = {}
CONTEXT_TTL = 300  # 5 minutes


def _get_cached_context(user_id: str) -> str:
    """
    Get Riley's compiled brief from MongoDB briefs collection.
    Falls back to building fresh if not found or expired.
    """
    cached = _get_cached_brief(user_id)
    if cached:
        log.info(f"[BRIEF HIT] Loaded brief from MongoDB for {user_id}")
        return cached

    log.info(f"[BRIEF MISS] Building fresh brief for {user_id}")
    brief = build_riley_context(user_id)
    _save_brief(user_id, brief)
    return brief


def chat_with_riley_cached(
    user_id: str,
    message: str,
    history: list[dict],
    image_b64: Optional[str] = None,
) -> str:
    """
    Main chat entry point with tool-calling support.

    TEXT path (no image):
      1. Call meta/llama-3.1-70b-instruct with tools + tool_choice='auto'
      2. If model returns tool_calls → execute each tool, append results,
         make a second call to get Riley's natural-language response
      3. Keep existing pre-built context (zone1/2/3 + trends) as-is

    VISION path (image attached):
      Unchanged — vision models don't support tool calling on NIM.
      Falls through to direct nvidia_call() with 11b/90b vision models.
    """
    has_image = bool(image_b64)

    # ── Pre-built context (unchanged) ────────────────────────────
    context = _get_cached_context(user_id)

    focus_item    = detect_focus_item(message)
    focus_context = ""
    if focus_item and HAS_KG and kg:
        try:
            if hasattr(kg, "query_item_pairings"):
                pairings = kg.query_item_pairings(focus_item)
                if pairings:
                    pairs_str = ", ".join(
                        p.get("paired_item", "?") for p in pairings[:5]
                    )
                    focus_context = (
                        f"\n\nFor {focus_item}, trending pairings: {pairs_str}"
                    )
        except Exception as e:
            log.warning(f"KG focus query failed: {e}")

    tailor_intent = detect_tailor_intent(message, has_image)
    tailor_signal = "\n\nTAILOR_INTENT_DETECTED" if (tailor_intent and has_image) else ""

    model, max_tokens = _select_model(message, has_image)
    if has_image:
        model = MODEL_VISION_90B

    system_msg = (
        f"{PERSONALITY_LAW}\n\nUSER CONTEXT:\n{context}{focus_context}"
    )
    messages = [{"role": "system", "content": system_msg}]
    if history:
        messages.extend(history[-10:])

    # ── Build user message ────────────────────────────────────────
    if has_image:
        user_content = [
            {"type": "text", "text": message or "What do you think of this?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": message})

    # ── VISION PATH — unchanged, no tool calling ──────────────────
    if has_image:
        try:
            response = nvidia_call(
                client=_get_client(), model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.7,
            )
        except Exception as e:
            log.error(f"Vision model call failed: {e}")
            try:
                response = nvidia_call(
                    client=_get_client(), model=MODEL_VISION_90B,
                    messages=messages, max_tokens=max_tokens, temperature=0.7,
                )
            except Exception:
                return "Send that again, something dropped on my end ✨"

        signal_type = "tailor_intent" if tailor_intent else (
            f"focus_{focus_item}" if focus_item else "chat_message"
        )
        log.info(f"[RSI] Signal: {signal_type} for {user_id}")
        try:
            _log_session(user_id, message, response or "", focus_item, tailor_intent)
        except Exception as e:
            log.warning(f"Session log failed (non-blocking): {e}")
        return (response or "").strip() + tailor_signal

    # ── TEXT PATH — tool-calling enabled ─────────────────────────
    try:
        from shaaru_retry import nvidia_call_raw
        from shaaru_tools import TOOL_SCHEMAS, execute_tool
        import json as _json

        client = _get_client()

        # ── Call 1: model decides whether to use a tool ──────────
        log.info(f"[TOOLS] Call 1 — model={model}, tools={[t['function']['name'] for t in TOOL_SCHEMAS]}")
        msg1 = nvidia_call_raw(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        tool_calls = getattr(msg1, "tool_calls", None) or []

        if tool_calls:
            # ── Execute each tool and collect results ────────────
            # Append the assistant turn (must include tool_calls for the API)
            messages.append({
                "role":       "assistant",
                "content":    msg1.content or "",   # None when tool_calls fire
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                args = tc.function.arguments

                # ── Console log for test visibility ──────────────
                print(
                    f"\n[TOOL CALL] name={name} | "
                    f"args={args}"
                )

                result_json = execute_tool(name, args)

                print(f"[TOOL RESULT] {name} → {result_json[:300]}")

                # Append tool result message
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_json,
                })

            # ── Call 2: model generates Riley's natural response ─
            log.info(f"[TOOLS] Call 2 — getting Riley's final response after {len(tool_calls)} tool(s)")
            response = nvidia_call(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )

        else:
            # Model chose not to call any tool — use Call 1 content directly
            log.info(f"[TOOLS] No tool calls — using direct response")
            response = msg1.content or ""

    except Exception as e:
        log.error(f"Tool-call path failed ({model}): {e} — falling back to plain call")
        # Graceful fallback to pre-tool-calling behaviour
        try:
            response = nvidia_call(
                client=_get_client(), model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.7,
            )
        except Exception as e2:
            log.error(f"Fallback call also failed: {e2}")
            return "Send that again, something dropped on my end ✨"

    # ── Session logging ───────────────────────────────────────────
    signal_type = "tailor_intent" if tailor_intent else (
        f"focus_{focus_item}" if focus_item else "chat_message"
    )
    log.info(f"[RSI] Signal: {signal_type} for {user_id}")

    try:
        _log_session(user_id, message, response or "", focus_item, tailor_intent)
    except Exception as e:
        log.warning(f"Session log failed (non-blocking): {e}")

    return (response or "").strip() + tailor_signal


# ══════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════

def clear_context_cache(user_id: Optional[str] = None):
    """Clear cached context for a user or all users."""
    if user_id:
        _context_cache.pop(user_id, None)
    else:
        _context_cache.clear()


def get_greeting(user_id: str) -> str:
    """Generate a personalized opening greeting based on profile."""
    profile = get_profile(user_id)
    if not profile:
        return "Hey bestie! ✨ Let's figure out your style — tell me a bit about yourself?"

    name_part = ""
    if profile.get("face_shape") and profile.get("skin_tone_label"):
        name_part = (
            f"With your {profile['face_shape']} face shape and "
            f"{profile['skin_tone_label']} skin tone, "
        )

    gaps = profile.get("wardrobe_gaps", [])
    gap_part = ""
    if gaps:
        gap_part = f" I noticed you're missing a good {gaps[0]} — let's fix that today."

    return (
        f"Hey bestie! ✨ {name_part}"
        f"I've got some ideas for you.{gap_part}"
    )


async def _parse_query_intent(query: str, model: str = "meta/llama-3.1-8b-instruct") -> dict:
    """Extract techniques, aesthetics, and designers from query string using fast 8b / keyword matching."""
    text = query.lower()
    designers = ["abhinav_mishra", "torani", "injiri", "house_of_masaba", "raw_mango", "anavila", "sabyasachi", "rimzim_dadu", "pero", "anita_dongre"]
    detected_designers = [d for d in designers if d.replace("_", " ") in text or d in text]
    
    aesthetics = ["Mirror Maximalism", "Folk Maximalist", "Artisan Craft", "Graphic Pop Indian", "Handloom Minimal", "Heritage Couture", "Avant-Garde", "Festive Occasion"]
    detected_aesthetics = [a for a in aesthetics if a.lower() in text]
    
    techniques = ["mirror work", "sheesha", "zardozi", "resham", "kantha", "block print", "natural dye", "gota", "dabka", "chikankari", "phulkari", "sequin", "crystal", "zari", "bandhani", "ikat"]
    detected_techniques = [t for t in techniques if t in text]
    
    return {
        "techniques_detected": detected_techniques,
        "aesthetics_detected": detected_aesthetics or [None],
        "designers_detected": detected_designers or [None]
    }


def _product_text_for_fabric(product: dict) -> str:
    """Caption or raw_description from a cached/enriched product."""
    if not product:
        return ""
    cap = product.get("caption", "")
    if isinstance(cap, dict):
        cap = cap.get("text", "")
    return (product.get("raw_description") or cap or "").strip()


def _parse_fabric_from_description(text: str) -> str:
    """
    Extract primary base fabric from Shopify-style product copy.
    e.g. "Jacket & Pants - Velvet, Kurta - Special Silk" -> Velvet
    """
    if not text:
        return ""
    text = re.sub(r"&amp;", "&", text)
    section = text
    m = re.search(
        r"fabric\s*description\s*[:\-]\s*(.+?)(?:embroidery|color|care|no\.\s*of|$)",
        text,
        re.I | re.S,
    )
    if m:
        section = m.group(1)
    jacket = re.search(r"jacket[^,\n\-]*-\s*([^,\n]+)", section, re.I)
    if jacket:
        return jacket.group(1).strip()
    dash = re.search(r"-\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)", section)
    if dash:
        return dash.group(1).strip()
    for fabric in (
        "Velvet", "Special Silk", "Raw Silk", "Matka Silk", "Dupion Silk",
        "Chanderi", "Georgette", "Cotton Silk", "Satin Silk",
    ):
        if fabric.lower() in text.lower():
            return fabric
    return ""


async def answer(query: str, user_context: Optional[dict] = None) -> dict:
    """
    Two-phase response: Phase 1 instant Neo4j foundation only (<1s).
    Phase 2 background extraction fired by caller on cache miss.
    """
    t0 = time.time()
    if check_cache is None:
        return {"instant_answer": {"cache_hit": False}, "needs_enrichment": False}
        
    intent = await _parse_query_intent(query, model="meta/llama-3.1-8b-instruct")
    
    # Phase 1 — instant: Neo4j foundation only
    cache_result = check_cache(
        query=query,
        techniques=intent.get("techniques_detected", []),
        aesthetic=intent.get("aesthetics_detected", [None])[0]
    )
    
    # Fetch Mongo intelligence for Phase 1 enrichment (reusing warm connection)
    db = _get_db()
    
    # 1. Embellishment / Fabric sourcing matching technique
    fabric_sourcing = {}
    for tech in intent.get("techniques_detected", []):
        tech_query = tech.lower().replace(" ", ".*")
        doc = db.embellishment_sourcing.find_one({
            "$or": [
                {"embellishment_id": {"$regex": tech_query, "$options": "i"}},
                {"type": {"$regex": tech_query, "$options": "i"}},
                {"technique": {"$regex": tech_query, "$options": "i"}}
            ]
        })
        if not doc and ("mirror" in tech.lower() or "sheesha" in tech.lower()):
            doc = db.embellishment_sourcing.find_one({"embellishment_id": {"$regex": "mirror", "$options": "i"}})
        if doc and "sourcing" in doc:
            fabric_sourcing = doc["sourcing"]
            break
            
    if not fabric_sourcing:
        fab_doc = db.fabric_intelligence.find_one({"verified": True})
        if fab_doc and "sourcing" in fab_doc:
            fabric_sourcing = fab_doc["sourcing"]
        
    # 2. Editorial phrases
    editorial_phrases = []
    for ed_doc in db.editorial_vocab.find().limit(10):
        editorial_phrases.extend(ed_doc.get("sensory_adjectives", []))
        editorial_phrases.extend(ed_doc.get("draping_descriptions", []))
        editorial_phrases.extend(ed_doc.get("craft_heritage_phrases", []))
    editorial_phrases = list(dict.fromkeys(editorial_phrases))  # Deduplicate preserving order
    
    # 3. Graph context
    products = cache_result.get("products", [])
    graph_context = {
        "match_type": cache_result.get("match_type", "none"),
        "matched_on": cache_result.get("matched_on", ""),
        "designer_detected": intent.get("designers_detected", [None])[0],
        "techniques_detected": intent.get("techniques_detected", []),
        "aesthetics_detected": intent.get("aesthetics_detected", [])
    }
    
    # 4. Tailor brief (if tailor intent detected in query)
    tailor_brief = {}
    if any(k in query.lower() for k in ["made", "make", "stitch", "tailor", "build", "recreate", "construct", "sew"]):
        tech_list = intent.get("techniques_detected", [])
        tech_name = tech_list[0] if tech_list else "mirror work"
        markets = []
        if isinstance(fabric_sourcing, dict):
            for city, s_data in fabric_sourcing.items():
                if isinstance(s_data, dict) and "markets" in s_data:
                    markets.extend(s_data["markets"])
        if not markets:
            markets = ["Chickpet", "Commercial Street"]

        primary_product = products[0] if products else {}
        product_text = _product_text_for_fabric(primary_product)
        fabric_name = _parse_fabric_from_description(product_text) or "Raw Silk / Matka Silk"
        fabric_weight = "220-280 GSM (medium-heavy structured weight)"

        tailor_brief = {
            "silhouette_and_length": "Straight-cut traditional sherwani achkan, knee-length or slightly below knee",
            "base_fabric": {
                "name": fabric_name,
                "weight": fabric_weight,
                "suggested_source": list(dict.fromkeys(markets))
            },
            "lining_recommendation": "Cotton-silk or pure Shantoon lining for stiff structure and breathable drape",
            "embroidery_technique_and_origin": f"{tech_name.title()} (Abla bharat) with Resham thread couching; regional origin: Kutch / Gujarat",
            "embroidery_placement": "Dense geometric mirror scatter across front chest panels, band collar, and sleeve cuffs",
            "mirror_type": "Traditional foil-backed sheesha glass pieces (10mm - 15mm round abla); avoid synthetic plastic sequins",
            "collar_construction": "Stiff mandarin / band collar (Bandgala), 1.75 to 2 inches height with fused canvas backing",
            "closure_type": "Concealed front button placket with inner metallic hooks",
            "sleeve_construction": "Straight-set tailored full sleeves with functional or decorative cuff buttoning",
            "tailor_instructions_verbatim": (
                f"Masterji, straight-cut sherwani banani hai {fabric_name.lower()} mein. "
                "Front panel aur collar pe dense sheesha abla work chahiye resham thread ke saath. "
                "Collar achhe se stiff canvas se pasting karna 1.75 inch ka, aur andar concealed placket mein hooks lagana. "
                "Plastic mirror bilkul mat use karna, asli kaanch ke abla lagana."
            ),
        }
        if product_text:
            tailor_brief["source_product_text"] = product_text
    
    response_speed = f"{(time.time() - t0)*1000:.1f}ms"
    
    # Phase 2 — background: extract only on cache miss
    return {
        "response_speed": response_speed,
        "products": products,
        "fabric_sourcing": fabric_sourcing,
        "editorial_phrases": editorial_phrases,
        "graph_context": graph_context,
        "tailor_brief": tailor_brief,
        "instant_answer": cache_result,
        "needs_enrichment": not cache_result.get("cache_hit", False),
        "enrichment_args": {
            "query": query,
            "techniques": intent.get("techniques_detected", []),
            "aesthetic": intent.get("aesthetics_detected", [None])[0],
            "designer_id": intent.get("designers_detected", [None])[0]
        }
    }

