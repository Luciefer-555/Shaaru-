"""
api.py — FastAPI backend for the SHAARU fashion AI platform.

Endpoints:
  POST /api/chat          — Chat with Shaaru
  GET  /api/profile/{uid} — Get user profile
  GET  /api/products/seed — Discover products
  POST /api/wardrobe/upload — Upload wardrobe item
  GET  /api/wardrobe/{uid}  — Get wardrobe
  POST /api/profile/update  — Update profile
"""

import os
import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict

from dotenv import load_dotenv
load_dotenv()

import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext

from auth import create_access_token, get_current_user

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

load_dotenv()
log = logging.getLogger("shaaru.api")
logging.basicConfig(level=logging.INFO)

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="SHAARU", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Scheduler (auto-start on boot) ───────────────────────────────
try:
    from trend_scheduler import start_scheduler
    start_scheduler()
except Exception as e:
    log.warning(f"[API] Trend scheduler not started: {e}")

@app.on_event("startup")
async def startup_event():
    from pipeline.knowledge.graph_query import get_driver
    try:
        get_driver()
        log.info("[API] Neo4j driver initialized on startup.")
    except Exception as e:
        log.warning(f"[API] Neo4j startup warning: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    from pipeline.knowledge.graph_query import close_driver
    try:
        close_driver()
        log.info("[API] Neo4j driver closed on shutdown.")
    except Exception as e:
        log.warning(f"[API] Neo4j shutdown warning: {e}")

# ── Imports (graceful) ───────────────────────────────────────────
from shaaru_brain import chat_with_riley_cached, detect_tailor_intent

from tailor_router import tailor_router
app.include_router(tailor_router, prefix="/api")

try:
    from cv_router import router as cv_router
    app.include_router(cv_router)
    print("[OK] CV router loaded")
except Exception as e:
    print(f"[FAIL] CV router: {e}")

try:
    from voice_router import router as voice_router
    app.include_router(voice_router)
    print("[OK] Voice router loaded")
except Exception as e:
    print(f"[FAIL] Voice router: {e}")

try:
    from face_analysis import analyze_face_b64
except (ImportError, SyntaxError):
    analyze_face_b64 = None

try:
    from comfort_profile import get_profile, save_profile
except (ImportError, SyntaxError):
    def get_profile(uid): return {}
    def save_profile(uid, data): return False

try:
    from product_engine import get_products_for_discover
except (ImportError, SyntaxError):
    get_products_for_discover = None

try:
    from tailor_router import tailor_router
    print("[OK] Tailor router loaded")
except Exception as e:
    print(f"[FAIL] Tailor router: {e}")

# ── MongoDB (lazy) ───────────────────────────────────────────────
from shaaru_brain import _get_db

# ── Gender filter map ────────────────────────────────────────────
GENDER_MAP = {
    "he/him": {"male", "unisex"},
    "she/her": {"female", "unisex"},
    "they/them": {"male", "female", "unisex"},
    "male": {"male", "unisex"},
    "female": {"female", "unisex"},
    "mixed": {"male", "female", "unisex"},
}


def _require_same_user(requested_user_id: str, token_user: dict) -> None:
    token_user_id = token_user.get("user_id") or token_user.get("sub")
    if requested_user_id != token_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access another user's data",
        )


# ══════════════════════════════════════════════════════════════════
#  Request / Response Models
# ══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    user_id: str
    message: str
    image_b64: Optional[str] = None
    session_id: Optional[str] = None

class ChatMessageRequest(BaseModel):
    user_id: str
    message: str
    history: list = []
    image_base64: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tailor_intent: bool = False

class ProfileUpdateRequest(BaseModel):
    user_id: str
    face_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_tone_label: Optional[str] = None
    monk_scale: Optional[int] = None
    body_type: Optional[str] = None
    preferred_styles: Optional[list] = None
    avoided_styles: Optional[list] = None
    preferred_colors: Optional[list] = None
    avoided_colors: Optional[list] = None
    fit_preference: Optional[str] = None
    occasion_needs: Optional[list] = None
    location: Optional[str] = None
    climate: Optional[str] = None
    budget_range: Optional[str] = None
    pronouns: Optional[str] = None
    adventure_score: Optional[float] = None

class WardrobeUploadRequest(BaseModel):
    user_id: str
    image_b64: str


# ══════════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_history(user_id: str, session_id: str, limit: int = 10) -> list:
    """Fetch recent conversation history from MongoDB."""
    db = _get_db()
    if db is None:
        return []
    try:
        docs = (
            db["conversations"]
            .find({"user_id": user_id, "session_id": session_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        history = []
        for doc in reversed(list(docs)):
            history.append({"role": "user", "content": doc.get("message", "")})
            history.append({"role": "assistant", "content": doc.get("response", "")})
        return history
    except Exception as e:
        log.warning(f"Error fetching history: {e}")
        return []


def _save_conversation(user_id: str, session_id: str,
                       message: str, response: str, tailor_intent: bool):
    """Save a conversation turn to MongoDB."""
    db = _get_db()
    if db is None:
        return
    try:
        db["conversations"].insert_one({
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "response": response,
            "tailor_intent": tailor_intent,
            "created_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        log.warning(f"Error saving conversation: {e}")


# ══════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

class AuthRequest(BaseModel):
    user_id: str
    password: str

@app.post("/api/auth/register")
async def register(req: AuthRequest):
    db = _get_db()
    if db is None:
        raise HTTPException(500, "Database unavailable")
        
    existing = db["users"].find_one({"user_id": req.user_id})
    if existing and "hashed_password" in existing:
        raise HTTPException(400, "User already registered")
        
    hashed = pwd_context.hash(req.password)
    db["users"].update_one(
        {"user_id": req.user_id},
        {"$set": {"hashed_password": hashed}},
        upsert=True
    )
    
    token = create_access_token({"user_id": req.user_id, "sub": req.user_id})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/auth/login")
async def login(req: AuthRequest):
    db = _get_db()
    if db is None:
        raise HTTPException(500, "Database unavailable")
        
    user = db["users"].find_one({"user_id": req.user_id})
    if not user or "hashed_password" not in user:
        raise HTTPException(400, "Incorrect username or password")
        
    if not pwd_context.verify(req.password, user["hashed_password"]):
        raise HTTPException(400, "Incorrect username or password")
        
    token = create_access_token({"user_id": req.user_id, "sub": req.user_id})
    return {"access_token": token, "token_type": "bearer"}


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat_vision(req: ChatRequest):
    """Two-step vision route: analyse image → then pull real brands via riley_think."""
    try:
        if req.image_b64:
            # STEP 1 — Vision call: structured image analysis
            vision_message = f"""Look at this outfit image carefully and answer:

1. TECHNIQUE: What specific styling technique is shown?
   (e.g. "bandana waist wrap tied front point-down" not just "bandana")
2. SILHOUETTE: Name each garment and describe fit
   (e.g. "wide-leg pinstripe trousers, white satin button-down open 3 buttons")
3. ACCESSORIES: What accessories, how exactly worn?
4. VIBE: One word only — choose the PRIMARY silhouette vibe, not accessories:
   streetwear = oversized/baggy fits, wide-leg, casual layering
   minimal = clean lines, neutral tones, no embellishment  
   ethnic = Indian garments, handloom, embroidery, traditional cuts
   avant_garde = sculptural, experimental, unconventional construction
   editorial = sharp tailoring, high-fashion, structured
   maximalist = heavy embellishment, mirror work, sequins, bridal level
   Choose based on the CLOTHES, not the accessories.
5. HOW TO RECREATE: Step-by-step for the key technique only
6. PROPORTION LOGIC: Why does this combination work?

User's actual question: {req.message}

Answer all 6 points. On the VIBE line write exactly one word.

Do NOT suggest where to buy or recommend any stores, markets, 
or websites - that section will be handled separately."""

            vision_reply = chat_with_riley_cached(
                user_id=req.user_id,
                message=vision_message,
                history=[],
                image_b64=req.image_b64,
            )

            # STEP 2 — Direct Neo4j query, no LLM round trip
            from pipeline.knowledge.graph_query import get_brands_by_vibe

            vibe = "streetwear"  # fallback
            vibe_map = {
                "streetwear": "streetwear",
                "minimal": "minimal",
                "ethnic": "ethnic",
                "avant_garde": "avant_garde",
                "avant-garde": "avant_garde",
                "editorial": "editorial",
                "maximalist": "maximalist",
                "genderfluid": "genderfluid",
                "dark": "dark",
                "handcrafted": "handcrafted",
            }
            vision_lower = vision_reply.lower()
            for keyword, mapped in vibe_map.items():
                if keyword in vision_lower:
                    vibe = mapped
                    break

            # Keyword override — silhouette signals beat vibe label
            vision_lower_full = vision_reply.lower()
            streetwear_signals = [
                "wide-leg", "wide leg", "baggy", "oversized jacket",
                "pinstripe trouser", "bandana waist", "streetwear"
            ]
            ethnic_signals = [
                "kurta", "saree", "lehenga", "handloom", "embroidery", 
                "mirror work", "bandhani", "block print"
            ]
            if any(sig in vision_lower_full for sig in streetwear_signals):
                vibe = "streetwear"
            elif any(sig in vision_lower_full for sig in ethnic_signals):
                vibe = "ethnic"
            # else keep whatever vibe_map detected

            brands = get_brands_by_vibe(vibe)

            if brands:
                brand_lines = [
                    f"• {b['name']} — {b['url']} ({b['aesthetic']}, {b['region']})"
                    for b in brands
                ]
                brand_section = "\n".join(brand_lines)
            else:
                brand_section = "No brands found in graph for this vibe."

            combined = (
                f"{vision_reply}\n\n"
                f"---\n\n"
                f"**Where to shop this vibe ({vibe}):**\n"
                f"{brand_section}"
            )
            return {"reply": combined}
        else:
            # No image — straight to riley_think
            from riley_brain import riley_think
            result = riley_think(
                user_message=req.message,
                user_id=req.user_id,
                conversation_history=req.history or [],
            )
            return {"reply": result.get("reply", "")}
    except Exception as e:
        log.error(f"Error in /api/chat: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Riley ran into a problem. Try again.", "detail": str(e)})


@app.post("/api/chat/message")
async def chat_message(request: ChatMessageRequest, background_tasks: BackgroundTasks):
    try:
        from riley_brain import riley_think, _needs_tools
        from shaaru_brain import _get_db
        db = _get_db()
    
        session = db['chat_sessions'].find_one({'user_id': request.user_id})
        history = session.get('history', [])[-10:] if session else []

        result = riley_think(
            user_message=request.message,
            user_id=request.user_id,
            conversation_history=history,
            image_base64=request.image_base64
        )
        
        db['chat_sessions'].update_one(
            {'user_id': request.user_id},
            {'$push': {
                'history': {
                    '$each': [
                        {'role': 'user', 'content': request.message},
                        {'role': 'assistant', 'content': result['reply']}
                    ]
                }
            },
            '$set': {'updated_at': datetime.now(timezone.utc)}},
            upsert=True
        )
        
        response = {
            'reply': result['reply'],
            'tool_calls': result['tool_calls'],
            'model': result['model']
        }
        
        # If tailor flow was triggered, add redirect signal
        for tool in result['tool_calls']:
            if tool['tool'] == 'trigger_tailor_flow':
                if tool['result'].get('status') == 'tailor_flow_triggered':
                    response['action'] = {
                        'type': 'redirect',
                        'destination': '/tailor',
                        'garment': tool['result'].get('garment')
                    }

        # Two-phase response: Phase 1 instant Neo4j answer (<1s), Phase 2 background extraction
        if _needs_tools(request.message):
            try:
                from shaaru_brain import answer as shaaru_answer
                from pipeline.on_demand.extractor import handle_user_query
                brain_result = await shaaru_answer(query=request.message)
                if brain_result.get("needs_enrichment"):
                    background_tasks.add_task(handle_user_query, **brain_result["enrichment_args"])
                    log.info(f"[TwoPhase] Queued background enrichment for: '{request.message}'")
            except Exception as bg_e:
                log.warning(f"[TwoPhase] Background queue error: {bg_e}")

        return response
    except Exception as e:
        log.error(f"Error in /api/chat/message: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Riley ran into a problem. Try again.", "detail": str(e)})


@app.get("/api/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Fetch user's comfort/style profile."""
    profile = get_profile(user_id)
    if not profile:
        # Return defaults
        return {
            "user_id": user_id,
            "face_shape": None,
            "skin_tone_label": None,
            "body_type": None,
            "preferred_styles": [],
            "avoided_styles": [],
            "preferred_colors": [],
            "avoided_colors": [],
            "fit_preference": None,
            "occasion_needs": [],
            "pronouns": "they/them",
            "adventure_score": 0.3,
            "wardrobe_items": [],
            "wardrobe_gaps": [],
        }
    return profile


@app.get("/api/products/seed")
async def get_products(
    category: Optional[str] = None,
    aesthetic: Optional[str] = None,
    query: Optional[str] = None,
    pronouns: str = "they/them",
    adventure_score: float = 0.3,
    user_id: Optional[str] = None,
):
    """Get personalized product feed for Discover page."""
    if get_products_for_discover:
        try:
            products = get_products_for_discover(
                category=category,
                aesthetic=aesthetic,
                pronouns=pronouns,
                adventure_score=adventure_score,
                user_id=user_id or "anonymous",
            )
            return {"products": products, "total": len(products)}
        except Exception as e:
            log.error(f"Product engine error: {e}")

    # Fallback: load from products_seed.json
    products = _load_products_json()

    # Apply gender filter
    allowed = GENDER_MAP.get(pronouns, {"male", "female", "unisex"})
    products = [
        p for p in products
        if p.get("gender", "unisex") in allowed
    ]

    # Apply category/aesthetic filters
    if category:
        products = [
            p for p in products
            if category.lower() in (p.get("category", "")).lower()
            or category.lower() in (p.get("name", "")).lower()
        ]
    if aesthetic:
        products = [
            p for p in products
            if aesthetic.lower() in json.dumps(
                p.get("aesthetic_scores", {})
            ).lower()
        ]

    return {"products": products[:50], "total": len(products)}


def _load_products_json() -> list:
    """Load products from local seed file."""
    base_dir = os.path.dirname(__file__)
    seed_path = os.path.join(base_dir, "products_seed.json")
    if os.path.exists(seed_path):
        try:
            with open(seed_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Error loading products_seed.json: {e}")

    brand_entries_path = os.path.join(base_dir, "new_brand_entries.json")
    if os.path.exists(brand_entries_path):
        try:
            with open(brand_entries_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("new_products", [])
        except Exception as e:
            log.warning(f"Error loading new_brand_entries.json: {e}")

    return []


@app.post("/api/wardrobe/upload")
async def upload_wardrobe(req: WardrobeUploadRequest):
    """Analyze and save a wardrobe item from image."""
    if not analyze_face_b64:
        raise HTTPException(503, "Vision analysis not available")

    # Use vision to analyze the garment
    from shaaru_brain import _get_client, detect_focus_item
    try:
        from shaaru_retry import nvidia_call
    except (ImportError, SyntaxError):
        raise HTTPException(503, "Retry module not available")

    try:
        client = _get_client()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this garment image. Return JSON with: "
                            "name, category, colors[], aesthetic, brand (if visible), "
                            "estimated_price_inr. No markdown."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{req.image_b64}"
                        },
                    },
                ],
            }
        ]
        response = nvidia_call(
            client=client,
            model="meta/llama-3.2-11b-vision-instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )

        # Parse response
        item = {}
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                item = json.loads(text[start:end])
        except Exception:
            item = {"name": "Unknown garment", "category": "other"}

        item["user_id"] = req.user_id
        item["created_at"] = datetime.now(timezone.utc).isoformat()

        # Save to MongoDB
        db = _get_db()
        if db:
            db["wardrobe"].insert_one(item)
            item.pop("_id", None)

        return {"item": item, "success": True}

    except Exception as e:
        log.error(f"Wardrobe upload failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/api/wardrobe/{user_id}")
async def get_wardrobe(user_id: str):
    """Get all wardrobe items for a user, grouped by category."""
    db = _get_db()
    if db is None:
        return {"items": [], "categories": {}}

    try:
        docs = list(db["wardrobe"].find({"user_id": user_id}))
        for d in docs:
            d.pop("_id", None)

        # Group by category
        categories = {}
        for item in docs:
            cat = item.get("category", "other")
            categories.setdefault(cat, []).append(item)

        return {"items": docs, "categories": categories}
    except Exception as e:
        log.warning(f"Error fetching wardrobe: {e}")
        return {"items": [], "categories": {}}


@app.post("/api/profile/update")
async def update_profile(req: ProfileUpdateRequest):
    """Update user's comfort/style profile."""
    update_data = req.model_dump(exclude_none=True)
    user_id = update_data.pop("user_id")

    success = save_profile(user_id, update_data)
    if not success:
        raise HTTPException(500, "Failed to update profile")

    updated = get_profile(user_id)
    return updated


class SignalRequest(BaseModel):
    user_id: str
    product_id: str
    session_id: str

try:
    from knowledge_graph import get_kg
    HAS_KG = True
except ImportError:
    get_kg = lambda: None
    HAS_KG = False

def _process_signal(req: SignalRequest, signal_type: str):
    db = _get_db()
    if db is not None:
        db["sessions"].update_one(
            {"_id": req.session_id},
            {"$push": {f"signals.{signal_type}": req.product_id}}
        )
        db["briefs"].delete_many({"user_id": req.user_id})

    if HAS_KG and get_kg() and get_kg().is_connected:
        now = datetime.now(timezone.utc).isoformat()
        rel_type = "SAVED"
        if signal_type == "skipped": rel_type = "SKIPPED"
        elif signal_type == "purchased": rel_type = "PURCHASED"
        
        query = f"""
        MERGE (u:User {{user_id: $user_id}})
        MERGE (p:Product {{product_id: $product_id}})
        MERGE (u)-[r:{rel_type}]->(p)
        ON CREATE SET r.weight = 1, r.created_at = $now
        ON MATCH SET r.weight = r.weight + 1, r.updated_at = $now
        """
        try:
            get_kg().query(query, {"user_id": req.user_id, "product_id": req.product_id, "now": now})
        except Exception as e:
            log.warning(f"Neo4j {signal_type} signal failed: {e}")

    return {"status": "ok", "signal": signal_type, "user_id": req.user_id}

def _evaluate_last_response(user_id: str, signal_type: str, product_id: str):
    try:
        from shaaru_brain import _get_db
        from riley_evaluator import evaluate_response
        db = _get_db()
        session = db["sessions"].find_one(
            {"user_id": user_id},
            sort=[("started_at", -1)]
        )
        if session and session.get("messages"):
            last_riley_msg = next(
                (m["content"] for m in reversed(session["messages"]) 
                 if m["role"] == "riley"), None
            )
            if last_riley_msg:
                evaluate_response(
                    user_id=user_id,
                    session_id=str(session["_id"]),
                    response_text=last_riley_msg,
                    next_action=signal_type,
                    product_id=product_id,
                    db=db
                )
    except Exception as e:
        log.warning(f"Evaluation failed: {e}")

@app.post("/api/signal/save")
async def save_signal(req: SignalRequest):
    from signal_collector import collect_signal
    from taste_engine import update_taste_vector
    collect_signal(req.user_id, "saved", req.product_id)
    _evaluate_last_response(req.user_id, "saved", req.product_id)
    asyncio.to_thread(update_taste_vector, req.user_id, req.product_id, "save")
    return {"status": "ok", "signal": "saved", "user_id": req.user_id}

@app.post("/api/signal/skip")
async def skip_signal(req: SignalRequest):
    from signal_collector import collect_signal
    from taste_engine import update_taste_vector
    collect_signal(req.user_id, "skipped", req.product_id)
    _evaluate_last_response(req.user_id, "skipped", req.product_id)
    asyncio.to_thread(update_taste_vector, req.user_id, req.product_id, "skip")
    return {"status": "ok", "signal": "skipped", "user_id": req.user_id}

@app.post("/api/signal/purchase")
async def purchase_signal(req: SignalRequest):
    from signal_collector import collect_signal
    from taste_engine import update_taste_vector
    collect_signal(req.user_id, "purchased", req.product_id)
    _evaluate_last_response(req.user_id, "purchased", req.product_id)
    asyncio.to_thread(update_taste_vector, req.user_id, req.product_id, "purchase")
    return {"status": "ok", "signal": "purchased", "user_id": req.user_id}


# ══════════════════════════════════════════════════════════════════
#  ONBOARDING ENDPOINTS
# ══════════════════════════════════════════════════════════════════

class OnboardingInitRequest(BaseModel):
    name: str

class OnboardingTasteRequest(BaseModel):
    user_id: str
    name: Optional[str] = None
    height_cm: Optional[float] = None
    body_type: Optional[str] = None
    everyday: Optional[list] = None
    cozy: Optional[list] = None
    fashion_week: Optional[list] = None
    dream_outfit: Optional[list] = None
    color_palette: Optional[list] = None
    occasion: Optional[list] = None
    style_icon: Optional[str] = None
    preferred_brands: Optional[list] = None

class OnboardingCompleteRequest(BaseModel):
    user_id: str

class OnboardingAestheticsRequest(BaseModel):
    user_id: str
    aesthetics: list

@app.post("/api/onboarding/init")
async def onboarding_init(req: OnboardingInitRequest):
    user_id = str(uuid.uuid4())
    db = _get_db()
    if db is not None:
        db["users"].insert_one({
            "user_id": user_id,
            "name": req.name,
            "meta": {
                "onboarding_complete": False,
                "tier": "free",
                "sessions_count": 0
            },
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    return {"user_id": user_id, "name": req.name}

@app.post("/api/onboarding/photo")
async def onboarding_photo(user_id: str = Form(...), photo: UploadFile = File(...)):
    db = _get_db()
    if not db:
        raise HTTPException(500, "Database unavailable")
    try:
        content = await photo.read()
        import base64
        image_b64 = base64.b64encode(content).decode("utf-8")
        
        try:
            from face_analysis import analyze_face_b64
            profile = analyze_face_b64(image_b64, user_id)
            result = profile.to_dict()
            
            db["users"].update_one(
                {"user_id": user_id},
                {"$set": {
                    "face_data.monk_scale": result.get("monk_scale"),
                    "face_data.face_shape": result.get("face_shape"),
                    "face_data.undertone": result.get("skin_tone_label"),
                    "face_data.photo_verified": True
                }}
            )
            print(f"[OK] Face analysis successful for {user_id}: Monk {result.get('monk_scale')}, Shape {result.get('face_shape')}")
            return {"user_id": user_id, "face_data": result, "status": "success"}
        except Exception as e:
            print(f"[FAIL] face_analysis: {e}")
            return {"user_id": user_id, "status": "partial_success", "message": "Photo uploaded but analysis failed"}
            
    except Exception as e:
        log.error(f"Photo upload failed: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/onboarding/taste")
async def onboarding_taste(req: OnboardingTasteRequest):
    db = _get_db()
    if db is not None:
        db["users"].update_one(
            {"user_id": req.user_id},
            {"$set": {
                "taste": {
                    "everyday": req.everyday or [],
                    "cozy": req.cozy or [],
                    "fashion_week": req.fashion_week or [],
                    "dream_outfit": req.dream_outfit or [],
                    "color_palette": req.color_palette or [],
                    "occasion": req.occasion or [],
                    "style_icon": req.style_icon,
                    "preferred_brands": req.preferred_brands or []
                },
                "physical": {
                    "height_cm": req.height_cm,
                    "body_type": req.body_type
                }
            }}
        )
    return {"status": "ok", "user_id": req.user_id}

@app.post("/api/onboarding/complete")
async def onboarding_complete(req: OnboardingCompleteRequest):
    db = _get_db()
    if db is None:
        raise HTTPException(500, "Database unavailable")
        
    user = db["users"].find_one({"user_id": req.user_id})
    if not user:
        raise HTTPException(404, "User not found")
        
    taste = user.get("taste", {})
    visual = user.get("visual") or user.get("face_data", {})
    physical = user.get("physical", {})
    name = user.get("name", "Unknown")
    
    everyday = taste.get("everyday", [])
    primary_aesthetic = everyday[0] if everyday else "casual"
    
    dream_outfit = taste.get("dream_outfit", [])
    secondary_aesthetic = dream_outfit[0] if dream_outfit else "chic"
    
    monk_scale = visual.get("monk_scale", 5)
    face_shape = visual.get("face_shape", "oval")
    body_type = physical.get("body_type", "average")
    
    profile_hash = f"{monk_scale}_{face_shape}_{body_type}_{primary_aesthetic}"
    now = datetime.now(timezone.utc).isoformat()
    
    style_equation = {
        "primary_aesthetic": primary_aesthetic,
        "secondary_aesthetic": secondary_aesthetic,
        "profile_hash": profile_hash,
        "generated_at": now
    }
    
    db["users"].update_one(
        {"user_id": req.user_id},
        {"$set": {
            "style_equation": style_equation,
            "meta.onboarding_complete": True
        }}
    )
    
    db["briefs"].delete_many({"user_id": req.user_id})
    
    if HAS_KG and get_kg() and get_kg().is_connected:
        query = """
        MERGE (u:User {user_id: $user_id})
        SET u.name = $name,
            u.monk_scale = $monk_scale,
            u.body_type = $body_type,
            u.primary_aesthetic = $primary_aesthetic,
            u.created_at = $now
        """
        try:
            get_kg().query(query, {
                "user_id": req.user_id,
                "name": name,
                "monk_scale": monk_scale,
                "body_type": body_type,
                "primary_aesthetic": primary_aesthetic,
                "now": now
            })
        except Exception as e:
            log.warning(f"Neo4j onboarding complete failed: {e}")
            
    return {"status": "complete", "user_id": req.user_id, "style_equation": style_equation}


@app.post("/api/onboarding/aesthetics")
async def onboarding_aesthetics(req: OnboardingAestheticsRequest):
    db = _get_db()
    if db is None:
        raise HTTPException(500, "Database unavailable")
    db["users"].update_one(
        {"user_id": req.user_id},
        {"$set": {
            "taste.selected_aesthetics": req.aesthetics,
            "meta.onboarding_complete": True
        }}
    )
    return {"status": "ok", "user_id": req.user_id, "aesthetics": req.aesthetics}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "shaaru-brain"}

@app.get("/api/products")
async def fetch_products(aesthetic: Optional[str] = None, body_type: Optional[str] = None, occasion: Optional[str] = None, limit: int = 12):
    db = _get_db()
    if db is None:
        return {"products": [], "count": 0}
        
    query = {}
    if aesthetic:
        query["aesthetic"] = aesthetic
    if body_type:
        query["compatibility.body_types"] = body_type
    if occasion:
        query["compatibility.occasions"] = occasion
        
    cursor = db["products"].find(query).limit(limit)
    products = []
    for doc in cursor:
        doc.pop("_id", None)
        products.append(doc)
        
    return {"products": products, "count": len(products)}

@app.get("/api/trends")
async def fetch_trends():
    db = _get_db()
    if db is None:
        return {"rising": [], "seasonal_direction": "", "styling_guides_count": 0}

    trend = db["trends"].find_one(sort=[("captured_at", -1)])
    sg_count = db["styling_guides"].count_documents({})

    if trend:
        trend.pop("_id", None)
        trend["styling_guides_count"] = sg_count
        return trend
    return {"rising": [], "seasonal_direction": "", "styling_guides_count": sg_count}


@app.post("/api/trends/refresh")
async def trends_refresh():
    """Manually trigger the trend ingestion pipeline (runs in background thread)."""
    try:
        from trend_ingestion import run_pipeline
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
        return {"status": "pipeline started", "message": "Check logs for progress"}
    except Exception as e:
        raise HTTPException(500, f"Pipeline trigger failed: {e}")

@app.post("/api/demo/seed")
async def demo_seed():
    db = _get_db()
    if db is None:
        raise HTTPException(500, "Database unavailable")
        
    user_id = "demo_user_001"
    
    db["users"].delete_many({"user_id": user_id})
    db["briefs"].delete_many({"user_id": user_id})
    db["comfort_profiles"].delete_many({"user_id": user_id})
    
    demo_user = {
        "user_id": "demo_user_001",
        "name": "Riya",
        "visual": {"monk_scale":"M4","undertone":"warm","face_shape":"oval","hair_color":"dark brown","eye_color":"dark brown"},
        "physical": {"height_cm":163,"body_type":"pear"},
        "taste": {"everyday":["Casual","Minimalist"],"cozy":["Cottagecore"],"fashion_week":["Editorial"],"dream_outfit":["Quiet Luxury"],"color_palette":["earth tones","neutrals"],"occasion":["college","brunch"],"style_icon":"Janhvi Kapoor"},
        "style_equation": {"primary_aesthetic":"Quiet Luxury","secondary_aesthetic":"Minimalist"},
        "meta": {"tier":"free","onboarding_complete":True},
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    db["users"].insert_one(demo_user)
    
    return {"status": "demo ready", "user_id": user_id}


# ══════════════════════════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
