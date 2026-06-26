"""
face_analysis.py — Face analysis using NVIDIA vision models.

Analyzes face images to extract physical profile data:
face shape, skin tone (Monk scale), hair type, eye color, body type.
"""

import os
import json
import base64
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
log = logging.getLogger("shaaru.face")

# ── NVIDIA client (lazy) ─────────────────────────────────────────
_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set.")
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
    return _client

# ── Models ───────────────────────────────────────────────────────
MODEL_VISION = "meta/llama-3.2-11b-vision-instruct"
MODEL_VISION_FALLBACK = "meta/llama-3.2-90b-vision-instruct"

# ── Retry import ─────────────────────────────────────────────────
try:
    from shaaru_retry import nvidia_call
except (ImportError, SyntaxError, ValueError):
    def nvidia_call(client, model, messages, max_tokens=1024,
                    temperature=0.7, **kwargs):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return resp.choices[0].message.content

# ── MongoDB (lazy) ───────────────────────────────────────────────
_db = None

def _get_db():
    global _db
    if _db is None:
        try:
            from pymongo import MongoClient
            uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGODB_DB", "shaaru")
            _db = MongoClient(uri, serverSelectionTimeoutMS=3000)[db_name]
        except Exception as e:
            log.error(f"MongoDB connection failed: {e}")
            return None
    return _db


# ══════════════════════════════════════════════════════════════════
#  FaceProfile dataclass
# ══════════════════════════════════════════════════════════════════

@dataclass
class FaceProfile:
    face_shape: Optional[str] = None
    skin_tone: Optional[str] = None
    skin_tone_label: Optional[str] = None
    monk_scale: Optional[int] = None        # 1-10
    hair_type: Optional[str] = None
    eye_color: Optional[str] = None
    body_type: Optional[str] = None
    confidence_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FaceProfile":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ══════════════════════════════════════════════════════════════════
#  Vision analysis prompt
# ══════════════════════════════════════════════════════════════════

FACE_ANALYSIS_PROMPT = """Analyze this person's physical features for fashion styling purposes.
Return ONLY a JSON object with these exact fields (no markdown, no explanation):

{
  "face_shape": "oval|round|square|heart|oblong|diamond",
  "skin_tone": "hex color code like #D4A574",
  "skin_tone_label": "fair|light|medium|olive|tan|brown|deep brown|dark",
  "monk_scale": 1-10 integer (1=lightest, 10=darkest),
  "hair_type": "straight|wavy|curly|coily|bald",
  "eye_color": "brown|dark brown|hazel|green|blue|black|amber",
  "body_type": "ectomorph|mesomorph|endomorph|athletic|pear|apple|hourglass|rectangle",
  "confidence_score": 0.0-1.0 float (how confident you are in this analysis)
}

Be specific and accurate. If you cannot determine a feature, use your best estimate
and lower the confidence_score accordingly."""


# ══════════════════════════════════════════════════════════════════
#  Core analysis functions
# ══════════════════════════════════════════════════════════════════

def _parse_vision_response(response_text: str) -> dict:
    """Extract JSON from the vision model's response."""
    text = response_text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            clean = part.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                continue

    # Try finding { ... } in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    log.warning(f"Could not parse vision response: {text[:200]}")
    return {}


def _call_vision_model(image_b64: str) -> dict:
    """Send image to NVIDIA vision model and parse response."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FACE_ANALYSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
            ],
        }
    ]

    client = _get_client()

    # Try primary model
    try:
        response = nvidia_call(
            client=client,
            model=MODEL_VISION,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        parsed = _parse_vision_response(response)
        if parsed:
            return parsed
    except Exception as e:
        log.warning(f"Primary vision model failed: {e}")

    # Fallback to 90b
    try:
        log.info("[FACE] Falling back to 90b vision model...")
        response = nvidia_call(
            client=client,
            model=MODEL_VISION_FALLBACK,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return _parse_vision_response(response)
    except Exception as e:
        log.error(f"Vision fallback also failed: {e}")
        return {}


def _save_to_mongo(user_id: str, profile: FaceProfile) -> bool:
    """Save face profile to MongoDB."""
    db = _get_db()
    if db is None:
        return False
    try:
        data = profile.to_dict()
        data["user_id"] = user_id
        db["face_profiles"].update_one(
            {"user_id": user_id},
            {"$set": data},
            upsert=True,
        )
        log.info(f"[FACE] Saved face profile for {user_id}")
        return True
    except Exception as e:
        log.error(f"[FACE] MongoDB save failed: {e}")
        return False


def analyze_face(image_path: str, user_id: str) -> FaceProfile:
    """
    Analyze a face image from a file path.

    Args:
        image_path: Path to the image file.
        user_id:    User identifier for saving results.

    Returns:
        FaceProfile with detected features.
    """
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        log.error(f"[FACE] Image file not found: {image_path}")
        return FaceProfile(confidence_score=0.0)
    except Exception as e:
        log.error(f"[FACE] Error reading image: {e}")
        return FaceProfile(confidence_score=0.0)

    return analyze_face_b64(image_b64, user_id)


def analyze_face_b64(image_b64: str, user_id: str) -> FaceProfile:
    """
    Analyze a face image from base64-encoded data.

    Args:
        image_b64: Base64-encoded image string.
        user_id:   User identifier for saving results.

    Returns:
        FaceProfile with detected features.
    """
    result = _call_vision_model(image_b64)

    if not result:
        log.warning(f"[FACE] Vision analysis returned empty for {user_id}")
        return FaceProfile(confidence_score=0.0)

    # Validate monk_scale
    monk = result.get("monk_scale")
    if monk is not None:
        try:
            monk = int(monk)
            monk = max(1, min(10, monk))
        except (ValueError, TypeError):
            monk = None
    result["monk_scale"] = monk

    # Validate confidence
    conf = result.get("confidence_score", 0.5)
    try:
        conf = float(conf)
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = 0.5
    result["confidence_score"] = conf

    profile = FaceProfile.from_dict(result)

    # Save to MongoDB
    _save_to_mongo(user_id, profile)

    log.info(
        f"[FACE] Analysis complete for {user_id}: "
        f"shape={profile.face_shape}, tone={profile.skin_tone_label}, "
        f"confidence={profile.confidence_score:.2f}"
    )

    return profile
