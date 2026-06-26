"""
shaaru_tools.py — Riley tool-calling definitions and executors.

Defines four tools in OpenAI/NIM function-calling format.
Each tool has:
  TOOL_SCHEMAS  — JSON schema sent to the model (tools= array)
  execute_tool  — dispatches tool_call names to Python functions

Tools:
  search_products          — MongoDB products query (wraps _get_product_recommendations)
  get_behavioral_signals   — Neo4j behavioral edge query
  trigger_tailor_flow      — gate only, does NOT call tailor_engine
  search_products_semantic — MongoDB Atlas vector search (from product_embeddings.py)

Usage:
  from shaaru_tools import TOOL_SCHEMAS, execute_tool
"""

import json
import logging

log = logging.getLogger("shaaru.tools")

# ══════════════════════════════════════════════════════════════════
#  TOOL SCHEMAS — sent to the model in tools= array
# ══════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the SHAARU product catalog for items matching the user's "
                "style profile and a specific query. Returns up to 5 products with "
                "name, brand, color, silhouette, and price. Use when the user asks "
                "to find, show, or recommend products."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The fashion item or style the user is looking for, e.g. 'linen blazer', 'kurta for wedding'."
                    },
                    "monk_scale": {
                        "type": "integer",
                        "description": "User's Monk skin tone scale (1–10). Use the value from the user profile context if known.",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "body_type": {
                        "type": "string",
                        "description": "User's body type from their profile (e.g. 'pear', 'hourglass', 'athletic'). Use from context if known."
                    },
                    "occasion": {
                        "type": "string",
                        "description": "Target occasion for the item, e.g. 'casual', 'wedding', 'brunch', 'college'. Optional."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_behavioral_signals",
            "description": (
                "Fetch this user's recent behavioral signals from the knowledge graph: "
                "which products they have saved, skipped, and purchased, with weighted "
                "recency. Use to understand what the user has been gravitating toward "
                "before making a recommendation or discussing their taste."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's unique identifier."
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_tailor_flow",
            "description": (
                "Signal that the user wants to have a garment made or tailored. "
                "This does NOT start construction — it returns a confirmation payload "
                "so the user can be asked to confirm before proceeding. "
                "Use when the user says 'make this', 'I want this made', 'stitch', "
                "'tailor', 'recreate', 'build me', or similar intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's unique identifier."
                    },
                    "garment_description": {
                        "type": "string",
                        "description": "A short description of the garment the user wants made, e.g. 'that embroidered kurta' or 'the linen blazer in the image'."
                    }
                },
                "required": ["user_id", "garment_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products_semantic",
            "description": (
                "Search for products using semantic (vector) similarity against a natural "
                "language query. More expressive than exact-match — good for queries like "
                "'minimalist linen shirt', 'oversized streetwear', 'earthy toned ethnic wear'. "
                "Use when the user describes an aesthetic or feel rather than a specific item name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the desired product style, e.g. 'minimalist linen shirt'."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user's unique identifier. Used for logging only."
                    }
                },
                "required": ["query", "user_id"]
            }
        }
    }
]


# ══════════════════════════════════════════════════════════════════
#  TOOL EXECUTORS
# ══════════════════════════════════════════════════════════════════

def _exec_search_products(args: dict) -> dict:
    """
    Wraps _get_product_recommendations() from shaaru_brain.py.
    Queries MongoDB products collection with profile-aware filters.
    """
    from shaaru_brain import _get_db

    query    = args.get("query", "")
    monk     = args.get("monk_scale")
    body     = args.get("body_type", "")
    occasion = args.get("occasion", "")

    try:
        db = _get_db()

        # Build MongoDB filter
        mongo_query: dict = {"availability.in_stock": True}
        if monk:
            mongo_query["compatibility.monk_scales"] = monk
        if body:
            mongo_query["compatibility.body_types"] = body
        if occasion:
            mongo_query["compatibility.occasions"] = occasion

        # Keyword filter from query
        if query:
            import re
            kw = re.escape(query)
            mongo_query["$or"] = [
                {"category":     {"$regex": kw, "$options": "i"}},
                {"product_name": {"$regex": kw, "$options": "i"}},
                {"aesthetic":    {"$regex": kw, "$options": "i"}},
                {"tags":         {"$regex": kw, "$options": "i"}},
            ]

        products = list(
            db["products"]
            .find(mongo_query, {
                "product_name": 1, "brand": 1,
                "pricing.price_inr": 1, "color": 1,
                "silhouette": 1, "aesthetic": 1,
                "product_url": 1,
            })
            .limit(5)
        )

        # Fallback — drop profile filters, keep keyword only
        if not products and query:
            import re
            kw = re.escape(query)
            products = list(
                db["products"]
                .find(
                    {"$or": [
                        {"category":     {"$regex": kw, "$options": "i"}},
                        {"product_name": {"$regex": kw, "$options": "i"}},
                        {"aesthetic":    {"$regex": kw, "$options": "i"}},
                    ]},
                    {"product_name": 1, "brand": 1,
                     "pricing.price_inr": 1, "color": 1,
                     "silhouette": 1, "aesthetic": 1}
                )
                .limit(5)
            )

        # Final fallback — return top-5 in-stock products
        if not products:
            products = list(
                db["products"]
                .find({"availability.in_stock": True},
                      {"product_name": 1, "brand": 1,
                       "pricing.price_inr": 1, "color": 1,
                       "silhouette": 1, "aesthetic": 1})
                .limit(5)
            )

        result = []
        for p in products:
            p.pop("_id", None)
            result.append({
                "name":      p.get("product_name", ""),
                "brand":     p.get("brand", ""),
                "color":     p.get("color", ""),
                "silhouette": p.get("silhouette", ""),
                "aesthetic": p.get("aesthetic", ""),
                "price_inr": p.get("pricing", {}).get("price_inr", 0),
            })

        return {"products": result, "count": len(result), "query": query}

    except Exception as e:
        log.error(f"[TOOL] search_products failed: {e}")
        return {"products": [], "count": 0, "error": str(e)}


def _exec_get_behavioral_signals(args: dict) -> dict:
    """
    Wraps _query_behavioral_edges() from shaaru_brain.py.
    Pulls top weighted SAVED/PURCHASED/SKIPPED edges from Neo4j.
    """
    user_id = args.get("user_id", "")
    try:
        from shaaru_brain import _query_behavioral_edges
        edges = _query_behavioral_edges(user_id)
        return {
            "user_id": user_id,
            "behavioral_signals": edges,
            "count": len(edges),
        }
    except Exception as e:
        log.error(f"[TOOL] get_behavioral_signals failed: {e}")
        return {"user_id": user_id, "behavioral_signals": [], "error": str(e)}


def _exec_trigger_tailor_flow(args: dict) -> dict:
    """
    Gate only — does NOT call tailor_engine.
    Returns a confirmation payload so the frontend can prompt the user.
    """
    user_id     = args.get("user_id", "")
    description = args.get("garment_description", "")
    return {
        "action":   "confirm_tailor",
        "user_id":  user_id,
        "garment":  description,
        "message":  (
            f"Ready to build your brief for '{description}'. "
            "Confirm and upload a reference image to start."
        ),
    }


def _exec_search_products_semantic(args: dict) -> dict:
    """
    Wraps search_products_semantic() from product_embeddings.py.
    Uses nvidia/nv-embedqa-e5-v5 + MongoDB Atlas $vectorSearch.
    """
    query   = args.get("query", "")
    user_id = args.get("user_id", "")
    try:
        from product_embeddings import search_products_semantic
        raw = search_products_semantic(query, limit=5)

        result = []
        for p in raw:
            p.pop("_id", None)
            result.append({
                "name":      p.get("product_name", ""),
                "brand":     p.get("brand", ""),
                "color":     p.get("color", ""),
                "aesthetic": p.get("aesthetic", ""),
                "score":     round(p.get("score", 0.0), 3),
            })

        return {"products": result, "count": len(result), "query": query}

    except Exception as e:
        log.error(f"[TOOL] search_products_semantic failed: {e}")
        return {"products": [], "count": 0, "error": str(e)}


# ── Dispatch table ───────────────────────────────────────────────

_EXECUTORS = {
    "search_products":          _exec_search_products,
    "get_behavioral_signals":   _exec_get_behavioral_signals,
    "trigger_tailor_flow":      _exec_trigger_tailor_flow,
    "search_products_semantic": _exec_search_products_semantic,
}


def execute_tool(tool_name: str, arguments_json: str) -> str:
    """
    Parse arguments, dispatch to the correct executor, and return
    the result as a JSON string (suitable for a tool-role message).

    Args:
        tool_name:       Name of the function the model called.
        arguments_json:  Raw arguments string from tool_call.function.arguments.

    Returns:
        JSON-encoded result string.
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        log.warning(f"[TOOL] JSON parse failed for '{tool_name}' args: {e}")
        args = {}

    executor = _EXECUTORS.get(tool_name)
    if executor is None:
        log.warning(f"[TOOL] Unknown tool called: {tool_name}")
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    log.info(f"[TOOL] Executing '{tool_name}' with args: {args}")
    result = executor(args)
    log.info(f"[TOOL] '{tool_name}' result: {result}")

    return json.dumps(result, ensure_ascii=False, default=str)
