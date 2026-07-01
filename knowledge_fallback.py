"""
knowledge_fallback.py — Four-layer fashion knowledge resolution engine.

Layer 1: Neo4j knowledge graph cache + local catalogs
Layer 2: On-demand graph extraction / knowledge verification
Layer 3: Tavily web search fallback
Layer 4: Model general knowledge fallback

Auto-caches new verified facts back to Neo4j.
Persists fallback audit events to MongoDB.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database import get_db
from pipeline.knowledge.graph_query import run_query, run_write_query
from shaaru_brain import _get_client, MODEL_TEXT

log = logging.getLogger("shaaru.knowledge.fallback")


def _log_fallback_event(user_id: str, query: str, entity_type: str, layer: int, source: str, details: Any = None):
    try:
        db = get_db()
        if db is not None:
            db["knowledge_fallback_events"].insert_one({
                "user_id": user_id,
                "query": query,
                "entity_type": entity_type,
                "layer_resolved": layer,
                "source": source,
                "details": details or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            log.info(f"[KNOWLEDGE FALLBACK] Resolved '{query}' ({entity_type}) at Layer {layer} ({source})")
    except Exception as e:
        log.warning(f"[KNOWLEDGE FALLBACK] Failed to audit event: {e}")


def _cache_to_neo4j(entity_type: str, data: Dict[str, Any]):
    """Write newly discovered facts back to Neo4j so future lookups hit Layer 1."""
    try:
        name = data.get("name")
        if not name:
            return
        if entity_type in ("brand", "designer"):
            cypher = """
            MERGE (b:Brand {name: $name})
            SET b.category = coalesce($category, b.category, 'designer'),
                b.aesthetic_hint = coalesce($description, b.aesthetic_hint, ''),
                b.region = coalesce($region, b.region, 'India'),
                b.active = true,
                b.source = coalesce($source, 'fallback_engine')
            RETURN b.name
            """
            run_write_query(cypher, {
                "name": name,
                "category": data.get("category", "designer"),
                "description": data.get("description", ""),
                "region": data.get("region", "India"),
                "source": data.get("source", "fallback")
            })
        elif entity_type in ("aesthetic", "vibe"):
            cypher = """
            MERGE (a:Aesthetic {name: $name})
            SET a.description = coalesce($description, a.description, ''),
                a.source = coalesce($source, 'fallback_engine')
            RETURN a.name
            """
            run_write_query(cypher, {
                "name": name,
                "description": data.get("description", ""),
                "source": data.get("source", "fallback")
            })
        elif entity_type == "fabric":
            cypher = """
            MERGE (f:Fabric {name: $name})
            SET f.best_for = coalesce($description, f.best_for, ''),
                f.source = coalesce($source, 'fallback_engine')
            RETURN f.name
            """
            run_write_query(cypher, {
                "name": name,
                "description": data.get("description", ""),
                "source": data.get("source", "fallback")
            })
    except Exception as e:
        log.warning(f"[KNOWLEDGE FALLBACK] Failed to cache to Neo4j: {e}")


def resolve_fashion_knowledge(query: str, entity_type: str = "general", user_id: str = "system") -> Dict[str, Any]:
    """
    Execute 4-layer resolution for a fashion query (brand, designer, fabric, aesthetic).
    """
    q_clean = query.strip().lower()
    if not q_clean:
        return {"error": "Empty query"}

    # ─────────────────────────────────────────────────────────────
    # Layer 1: Neo4j knowledge graph cache + local designers.json
    # ─────────────────────────────────────────────────────────────
    try:
        results = []
        if entity_type in ("brand", "designer"):
            cypher = """
            MATCH (b:Brand)
            WHERE toLower(b.name) CONTAINS $q OR toLower(b.category) CONTAINS $q OR toLower(coalesce(b.aesthetic_hint, '')) CONTAINS $q
            RETURN b.name AS name, b.url AS url, b.category AS category, b.region AS region, b.aesthetic_hint AS description
            LIMIT 5
            """
            results = run_query(cypher, {"q": q_clean})
            
            # Also check designers.json local catalog
            if not results:
                designers_path = os.path.join(os.path.dirname(__file__), 'pipeline', 'config', 'designers.json')
                if os.path.exists(designers_path):
                    with open(designers_path, encoding='utf-8') as f:
                        catalog = json.load(f)
                    for d in catalog:
                        if not d.get('active', True):
                            continue
                        if (q_clean in d.get('name', '').lower() or
                            q_clean in d.get('aesthetic_hint', '').lower() or
                            q_clean in d.get('category', '').lower() or
                            q_clean in d.get('id', '').lower()):
                            results.append({
                                "name": d.get("name"),
                                "category": d.get("category"),
                                "description": d.get("aesthetic_hint"),
                                "region": d.get("region", "India")
                            })
        elif entity_type in ("aesthetic", "vibe"):
            cypher = """
            MATCH (a:Aesthetic)
            WHERE toLower(a.name) CONTAINS $q OR toLower(coalesce(a.description, '')) CONTAINS $q
            RETURN a.name AS name, a.description AS description
            LIMIT 5
            """
            results = run_query(cypher, {"q": q_clean})
            if not results:
                cypher_vibe = """
                MATCH (v:Vibe)-[:EXPRESSES_THROUGH]->(a:Aesthetic)
                WHERE toLower(v.name) CONTAINS $q
                RETURN a.name AS name, a.description AS description
                LIMIT 5
                """
                results = run_query(cypher_vibe, {"q": q_clean})
        elif entity_type == "fabric":
            cypher = """
            MATCH (f:Fabric)
            WHERE toLower(f.name) CONTAINS $q
            RETURN f.name AS name, f.gsm_range AS gsm_range, f.drape_score AS drape_score, f.best_for AS description
            LIMIT 5
            """
            results = run_query(cypher, {"q": q_clean})

        if results:
            _log_fallback_event(user_id, query, entity_type, 1, "neo4j_cache", {"count": len(results)})
            return {"status": "resolved", "layer": 1, "source": "neo4j_cache", "results": results}
    except Exception as e:
        log.warning(f"[Layer 1] Neo4j cache check failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # Layer 2: On-demand graph extraction / local verification DBs
    # ─────────────────────────────────────────────────────────────
    try:
        db = get_db()
        if db is not None:
            # Check MongoDB collections for pre-harvested knowledge
            coll_name = "products" if entity_type in ("brand", "designer") else ("trends" if entity_type in ("aesthetic", "vibe") else "fabrics")
            if coll_name in db.list_collection_names():
                found = list(db[coll_name].find({"$text": {"$search": query}}, {"_id": 0}).limit(3))
                if found:
                    extracted = []
                    for item in found:
                        rec = {
                            "name": item.get("name") or item.get("product_name") or item.get("title") or query,
                            "description": item.get("description") or item.get("details") or str(item),
                            "category": item.get("category", entity_type)
                        }
                        extracted.append(rec)
                        _cache_to_neo4j(entity_type, {**rec, "source": "layer2_extraction"})
                    _log_fallback_event(user_id, query, entity_type, 2, "graph_extraction", {"count": len(extracted)})
                    return {"status": "resolved", "layer": 2, "source": "graph_extraction", "results": extracted}
    except Exception as e:
        log.warning(f"[Layer 2] On-demand extraction failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # Layer 3: Tavily Web Search
    # ─────────────────────────────────────────────────────────────
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from tavily import TavilyClient
            tclient = TavilyClient(api_key=tavily_key)
            search_query = f"{query} fashion {entity_type} India clothing brand style fabric"
            t_res = tclient.search(search_query, max_results=3)
            snippets = [r.get("content", "") for r in t_res.get("results", []) if r.get("content")]
            
            if snippets:
                client = _get_client()
                prompt = f"""Extract precise fashion details about '{query}' ({entity_type}) based on these search results:
{chr(10).join(snippets[:3])}

Return a single JSON object with:
- "name": proper fashion entity name
- "description": concise 1-2 sentence description
- "category": specific category or classification
"""
                raw = client.chat.completions.create(
                    model=MODEL_TEXT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
                content = raw.choices[0].message.content
                parsed = json.loads(content)
                if parsed and parsed.get("name"):
                    parsed["source"] = "tavily_web_search"
                    _cache_to_neo4j(entity_type, parsed)
                    _log_fallback_event(user_id, query, entity_type, 3, "tavily_web_search", parsed)
                    return {"status": "resolved", "layer": 3, "source": "tavily_web_search", "results": [parsed]}
        except Exception as e:
            log.warning(f"[Layer 3] Tavily search failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # Layer 4: Model General Knowledge Fallback
    # ─────────────────────────────────────────────────────────────
    try:
        client = _get_client()
        prompt = f"""You are a professional fashion curator and textile expert. Provide factual, expert fashion knowledge about '{query}' as a '{entity_type}'.
Return ONLY a JSON object:
{{
  "name": "{query}",
  "description": "2-sentence authoritative analysis of aesthetic, construction, brand heritage, or textile behavior.",
  "category": "{entity_type}"
}}"""
        raw = client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        content = raw.choices[0].message.content
        parsed = json.loads(content)
        if not parsed.get("name"):
            parsed["name"] = query
        parsed["source"] = "model_general_knowledge"
        
        # Cache back to Neo4j so future lookups hit Layer 1
        _cache_to_neo4j(entity_type, parsed)
        _log_fallback_event(user_id, query, entity_type, 4, "model_general_knowledge", parsed)
        return {"status": "resolved", "layer": 4, "source": "model_general_knowledge", "results": [parsed]}
    except Exception as e:
        log.error(f"[Layer 4] General knowledge fallback failed: {e}")
        return {"status": "error", "message": "Could not resolve fashion knowledge"}
