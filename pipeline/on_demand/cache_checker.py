"""
Before extracting anything — check if we already know it.
Neo4j is the cache. If product is there with caption,
return it instantly. No extraction needed.
"""

import os
import sys
import re

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from knowledge_graph import KnowledgeGraph

kg = None


def get_kg():
    global kg
    if kg is None:
        kg = KnowledgeGraph()
    return kg

_PRODUCT_QUERY_SKIP = frozenset({
    "the", "this", "that", "from", "with", "want", "made", "make", "get",
    "tell", "give", "full", "best", "friend", "wedding", "everything",
    "what", "where", "how", "like", "love", "called", "work", "technique",
    "materials", "india", "source", "tailor", "brief", "actually", "https",
    "http", "www", "com", "products", "official", "insane", "something",
    "exactly", "fabric", "mirror", "sherwani", "abhinav", "mishra",
})


def _extract_product_handles(query: str) -> list[str]:
    """Pull URL handles and explicit product names from the user query."""
    handles: list[str] = []
    for m in re.finditer(r"/products/([a-z0-9\-]+)", query, re.I):
        handles.append(m.group(1).lower())
    for m in re.finditer(r"\bcalled\s+([a-z0-9\-]+)", query, re.I):
        handles.append(m.group(1).lower())
    for m in re.finditer(r"\b([A-Z]{2,}[A-Z0-9\-]*)\b", query):
        handles.append(m.group(1).lower())
    deduped = []
    for h in handles:
        if h not in _PRODUCT_QUERY_SKIP and len(h) >= 3 and h not in deduped:
            deduped.append(h)
    return deduped


def _product_return_query():
    return """
        MATCH (p:Product)
        WHERE p.caption IS NOT NULL
          AND p.caption <> '{}'
          AND (
            toLower(p.title) CONTAINS $handle
            OR toLower(p.source_url) CONTAINS $handle
          )
        WITH p LIMIT 5
        OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(a:Aesthetic)
        OPTIONAL MATCH (p)-[:SUITS_OCCASION]->(o:Occasion)
        RETURN p.title as title,
               p.source_url as url,
               p.image_url as image,
               p.caption as caption,
               p.designer as designer,
               p.source_id as source_id,
               collect(distinct o.name) as occasions,
               a.name as aesthetic
    """


def check_cache(
    query: str,
    techniques: list = None,
    aesthetic: str = None,
    designer_id: str = None
) -> dict:
    """
    Checks Neo4j for existing knowledge matching the query.
    Returns cached results if found.
    """
    if not get_kg().is_connected:
        return {"cache_hit": False}
        
    try:
        with get_kg().driver.session() as session:
            # Strategy 0 — title / URL handle match (e.g. "raqs", "shrine")
            for handle in _extract_product_handles(query):
                result = session.run(_product_return_query(), handle=handle)
                rows = [dict(r) for r in result]
                if rows:
                    return {
                        "cache_hit": True,
                        "match_type": "product_handle",
                        "matched_on": handle,
                        "products": rows,
                    }

            # Strategy 1 — technique match
            if techniques:
                for technique in techniques:
                    result = session.run("""
                        MATCH (p:Product)-[:HAS_TECHNIQUE]->(t:Technique)
                        WHERE toLower(t.name) CONTAINS toLower($technique)
                        AND p.caption IS NOT NULL
                        AND p.caption <> '{}'
                        WITH p LIMIT 5
                        OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
                        OPTIONAL MATCH (p)-[:BELONGS_TO]->(a:Aesthetic)
                        OPTIONAL MATCH (p)-[:SUITS_OCCASION]->(o:Occasion)
                        RETURN p.title as title,
                               p.source_url as url,
                               p.image_url as image,
                               p.caption as caption,
                               p.designer as designer,
                               p.source_id as source_id,
                               collect(distinct o.name) as occasions,
                               a.name as aesthetic
                    """, technique=technique)
                    
                    rows = [dict(r) for r in result]
                    if rows:
                        return {
                            "cache_hit": True,
                            "match_type": "technique",
                            "matched_on": technique,
                            "products": rows
                        }
            
            # Strategy 2 — aesthetic match
            if aesthetic:
                result = session.run("""
                    MATCH (p:Product)-[:BELONGS_TO]->(a:Aesthetic)
                    WHERE toLower(a.name) CONTAINS toLower($aesthetic)
                    AND p.caption IS NOT NULL
                    WITH p, a LIMIT 5
                    OPTIONAL MATCH (p)-[:SUITS_OCCASION]->(o:Occasion)
                    RETURN p.title as title,
                           p.source_url as url,
                           p.image_url as image,
                           p.caption as caption,
                           p.designer as designer,
                           a.name as aesthetic,
                           collect(distinct o.name) as occasions
                """, aesthetic=aesthetic)
                
                rows = [dict(r) for r in result]
                if rows:
                    return {
                        "cache_hit": True,
                        "match_type": "aesthetic",
                        "matched_on": aesthetic,
                        "products": rows
                    }
            
            # Strategy 3 — keyword in title
            result = session.run("""
                MATCH (p:Product)
                WHERE toLower(p.title) CONTAINS toLower($query)
                AND p.caption IS NOT NULL
                WITH p LIMIT 5
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(a:Aesthetic)
                RETURN p.title as title,
                       p.source_url as url,
                       p.image_url as image,
                       p.caption as caption,
                       p.designer as designer,
                       a.name as aesthetic
            """, query=query)
            
            rows = [dict(r) for r in result]
            if rows:
                return {
                    "cache_hit": True,
                    "match_type": "keyword",
                    "matched_on": query,
                    "products": rows
                }
    except Exception as e:
        print(f"[CacheChecker] Neo4j query error: {e}")
        
    return {"cache_hit": False}
