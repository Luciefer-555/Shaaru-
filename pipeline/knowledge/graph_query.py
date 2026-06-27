"""
pipeline/knowledge/graph_query.py

Neo4j graph query helpers for Riley's brand-awareness tool.
Traversal patterns:
    Vibe     -[:EXPRESSES_THROUGH]-> Aesthetic <-[:HAS_AESTHETIC]- Brand
    Occasion -[:CALLS_FOR]->         Aesthetic <-[:HAS_AESTHETIC]- Brand
"""

import os

from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired
from dotenv import load_dotenv

load_dotenv()


def _get_driver():
    """Fresh driver on every call - avoids defunct connection errors."""
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )


def get_brands_by_vibe(vibe: str, region: str = None) -> list[dict]:
    """
    Traverse: Vibe -> EXPRESSES_THROUGH -> Aesthetic <- HAS_AESTHETIC <- Brand
    Optionally filter by region.
    Returns list of {name, url, category, aesthetic, region}
    """
    # "India" means no region filter - treat as national query
    if region and region.lower() in ("india", "in", "india."):
        region = None

    for attempt in range(2):
        driver = None
        try:
            driver = _get_driver()
            with driver.session() as session:
                if region:
                    result = session.run(
                        """
                        MATCH (v:Vibe)-[:EXPRESSES_THROUGH]->(a:Aesthetic)
                              <-[:HAS_AESTHETIC]-(b:Brand)
                        WHERE toLower(v.name) CONTAINS toLower($vibe)
                        AND   toLower(b.region) CONTAINS toLower($region)
                        AND   b.active = true
                        RETURN b.name      AS name,
                               b.url       AS url,
                               b.category  AS category,
                               a.name      AS aesthetic,
                               b.region    AS region
                        LIMIT 6
                        """,
                        vibe=vibe,
                        region=region,
                    )
                else:
                    result = session.run(
                        """
                        MATCH (v:Vibe)-[:EXPRESSES_THROUGH]->(a:Aesthetic)
                              <-[:HAS_AESTHETIC]-(b:Brand)
                        WHERE toLower(v.name) CONTAINS toLower($vibe)
                        AND   b.active = true
                        RETURN b.name      AS name,
                               b.url       AS url,
                               b.category  AS category,
                               a.name      AS aesthetic,
                               b.region    AS region
                        LIMIT 8
                        """,
                        vibe=vibe,
                    )
                return [dict(r) for r in result]
        except SessionExpired as e:
            if attempt == 1:
                print(f"[graph_query] get_brands_by_vibe error: {e}")
                return []
        except Exception as e:
            print(f"[graph_query] get_brands_by_vibe error: {e}")
            return []
        finally:
            if driver:
                driver.close()

    return []


def get_brands_by_occasion(occasion: str) -> list[dict]:
    """
    Traverse: Occasion -> CALLS_FOR -> Aesthetic <- HAS_AESTHETIC <- Brand
    """
    for attempt in range(2):
        driver = None
        try:
            driver = _get_driver()
            with driver.session() as session:
                result = session.run(
                    """
                    MATCH (o:Occasion)-[:CALLS_FOR]->(a:Aesthetic)
                          <-[:HAS_AESTHETIC]-(b:Brand)
                    WHERE toLower(o.name) CONTAINS toLower($occasion)
                    AND   b.active = true
                    RETURN b.name      AS name,
                           b.url       AS url,
                           b.category  AS category,
                           a.name      AS aesthetic,
                           b.region    AS region
                    LIMIT 8
                    """,
                    occasion=occasion,
                )
                return [dict(r) for r in result]
        except SessionExpired as e:
            if attempt == 1:
                print(f"[graph_query] get_brands_by_occasion error: {e}")
                return []
        except Exception as e:
            print(f"[graph_query] get_brands_by_occasion error: {e}")
            return []
        finally:
            if driver:
                driver.close()

    return []
