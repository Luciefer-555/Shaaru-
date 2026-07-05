"""
pipeline/knowledge/graph_query.py

Neo4j graph query helpers for Riley's brand-awareness tool.
"""

import os
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired
from dotenv import load_dotenv

load_dotenv()

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri:
            return None
        _driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
            connection_timeout=10
        )
    return _driver

def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None

def run_query(cypher: str, parameters: dict = None) -> list:
    driver = get_driver()
    if not driver:
        return []
    for attempt in range(2):
        try:
            with driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [dict(record) for record in result]
        except SessionExpired:
            if attempt == 1:
                raise
        except Exception as e:
            print(f"[graph_query] query error: {e}")
            return []
    return []

def run_write_query(cypher: str, parameters: dict = None):
    driver = get_driver()
    if not driver:
        return None
    for attempt in range(2):
        try:
            with driver.session() as session:
                return session.run(cypher, parameters or {})
        except SessionExpired:
            if attempt == 1:
                raise
        except Exception as e:
            print(f"[graph_query] write query error: {e}")
            return None
    return None

def get_brands_by_vibe(vibe: str, region: str = None) -> list[dict]:
    if region and region.lower() in ("india", "in", "india."):
        region = None
    if region:
        cypher = """
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
        """
        return run_query(cypher, {"vibe": vibe, "region": region})
    else:
        cypher = """
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
        """
        return run_query(cypher, {"vibe": vibe})

def get_brands_by_occasion(occasion: str) -> list[dict]:
    cypher = """
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
    """
    return run_query(cypher, {"occasion": occasion})
