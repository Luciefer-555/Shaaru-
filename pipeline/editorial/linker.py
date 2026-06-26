"""
Links editorial articles to known designers, aesthetics, and products in Neo4j.
Enriches the Knowledge Graph with editorial backing.
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()


def link_articles_to_kg(limit: int = 100):
    client = MongoClient(os.environ["MONGODB_URI"])
    db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
    col = db["editorial"]
    
    if not kg.is_connected:
        print("[Linker] Neo4j not connected", flush=True)
        return
        
    articles = list(col.find({"linked_to_kg": {"$ne": True}}).limit(limit))
    print(f"Linking {len(articles)} editorial articles to Neo4j...", flush=True)
    
    with kg.driver.session() as session:
        for article in articles:
            text = (article.get("title", "") + " " + article.get("content", "")).lower()
            
            # Match designers
            designers = ["abhinav mishra", "sabyasachi", "raw mango", "masaba", "torani", "anavila", "pero", "rimzim dadu", "anita dongre", "injiri"]
            for d in designers:
                if d in text:
                    session.run("""
                        MERGE (e:Editorial {url: $url})
                        SET e.title = $title, e.source = $source
                        WITH e
                        MATCH (b:Brand) WHERE toLower(b.name) CONTAINS $brand
                        MERGE (e)-[:MENTIONS_BRAND]->(b)
                    """, url=article["url"], title=article["title"], source=article["source"], brand=d)
            
            col.update_one({"_id": article["_id"]}, {"$set": {"linked_to_kg": True}})
            print(f"  [OK] Linked article: {article.get('title', '')[:40]}", flush=True)
            
    print("Editorial linking complete", flush=True)


def link_editorial_to_graph(limit: int = 100):
    link_articles_to_kg(limit=limit)
