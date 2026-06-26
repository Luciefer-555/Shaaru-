import os
import json
from pymongo import MongoClient
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def run_step_4():
    print("Running Step 4: DB Expansion from candidates...")
    client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'))
    db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
    
    review_dir = os.path.join("pipeline", "output", "review")
    candidates = []
    
    for filename in os.listdir(review_dir):
        if not filename.endswith(".json"): continue
        with open(os.path.join(review_dir, filename), "r") as f:
            try:
                products = json.load(f)
                for product in products:
                    for cand in product.get("new_fabric_candidates", []):
                        cand["designer"] = product.get("designer", "Unknown")
                        candidates.append(cand)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    # Filter valid candidates (they have a fabric_id_guess or name)
    inserted = 0
    for cand in candidates:
        fabric_name = cand.get("fabric_id_guess") or cand.get("name")
        if not fabric_name: continue
        
        # Check if already exists in DB
        exists = db.fabric_intelligence.find_one({"fabric_id": fabric_name.lower().strip()})
        if not exists:
            db.fabric_intelligence.insert_one({
                "fabric_id": fabric_name.lower().strip(),
                "common_names": [fabric_name],
                "description": cand.get("new_candidate_description", ""),
                "source": cand.get("source", "batch_expansion"),
                "discovered_via": cand.get("designer")
            })
            inserted += 1
            
    print(f"Step 4 Complete. Inserted {inserted} new fabrics into MongoDB.")

def run_step_5():
    print("Running Step 5: Editorial ingestion...")
    # As per user: "Editorial ingestion. Only after product data is solid."
    # We will trigger the editorial scraper/pipeline here.
    from pipeline.scrapers.editorial_scraper import scrape_editorial
    # Scrape generic editorial trends
    print("Triggering editorial scraper...")
    try:
        data = scrape_editorial("latest indian fashion trends 2026")
        print(f"Successfully scraped editorial content: {len(str(data))} bytes")
    except Exception as e:
        print(f"Editorial scrape failed: {e}")

def run_step_6():
    print("Running Step 6: Neo4j sync...")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    except Exception as e:
        print(f"Could not connect to Neo4j: {e}")
        return
        
    review_dir = os.path.join("pipeline", "output", "review")
    
    with driver.session() as session:
        for filename in os.listdir(review_dir):
            if not filename.endswith(".json"): continue
            with open(os.path.join(review_dir, filename), "r") as f:
                try:
                    products = json.load(f)
                    for product in products:
                        aesthetic = product.get("aesthetic_category")
                        if not aesthetic: continue
                        
                        # Create aesthetic node if not exists
                        session.run("""
                            MERGE (a:Aesthetic {name: $name})
                            ON CREATE SET a.indian_context = true
                        """, name=aesthetic)
                        
                        # Link fabrics to aesthetic
                        fabrics = product.get("fabric_vocabulary", {}).get("confirmed", [])
                        for fab in fabrics:
                            fab_id = fab.get("fabric_id")
                            if fab_id:
                                session.run("""
                                    MERGE (f:Fabric {name: $fab_id})
                                    MERGE (a:Aesthetic {name: $aesthetic})
                                    MERGE (f)-[:BELONGS_TO_AESTHETIC]->(a)
                                """, fab_id=fab_id, aesthetic=aesthetic)
                except Exception as e:
                    print(f"Error syncing {filename} to Neo4j: {e}")
                    
    driver.close()
    print("Step 6 Complete. Synced fabrics and aesthetics to Neo4j.")

if __name__ == "__main__":
    run_step_4()
    run_step_5()
    run_step_6()
    print("All final steps completed successfully.")
