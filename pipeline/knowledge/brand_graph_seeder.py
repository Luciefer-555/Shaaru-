import os
import json
from dotenv import load_dotenv
from neo4j import GraphDatabase

def main():
    # Load environment variables
    load_dotenv()
    
    # Resolve path to designers.json relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.normpath(os.path.join(script_dir, "..", "config", "designers.json"))
    
    if not os.path.exists(config_path):
        fallback_path = os.path.abspath("pipeline/config/designers.json")
        if os.path.exists(fallback_path):
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Cannot find designers.json at {config_path} or {fallback_path}")
            
    print(f"Loading designers from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        brands = json.load(f)
        
    print(f"Loaded {len(brands)} brands.")
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not user or not password:
        raise ValueError("Missing Neo4j credentials in environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)")
        
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    seed_query = """
    UNWIND $brands AS brand
    MERGE (b:Brand {id: brand.id})
    SET b += brand
    WITH b, brand
    MERGE (a:Aesthetic {name: brand.aesthetic_hint})
    MERGE (b)-[:HAS_AESTHETIC]->(a)
    WITH b, brand
    MERGE (r:Region {name: brand.region})
    MERGE (b)-[:LOCATED_IN]->(r)
    WITH b, brand
    MERGE (c:Category {name: brand.category})
    MERGE (b)-[:BELONGS_TO_CATEGORY]->(c)
    """
    
    with driver.session() as session:
        print("Seeding brands into Neo4j...")
        session.run(seed_query, brands=brands)
        
        brand_count = session.run("MATCH (b:Brand) RETURN count(b) AS c").single()["c"]
        aesthetic_count = session.run("MATCH (a:Aesthetic) RETURN count(a) AS c").single()["c"]
        region_count = session.run("MATCH (r:Region) RETURN count(r) AS c").single()["c"]
        category_count = session.run("MATCH (c:Category) RETURN count(c) AS c").single()["c"]
        
        print("\n--- Seeding Results ---")
        print(f"Total Brand nodes created: {brand_count}")
        print(f"Total Aesthetic nodes created: {aesthetic_count}")
        print(f"Total Region nodes created: {region_count}")
        print(f"Total Category nodes created: {category_count}")
        
        print("\nAll Brands: MATCH (b:Brand)-[:HAS_AESTHETIC]->(a) RETURN b.name, a.name ORDER BY b.name")
        sample_res = session.run("MATCH (b:Brand)-[:HAS_AESTHETIC]->(a) RETURN b.name, a.name ORDER BY b.name")
        for record in sample_res:
            print(f"  {record['b.name']} -> {record['a.name']}")
            
    driver.close()

if __name__ == "__main__":
    main()
