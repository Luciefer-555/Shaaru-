import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

def main():
    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not user or not password:
        raise ValueError("Missing Neo4j credentials in environment variables")
        
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    vibes_data = [
        {"name": "maximalist", "aesthetics": ["Mirror Maximalism", "Folk Maximalist", "Quirky Maximalist", "Maximalist Prints Embroidery", "Graphic Pop Indian", "Leopard Print Scarves Accessories"]},
        {"name": "minimal", "aesthetics": ["Handloom Minimal", "Contemporary Indian Linen Drape", "Premium Clean Fits Not Streetwear", "Clean Basics Henley Tees"]},
        {"name": "streetwear", "aesthetics": ["Japanese Inspired Indian Streetwear", "Mumbai Graphic Streetwear", "High Fashion Streetwear", "Limited Drop Genderless", "Mumbai Streetwear Store", "Darkwear Band Tees Subculture", "Craft Streetwear", "Indian Handcrafted Shoes Streetwear"]},
        {"name": "avant_garde", "aesthetics": ["Desi Avant-Garde", "Avant-Garde", "Sculptural Couture", "Tech Futurist Couture", "Post Sneaker Era Fashion Footwear"]},
        {"name": "ethnic", "aesthetics": ["Heritage Couture", "Artisan Craft", "Folk Maximalist", "Festive Occasion", "Ethno-Western Sustainable Menswear", "Psychedelic Indian Print", "Contemporary Handcrafted Shirts"]},
        {"name": "editorial", "aesthetics": ["Editorial Indian Menswear", "Sculptural Couture", "Tech Futurist Couture", "Heritage Bridal Couture"]},
        {"name": "clean", "aesthetics": ["Minimalist Indian Streetwear Denim", "Contemporary Indian Menswear", "Made In India Apparel", "Affordable Women's Western Everyday Wear"]},
        {"name": "genderfluid", "aesthetics": ["Indian Genderfluid", "Limited Drop Genderless"]},
        {"name": "dark", "aesthetics": ["Dark Feminine Clothing", "Darkwear Band Tees Subculture"]},
        {"name": "handcrafted", "aesthetics": ["Artisan Craft", "Contemporary Handcrafted Shirts", "Handmade Indian Jewellery Apparel", "Indian Handcrafted Shoes Streetwear", "Urban Handmade Accessories Wallets"]},
        {"name": "fast_fashion", "aesthetics": ["Rapid Drop Fast Fashion Women", "Budget Casual Western Wear", "Gen Z D2C Fashion + Footwear (Shark Tank S3)", "Affordable Women's Western Everyday Wear"]}
    ]
    
    occasions_data = [
        {"name": "wedding", "aesthetics": ["Heritage Couture", "Heritage Bridal Couture", "Mirror Maximalism", "Sculptural Couture"]},
        {"name": "festive", "aesthetics": ["Mirror Maximalism", "Folk Maximalist", "Festive Occasion", "Quirky Maximalist", "Graphic Pop Indian", "Psychedelic Indian Print", "Heritage Couture"]},
        {"name": "everyday", "aesthetics": ["Handloom Minimal", "Contemporary Indian Menswear", "Made In India Apparel", "Minimalist Indian Streetwear Denim", "Clean Basics Henley Tees", "Affordable Women's Western Everyday Wear", "Budget Casual Western Wear", "Contemporary Indian Linen Drape", "Premium Clean Fits Not Streetwear"]},
        {"name": "street", "aesthetics": ["Japanese Inspired Indian Streetwear", "Mumbai Graphic Streetwear", "High Fashion Streetwear", "Craft Streetwear", "Darkwear Band Tees Subculture", "Mumbai Streetwear Store", "Indian Genderfluid"]},
        {"name": "editorial", "aesthetics": ["Sculptural Couture", "Tech Futurist Couture", "Desi Avant-Garde", "Avant-Garde", "Editorial Indian Menswear"]},
        {"name": "party", "aesthetics": ["Mirror Maximalism", "Quirky Maximalist", "High Fashion Streetwear", "Graphic Pop Indian", "Edgy Statement Jewellery", "Edgy Chain Accessories Keychains"]},
        {"name": "casual", "aesthetics": ["Ethno-Western Sustainable Menswear", "Rapid Drop Fast Fashion Women", "Gen Z D2C Fashion + Footwear (Shark Tank S3)", "Minimalist Indian Streetwear Denim"]}
    ]
    
    silhouettes_data = [
        {"name": "draped", "aesthetics": ["Handloom Minimal", "Contemporary Indian Linen Drape", "Artisan Craft", "Ethno-Western Sustainable Menswear"]},
        {"name": "structured", "aesthetics": ["Heritage Couture", "Heritage Bridal Couture", "Sculptural Couture", "Structured Leather Bags Briefcases", "Bold Structured Statement Bags"]},
        {"name": "oversized", "aesthetics": ["Japanese Inspired Indian Streetwear", "Mumbai Graphic Streetwear", "Craft Streetwear", "Darkwear Band Tees Subculture"]},
        {"name": "experimental", "aesthetics": ["Desi Avant-Garde", "Avant-Garde", "Tech Futurist Couture", "Sculptural Couture", "Post Sneaker Era Fashion Footwear"]},
        {"name": "fitted", "aesthetics": ["Premium Clean Fits Not Streetwear", "Contemporary Indian Menswear", "Editorial Indian Menswear"]},
        {"name": "traditional", "aesthetics": ["Heritage Couture", "Festive Occasion", "Folk Maximalist", "Artisan Craft", "Mirror Maximalism"]}
    ]
    
    vibe_query = """
    UNWIND $data AS item
    MERGE (v:Vibe {name: item.name})
    WITH v, item
    UNWIND item.aesthetics AS aesthetic_name
    MERGE (a:Aesthetic {name: aesthetic_name})
    MERGE (v)-[:EXPRESSES_THROUGH]->(a)
    """
    
    occasion_query = """
    UNWIND $data AS item
    MERGE (o:Occasion {name: item.name})
    WITH o, item
    UNWIND item.aesthetics AS aesthetic_name
    MERGE (a:Aesthetic {name: aesthetic_name})
    MERGE (o)-[:CALLS_FOR]->(a)
    """
    
    silhouette_query = """
    UNWIND $data AS item
    MERGE (s:Silhouette {name: item.name})
    WITH s, item
    UNWIND item.aesthetics AS aesthetic_name
    MERGE (a:Aesthetic {name: aesthetic_name})
    MERGE (s)-[:SUITS]->(a)
    """

    with driver.session() as session:
        print("Seeding Vibe nodes and relationships...")
        session.run(vibe_query, data=vibes_data)
        vibe_count = session.run("MATCH (v:Vibe) RETURN count(v) AS c").single()["c"]
        print(f"Total Vibe nodes: {vibe_count}")
        print("Sample Vibe wiring:")
        for r in session.run("MATCH (v:Vibe)-[:EXPRESSES_THROUGH]->(a:Aesthetic) RETURN v.name, a.name LIMIT 4"):
            print(f"  ({r['v.name']}) -[:EXPRESSES_THROUGH]-> ({r['a.name']})")

        print("\nSeeding Occasion nodes and relationships...")
        session.run(occasion_query, data=occasions_data)
        occasion_count = session.run("MATCH (o:Occasion) RETURN count(o) AS c").single()["c"]
        print(f"Total Occasion nodes: {occasion_count}")
        print("Sample Occasion wiring:")
        for r in session.run("MATCH (o:Occasion)-[:CALLS_FOR]->(a:Aesthetic) RETURN o.name, a.name LIMIT 4"):
            print(f"  ({r['o.name']}) -[:CALLS_FOR]-> ({r['a.name']})")

        print("\nSeeding Silhouette nodes and relationships...")
        session.run(silhouette_query, data=silhouettes_data)
        silhouette_count = session.run("MATCH (s:Silhouette) RETURN count(s) AS c").single()["c"]
        print(f"Total Silhouette nodes: {silhouette_count}")
        print("Sample Silhouette wiring:")
        for r in session.run("MATCH (s:Silhouette)-[:SUITS]->(a:Aesthetic) RETURN s.name, a.name LIMIT 4"):
            print(f"  ({r['s.name']}) -[:SUITS]-> ({r['a.name']})")

        test_query = """
        MATCH (v:Vibe {name: 'streetwear'})-[:EXPRESSES_THROUGH]->(a:Aesthetic)<-[:HAS_AESTHETIC]-(b:Brand)
        RETURN v.name as vibe, a.name as aesthetic, b.name as brand
        ORDER BY b.name
        """
        print("\n================== PROOF QUERY OUTPUT ==================")
        print("MATCH (v:Vibe {name: 'streetwear'})-[:EXPRESSES_THROUGH]->(a:Aesthetic)<-[:HAS_AESTHETIC]-(b:Brand)")
        print("RETURN v.name as vibe, a.name as aesthetic, b.name as brand\n")
        
        results = session.run(test_query)
        found = False
        for r in results:
            found = True
            print(f"  vibe: '{r['vibe']}' | aesthetic: '{r['aesthetic']}' | brand: '{r['brand']}'")
        if not found:
            print("  No records found.")

    driver.close()

if __name__ == "__main__":
    main()
