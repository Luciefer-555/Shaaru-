from knowledge_graph import get_kg

def main():
    if not get_kg().is_connected:
        print("Neo4j not connected.")
        return

    labels = ["Product", "Technique", "Fabric", "Brand", "Occasion", "Aesthetic"]
    print("-- NEO4J NODE CENSUS --")
    for lbl in labels:
        res = get_kg().query(f"MATCH (n:{lbl}) RETURN count(n) AS c")
        print(f"  {lbl:<12}: {res[0]['c']}")

    res_cap = get_kg().query("MATCH (p:Product) WHERE p.caption IS NOT NULL AND p.caption <> '' RETURN count(p) AS c")
    print(f"  Product (w/ caption): {res_cap[0]['c']}")

    rels = ["HAS_TECHNIQUE", "HAS_FABRIC", "MADE_BY", "SUITS_OCCASION", "BELONGS_TO", "COMPLEMENTS", "REQUIRES_FABRIC"]
    print("\n-- NEO4J RELATIONSHIP CENSUS --")
    for r in rels:
        res = get_kg().query(f"MATCH ()-[rel:{r}]->() RETURN count(rel) AS c")
        print(f"  {r:<18}: {res[0]['c']}")

    print("\n-- DESIGNER PRODUCT BREAKDOWN --")
    res_des = get_kg().query("""
        MATCH (p:Product)
        RETURN p.designer AS designer, count(p) AS count
        ORDER BY count DESC
    """)
    designers_found = set()
    for row in res_des:
        des = row['designer']
        if des:
            designers_found.add(des)
            print(f"  {des:<20}: {row['count']}")

    print(f"\nTotal Designers with >0 products: {len(designers_found)}")

if __name__ == "__main__":
    main()
