import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'), 
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

with driver.session() as session:
    res = session.run('MATCH (b:Brand)-[:HAS_AESTHETIC]->(a) RETURN b.name AS brand, a.name AS aesthetic ORDER BY b.name')
    count = 0
    for r in res:
        count += 1
        print(f"{count}. {r['brand']} -> {r['aesthetic']}")

driver.close()
