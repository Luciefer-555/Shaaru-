from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
load_dotenv()
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')))
with d.session() as s:
    print('--- NODE LABELS ---')
    for r in s.run('CALL db.labels() YIELD label RETURN label'):
        count = s.run(f'MATCH (n:`{r["label"]}`) RETURN count(n) as c').single()['c']
        print(f'  {r["label"]}: {count}')
    print('--- RELATIONSHIP TYPES ---')
    for r in s.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType'):
        count = s.run(f'MATCH ()-[r:`{r["relationshipType"]}`]->() RETURN count(r) as c').single()['c']
        print(f'  {r["relationshipType"]}: {count}')
d.close()
