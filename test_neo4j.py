from dotenv import load_dotenv
load_dotenv()
from knowledge_graph import kg
results = kg.query('''
MATCH (u:User)-[r]->(p:Product)
WHERE u.user_id = "demo_user_001"
RETURN type(r) as rel_type, p.product_id as product, r.count as count
LIMIT 10
''')
for r in results:
    print(r)
if not results:
    print('No edges found in Neo4j')
