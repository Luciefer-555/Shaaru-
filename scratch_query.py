import os
from dotenv import load_dotenv
from pymongo import MongoClient
from neo4j import GraphDatabase
import pprint

load_dotenv()

print("--- MONGODB ---")
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client[os.getenv("MONGODB_DB")]

print("\n// 1. Get all fabric names + count")
pprint.pprint(list(db.fabrics.aggregate([
  { "$group": { "_id": "$name", "count": { "$sum": 1 } } },
  { "$sort": { "count": -1 } }
])))

print("\n// 2. Get all construction types")
pprint.pprint(list(db.constructions.aggregate([
  { "$group": { "_id": "$name", "type": { "$first": "$type" } } },
  { "$sort": { "_id": 1 } }
])))

print("\n// 3. Check what fields exist on fabric documents")
pprint.pprint(db.fabrics.find_one())

print("\n// 4. Check what fields exist on construction documents")
pprint.pprint(db.constructions.find_one())

print("\n// 5. Check if there are any other relevant collections")
pprint.pprint(db.list_collection_names())

print("\n--- NEO4J ---")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_pw = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))

def run_cypher(query):
    with driver.session() as session:
        result = session.run(query)
        try:
            return [dict(record) for record in result]
        except Exception as e:
            return str(e)

print("\n// 1. See all node labels")
pprint.pprint(run_cypher("CALL db.labels()"))

print("\n// 2. See all relationship types")
pprint.pprint(run_cypher("CALL db.relationshipTypes()"))

print("\n// 3. Get all fabric nodes with properties")
pprint.pprint(run_cypher("MATCH (f:Fabric) RETURN properties(f) AS f LIMIT 20"))

print("\n// 4. Get all construction nodes")
pprint.pprint(run_cypher("MATCH (c:Construction) RETURN properties(c) AS c LIMIT 20"))

print("\n// 5. See how fabrics and constructions relate")
pprint.pprint(run_cypher("MATCH (f:Fabric)-[r]->(c:Construction) RETURN f.name AS fabric, type(r) AS rel, c.name AS construction LIMIT 30"))

print("\n// 6. Check if techniques are separate nodes")
pprint.pprint(run_cypher("MATCH (t:Technique) RETURN properties(t) AS t LIMIT 20"))

print("\n// 7. See the full schema")
try:
    pprint.pprint(run_cypher("CALL apoc.meta.schema()"))
except Exception as e:
    print("APOC schema failed, falling back...")
    pprint.pprint(run_cypher("CALL db.schema.visualization()"))

driver.close()
