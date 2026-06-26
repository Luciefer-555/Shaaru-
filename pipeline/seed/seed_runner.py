"""
Run once to seed all designers.
After this runs — never run batch pipeline again.
"""

import asyncio
import json
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path: sys.path.append(root_dir)

from dotenv import load_dotenv
load_dotenv()

from pipeline.seed.seeder import seed_designer
from pipeline.validators.quality_gate import load_quality_gates
from pipeline.db.db_loader import load_all_references
from knowledge_graph import KnowledgeGraph


def load_designers():
    with open("pipeline/config/designers.json", encoding="utf-8") as f:
        return json.load(f)


async def main():
    print("SHAARU SEED PIPELINE — Starting")
    print("Seeding 15 products per designer...")
    
    designers = load_designers()
    gates = load_quality_gates("pipeline/config/quality_gates.json")
    db_refs = load_all_references(
        mongo_uri=os.environ["MONGODB_URI"],
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_user=os.environ["NEO4J_USER"],
        neo4j_password=os.environ["NEO4J_PASSWORD"]
    )
    kg = KnowledgeGraph()
    
    results = []
    
    for designer_config in designers:
        if not designer_config.get("active", False):
            continue
        
        print(f"\nSeeding {designer_config['name']}...")
        
        result = await seed_designer(
            designer_config=designer_config,
            db_refs=db_refs,
            gates=gates,
            kg=kg
        )
        
        results.append(result)
        print(
            f"[OK] {designer_config['name']}: "
            f"{result['seeded_count']}/15 seeded"
        )
    
    print("\n" + "="*50)
    print("SEED COMPLETE")
    print("="*50)
    for r in results:
        print(
            f"  {r['designer_id']}: "
            f"{r['seeded_count']} products"
        )
    
    total = sum(r["seeded_count"] for r in results)
    print(f"\nTotal seeded: {total} products")
    print("Shaaru is ready for day 1.")


if __name__ == "__main__":
    asyncio.run(main())
