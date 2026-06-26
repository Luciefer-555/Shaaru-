import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(".")

from pipeline.db.db_loader import load_fabric_reference
from knowledge_graph import KnowledgeGraph

def seed_fabrics():
    kg = KnowledgeGraph()
    if not kg.is_connected:
        print("Neo4j not connected.")
        return

    _, fabs, _ = load_fabric_reference(os.getenv("MONGODB_URI"))
    print(f"Loaded {len(fabs)} fabrics from MongoDB.")

    cypher = """
    MERGE (f:Fabric {name: $id})
    SET f.common_names      = $cn,
        f.fiber_composition = $fb,
        f.weave             = $wv,
        f.hand_feel         = $hf,
        f.structure_score   = $ss,
        f.drape_score       = $ds
    """

    count = 0
    with kg.driver.session() as session:
        for k, v in fabs.items():
            session.run(cypher, {
                "id": k,
                "cn": v.get("common_names", []),
                "fb": v.get("fiber_composition", ""),
                "wv": v.get("weave", ""),
                "hf": v.get("hand_feel", ""),
                "ss": v.get("structure_score", 0),
                "ds": v.get("drape_score", 0)
            })
            count += 1

    res = kg.query("MATCH (f:Fabric) RETURN count(f) AS c")
    print(f"Successfully seeded {count} fabrics! Total Neo4j Fabric nodes: {res[0]['c']}")

if __name__ == "__main__":
    seed_fabrics()
