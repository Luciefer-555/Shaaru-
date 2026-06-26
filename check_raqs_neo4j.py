from dotenv import load_dotenv
load_dotenv()
import json
from knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
r = kg.query("MATCH (p:Product) WHERE toLower(p.title) CONTAINS 'raqs' RETURN p.title as title, p.source_url as source_url, p.caption as caption LIMIT 3")
print(json.dumps(r, indent=2, default=str))
