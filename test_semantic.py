import logging
logging.basicConfig(level=logging.DEBUG)
from dotenv import load_dotenv
load_dotenv()
from product_embeddings import search_products_semantic
print("Starting search...")
results = search_products_semantic('flowy kurta for summer office wear', limit=3)
print(f"Found {len(results)} results")
for r in results:
    print(f"[{r.get('score', 0):.3f}] {r.get('product_name', '')} - {r.get('brand', '')}")
