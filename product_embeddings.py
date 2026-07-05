import os
import requests
import logging
from datetime import datetime, timezone
from database import get_db

log = logging.getLogger("shaaru.embeddings")

def generate_embedding(text: str, input_type: str = "passage") -> list[float] | None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        log.error("NVIDIA_API_KEY not found.")
        return None
        
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    # For NVIDIA nv-embedqa-e5-v5, input must be a list of strings
    payload = {
        "input": [text],
        "model": "nvidia/nv-embedqa-e5-v5",
        "input_type": input_type,
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        log.error(f"[FAIL] Embedding failed: {e}")
        return None

def generate_product_embedding(product: dict) -> list[float] | None:
    tags = product.get("tags", [])
    if not isinstance(tags, list):
        tags = []
        
    # Build rich text representation
    parts = [
        product.get('product_name', ''),
        product.get('brand', ''),
        product.get('category', ''),
        product.get('description', ''),
        ' '.join(tags),
        product.get('aesthetic', ''),
        product.get('occasion', '')
    ]
    text = " ".join([str(p) for p in parts if p]).strip()
    
    embedding = generate_embedding(text, input_type="passage")
    if embedding:
        print(f"[OK] Generated embedding for {product.get('product_name', 'Unknown')}")
    else:
        print(f"[FAIL] Failed to generate embedding for {product.get('product_name', 'Unknown')}")
    return embedding

def embed_all_products() -> dict:
    db = get_db()
    if db is None:
        return {"error": "Database unavailable"}
        
    # Find products that don't have embeddings yet or have empty embeddings
    cursor = db["products"].find({
        "$or": [
            {"embedding": {"$exists": False}},
            {"embedding": {"$size": 0}}
        ]
    })
    products = list(cursor)
    
    stats = {"embedded": 0, "skipped": 0, "failed": 0}
    
    for p in products:
        embedding = generate_product_embedding(p)
        if embedding:
            try:
                db["products"].update_one(
                    {"_id": p["_id"]},
                    {"$set": {
                        "embedding": embedding,
                        "embedded_at": datetime.now(timezone.utc)
                    }}
                )
                stats["embedded"] += 1
            except Exception as e:
                log.error(f"[FAIL] Failed to save embedding for {p.get('product_name')}: {e}")
                stats["failed"] += 1
        else:
            stats["failed"] += 1
            
    # Calculate how many were skipped (already had embedding)
    total_embedded = db["products"].count_documents({"embedding": {"$exists": True}})
    stats["skipped"] = total_embedded - stats["embedded"]
    
    return stats

def search_products_semantic(query: str, limit: int = 5, filters: dict = None) -> list[dict]:
    query_embedding = generate_embedding(query, input_type="query")
    if not query_embedding:
        return []
        
    db = get_db()
    if db is None:
        return []
        
    vector_search_stage = {
        "$vectorSearch": {
            "index": "product_embedding_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 150,
            "limit": limit
        }
    }
    if filters:
        vector_search_stage["$vectorSearch"]["filter"] = filters

    pipeline = [
        vector_search_stage,
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}}
    ]
    
    try:
        cursor = db["products"].aggregate(pipeline)
        return list(cursor)
    except Exception as e:
        log.error(f"[FAIL] Vector search failed: {e}")
        return []
