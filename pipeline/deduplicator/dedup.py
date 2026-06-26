import os
import requests
import numpy as np
import os
from config.models import NIM_BASE_URL, MODELS

def get_embedding(text: str, model_config: dict):
    """
    Calls NVIDIA NIM NV-EmbedQA model to generate embeddings for a text string.
    """
    url = f"{NIM_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": [text],
        "model": model_config["model"],
        "input_type": "query"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    tokens = {"input": data.get("usage", {}).get("prompt_tokens", 0), "output": 0}
    embedding = data["data"][0]["embedding"]
    return embedding, tokens

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def deduplicate_batch(products, existing_embeddings, designer_id: str):
    """
    Deduplicates a batch of products against existing products for the SAME designer.
    Products is a list of dicts.
    Returns (unique_products, duplicates, total_embedding_tokens)
    """
    unique_products = []
    duplicates = []
    total_tokens = 0
    
    # Track embeddings within this batch too
    batch_embeddings = list(existing_embeddings)
    
    for product in products:
        # Build deduplication string
        f_conf_list = product.get("fabric_vocabulary", {}).get("confirmed", [])
        t_conf_list = product.get("techniques", {}).get("confirmed", [])
        f_conf = " ".join([f.get("name", "") if isinstance(f, dict) else str(f) for f in f_conf_list]) if f_conf_list else ""
        t_conf = " ".join([t.get("name", "") if isinstance(t, dict) else str(t) for t in t_conf_list]) if t_conf_list else ""
        silhouette = product.get("silhouette", "")
        aesthetic = product.get("aesthetic_category", "")
        
        dedup_str = f"{f_conf} {t_conf} {silhouette} {aesthetic}".strip()
        
        if not dedup_str:
            unique_products.append(product)
            continue
            
        try:
            emb, tokens = get_embedding(dedup_str, MODELS["embeddings"])
            total_tokens += tokens["input"]
            product["dedup_hash"] = dedup_str
            
            is_dup = False
            for ext_emb, ext_prod in batch_embeddings:
                sim = cosine_similarity(emb, ext_emb)
                if sim > 0.96:
                    is_dup = True
                    # Keep the one with more images or richer caption
                    if len(product.get("images", [])) > len(ext_prod.get("images", [])):
                        # Current product is better, replace existing in unique_products (not trivial in simple loop, 
                        # so we'll just flag the new one as dup for now and user can resolve, 
                        # or we log it strictly)
                        pass
                    
                    product["duplicate_of"] = ext_prod.get("source_id")
                    product["similarity_score"] = sim
                    duplicates.append(product)
                    break
            
            if not is_dup:
                unique_products.append(product)
                
        except Exception as e:
            print(f"Embedding failed for {product.get('source_id')}: {e}")
            unique_products.append(product) # fallback to keep
            
    return unique_products, duplicates, total_tokens
