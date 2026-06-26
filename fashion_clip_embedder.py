"""
FashionCLIP embedding service — patrickjohncyh/fashion-clip (CLIP ViT-B/32 fine-tuned on fashion).
Output: 512-dim L2-normalized vectors. Lazy-loads on first call.
"""
import logging
import requests
from io import BytesIO
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    from transformers import CLIPModel, CLIPProcessor
    import torch
    logger.info("Loading FashionCLIP — first call only...")
    _processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    _model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device).eval()
    logger.info(f"FashionCLIP loaded on {device}")
    return _model, _processor


def embed_text(text: str) -> Optional[list[float]]:
    """Embed a text query → 512-dim L2-normalized vector."""
    try:
        import torch
        model, processor = _load_model()
        device = next(model.parameters()).device
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().tolist()
    except Exception as e:
        logger.error(f"embed_text failed for '{text}': {e}")
        return None


def embed_image_url(url: str) -> Optional[list[float]]:
    """Fetch image from URL, embed → 512-dim L2-normalized vector."""
    try:
        import torch
        from PIL import Image
        model, processor = _load_model()
        device = next(model.parameters()).device
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().tolist()
    except Exception as e:
        logger.error(f"embed_image_url failed for '{url}': {e}")
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two L2-normalized vectors = cosine similarity."""
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


def find_similar_products(
    query_vector: list[float],
    products: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rank products by cosine similarity to query_vector.
    Products without embeddings score 0.0 and fall to the bottom.
    Strips raw embedding vectors from returned dicts.
    """
    scored: list[tuple[float, dict]] = []
    for p in products:
        emb = p.get("image_embedding")
        score = cosine_similarity(query_vector, emb) if emb and len(emb) == len(query_vector) else 0.0
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, product in scored[:top_k]:
        clean = {k: v for k, v in product.items() if k != "image_embedding"}
        clean["_similarity_score"] = round(score, 4)
        results.append(clean)
    return results
