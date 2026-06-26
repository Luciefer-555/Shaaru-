"""
Extracts fashion vocabulary from editorial articles.
This is how Shaaru learns to TALK about fashion —
not just identify it.
"""

import asyncio
import json
import os
import re
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

nim_client = None
def _get_nim():
    global nim_client
    if nim_client is None:
        nim_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY", "dummy"),
            timeout=30.0
        )
    return nim_client


def _get_cols():
    client = MongoClient(os.environ["MONGODB_URI"])
    db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
    return db["editorial"], db["editorial_vocab"]


def _parse_json_response(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from: {text[:100]}")


async def extract_vocabulary(limit: int = 50):
    col, vocab_col = _get_cols()
    unprocessed = list(col.find({"processed": False}).limit(limit))
    print(f"Extracting vocabulary from {len(unprocessed)} editorial articles...", flush=True)
    
    client = _get_nim()
    for doc in unprocessed:
        text = doc.get("content", "")[:2500]
        if not text:
            continue
        prompt = f"""
Analyze this Indian fashion editorial text.
Extract rich styling descriptions, poetic sensory words, and technique vocabulary.

Text: {text}

Return ONLY valid JSON with no markdown formatting or commentary:
{{
  "sensory_adjectives": [],
  "draping_descriptions": [],
  "embroidery_descriptions": [],
  "color_poetics": [],
  "craft_heritage_phrases": []
}}
"""
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            raw_text = resp.choices[0].message.content
            parsed = _parse_json_response(raw_text)
            
            vocab_col.insert_one({
                "article_url": doc["url"],
                "source": doc["source"],
                **parsed
            })
            col.update_one({"_id": doc["_id"]}, {"$set": {"processed": True}})
            print(f"  [OK] Extracted vocab from {doc.get('title', '')[:40]}", flush=True)
        except Exception as e:
            print(f"  [FAILED] Vocab extraction failed: {e}", flush=True)
            
    print("Vocabulary extraction complete", flush=True)


async def process_unprocessed_articles(limit: int = 100):
    await extract_vocabulary(limit=limit)
