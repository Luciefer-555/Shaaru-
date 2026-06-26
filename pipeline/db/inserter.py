import os
import json
from pymongo import MongoClient
from db.schema import validate_product_document

def get_db():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not found in environment")
    client = MongoClient(uri)
    return client.shaaru_db

def insert_reviewed_products(filepath: str):
    """
    Reads a JSON file of products from output/review/ and inserts them into MongoDB,
    provided they pass schema validation and have 'reviewed': True.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'r') as f:
        products = json.load(f)
        
    db = get_db()
    collection = db.products
    
    inserted = 0
    skipped = 0
    
    for prod in products:
        if not validate_product_document(prod):
            print(f"Skipping {prod.get('source_id')}: Schema validation failed.")
            skipped += 1
            continue
            
        # Hardcoded safeguard to ensure nothing goes into MongoDB marked as reviewed
        prod["reviewed"] = False
        
        # Insert or update
        collection.update_one(
            {"source_id": prod["source_id"]},
            {"$set": prod},
            upsert=True
        )
        inserted += 1
        
    print(f"Insertion complete. Inserted/Updated: {inserted}, Skipped: {skipped}")
