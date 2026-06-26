# pipeline/db/db_loader.py
# Runs at pipeline startup, caches all reference data

from pymongo import MongoClient
from neo4j import GraphDatabase
import json
import os

def load_fabric_reference(mongo_uri):
    """
    Pulls all fabrics from fabric_intelligence.
    Returns two things:
    1. injection_list: formatted for prompt injection
    2. lookup_dict: fabric_id → full document for post-extraction enrichment
    """
    client = MongoClient(mongo_uri)
    db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
    
    fabrics = list(db.fabric_intelligence.find(
        {},
        {
            "_id": 0,
            "fabric_id": 1,
            "common_names": 1,
            "drape_score": 1,
            "structure_score": 1,
            "hand_feel": 1,
            "best_for": 1,
            "avoid_for": 1,
            "embellishment_compatibility": 1,
            "seasonal": 1,
            "weave": 1,
            "fiber_composition": 1
        }
    ))
    
    # For prompt injection — flat list of all names
    injection_list = []
    for f in fabrics:
        entry = f["fabric_id"]
        if f.get("common_names"):
            entry += f" (also known as: {', '.join(f['common_names'])})"
        injection_list.append(entry)
    
    # For post-extraction enrichment — full dict
    lookup_dict = {f["fabric_id"]: f for f in fabrics}
    # Also index by common names for fuzzy matching
    name_index = {}
    for f in fabrics:
        for name in f.get("common_names", []):
            name_index[name.lower()] = f["fabric_id"]
    
    return injection_list, lookup_dict, name_index


def load_embellishment_reference(mongo_uri):
    """
    Pulls all embellishments from embellishment_sourcing.
    NOTE: Only 6 records currently — will grow.
    """
    client = MongoClient(mongo_uri)
    db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
    
    embellishments = list(db.embellishment_sourcing.find(
        {},
        {
            "_id": 0,
            "embellishment_id": 1,
            "type": 1,
            "technique": 1,
            "best_for": 1,
            "avoid_for": 1
        }
    ))
    
    injection_list = [e["embellishment_id"] for e in embellishments]
    lookup_dict = {e["embellishment_id"]: e for e in embellishments}
    
    return injection_list, lookup_dict


def load_construction_reference(mongo_uri):
    """
    Pulls Indian constructions from garment_construction.
    Filters for tradition = 'indian' where possible.
    """
    client = MongoClient(mongo_uri)
    db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
    
    # Try to get Indian tradition first
    constructions = list(db.garment_construction.find(
        {"tradition": "indian"},
        {"_id": 0, "garment_id": 1, "category": 1, "recommended_fabrics": 1}
    ))
    
    # If tradition field not reliable, get all
    if len(constructions) < 5:
        constructions = list(db.garment_construction.find(
            {},
            {"_id": 0, "garment_id": 1, "category": 1, "recommended_fabrics": 1, "tradition": 1}
        ))
    
    injection_list = [c["garment_id"] for c in constructions]
    lookup_dict = {c["garment_id"]: c for c in constructions}
    
    return injection_list, lookup_dict


def load_aesthetic_reference(neo4j_uri, neo4j_user, neo4j_password):
    """
    Pulls all 47 aesthetics with indian_context from Neo4j.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Aesthetic)
            RETURN a.name as name, 
                   a.description as description,
                   a.indian_context as indian_context
        """)
        aesthetics = [dict(r) for r in result]
    
    driver.close()
    return aesthetics


def load_all_references(mongo_uri, neo4j_uri, neo4j_user, neo4j_password):
    """
    Master loader — call once at pipeline startup.
    Returns everything the prompts need.
    """
    fabric_list, fabric_lookup, fabric_name_index = load_fabric_reference(mongo_uri)
    embellishment_list, embellishment_lookup = load_embellishment_reference(mongo_uri)
    construction_list, construction_lookup = load_construction_reference(mongo_uri)
    aesthetics = load_aesthetic_reference(neo4j_uri, neo4j_user, neo4j_password)
    
    print(f"Loaded {len(fabric_list)} fabrics")
    print(f"Loaded {len(embellishment_list)} embellishments")
    print(f"Loaded {len(construction_list)} constructions")
    print(f"Loaded {len(aesthetics)} aesthetics from Neo4j")
    
    return {
        "fabrics": {
            "injection_list": fabric_list,
            "lookup": fabric_lookup,
            "name_index": fabric_name_index
        },
        "embellishments": {
            "injection_list": embellishment_list,
            "lookup": embellishment_lookup
        },
        "constructions": {
            "injection_list": construction_list,
            "lookup": construction_lookup
        },
        "aesthetics": aesthetics
    }
