import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client[os.getenv('MONGODB_DB', 'shaaru')]

new_fabrics = [
    {
        "fabric_id": "special_net_abhinav_mishra",
        "common_names": ["Special Net", "SPECIAL NET"],
        "weave": "Net",
        "drape_score": 7,
        "structure_score": 3,
        "embellishment_compatibility": { "embroidery": "high" },
        "best_for": ["lehengas", "dupattas", "overlays"],
        "verified": False
    },
    {
        "fabric_id": "special_silk_abhinav_mishra",
        "common_names": ["Special Silk"],
        "hand_feel": "smooth, medium weight, subtle sheen",
        "drape_score": 6,
        "structure_score": 5,
        "verified": False
    },
    {
        "fabric_id": "pure_georgette",
        "common_names": ["Pure Georgette"],
        "weave": "Crepe",
        "drape_score": 8,
        "structure_score": 2,
        "seasonal": ["all-year"],
        "verified": False
    },
    {
        "fabric_id": "satin_silk",
        "common_names": ["Satin Silk"],
        "weave": "Satin_weave",
        "drape_score": 7,
        "structure_score": 4,
        "seasonal": ["winter", "transitional"],
        "verified": False
    },
    {
        "fabric_id": "heavy_chanderi",
        "common_names": ["Heavy chanderi"],
        "weave": "Plain",
        "drape_score": 5,
        "structure_score": 6,
        "seasonal": ["winter", "transitional"],
        "verified": False
    },
    {
        "fabric_id": "crepe_georgette",
        "common_names": ["Crepe georgette"],
        "weave": "Crepe",
        "drape_score": 8,
        "structure_score": 2,
        "seasonal": ["all-year"],
        "verified": False
    },
    {
        "fabric_id": "neo_tech_fabric",
        "common_names": ["Neo Tech Fabric"],
        "hand_feel": "unknown — needs physical verification",
        "verified": False,
        "needs_investigation": True
    }
]

for fab in new_fabrics:
    db.fabric_intelligence.update_one(
        {"fabric_id": fab["fabric_id"]},
        {"$set": fab},
        upsert=True
    )
print("Seeded 7 new fabrics")
