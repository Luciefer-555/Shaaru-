from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("MONGODB_DB", "shaaru_db")]

embellishments = [
    {"embellishment_id": "sheesha_mirror", "technique": "mirror-work", "type": "embroidery"},
    {"embellishment_id": "resham_thread", "technique": "silk-thread", "type": "embroidery"},
    {"embellishment_id": "dabka", "technique": "coiled-wire", "type": "embroidery"},
    {"embellishment_id": "nakshi", "technique": "carved-wire", "type": "embroidery"},
    {"embellishment_id": "marodi", "technique": "twisted-wire", "type": "embroidery"},
    {"embellishment_id": "gota_patti", "technique": "ribbon-applique", "type": "applique"},
    {"embellishment_id": "mukaish", "technique": "metal-dot", "type": "embroidery"},
    {"embellishment_id": "zari_flat", "technique": "flat-wire", "type": "embroidery"},
    {"embellishment_id": "kantha_stitch", "technique": "running-stitch", "type": "embroidery"},
    {"embellishment_id": "chikankari", "technique": "shadow-work", "type": "embroidery"},
    {"embellishment_id": "bagh", "technique": "geometric-thread", "type": "embroidery"},
    {"embellishment_id": "kutch_mirror", "technique": "mirror-thread", "type": "embroidery"},
    {"embellishment_id": "kashmiri_ari", "technique": "chain-stitch", "type": "embroidery"},
    {"embellishment_id": "sequin_dori", "technique": "sequin-thread", "type": "embroidery"},
    {"embellishment_id": "thread_work_basic", "technique": "basic-thread", "type": "embroidery"},
    {"embellishment_id": "zardozi_thread_gold", "technique": "hand-embroidery", "type": "embroidery"},
    {"embellishment_id": "swarovski_crystal", "technique": "crystal-work", "type": "embellishment"}
]

collection = db.embellishment_sourcing
for e in embellishments:
    # Update if exists, otherwise insert
    collection.update_one(
        {"embellishment_id": e["embellishment_id"]},
        {"$set": e},
        upsert=True
    )

print(f"Embellishments count: {collection.count_documents({})}")
