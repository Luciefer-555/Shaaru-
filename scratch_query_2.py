import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pprint

load_dotenv()

print("--- MONGODB ---")
mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client[os.getenv("MONGODB_DB")]

print("\n// 1. See full fabric_intelligence document structure")
pprint.pprint(db.fabric_intelligence.find_one())

print("\n// 2. Get all fabric names from fabric_intelligence")
pprint.pprint(db.fabric_intelligence.distinct("name"))

print("\n// 3. See full garment_construction document structure")
pprint.pprint(db.garment_construction.find_one())

print("\n// 4. Get all construction names")
pprint.pprint(db.garment_construction.distinct("name"))

print("\n// 5. See what's in embellishment_sourcing")
pprint.pprint(db.embellishment_sourcing.find_one())

print("\n// 6. See what's in styling_guides")
pprint.pprint(db.styling_guides.find_one())

print("\n// 7. Count of each")
print("fabric_intelligence count:", db.fabric_intelligence.count_documents({}))
print("garment_construction count:", db.garment_construction.count_documents({}))
print("embellishment_sourcing count:", db.embellishment_sourcing.count_documents({}))
