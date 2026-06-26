import argparse
import glob
import json
import os
import sys

# Ensure pipeline directory is in python path to import validators
sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))
from validators.quality_gate import load_quality_gates, validate_product

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="abhinav_mishra")
parser.add_argument("--recheck-only", action="store_true")
args = parser.parse_args()

review_dir = os.path.join("pipeline", "output", "review")
files = sorted(glob.glob(os.path.join(review_dir, f"{args.source}_*.json")))

if not files:
    print(f"No review files found for source: {args.source}")
    sys.exit(1)

gates_path = os.path.join("pipeline", "config", "quality_gates.json")
gates = load_quality_gates(gates_path)

total_data = []

seen_ids = set()
for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
        
    if args.recheck_only:
        for doc in batch_data:
            doc.pop("validation_failures", None)
            doc.pop("needs_manual_review", None)
            validate_product(doc, args.source, gates)
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
    for doc in batch_data:
        sid = str(doc.get("source_id", "") or doc.get("id", ""))
        if sid and sid in seen_ids: continue
        if sid: seen_ids.add(sid)
        total_data.append(doc)

passed = [p for p in total_data if p.get("quality_gate_passed")]
failed = [p for p in total_data if not p.get("quality_gate_passed")]

new_fabrics = set()
for p in total_data:
    for c in p.get("new_fabric_candidates", []):
        if isinstance(c, str):
            new_fabrics.add(c)
        elif isinstance(c, dict):
            name = c.get("name") or c.get("fabric_name", "")
            if name:
                new_fabrics.add(name)

print(f"\n==========================================")
print(f"SCORECARD ({args.source})")
print(f"Total: {len(total_data)} | Passed: {len(passed)} | Failed: {len(failed)}")
print(f"==========================================")

if failed:
    print("\nRemaining legitimate failures:")
    for p in failed:
        title = p.get("title") or p.get("id", "Unknown")
        fail = p.get("validation_failures", [])
        print(f"  → {title}: {fail}")

print(f"\nNew fabric candidates discovered ({len(new_fabrics)}): {sorted(list(new_fabrics))}")
