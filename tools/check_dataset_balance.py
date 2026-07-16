#!/usr/bin/env python3
"""
tools/check_dataset_balance.py — SHAARU YOLO26 Dataset Balance Auditor

Before kicking off fine-tuning training (`train_yolo26_shaaru.py`), this script audits
the labeled dataset across all 36 Shaaru apparel classes to ensure reasonable class
balance and flags any thin or missing categories (e.g. sharara_set, cape, leg_warmer).

Usage:
  python tools/check_dataset_balance.py --config pipeline/cv/shaaru_apparel_36.yaml --min-instances 15
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Canonical 36 classes (fallback if YAML not provided)
CANONICAL_CLASSES_36 = [
    "anarkali_dress", "bag_wallet", "belt", "cape", "cardigan",
    "co_ord_set", "coat", "dress", "dupatta", "glasses",
    "glove", "hat", "headband_hair_accessory", "jacket", "jumpsuit",
    "kurta", "leg_warmer", "lehenga_set", "pants", "romper",
    "salwar_kameez_set", "saree", "scarf", "sharara_set", "shirt_blouse",
    "shoe", "shorts", "skirt", "sock", "sweater",
    "tie", "tights_stockings", "top_t_shirt_sweatshirt", "umbrella", "vest",
    "watch"
]

def audit_dataset_balance(
    yaml_path: str,
    min_instances: int = 15,
    warn_only: bool = False
) -> Tuple[bool, Dict[str, int], List[str]]:
    """
    Audits the dataset specified in `yaml_path` for class representation.
    Returns: (passed_all_checks, class_counts_dict, flagged_thin_classes)
    """
    if not os.path.exists(yaml_path):
        print(f"[ERROR] Dataset configuration YAML not found at: {yaml_path}", file=sys.stderr)
        return False, {}, CANONICAL_CLASSES_36

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    names = config.get("names", {})
    if isinstance(names, list):
        class_map = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        class_map = {int(k): str(v) for k, v in names.items()}
    else:
        class_map = {i: name for i, name in enumerate(CANONICAL_CLASSES_36)}

    dataset_root = Path(yaml_path).parent / config.get("path", ".")
    if not dataset_root.exists():
        # Fallback to current working dir check if path is relative
        dataset_root = Path(config.get("path", "."))

    # Identify label folders (train + val)
    train_rel = config.get("train", "images/train")
    val_rel = config.get("val", "images/val")
    
    # In YOLO format, labels correspond to images/ -> labels/
    label_dirs = []
    for rel_path in [train_rel, val_rel]:
        img_dir = (dataset_root / rel_path).resolve()
        # Typical pattern: images/train -> labels/train
        if "images" in str(img_dir):
            lbl_dir = Path(str(img_dir).replace("images", "labels"))
        else:
            lbl_dir = img_dir.parent / "labels" / img_dir.name
        if lbl_dir.exists():
            label_dirs.append(lbl_dir)

    counts = {name: 0 for name in class_map.values()}
    total_instances = 0
    total_files = 0

    for lbl_dir in label_dirs:
        for txt_file in lbl_dir.glob("*.txt"):
            total_files += 1
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_idx = int(parts[0])
                        cls_name = class_map.get(cls_idx, f"unknown_{cls_idx}")
                        if cls_name in counts:
                            counts[cls_name] += 1
                        else:
                            counts[cls_name] = 1
                        total_instances += 1
            except Exception as e:
                print(f"[WARNING] Error reading label file {txt_file}: {e}", file=sys.stderr)

    # Analyze balance & flag thin classes
    flagged = []
    print("\n======================================================================")
    print("                SHAARU YOLO26 DATASET BALANCE AUDIT                   ")
    print("======================================================================")
    print(f"Dataset Root    : {dataset_root.resolve()}")
    print(f"Total Files     : {total_files} label (.txt) files")
    print(f"Total Instances : {total_instances} bounding boxes across {len(class_map)} classes")
    print(f"Minimum Req.    : {min_instances} instances per class\n")

    print(f"{'Class ID':<10} | {'Class Name':<25} | {'Instances':<10} | {'Share (%)':<10} | {'Status'}")
    print("-" * 75)

    sorted_classes = sorted(class_map.items(), key=lambda x: x[0])
    for cls_idx, cls_name in sorted_classes:
        cnt = counts.get(cls_name, 0)
        pct = (cnt / total_instances * 100.0) if total_instances > 0 else 0.0
        if cnt < min_instances:
            status = "[FLAG / THIN]"
            flagged.append(cls_name)
        else:
            status = "[OK]"
        print(f"{cls_idx:<10} | {cls_name:<25} | {cnt:<10} | {pct:<9.2f}% | {status}")

    print("======================================================================\n")

    if flagged:
        print(f"[ALERT] Found {len(flagged)} class(es) with < {min_instances} instances:")
        print(f"        {', '.join(flagged)}")
        print("        To avoid an imbalanced model checkpoint where rare ethnic/specialty")
        print("        classes fail, please expand sourcing or batch labeling for these categories")
        print("        before initiating fine-tuning.\n")
        if not warn_only:
            return False, counts, flagged

    print("[PASS] Dataset class balance audit verified! Ready for YOLO26 fine-tuning.")
    return True, counts, flagged

def main():
    parser = argparse.ArgumentParser(description="Audit Shaaru 36-class dataset balance")
    parser.add_argument("--config", default="pipeline/cv/shaaru_apparel_36.yaml", help="Path to dataset YAML config")
    parser.add_argument("--min-instances", type=int, default=15, help="Minimum acceptable bounding box instances per class")
    parser.add_argument("--warn-only", action="store_true", help="Warn without exiting non-zero if classes are thin")
    args = parser.parse_args()

    passed, _, flagged = audit_dataset_balance(args.config, args.min_instances, args.warn_only)
    if not passed and not args.warn_only:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
