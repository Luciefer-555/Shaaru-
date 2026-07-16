#!/usr/bin/env python3
"""
tools/export_roboflow_dataset.py — SHAARU Dataset Export for Roboflow / CVAT

Exports logged user correction data (`cv_corrections.jsonl` / MongoDB) and verified
catalog/scanner crops into standard YOLO / COCO annotation structure (.yaml + images/ + labels/)
ready for Roboflow or CVAT batch labeling ingestion.

Usage:
  python tools/export_roboflow_dataset.py --output dataset/shaaru_roboflow_export
"""

import argparse
import base64
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

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

def get_class_map(yaml_path: str) -> Dict[str, int]:
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", {})
        if isinstance(names, dict):
            return {str(v): int(k) for k, v in names.items()}
        elif isinstance(names, list):
            return {name: i for i, name in enumerate(names)}
    return {name: i for i, name in enumerate(CANONICAL_CLASSES_36)}

def convert_bbox_to_yolo(bbox: dict | list) -> Optional[Tuple[float, float, float, float]]:
    """Converts [x_min, y_min, x_max, y_max] (normalized 0..1) to YOLO [x_center, y_center, width, height]."""
    try:
        if isinstance(bbox, list) and len(bbox) >= 4:
            xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        elif isinstance(bbox, dict):
            xmin = float(bbox.get("x_min", bbox.get("xmin", 0.0)))
            ymin = float(bbox.get("y_min", bbox.get("ymin", 0.0)))
            xmax = float(bbox.get("x_max", bbox.get("xmax", 1.0)))
            ymax = float(bbox.get("y_max", bbox.get("ymax", 1.0)))
        else:
            return None

        # Clamp normalized coordinates to [0, 1]
        xmin, ymin = max(0.0, min(1.0, xmin)), max(0.0, min(1.0, ymin))
        xmax, ymax = max(0.0, min(1.0, xmax)), max(0.0, min(1.0, ymax))
        if xmax <= xmin or ymax <= ymin:
            return None

        w = xmax - xmin
        h = ymax - ymin
        x_center = xmin + (w / 2.0)
        y_center = ymin + (h / 2.0)
        return (round(x_center, 6), round(y_center, 6), round(w, 6), round(h, 6))
    except Exception:
        return None

def decode_and_save_image(b64_str: str, out_path: Path) -> bool:
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        return True
    except Exception as e:
        print(f"[WARNING] Failed to decode b64 image: {e}", file=sys.stderr)
        return False

def export_roboflow_dataset(
    output_dir: str,
    config_yaml: str = "pipeline/cv/shaaru_apparel_36.yaml",
    jsonl_path: str = "cv_corrections.jsonl"
):
    out_root = Path(output_dir)
    img_dir = out_root / "images" / "train"
    lbl_dir = out_root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    class_map = get_class_map(config_yaml)
    exported_count = 0
    skipped_count = 0

    print("======================================================================")
    print("           SHAARU ROBOFLOW / CVAT DATASET EXPORTER                    ")
    print("======================================================================")
    print(f"Target Output   : {out_root.resolve()}")
    print(f"Taxonomy Config : {config_yaml}")
    print(f"Source JSONL    : {jsonl_path}\n")

    # 1. Export from local cv_corrections.jsonl if present
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    track_id = record.get("track_id", f"corr_{i}")
                    # Determine target label
                    corrected = record.get("corrected", {})
                    original = record.get("original", {})
                    label_name = corrected.get("label") or original.get("label")
                    if not label_name or label_name not in class_map:
                        skipped_count += 1
                        continue

                    class_id = class_map[label_name]
                    bbox = record.get("bbox") or corrected.get("bbox") or original.get("bbox")
                    yolo_bbox = convert_bbox_to_yolo(bbox) if bbox else (0.5, 0.5, 1.0, 1.0)
                    if not yolo_bbox:
                        yolo_bbox = (0.5, 0.5, 1.0, 1.0)

                    img_b64 = record.get("image_crop_b64") or corrected.get("image_crop_b64") or original.get("image_crop_b64")
                    if not img_b64:
                        skipped_count += 1
                        continue

                    file_stem = f"shaaru_{track_id}_{i}"
                    img_file = img_dir / f"{file_stem}.jpg"
                    lbl_file = lbl_dir / f"{file_stem}.txt"

                    if decode_and_save_image(img_b64, img_file):
                        with open(lbl_file, "w", encoding="utf-8") as lf:
                            lf.write(f"{class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")
                        exported_count += 1
                    else:
                        skipped_count += 1
                except Exception as ex:
                    print(f"[WARNING] Skipping malformed line {i}: {ex}", file=sys.stderr)
                    skipped_count += 1

    # 2. Copy data.yaml to export root
    data_yaml_out = out_root / "data.yaml"
    export_cfg = {
        "path": ".",
        "train": "images/train",
        "val": "images/train",
        "nc": len(class_map),
        "names": {v: k for k, v in class_map.items()}
    }
    with open(data_yaml_out, "w", encoding="utf-8") as yf:
        yaml.dump(export_cfg, yf, default_flow_style=False)

    print(f"[SUCCESS] Exported {exported_count} labeled image/box pairs to {out_root}")
    print(f"[INFO]    Skipped {skipped_count} records lacking valid image_crop_b64 or class mapping.")
    print(f"[INFO]    Generated Roboflow / CVAT compatible configuration: {data_yaml_out}")
    print("======================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Export Shaaru corrections to Roboflow/CVAT format")
    parser.add_argument("--output", default="dataset/shaaru_roboflow_export", help="Output export directory")
    parser.add_argument("--config", default="pipeline/cv/shaaru_apparel_36.yaml", help="Path to 36-class YAML")
    parser.add_argument("--jsonl", default="cv_corrections.jsonl", help="Path to correction log JSONL")
    args = parser.parse_args()

    export_roboflow_dataset(args.output, args.config, args.jsonl)

if __name__ == "__main__":
    main()
