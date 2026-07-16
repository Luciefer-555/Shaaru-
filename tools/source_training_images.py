#!/usr/bin/env python3
"""
tools/source_training_images.py — SHAARU Tavily Image Harvester & Quality Filter

Harvests high-resolution isolated product/catalog images via Tavily API across our 36-class
apparel taxonomy (`shaaru_apparel_36.yaml`), prioritizing thin/zero-coverage ethnic categories.

Enforces strict visual quality and diversity filters:
1. Minimum resolution >= 640px on the shorter side (min(w, h) >= 640) for YOLO26 640px imgsz.
2. Uncluttered/Plain background check via corner pixel variance (rejects busy street/event photos).
3. Deduplication via perceptual image hashing (rejects near-duplicate product shots).
4. Domain dominance check (flags if any single website/vendor provides >40% of a class's images).

Outputs:
- Staging directory: staging/<class_name>/img_001.jpg ...
- Manifest CSV: staging/manifest.csv (filename, class, source_url, domain, width, height, status)

Usage:
  python tools/source_training_images.py --staging staging --target-per-class 25 --min-usable 15
"""

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image, ImageStat

# Priority 1: Zero coverage / flagged thin classes from dataset balance audit
PRIORITY_THIN_CLASSES = [
    "lehenga_set", "sharara_set", "dupatta", "saree", "anarkali_dress",
    "salwar_kameez_set", "co_ord_set", "cape", "leg_warmer", "tights_stockings",
    "headband_hair_accessory", "watch", "glove", "belt", "scarf",
    "hat", "glasses", "tie", "umbrella"
]

# Priority 2: Standard fill classes
STANDARD_FILL_CLASSES = [
    "kurta", "shirt_blouse", "top_t_shirt_sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "coat", "dress", "jumpsuit",
    "romper", "pants", "shorts", "skirt", "shoe",
    "sock", "bag_wallet"
]

# Specific search queries per class tailored for clean catalog/e-commerce photography
CLASS_SEARCH_QUERIES = {
    "lehenga_set": [
        "lehenga set product photo isolated white background",
        "lehenga choli flat lay product image",
        "lehenga set women e-commerce plain background"
    ],
    "sharara_set": [
        "sharara suit women product image plain background",
        "sharara set isolated studio product photo"
    ],
    "dupatta": [
        "dupatta only product photo plain background",
        "dupatta draped over shoulder isolated studio shot",
        "dupatta flat lay full length e-commerce"
    ],
    "saree": [
        "saree draped product photo front view plain background",
        "saree isolated white background e-commerce"
    ],
    "anarkali_dress": [
        "anarkali dress full length product photo plain background",
        "anarkali suit isolated studio view"
    ],
    "salwar_kameez_set": [
        "salwar kameez suit set women product photo plain background",
        "salwar suit isolated white background"
    ],
    "co_ord_set": [
        "women co ord set outfit product photo plain background",
        "two piece matching set fashion isolated"
    ],
    "cape": [
        "fashion cape outerwear product photo isolated",
        "women cape jacket plain background e-commerce"
    ],
    "leg_warmer": [
        "knit leg warmers product photo isolated white background",
        "dance leg warmers flat lay e-commerce"
    ],
    "tights_stockings": [
        "women tights stockings product photo isolated",
        "nylon stockings flat lay plain background"
    ],
    "headband_hair_accessory": [
        "headband hair accessory product photo isolated white background",
        "fashion hairband flat lay studio shot"
    ],
    "watch": [
        "wristwatch luxury product photo isolated white background",
        "watch front view plain studio background"
    ],
    "glove": [
        "fashion gloves product photo isolated white background",
        "leather winter gloves flat lay e-commerce"
    ],
    "belt": [
        "fashion waist belt product photo isolated plain background",
        "leather belt flat lay studio shot"
    ],
    "scarf": [
        "fashion scarf product photo isolated white background",
        "silk scarf flat lay full length e-commerce"
    ],
    "hat": [
        "fashion hat fedora cap product photo isolated white background",
        "hat headwear studio shot plain background"
    ],
    "glasses": [
        "eyeglasses sunglasses product photo isolated white background",
        "eyewear front view studio background"
    ],
    "tie": [
        "necktie silk product photo isolated white background",
        "fashion tie flat lay e-commerce"
    ],
    "umbrella": [
        "fashion rain umbrella open product photo isolated white background",
        "umbrella studio shot plain background"
    ],
}

# Generate standard fallback queries for remaining classes
for _cls in STANDARD_FILL_CLASSES:
    if _cls not in CLASS_SEARCH_QUERIES:
        readable = _cls.replace("_", " ")
        CLASS_SEARCH_QUERIES[_cls] = [
            f"{readable} product photo plain background e-commerce",
            f"{readable} isolated white background studio shot"
        ]

def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Computes perceptual difference hash (dHash) for near-duplicate image detection."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(resized.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(pixel_left > pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(diff):
        if value:
            decimal_value += 2 ** (index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, "0"))
            decimal_value = 0
    return "".join(hex_string)

def check_image_quality(img: Image.Image) -> Tuple[bool, str]:
    """
    Applies rigorous visual quality filters:
    1. Minimum resolution ~640px on shorter side.
    2. Aspect ratio check (rejects extreme banners/icons).
    3. Uncluttered background check via corner pixel standard deviation.
    """
    w, h = img.size
    shorter_side = min(w, h)
    if shorter_side < 640:
        return False, f"Resolution too low ({w}x{h}, min shorter side 640px)"

    aspect_ratio = w / float(h) if h > 0 else 0
    if aspect_ratio < 0.35 or aspect_ratio > 2.8:
        return False, f"Extreme aspect ratio ({aspect_ratio:.2f})"

    # Check background clutter by sampling 50x50 corner boxes
    try:
        rgb_img = img.convert("RGB")
        cw = min(50, w // 4)
        ch = min(50, h // 4)
        corners = [
            rgb_img.crop((0, 0, cw, ch)),
            rgb_img.crop((w - cw, 0, w, ch)),
            rgb_img.crop((0, h - ch, cw, h)),
            rgb_img.crop((w - cw, h - ch, w, h)),
        ]
        # Calculate standard deviation across corner regions
        corner_stds = []
        for c in corners:
            stat = ImageStat.Stat(c)
            # mean of std dev of R, G, B channels
            mean_std = sum(stat.stddev) / len(stat.stddev) if stat.stddev else 0.0
            corner_stds.append(mean_std)
        avg_corner_std = sum(corner_stds) / len(corner_stds) if corner_stds else 0.0
        
        # If corner std > 55.0, background is highly cluttered/busy (street/event/lifestyle)
        if avg_corner_std > 55.0:
            return False, f"Cluttered/Busy background detected (corner std {avg_corner_std:.1f} > 55.0)"
    except Exception as e:
        pass

    return True, "OK"

def harvest_class_images(
    tavily_client,
    cls_name: str,
    queries: List[str],
    target_count: int,
    staging_dir: Path,
    existing_hashes: Set[str],
    manifest_rows: List[dict]
) -> Tuple[int, Dict[str, int]]:
    """
    Queries Tavily for a single class, downloads/filters images, checks diversity, and saves to staging/.
    Returns (qualified_count, domain_counts_dict).
    """
    cls_dir = staging_dir / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    qualified_count = 0
    domain_counts: Dict[str, int] = {}
    seen_urls: Set[str] = set()

    for query in queries:
        if qualified_count >= target_count:
            break
        print(f"  -> Searching Tavily: '{query}'...")
        try:
            res = tavily_client.search(
                query=query,
                search_depth="advanced",
                include_images=True,
                max_results=20
            )
        except Exception as e:
            err_str = str(e)
            if "ForbiddenError" in err_str or "exceeds your plan" in err_str or "quota" in err_str.lower():
                print(f"\n[CRITICAL ERROR] Tavily API Quota Exceeded: {err_str}", file=sys.stderr)
                raise RuntimeError(f"Tavily Quota Exceeded: {err_str}")
            print(f"  [WARNING] Search error for '{query}': {err_str}", file=sys.stderr)
            continue

        images_list = res.get("images", [])
        if not images_list:
            continue

        for img_url in images_list:
            if qualified_count >= target_count:
                break
            if not img_url or not isinstance(img_url, str) or img_url in seen_urls:
                continue
            seen_urls.add(img_url)

            # Extract domain name
            try:
                domain = urlparse(img_url).netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = "unknown_domain"

            # Skip Pinterest/Instagram if possible in favor of clean catalog/e-commerce
            if any(social in domain for social in ("pinterest.", "instagram.", "facebook.", "tiktok.")):
                continue

            # Download and verify image
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(img_url, headers=headers, timeout=6)
                if resp.status_code != 200 or len(resp.content) < 5000:
                    continue

                img = Image.open(io.BytesIO(resp.content))
                # Check visual quality
                is_valid, reason = check_image_quality(img)
                if not is_valid:
                    continue

                # Check duplicate hash
                dhash = compute_dhash(img)
                if dhash in existing_hashes:
                    continue
                existing_hashes.add(dhash)

                # Save qualified image
                qualified_count += 1
                img_num = f"img_{qualified_count:03d}.jpg"
                out_file = cls_dir / img_num
                
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(out_file, "JPEG", quality=92)

                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                w, h = img.size
                manifest_rows.append({
                    "filename": f"{cls_name}/{img_num}",
                    "class": cls_name,
                    "source_url": img_url,
                    "domain": domain,
                    "width": w,
                    "height": h,
                    "status": "QUALIFIED"
                })
            except Exception:
                continue

    return qualified_count, domain_counts

def main():
    parser = argparse.ArgumentParser(description="Source and filter Shaaru training images via Tavily")
    parser.add_argument("--staging", default="staging", help="Output staging directory")
    parser.add_argument("--target-per-class", type=int, default=25, help="Target images per class")
    parser.add_argument("--min-usable", type=int, default=15, help="Minimum usable floor per class before aborting")
    args = parser.parse_args()

    try:
        import dotenv
        dotenv.load_dotenv()
    except Exception:
        pass

    # Initialize Tavily client
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_key:
        # Check custom_scraper or local fallback keys
        tavily_key = os.getenv("TAVILY_API_KEY_BACKUP", "")
    if not tavily_key:
        print("[ERROR] TAVILY_API_KEY is not set in environment or .env.", file=sys.stderr)
        sys.exit(1)

    try:
        from tavily import TavilyClient
        tclient = TavilyClient(api_key=tavily_key)
    except ImportError:
        print("[ERROR] tavily-python package is not installed.", file=sys.stderr)
        sys.exit(1)

    staging_dir = Path(args.staging)
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = staging_dir / "manifest.csv"

    manifest_rows: List[dict] = []
    existing_hashes: Set[str] = set()
    failed_thin_classes: List[str] = []
    domain_domination_warnings: List[str] = []
    class_summary: Dict[str, int] = {}

    print("======================================================================")
    print("           SHAARU TAVILY TRAINING IMAGE HARVESTER                     ")
    print("======================================================================")
    print(f"Staging Root     : {staging_dir.resolve()}")
    print(f"Target Quantity  : {args.target_per_class} images per class")
    print(f"Minimum Floor    : {args.min_usable} usable images per class\n")

    all_classes = PRIORITY_THIN_CLASSES + STANDARD_FILL_CLASSES

    try:
        for idx, cls_name in enumerate(all_classes, 1):
            queries = CLASS_SEARCH_QUERIES.get(cls_name, [f"{cls_name} product photo plain background"])
            print(f"[{idx}/{len(all_classes)}] Harvesting class: '{cls_name}' (queries: {len(queries)})...")
            
            cnt, dom_counts = harvest_class_images(
                tavily_client=tclient,
                cls_name=cls_name,
                queries=queries,
                target_count=args.target_per_class,
                staging_dir=staging_dir,
                existing_hashes=existing_hashes,
                manifest_rows=manifest_rows
            )
            class_summary[cls_name] = cnt

            # Check domain dominance (>40% from one vendor)
            if cnt > 0:
                top_domain, top_cnt = max(dom_counts.items(), key=lambda x: x[1])
                share = top_cnt / float(cnt)
                if share > 0.40 and cnt >= 5:
                    warn = f"{cls_name}: {top_domain} provided {top_cnt}/{cnt} ({share*100:.1f}%) images"
                    domain_domination_warnings.append(warn)
                    print(f"  ⚠️ DOMAIN DOMINANCE WARNING: {warn}")

            if cnt < args.min_usable:
                print(f"  ❌ THIN CLASS DETECTED: Only {cnt} usable images harvested (< {args.min_usable} floor)")
                failed_thin_classes.append(cls_name)
            else:
                print(f"  ✅ Harvested {cnt} high-quality 640px+ isolated catalog images.")

    except RuntimeError as quota_err:
        print("\n======================================================================")
        print(" [CRITICAL STOP] HARVESTING HALTED BY TAVILY QUOTA EXHAUSTION         ")
        print("======================================================================")
        print(f"Reason: {quota_err}\n")
        print("Tavily search requests failed because the current API plan limit is exceeded.")
        print("To proceed with automated image harvesting, please:")
        print("  1. Upgrade the Tavily plan quota or supply a fresh TAVILY_API_KEY inside .env")
        print("  2. Or place your sourced product images directly into staging/<class_name>/")
        print("     and rerun the manifest checker when ready.\n")
        # Save whatever rows we got before aborting
        if manifest_rows:
            with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "class", "source_url", "domain", "width", "height", "status"])
                writer.writeheader()
                writer.writerows(manifest_rows)
        sys.exit(2)

    # Save complete manifest CSV
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "source_url", "domain", "width", "height", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n======================================================================")
    print("                HARVEST & QUALITY FILTER SUMMARY                      ")
    print("======================================================================")
    print(f"Total Qualified Images : {len(manifest_rows)}")
    print(f"Manifest CSV Saved     : {manifest_csv.resolve()}\n")

    print(f"{'Class Name':<25} | {'Count':<10} | {'Status'}")
    print("-" * 55)
    for cls_name in all_classes:
        c = class_summary.get(cls_name, 0)
        st = "[FAIL (<15)]" if c < args.min_usable else "[OK]"
        print(f"{cls_name:<25} | {c:<10} | {st}")
    print("-" * 55)

    if domain_domination_warnings:
        print("\n[DOMAIN DOMINANCE WARNINGS (>40% from one vendor)]:")
        for dw in domain_domination_warnings:
            print(f"  - {dw}")
        print("  Please ensure single-vendor photography styles do not bias class detection.")

    if failed_thin_classes:
        print("\n[STOP AND REPORT TRIGGERED]:")
        print(f"The following {len(failed_thin_classes)} class(es) came back with fewer than {args.min_usable} usable images:")
        print(f"  -> {', '.join(failed_thin_classes)}")
        print("As instructed, we STOP HERE and will NOT proceed to export_roboflow_dataset.py or labeling.")
        sys.exit(1)

    print("\n[PASS] All 36 classes met the >=15 usable image floor with verified quality and diversity!")
    sys.exit(0)

if __name__ == "__main__":
    main()
