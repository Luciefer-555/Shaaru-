#!/usr/bin/env python3
"""
test_cv_engine_fabric.py — Evaluate cv_engine.py fixed fabric taxonomy & uncertainty gate
-----------------------------------------------------------------------------------------
Runs scan_frame() across the 13 benchmark images in test_garments/ TWICE per image to test:
- Run-to-run consistency (Run 1 vs Run 2 flip-flops) at temperature=0.0
- Accuracy against fixed taxonomy
- How often "uncertain" is returned
"""

import os
import sys
import time
import json
import base64
from pathlib import Path

# Ensure utf-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from cv_engine import scan_frame

TEST_DIR = Path("test_garments")
RESULTS_FILE = Path("test_cv_engine_fabric_results.json")

BENCHMARK_ITEMS = [
    {
        "filename": "01_denim_jeans.jpg",
        "category": "Denim",
        "expected_fabric": "denim",
        "before_baseline": "Denim / Heavy Cotton"
    },
    {
        "filename": "02_denim_jacket.jpg",
        "category": "Denim",
        "expected_fabric": "denim",
        "before_baseline": "Denim / Structured Cotton"
    },
    {
        "filename": "03_denim_coord.jpg",
        "category": "Denim",
        "expected_fabric": "denim",
        "before_baseline": "Structured Denim"
    },
    {
        "filename": "04_ribbed_knit_sweater.jpg",
        "category": "Knit",
        "expected_fabric": "ribbed knit",
        "before_baseline": "Ribbed Knit / Wool Blend"
    },
    {
        "filename": "05_jersey_knit_tshirt.jpg",
        "category": "Knit",
        "expected_fabric": "jersey knit",
        "before_baseline": "Cotton Jersey / T-Shirt Knit"
    },
    {
        "filename": "06_cable_knit_cardigan.jpg",
        "category": "Knit",
        "expected_fabric": "cable knit",
        "before_baseline": "Heavy Knit / Wool"
    },
    {
        "filename": "07_genuine_leather_jacket.jpg",
        "category": "Leather",
        "expected_fabric": "genuine leather",
        "before_baseline": "Genuine Leather / Heavy Sheen"
    },
    {
        "filename": "08_faux_vegan_leather_coat.jpg",
        "category": "Leather",
        "expected_fabric": "faux/pu leather",
        "before_baseline": "Faux Leather / Polyurethane"
    },
    {
        "filename": "09_woven_linen_shirt.jpg",
        "category": "Woven / Linen",
        "expected_fabric": "linen",
        "before_baseline": "Linen / Breathable Woven"
    },
    {
        "filename": "10_cotton_poplin_shirt.jpg",
        "category": "Woven / Poplin",
        "expected_fabric": "poplin",
        "before_baseline": "Cotton Poplin / Structured Woven"
    },
    {
        "filename": "11_torani_silk_organza_saree.png",
        "category": "Indian D2C",
        "expected_fabric": "organza|silk",
        "before_baseline": "Silk Organza"
    },
    {
        "filename": "12_picante_ruched_slip_dress.jpg",
        "category": "Indian D2C",
        "expected_fabric": "satin/crepe|silk",
        "before_baseline": "Satin Crepe / Silk Blend"
    },
    {
        "filename": "13_nicobar_linen_kurta.jpg",
        "category": "Indian D2C",
        "expected_fabric": "linen",
        "before_baseline": "Linen / Geometric Woven"
    }
]

def check_match(fab_type, expected_fabric):
    if fab_type == "uncertain":
        return "[UNCERTAIN]"
    expected_opts = expected_fabric.lower().split("|")
    if any(opt in fab_type for opt in expected_opts) or any(fab_type in opt for opt in expected_opts):
        return "[OK]"
    elif "leather" in fab_type and "leather" in expected_fabric:
        return "[OK]"
    elif "knit" in fab_type and "knit" in expected_fabric:
        return "[OK]"
    elif "denim" in fab_type and "denim" in expected_fabric:
        return "[OK]"
    else:
        return "[MISMATCH]"

def run_evaluation():
    print(f"[START] Evaluating cv_engine.py scan_frame() at temp=0.0 on {len(BENCHMARK_ITEMS)} images (2 runs each)...\n")
    print("=" * 135)
    print(f"{'FILENAME':<28} | {'EXPECTED':<15} | {'RUN 1 FABRIC':<16} | {'RUN 2 FABRIC':<16} | {'CONSISTENT?':<11} | {'STATUS (R1/R2)'}")
    print("-" * 135)
    
    results = []
    consistent_count = 0
    correct_r1 = 0
    correct_r2 = 0
    uncertain_r1 = 0
    uncertain_r2 = 0
    total_tested = 0
    flip_flops = []
    
    for item in BENCHMARK_ITEMS:
        fpath = TEST_DIR / item["filename"]
        if not fpath.exists():
            print(f"[WARN] Missing image: {fpath}")
            continue
            
        total_tested += 1
        
        try:
            with open(fpath, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
            # Run 1
            t0 = time.time()
            scan_res_1 = scan_frame(img_b64, user_id="test_eval_1")
            lat_1 = time.time() - t0
            items_1 = scan_res_1.get("items", [])
            top_1 = max(items_1, key=lambda x: x.get("confidence", 0.0)) if items_1 else {}
            fab_1 = str(top_1.get("fabric_type", "missing")).strip().lower()
            status_1 = check_match(fab_1, item["expected_fabric"])
            if status_1 == "[OK]": correct_r1 += 1
            if status_1 == "[UNCERTAIN]": uncertain_r1 += 1
            
            # Run 2
            t0 = time.time()
            scan_res_2 = scan_frame(img_b64, user_id="test_eval_2")
            lat_2 = time.time() - t0
            items_2 = scan_res_2.get("items", [])
            top_2 = max(items_2, key=lambda x: x.get("confidence", 0.0)) if items_2 else {}
            fab_2 = str(top_2.get("fabric_type", "missing")).strip().lower()
            status_2 = check_match(fab_2, item["expected_fabric"])
            if status_2 == "[OK]": correct_r2 += 1
            if status_2 == "[UNCERTAIN]": uncertain_r2 += 1
            
            is_consistent = (fab_1 == fab_2)
            if is_consistent:
                consistent_count += 1
                cons_str = "YES"
            else:
                cons_str = "NO (FLIP)"
                flip_flops.append((item["filename"], fab_1, fab_2))
                
            print(f"{item['filename']:<28} | {item['expected_fabric']:<15} | {fab_1:<16} | {fab_2:<16} | {cons_str:<11} | {status_1} / {status_2}")
            
            results.append({
                "filename": item["filename"],
                "expected_fabric": item["expected_fabric"],
                "run1": {"fabric_type": fab_1, "status": status_1, "confidence": top_1.get("confidence"), "latency_sec": round(lat_1, 2)},
                "run2": {"fabric_type": fab_2, "status": status_2, "confidence": top_2.get("confidence"), "latency_sec": round(lat_2, 2)},
                "is_consistent": is_consistent
            })
            
        except Exception as e:
            print(f"{item['filename']:<28} | {item['expected_fabric']:<15} | [ERR]            | [ERR]            | NO          | [ERR]")
            results.append({
                "filename": item["filename"],
                "error": str(e)
            })

    print("=" * 135)
    
    print("\n[REPORT] TEMPERATURE=0.0 RUN-TO-RUN CONSISTENCY & ACCURACY SUMMARY:")
    print("-" * 70)
    print(f"Total Tested         : {total_tested}")
    print(f"Run-to-Run Consistent: {consistent_count}/{total_tested} ({round(consistent_count/total_tested*100, 1) if total_tested else 0}%)")
    print(f"Accuracy (Run 1)     : {correct_r1}/{total_tested} ({round(correct_r1/total_tested*100, 1) if total_tested else 0}%) | Uncertain: {uncertain_r1}")
    print(f"Accuracy (Run 2)     : {correct_r2}/{total_tested} ({round(correct_r2/total_tested*100, 1) if total_tested else 0}%) | Uncertain: {uncertain_r2}")
    if flip_flops:
        print("\n[FLIP-FLOPS DETECTED]:")
        for fname, f1, f2 in flip_flops:
            print(f"  - {fname}: Run 1 = '{f1}' vs Run 2 = '{f2}'")
    else:
        print("\n[FLIP-FLOPS]: None! 100% Run-to-Run Consistency achieved.")
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature_tested": 0.0,
            "summary": {
                "total_tested": total_tested,
                "consistent_count": consistent_count,
                "consistency_pct": round(consistent_count/total_tested*100, 1) if total_tested else 0,
                "accuracy_r1_pct": round(correct_r1/total_tested*100, 1) if total_tested else 0,
                "accuracy_r2_pct": round(correct_r2/total_tested*100, 1) if total_tested else 0,
                "flip_flops": flip_flops
            },
            "detailed_results": results
        }, f, indent=2)
    print(f"\n[INFO] Detailed evaluation results saved to '{RESULTS_FILE}'.")

if __name__ == "__main__":
    run_evaluation()
