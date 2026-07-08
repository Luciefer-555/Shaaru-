#!/usr/bin/env python3
"""
test_fabric_model.py — Standalone Evaluation for Roboflow FabricClassv2 Model
-----------------------------------------------------------------------------
Evaluates the fabricclassv2/2 model against SHAARU's garment categories:
- Denim (jeans, jackets, co-ords)
- Knit (ribbed knit, jersey knit, cable knit)
- Leather (genuine biker, faux/vegan coat)
- Woven / Poplin / Linen shirts
- Indian D2C brand items (Torani Silk Organza, Studio Picante, Rimzim Dadu)

Compares Roboflow predictions side-by-side with SHAARU baseline scan logs
(Nemotron 3 Nano Omni / LLaMA 90B / Catalog DB ground truth).
"""

import os
import sys
import time
import json
import base64
import urllib.request
from pathlib import Path
from typing import Dict, Any, List

# Ensure utf-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try importing official SDK; fallback to drop-in HTTP Client if blocked by Python <3.13 metadata
try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    import requests
    class InferenceHTTPClient:
        """Drop-in replacement for Roboflow InferenceHTTPClient using standard requests."""
        def __init__(self, api_url: str, api_key: str):
            self.api_url = api_url.rstrip("/")
            self.api_key = api_key

        def infer(self, image_path_or_url: str, model_id: str) -> Dict[str, Any]:
            url = f"{self.api_url}/{model_id}?api_key={self.api_key}"
            if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
                try:
                    resp = requests.get(image_path_or_url, timeout=15)
                    resp.raise_for_status()
                    img_b64 = base64.b64encode(resp.content).decode("utf-8")
                    res = requests.post(url, data=img_b64, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
                except Exception:
                    res = requests.post(f"{url}&image={image_path_or_url}", timeout=30)
            else:
                with open(image_path_or_url, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                res = requests.post(url, data=img_b64, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
            
            res.raise_for_status()
            return res.json()


# ── CONFIGURATION ──────────────────────────────────────────────────────────
API_URL = "https://serverless.roboflow.com"
API_KEY = os.getenv("ROBOFLOW_API_KEY") or os.getenv('ROBOFLOW_API_KEY')
MODEL_ID = "fabricclassv2/2"
TEST_DIR = Path("test_garments")
RESULTS_FILE = Path("test_fabric_model_results.json")

# Curated Benchmark Dataset: 13 items covering all required test categories
BENCHMARK_ITEMS = [
    {
        "filename": "01_denim_jeans.jpg",
        "category": "Denim",
        "expected_fabric": "Denim",
        "shaaru_baseline": "Denim / Heavy Cotton (Nemotron)",
        "url": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=600&q=80"
    },
    {
        "filename": "02_denim_jacket.jpg",
        "category": "Denim",
        "expected_fabric": "Denim",
        "shaaru_baseline": "Denim / Structured Cotton (Nemotron)",
        "url": "https://images.unsplash.com/photo-1523205771623-e0faa4d2813d?w=600&q=80"
    },
    {
        "filename": "03_denim_coord.jpg",
        "category": "Denim",
        "expected_fabric": "Denim",
        "shaaru_baseline": "Structured Denim (SHAARU Catalog)",
        "url": "https://images.unsplash.com/photo-1582552938357-32b906df40cb?w=600&q=80"
    },
    {
        "filename": "04_ribbed_knit_sweater.jpg",
        "category": "Knit",
        "expected_fabric": "Ribbed Knit",
        "shaaru_baseline": "Ribbed Knit / Wool Blend (Nemotron)",
        "url": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=600&q=80"
    },
    {
        "filename": "05_jersey_knit_tshirt.jpg",
        "category": "Knit",
        "expected_fabric": "Jersey Knit",
        "shaaru_baseline": "Cotton Jersey / T-Shirt Knit (Nemotron)",
        "url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=600&q=80"
    },
    {
        "filename": "06_cable_knit_cardigan.jpg",
        "category": "Knit",
        "expected_fabric": "Cable Knit",
        "shaaru_baseline": "Heavy Knit / Wool (LLaMA 90B)",
        "url": "https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=600&q=80"
    },
    {
        "filename": "07_genuine_leather_jacket.jpg",
        "category": "Leather",
        "expected_fabric": "Genuine Leather",
        "shaaru_baseline": "Genuine Leather / Heavy Sheen (Nemotron)",
        "url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=600&q=80"
    },
    {
        "filename": "08_faux_vegan_leather_coat.jpg",
        "category": "Leather",
        "expected_fabric": "Faux Leather / Vegan",
        "shaaru_baseline": "Faux Leather / Polyurethane (Holy Headen DB)",
        "url": "https://images.unsplash.com/photo-1539533113208-f6df8cc8b543?w=600&q=80"
    },
    {
        "filename": "09_woven_linen_shirt.jpg",
        "category": "Woven / Linen",
        "expected_fabric": "Linen",
        "shaaru_baseline": "Linen / Breathable Woven (Meluku DB)",
        "url": "https://images.unsplash.com/photo-1598033129183-c4f50c736f10?w=600&q=80"
    },
    {
        "filename": "10_cotton_poplin_shirt.jpg",
        "category": "Woven / Poplin",
        "expected_fabric": "Cotton Poplin",
        "shaaru_baseline": "Cotton Poplin / Structured Woven (Nemotron)",
        "url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=600&q=80"
    },
    {
        "filename": "11_torani_silk_organza_saree.png",
        "category": "Indian D2C",
        "expected_fabric": "Silk Organza",
        "shaaru_baseline": "Silk Organza (Torani DB Confirmed)",
        "url": "https://cdn.shopify.com/s/files/1/0227/1271/3296/files/31-07-23TORANI10428_820776b7-3f9c-4dfa-917b-8353ad6d44a9.png?v=1698386197"
    },
    {
        "filename": "12_picante_ruched_slip_dress.jpg",
        "category": "Indian D2C",
        "expected_fabric": "Satin / Crepe",
        "shaaru_baseline": "Satin Crepe / Silk Blend (Studio Picante DB)",
        "url": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=600&q=80"
    },
    {
        "filename": "13_nicobar_linen_kurta.jpg",
        "category": "Indian D2C",
        "expected_fabric": "Linen Kurta",
        "shaaru_baseline": "Linen / Geometric Woven (Nicobar DB)",
        "url": "https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=600&q=80"
    }
]


def ensure_test_images():
    """Downloads benchmark images to TEST_DIR if not already present."""
    TEST_DIR.mkdir(exist_ok=True)
    print(f"[INFO] Verifying test image dataset in '{TEST_DIR}'...")
    for item in BENCHMARK_ITEMS:
        fpath = TEST_DIR / item["filename"]
        if not fpath.exists():
            print(f"   Downloading {item['filename']} ({item['category']})...")
            try:
                req = urllib.request.Request(item["url"], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response, open(fpath, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"   [WARN] Failed to download {item['filename']}: {e}")
    print("[OK] Test image dataset ready.\n")


def evaluate_model():
    client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
    
    results = []
    latencies = []
    
    print(f"[START] Starting evaluation of model '{MODEL_ID}' on {len(BENCHMARK_ITEMS)} items...\n")
    print("=" * 115)
    print(f"{'FILENAME':<30} | {'EXPECTED':<16} | {'ROBOFLOW GUESS':<16} | {'CONF':<6} | {'SHAARU BASELINE':<25} | {'LAT (s)':<7}")
    print("-" * 115)
    
    for item in BENCHMARK_ITEMS:
        fpath = str(TEST_DIR / item["filename"])
        if not os.path.exists(fpath):
            continue
            
        t0 = time.time()
        try:
            resp = client.infer(fpath, model_id=MODEL_ID)
            latency = time.time() - t0
            latencies.append(latency)
            
            # Extract top prediction
            predictions = resp.get("predictions", [])
            top_class = "unknown"
            top_conf = 0.0
            
            if isinstance(predictions, list) and len(predictions) > 0:
                pred = max(predictions, key=lambda x: x.get("confidence", 0))
                top_class = pred.get("class", pred.get("top", "unknown"))
                top_conf = pred.get("confidence", 0.0)
            elif isinstance(predictions, dict):
                top_class = predictions.get("top", "unknown")
                top_conf = predictions.get("confidence", 0.0)
            elif "top" in resp:
                top_class = resp["top"]
                top_conf = resp.get("confidence", 0.0)
                
            # Determine accuracy / alignment
            expected_lower = item["expected_fabric"].lower()
            guess_lower = top_class.lower()
            
            # Match heuristics
            is_match = False
            if guess_lower in expected_lower or expected_lower in guess_lower:
                is_match = True
            elif "denim" in guess_lower and "denim" in expected_lower:
                is_match = True
            elif "knit" in guess_lower and "knit" in expected_lower:
                is_match = True
            elif "leather" in guess_lower and "leather" in expected_lower:
                is_match = True
            elif "linen" in guess_lower and "linen" in expected_lower:
                is_match = True
            elif "cotton" in guess_lower and ("poplin" in expected_lower or "cotton" in expected_lower):
                is_match = True
            elif "silk" in guess_lower and "silk" in expected_lower:
                is_match = True
                
            status_tag = "[OK]" if is_match else ("[WARN]" if top_conf < 0.55 else "[ERR]")
            
            print(f"{item['filename']:<30} | {item['expected_fabric']:<16} | {status_tag} {top_class:<11} | {top_conf:<6.2f} | {item['shaaru_baseline']:<25} | {latency:<7.2f}")
            
            results.append({
                "filename": item["filename"],
                "category": item["category"],
                "expected_fabric": item["expected_fabric"],
                "shaaru_baseline": item["shaaru_baseline"],
                "roboflow_prediction": top_class,
                "confidence": top_conf,
                "latency_sec": round(latency, 3),
                "is_match": is_match,
                "raw_response": resp
            })
            
        except Exception as e:
            latency = time.time() - t0
            print(f"{item['filename']:<30} | {item['expected_fabric']:<16} | [ERR] {str(e)[:15]:<11} | 0.00   | {item['shaaru_baseline']:<25} | {latency:<7.2f}")
            results.append({
                "filename": item["filename"],
                "category": item["category"],
                "error": str(e),
                "latency_sec": round(latency, 3)
            })

    print("=" * 115)
    
    # ── REPORT SUMMARY ─────────────────────────────────────────────────────
    print("\n[REPORT] EVALUATION REPORT & METRICS ANALYSIS:")
    print("-" * 60)
    
    valid_results = [r for r in results if "error" not in r]
    total_valid = len(valid_results)
    
    if total_valid == 0:
        print("[ERROR] No valid inference results obtained. Please check API key and network.")
        return
        
    # 1. Accuracy per category
    cat_stats = {}
    for r in valid_results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        if r["is_match"]:
            cat_stats[cat]["correct"] += 1
            
    print("\n[1] ACCURACY PER CATEGORY:")
    for cat, stats in cat_stats.items():
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"    * {cat:<18}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
    # 2. Confidence Calibration & Confident Errors
    conf_correct = [r["confidence"] for r in valid_results if r["is_match"]]
    conf_wrong = [r["confidence"] for r in valid_results if not r["is_match"]]
    
    avg_conf_correct = sum(conf_correct) / len(conf_correct) if conf_correct else 0
    avg_conf_wrong = sum(conf_wrong) / len(conf_wrong) if conf_wrong else 0
    
    print("\n[2] CONFIDENCE CALIBRATION:")
    print(f"    * Avg Confidence (Correct Matches) : {avg_conf_correct:.2f}")
    print(f"    * Avg Confidence (Wrong Matches)   : {avg_conf_wrong:.2f}")
    
    confident_errors = [r for r in valid_results if not r["is_match"] and r["confidence"] >= 0.60]
    if confident_errors:
        print(f"    [WARN] Found {len(confident_errors)} CONFIDENTLY WRONG items (Conf >= 0.60):")
        for err in confident_errors:
            print(f"      - {err['filename']} ({err['category']}): Expected '{err['expected_fabric']}', but got '{err['roboflow_prediction']}' (Conf: {err['confidence']:.2f})")
    else:
        print("    [OK] No confident hallucinations (Conf >= 0.60 on wrong guesses). Well calibrated!")
        
    # 3. API Latency
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0
    min_lat = min(latencies) if latencies else 0
    
    print("\n[3] API LATENCY (cv_engine.py Concurrency Feasibility):")
    print(f"    * Avg Latency : {avg_lat:.2f}s")
    print(f"    * Min Latency : {min_lat:.2f}s")
    print(f"    * Max Latency : {max_lat:.2f}s")
    if avg_lat < 1.5:
        print("    [FAST] Latency is low enough to include as a 3rd concurrent race model in cv_engine.py!")
    else:
        print("    [MODERATE/SLOW] Might introduce lag if blocking; should run asynchronously or as fallback in cv_engine.py.")
        
    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": {
                "total_tested": total_valid,
                "overall_accuracy_pct": round((sum(1 for r in valid_results if r["is_match"]) / total_valid) * 100, 1),
                "avg_latency_sec": round(avg_lat, 3),
                "confident_errors_count": len(confident_errors)
            },
            "category_breakdown": cat_stats,
            "detailed_results": results
        }, f, indent=2)
    print(f"\n[INFO] Full detailed results saved to '{RESULTS_FILE}'.")


if __name__ == "__main__":
    ensure_test_images()
    evaluate_model()
