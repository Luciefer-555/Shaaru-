#!/usr/bin/env python3
"""
pipeline/train_yolo26_shaaru.py — SHAARU YOLO26 Fine-Tuning Script Scaffold

Automates the fine-tuning training loop for SHAARU's 36-class fashion & Indian ethnic wear
taxonomy (`shaaru_apparel_36.yaml`).

CRITICAL SAFEGUARD:
Before initiating PyTorch weights updates, this script automatically runs the dataset
balance auditor (`tools/check_dataset_balance.py`). If any of the 36 classes (e.g. sharara_set,
cape, leg_warmer) have fewer than `--min-instances` examples, training aborts immediately
to prevent producing an imbalanced checkpoint that fails on rarer categories.

Usage:
  python pipeline/train_yolo26_shaaru.py --data pipeline/cv/shaaru_apparel_36.yaml --epochs 50 --batch 16
"""

import argparse
import os
import sys
from pathlib import Path

# Dynamically find project root containing 'tools/check_dataset_balance.py'
_curr = Path(__file__).resolve().parent
while _curr != _curr.parent and not (_curr / "tools" / "check_dataset_balance.py").exists():
    _curr = _curr.parent
PROJECT_ROOT = _curr if (_curr / "tools" / "check_dataset_balance.py").exists() else Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.check_dataset_balance import audit_dataset_balance
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_dataset_balance", str(PROJECT_ROOT / "tools" / "check_dataset_balance.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit_dataset_balance = module.audit_dataset_balance

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 on Shaaru 36-class apparel taxonomy")
    parser.add_argument("--data", default="pipeline/cv/shaaru_apparel_36.yaml", help="Path to dataset YAML config")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Pretrained base checkpoint (yolo26n.pt / yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--min-instances", type=int, default=15, help="Minimum required instances per class before training")
    parser.add_argument("--force-train", action="store_true", help="Bypass balance check abort (NOT RECOMMENDED)")
    parser.add_argument("--project", default="pipeline/output/yolo26_runs", help="Save training runs dir")
    parser.add_argument("--name", default="shaaru_36class_run", help="Run name")
    args = parser.parse_args()

    data_yaml = str((PROJECT_ROOT / args.data).resolve() if not os.path.isabs(args.data) else Path(args.data))

    print("======================================================================")
    print("             SHAARU YOLO26 FINE-TUNING PIPELINE                       ")
    print("======================================================================")
    print(f"Dataset Config  : {data_yaml}")
    print(f"Base Checkpoint : {args.base_model}")
    print(f"Epochs / Batch  : {args.epochs} epochs | batch size {args.batch}")
    print(f"Balance Safeguard: Min {args.min_instances} instances per class\n")

    # 1. PRE-TRAINING DATASET BALANCE AUDIT
    print("[STEP 1/3] Running mandatory 36-class dataset balance audit...")
    passed, counts, flagged = audit_dataset_balance(
        yaml_path=data_yaml,
        min_instances=args.min_instances,
        warn_only=args.force_train
    )

    if not passed and not args.force_train:
        print("\n[ABORTED] Training stopped due to thin/missing classes identified above.")
        print(f"          Please add more labeled crops for: {', '.join(flagged)}")
        print("          Or run with --force-train if you explicitly wish to override.")
        sys.exit(1)

    if args.force_train:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        warn_msg = (
            f"======================================================================\n"
            f"[WARNING - TIMESTAMP: {ts}]\n"
            f"FORCE-TRAIN BYPASS ACTIVATED: The dataset balance check was explicitly overridden via --force-train.\n"
            f"Audit Status: passed={passed}, flagged_thin_classes ({len(flagged)}): {', '.join(flagged) if flagged else 'None'}\n"
            f"If the resulting YOLO26 model checkpoint exhibits poor detection accuracy on rare ethnic or\n"
            f"specialty categories, check this log first — the checkpoint was trained on an imbalanced dataset.\n"
            f"======================================================================\n"
        )
        print(warn_msg, file=sys.stderr)
        try:
            run_dir = Path(args.project) / args.name
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "balance_bypass_warning.log", "a", encoding="utf-8") as f:
                f.write(warn_msg + "\n")
        except Exception as log_err:
            print(f"[WARNING] Could not write bypass log to disk: {log_err}", file=sys.stderr)

    # 2. INITIALIZE ULTRALYTICS YOLO
    print("\n[STEP 2/3] Initializing Ultralytics PyTorch training engine...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics package not installed. Please run `pip install ultralytics`.", file=sys.stderr)
        sys.exit(1)

    base_ckpt = args.base_model
    if not os.path.exists(base_ckpt):
        print(f"[INFO] Base checkpoint '{base_ckpt}' not found locally; will download from Ultralytics assets.")

    model = YOLO(base_ckpt)

    # 3. KICK OFF TRAINING LOOP
    print("\n[STEP 3/3] Starting YOLO26 fine-tuning loop...")
    try:
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            # Fashion & apparel tailored hyperparams
            box=7.5,       # Box loss gain
            cls=0.5,       # Class loss gain (balanced across 36 classes)
            dfl=1.5,       # Distribution focal loss gain
            mosaic=1.0,    # Strong mosaic augmentation for multi-garment scenes
            mixup=0.15,    # Mixup to handle layered/overlapping fabrics (dupatta/jacket over kurta)
            copy_paste=0.1,# Copy-paste augmentation for rare accessories
            lr0=0.01,      # Initial learning rate
            lrf=0.01,      # Final learning rate fraction
            warmup_epochs=3.0,
            save=True,
            val=True
        )
        print("\n======================================================================")
        print(" [SUCCESS] YOLO26 Fine-Tuning Completed Successfully!")
        print(f" Best checkpoint saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
        print("======================================================================")
    except Exception as train_err:
        print(f"\n[ERROR] PyTorch training loop failed: {train_err}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
