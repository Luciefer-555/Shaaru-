#!/usr/bin/env python3
import sys
import json
from cv_engine import _filter_body_parts

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def run_tests():
    print("==========================================================================")
    print("TESTING PHASE 1 BODY-PART DENYLIST AND TAXONOMY ALLOWLIST (_filter_body_parts)")
    print("==========================================================================")

    test_items = [
        {"label": "hand", "description": "a human hand holding garment", "category": "top"},
        {"label": "fingertip", "description": "finger touching cloth", "category": "accessory"},
        {"label": "person", "description": "person standing in room", "category": "person"},
        {"label": "thumb", "description": "thumb in frame", "category": "bottom"},
        {"label": "blue denim jeans", "description": "classic blue jeans", "category": "bottom"},
        {"label": "handloom cotton kurta", "description": "traditional Indian handloom kurta", "category": "top"},
        {"label": "silk saree", "description": "soft body feel and drapes on body beautifully", "category": "dress"},
        {"label": "leather handbag", "description": "black leather bag", "category": "bag_wallet"},
        {"label": "skin tone wrist", "description": "wrist area", "category": "accessory"}
    ]

    filtered = _filter_body_parts(test_items)
    print("\nOriginal count:", len(test_items))
    print("Filtered count:", len(filtered))
    
    kept_labels = [it["label"] for it in filtered]
    print("Kept items:", kept_labels)

    assert "hand" not in kept_labels, "'hand' should be filtered out"
    assert "fingertip" not in kept_labels, "'fingertip' should be filtered out"
    assert "person" not in kept_labels, "'person' should be filtered out"
    assert "thumb" not in kept_labels, "'thumb' should be filtered out"
    assert "skin tone wrist" not in kept_labels, "'skin tone wrist' should be filtered out"
    
    assert "blue denim jeans" in kept_labels, "'blue denim jeans' should be preserved"
    assert "handloom cotton kurta" in kept_labels, "'handloom cotton kurta' should be preserved despite 'handloom'"
    assert "silk saree" in kept_labels, "'silk saree' should be preserved despite 'drapes on body'"
    assert "leather handbag" in kept_labels, "'leather handbag' should be preserved"

    print("\n[SUCCESS] All Phase 1 denylist and allowlist test cases passed perfectly!")

if __name__ == "__main__":
    run_tests()
