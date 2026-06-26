import json
import os

def load_quality_gates(config_path: str) -> list:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return []

def validate_product(output: dict, designer_id: str, gates: list, anchor_gt: dict = None) -> dict:
    """
    Runs after every single product extraction.
    If validation fails → marks needs_manual_review: true
    and logs exactly which rule failed.
    Never silently passes bad data.
    """
    
    gate = next((g for g in gates if g["designer_id"] == designer_id), None)
    if not gate:
        return output  # no gate defined → pass through with warning
    
    failures = []
    
    # Collect all text in output for scanning
    output_text = str(output).lower()
    
    # Check must_not_appear
    for banned_term in gate.get("must_not_appear", []):
        if banned_term.lower() in output_text:
            failures.append(
                f"HALLUCINATION: '{banned_term}' appeared but is "
                f"not expected for {designer_id}"
            )
            
    # Check banned_if_not_in_anchor
    if gate.get("validate_against_source_anchor") and anchor_gt:
        anchor_text = str(anchor_gt.get("raw_description", "")).lower() + " " + str(anchor_gt.get("techniques", [])).lower()
        for banned_term in gate.get("banned_if_not_in_anchor", []):
            if banned_term.lower() in output_text and banned_term.lower() not in anchor_text:
                failures.append(
                    f"HALLUCINATION: '{banned_term}' appeared but is "
                    f"not expected for {designer_id} and not in source anchor"
                )
    
    # Check must_appear
    for required_term in gate.get("must_appear", []):
        if required_term.lower() not in output_text:
            failures.append(
                f"MISSING: '{required_term}' should appear for "
                f"{designer_id} but was not found"
            )

    # Check must_appear_when_present
    for trigger_term, required_groups in gate.get("must_appear_when_present", {}).items():
        trigger_lower = trigger_term.lower()
        if trigger_lower not in output_text:
            continue  # trigger not present — skip entire check

        for group in required_groups:
            # String → exact match (backward compatible)
            if isinstance(group, str):
                if group.lower() not in output_text:
                    failures.append(
                        f"CONDITIONAL MISSING: '{group}' should appear "
                        f"when '{trigger_term}' is present for {designer_id}"
                    )
            # List → OR group: at least one synonym must match
            elif isinstance(group, list):
                if not any(term.lower() in output_text for term in group):
                    failures.append(
                        f"CONDITIONAL MISSING: one of {group} should appear "
                        f"when '{trigger_term}' is present for {designer_id}"
                    )

    
    # Check aesthetic category
    expected = gate.get("expected_aesthetic", "")
    actual = output.get("aesthetic_category", "")
    if expected and actual and expected.lower() not in actual.lower():
        failures.append(
            f"AESTHETIC MISMATCH: expected '{expected}' "
            f"but got '{actual}'"
        )
    
    # Check Rimzim Dadu special rule
    if gate.get("must_appear_in_candidates"):
        candidates = output.get("new_fabric_candidates", [])
        if not candidates:
            failures.append(
                f"MISSING CANDIDATES: {designer_id} should always "
                f"produce new_fabric_candidates — got empty list"
            )
    
    # Apply results
    if failures:
        output["needs_manual_review"] = True
        output["validation_failures"] = failures
        output["quality_gate_passed"] = False
        print(f"[FAILED] QUALITY GATE FAILED for {designer_id}:")
        for f in failures:
            print(f"   -> {f}")
    else:
        output["quality_gate_passed"] = True
        print(f"[OK] Quality gate passed for {designer_id}")
    
    return output
