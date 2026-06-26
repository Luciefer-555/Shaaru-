def enforce_unknown_fabric_candidates(output: dict, unknown_fabrics: list, db_fabric_ids: set) -> dict:
    """
    Forces the LLM output to respect the candidate pipeline.
    If an unknown fabric is found in the ground truth, it MUST go to candidates
    and MUST be removed from 'confirmed'.
    """
    if not unknown_fabrics:
        return output
        
    if "new_fabric_candidates" not in output:
        output["new_fabric_candidates"] = []
        
    fabric_vocab = output.get("fabric_vocabulary", {})
    confirmed = fabric_vocab.get("confirmed", [])
    
    for unknown in unknown_fabrics:
        if unknown in confirmed:
            confirmed.remove(unknown)
            
        # Ensure it's not already added
        already_present = any(c.get("name") == unknown for c in output["new_fabric_candidates"])
        if not already_present:
            output["new_fabric_candidates"].append({
                "name": unknown,
                "source": "brand_confirmed_unknown",
                "reason": "Brand explicitly named this fabric, but it was not found in our known database IDs."
            })
            output["needs_manual_review"] = True
            
    return output

def _parse_llm_json(raw_text: str, label: str = "") -> dict:
    import json
    if not raw_text or raw_text == "{}":
        return {}
        
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end >= start:
            raw_text = raw_text[start:end+1]
        return json.loads(raw_text)
    except Exception as e:
        print(f"[{label}] JSON parsing failed: {e}")
        return {}
