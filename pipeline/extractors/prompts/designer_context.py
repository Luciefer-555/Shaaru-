def build_designer_context_block(designer_config: dict, gate: dict) -> str:
    if not gate:
        return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGNER CONTEXT — READ BEFORE ANALYZING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Designer: {designer_config.get('name', 'Unknown')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGNER CONTEXT — READ BEFORE ANALYZING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Designer: {designer_config.get('name', 'Unknown')}
Known for: {designer_config.get('aesthetic_hint', 'N/A')}

EXPECTED to find on this designer's garments:
  Aesthetic:  {gate.get('expected_aesthetic', 'refer to visual evidence only')}
  Fabrics:    {', '.join(gate.get('primary_fabrics', []) or ['refer to visual evidence only'])}
  Techniques: {', '.join(gate.get('primary_techniques', []) or ['refer to visual evidence only'])}

MUST NOT appear on this designer's garments
(if you think you see these, look again — you are likely hallucinating):
  {', '.join(gate.get('must_not_appear', []))}

This designer context is a CALIBRATION GUIDE — 
not a permission to ignore visual evidence.
If you genuinely see something unexpected, report it 
with confidence: "uncertain" and flag in confidence_notes.
But if you were about to write one of the MUST NOT terms above,
stop and reconsider carefully.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

def build_fabric_colour_prompt_with_context(anchor_block: str, designer_config: dict, gates: list, product_metadata: dict = None) -> str:
    from extractors.vision_client import build_fabric_colour_prompt
    
    gate = next((g for g in gates if g["designer_id"] == designer_config["id"]), None)
    context_block = build_designer_context_block(designer_config, gate)
    
    original_prompt = build_fabric_colour_prompt(anchor_block)
    
    return f"{context_block}\n\n{original_prompt}"


def build_structure_embellishment_prompt_with_context(anchor_block: str, designer_config: dict, gates: list, product_metadata: dict = None) -> str:
    from extractors.vision_client import build_structure_embellishment_prompt
    
    gate = next((g for g in gates if g["designer_id"] == designer_config["id"]), None)
    context_block = build_designer_context_block(designer_config, gate)
    
    original_prompt = build_structure_embellishment_prompt(anchor_block)
    
    return f"{context_block}\n\n{original_prompt}"
