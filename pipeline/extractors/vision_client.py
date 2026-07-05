# extractors/vision_client.py
#
# Dual-model vision pipeline — NVIDIA NIM only.
#
# MODEL A  qwen/qwen2-vl-72b-instruct
#          Speciality: fabric texture, colour accuracy, drape behaviour,
#                      surface sheen, weave structure, weight estimation.
#          Why: Qwen2-VL has superior fine-grained texture discrimination.
#
# MODEL B  meta/llama-3.2-90b-vision-instruct  (11B fallback)
#          Speciality: spatial layout, component identification, garment type,
#                      silhouette, embellishment placement and scatter pattern.
#          Why: Llama vision is stronger at spatial/structural reasoning.
#
# Both run via asyncio.gather() — no sequential wait.
# merge_vision_outputs() reconciles into one dict.

import asyncio
import logging
import os
from typing import Optional

from openai import AsyncOpenAI

from extractors.prompts.designer_context import (
    build_fabric_colour_prompt_with_context,
    build_structure_embellishment_prompt_with_context,
)

logger = logging.getLogger(__name__)

_nvidia = AsyncOpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    timeout=12.0,
    max_retries=0,
)

_PROFESSIONAL_SYSTEM = (
    "You are a professional fashion technologist performing structured garment "
    "analysis for a B2B fashion intelligence platform. Analyze commercial product "
    "photography of clothing and textiles. Output only valid JSON as instructed. "
    "This is a technical analysis task."
)

# ─────────────────────────────────────────────────────────────────────────────
# FOCUSED PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def build_fabric_colour_prompt(anchor_block: str) -> str:
    return f"""{anchor_block}

You are analyzing ONLY the following attributes from this garment image.
Do not comment on garment structure, silhouette, or embellishment placement.
Focus entirely on material and colour intelligence.

Output a single JSON object with exactly these keys:

{{
  "fabrics": [
    {{
      "component": "<jacket | inner kurta | pants | etc>",
      "fabric_id_guess": "<closest known fabric_id or null>",
      "is_new_candidate": true | false,
      "new_candidate_description": "<if new: fiber estimate, weave, hand feel, sheen>",
      "weave_structure": "pile_weave | plain_weave | twill | satin_weave | null",
      "surface_texture": "pile_soft | smooth | matte | grainy | slubbed | null",
      "sheen_level": "high | medium | low | none",
      "opacity": "opaque | semi-sheer | sheer",
      "drape_observation": "<how does this fabric hang — stiff, fluid, structured, flowing>",
      "drape_score_estimate": <1-10 integer, 1=boardlike 10=liquid>,
      "weight_estimate": "light | medium | heavy",
      "seasonal_read": "summer | winter | transitional | all-year",
      "confidence": "confirmed | probable | uncertain"
    }}
  ],
  "color_intelligence": {{
    "color_name_from_image": "<what colour you actually see>",
    "color_family": "pastel | earth | jewel | neutral | dark | bright",
    "temperature": "warm | cool | neutral",
    "metallic_accent": "gold | silver | rose-gold | none",
    "contrast_level": "high | medium | low",
    "color_notes": "<describe the exact tone — e.g. warm champagne beige, not just beige>"
  }},
  "surface_patterns": {{
    "has_surface_print": true | false,
    "print_method": "digital | block | screen | woven-jacquard | null",
    "motif_type": "<describe if present, else null>",
    "comes_from_embellishment_not_print": true | false
  }}
}}

No markdown. No preamble. Start with {{ end with }}.
"""


def build_structure_embellishment_prompt(anchor_block: str) -> str:
    return f"""{anchor_block}

You are analyzing ONLY the following attributes from this garment image.
Do not comment on fabric type, colour, or drape.
Focus entirely on garment structure and embellishment.

MIRROR WORK — READ THIS BEFORE ANALYZING:
If mirrors are present, identify the scatter pattern from EXACTLY this list:
  "constellation" = irregular random scatter, like stars. No grid. No repeat.
  "jaal"          = dense regular grid mesh. Clear repeat unit.
  "boota"         = isolated single motif islands with space between.
  "border"        = mirrors at hem / cuffs / neckline only.
  "all-over"      = even coverage, no clear directional logic.
  "diagonal"      = mirrors follow a clear diagonal axis.
Look at the SPACING and ARRANGEMENT carefully before deciding.
For Abhinav Mishra garments, if spacing is irregular, default to "constellation".

EMBELLISHMENT HALLUCINATION GUARD:
  zardozi  = requires visible GOLD METAL WIRE. Resham thread ≠ zardozi.
  dabka    = requires visible COILED METAL WIRE. Do not confuse with resham.
  gota     = requires visible FLAT METALLIC RIBBON strips.

Output a single JSON object with exactly these keys:

{{
  "garment_type": "<specific — e.g. 3-piece-sherwani-set | achkan-jacket-set>",
  "silhouette": "<straight-cut-3-piece | flared-achkan | slim-fit-achkan | bandgala-suit | kurta-set>",
  "gender": "menswear | womenswear | unisex",
  "components": ["<jacket>", "<inner kurta>", "<straight pants>"],
  "embellishments": [
    {{
      "embellishment_id": "<resham_thread | sheesha_mirror | zardozi_thread_gold | etc, or null>",
      "component": "<jacket | inner kurta | pants>",
      "confidence": "confirmed | probable | uncertain",
      "mirror_details": {{
        "scatter_pattern": "<one of the 6 options above, or null if no mirrors>",
        "mirror_size": "micro | small | medium | large | null",
        "connecting_thread": "resham | zari | none | null",
        "coverage_estimate_percent": <0-100 integer, how much of the surface>
      }},
      "is_new_candidate": true | false,
      "new_candidate_description": "<if new>"
    }}
  ],
  "embroidery_density": "all-over | heavy-front | border-only | scattered | medium | light",
  "embroidery_density_notes": "<front + back coverage description>",
  "outfit_completeness": {{
    "is_set": true | false,
    "includes": ["<list all visible pieces>"],
    "missing_for_complete_look": ["<e.g. dupatta, if relevant>"]
  }},
  "styling_dna": {{
    "aesthetic_category": "<single label, specific — e.g. Sheesha Heritage Maximalism>",
    "aesthetic_justification": "<one sentence citing specific visual evidence>",
    "occasion_suitability": ["<wedding>", "<formal event>"],
    "occasion_not_suitable_for": ["<e.g. outdoor summer mehendi — velvet + heat>"],
    "trend_position": "classic | trending | avant-garde | legacy"
  }}
}}

No markdown. No preamble. Start with {{ end with }}.
"""


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CALLERS
# ─────────────────────────────────────────────────────────────────────────────

async def _call_model(
    model: str,
    image_b64: str,
    prompt: str,
    mime_type: str = "image/jpeg",
    label: str = "",
) -> Optional[str]:
    try:
        import asyncio
        coro = _nvidia.chat.completions.create(
            model=model,
            max_tokens=4096,
            messages=[
                {
                    "role": "system",
                    "content": _PROFESSIONAL_SYSTEM,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
        response = await asyncio.wait_for(coro, timeout=12.0)
        content = response.choices[0].message.content
        if not content or len(content.strip()) < 20:
            logger.warning("%s returned empty response", label or model)
            return None
        logger.info("Vision OK — %s", label or model)
        return content

    except Exception as e:
        print(f"Vision FAILED (1st try) — {label or model}: {e}")
        logger.warning("Vision FAILED — %s: %s. Retrying after 5 seconds...", label or model, e)
        await asyncio.sleep(5)
        try:
            import asyncio
            coro = _nvidia.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "system",
                        "content": _PROFESSIONAL_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            )
            response = await asyncio.wait_for(coro, timeout=12.0)
            content = response.choices[0].message.content
            if not content or len(content.strip()) < 20:
                logger.warning("%s returned empty response on retry", label or model)
                return None
            logger.info("Vision OK (Retry) — %s", label or model)
            return content
        except Exception as retry_e:
            print(f"Vision FAILED (Retry) — {label or model}: {retry_e}")
            logger.error("Vision FAILED (Retry) — %s: %s", label or model, retry_e)
            return None


async def _call_structure_model(
    image_b64: str,
    prompt: str,
    mime_type: str,
) -> tuple[Optional[str], str]:
    """90B primary, 11B fallback."""
    result = await _call_model(
        "meta/llama-3.2-90b-vision-instruct",
        image_b64, prompt, mime_type,
        label="Llama-90B-structure",
    )
    if result:
        return result, "meta/llama-3.2-90b-vision-instruct"

    logger.warning("Structure model 90B failed — falling back to 11B")
    result = await _call_model(
        "meta/llama-3.2-11b-vision-instruct",
        image_b64, prompt, mime_type,
        label="Llama-11B-structure-fallback",
    )
    return result, "meta/llama-3.2-11b-vision-instruct"


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_vision_outputs(
    fabric_output: dict,
    structure_output: dict,
) -> dict:
    """
    Merges Model A (fabric/colour) + Model B (structure/embellishment)
    into a single unified vision output dict.

    Conflict resolution:
    - fabric_output owns: fabrics, color_intelligence, surface_patterns
    - structure_output owns: garment_type, silhouette, gender, components,
                             embellishments, embroidery_density, styling_dna,
                             outfit_completeness
    - Neither can override the anchor ground truth — that happens in tailor_engine.
    """
    merged = {}

    # ── From fabric model ──────────────────────────────────────────────────
    merged["fabrics"]            = fabric_output.get("fabrics", [])
    merged["color_intelligence"] = fabric_output.get("color_intelligence", {})
    merged["surface_patterns"]   = fabric_output.get("surface_patterns", {})

    # ── From structure model ───────────────────────────────────────────────
    merged["garment_type"]       = structure_output.get("garment_type", "")
    merged["silhouette"]         = structure_output.get("silhouette", "")
    merged["gender"]             = structure_output.get("gender", "menswear")
    merged["components"]         = structure_output.get("components", [])
    merged["embellishments"]     = structure_output.get("embellishments", [])
    merged["embroidery_density"] = structure_output.get("embroidery_density", "")
    merged["embroidery_density_notes"] = structure_output.get("embroidery_density_notes", "")
    merged["outfit_completeness"] = structure_output.get("outfit_completeness", {})
    merged["styling_dna"]        = structure_output.get("styling_dna", {})

    # ── Cross-check: drape from fabric model enriches structure ───────────
    # Build a component → drape map for the combiner to use
    drape_map = {}
    for f in merged["fabrics"]:
        comp = f.get("component", "")
        if comp:
            drape_map[comp] = {
                "drape_observation": f.get("drape_observation"),
                "drape_score_estimate": f.get("drape_score_estimate"),
                "weight_estimate": f.get("weight_estimate"),
                "seasonal_read": f.get("seasonal_read"),
            }
    merged["drape_by_component"] = drape_map

    # ── Audit trail ───────────────────────────────────────────────────────
    merged["_models_used"] = {
        "fabric_colour": fabric_output.get("_model_used", "nvidia/nemotron-nano-12b-v2-vl"),
        "structure":     structure_output.get("_model_used", "llama-3.2-90b"),
    }

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

async def call_dual_vision(
    image_b64: str,
    anchor_block: str,
    mime_type: str = "image/jpeg",
) -> tuple[dict, dict]:
    """
    Runs both vision models concurrently.

    Returns:
        (merged_output, model_audit)
        merged_output: unified vision dict ready for combiner
        model_audit:   {"fabric_colour": model_name, "structure": model_name}

    Usage in tailor_engine.py:
        from extractors.vision_client import call_dual_vision
        vision_merged, audit = await call_dual_vision(image_b64, anchor_block)
    """
    from extractors.combiner_utils import _parse_llm_json

    fabric_prompt    = build_fabric_colour_prompt(anchor_block)
    structure_prompt = build_structure_embellishment_prompt(anchor_block)

    # Run both concurrently
    (fabric_raw, structure_result) = await asyncio.gather(
        _call_model(
            "nvidia/nemotron-nano-12b-v2-vl",
            image_b64, fabric_prompt, mime_type,
            label="Nemotron-Nano-12B-fabric-colour",
        ),
        _call_structure_model(image_b64, structure_prompt, mime_type),
    )

    structure_raw, structure_model = structure_result

    # Parse both
    fabric_dict    = _parse_llm_json(fabric_raw or "", label="fabric_colour_model")
    structure_dict = _parse_llm_json(structure_raw or "", label="structure_model")

    # Tag for audit trail
    fabric_dict["_model_used"]    = "nvidia/nemotron-nano-12b-v2-vl"
    structure_dict["_model_used"] = structure_model

    # Handle partial failures gracefully
    if not fabric_dict:
        logger.error("Fabric/colour model returned no parseable output")
        fabric_dict = {"_model_used": "failed", "_partial": True}

    if not structure_dict:
        logger.error("Structure model returned no parseable output")
        structure_dict = {"_model_used": "failed", "_partial": True}

    merged = merge_vision_outputs(fabric_dict, structure_dict)
    audit  = merged.get("_models_used", {})

    return merged, audit
