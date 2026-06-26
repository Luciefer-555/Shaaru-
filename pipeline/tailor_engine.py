import asyncio
import json
import traceback
import logging
from config.models import get_async_client, MODELS
from extractors.vision_client import call_dual_vision

logger = logging.getLogger(__name__)
from extractors.prompts.vision_prompt import build_vision_prompt
from extractors.prompts.caption_prompt import build_caption_prompt
from extractors.prompts.combiner_prompt import build_combiner_prompt
from extractors.source_anchor import parse_product_metadata, build_anchor_block
from extractors.combiner_utils import enforce_unknown_fabric_candidates, _parse_llm_json
from validators.quality_gate import validate_product

async def _call_vision_model(image_b64: str, prompt: str) -> str:
    client = get_async_client(MODELS["vision_primary"]["provider"])
    model_name = MODELS["vision_primary"]["model"]
    
    # Using Gemini 1.5 Pro via OpenAI compatible endpoint
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ]
    for attempt in range(10):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=2048,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=45.0
                ),
                timeout=45.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision model error on attempt {attempt+1}: {e}")
            await asyncio.sleep(15)
    return "{}"

async def _extract_text_attributes(text: str) -> dict:
    from extractors.text_extractor import TEXT_PROMPT
    client = get_async_client(MODELS["text_extractor"]["provider"])
    model_name = MODELS["text_extractor"]["model"]
    
    prompt = TEXT_PROMPT.format(title="Product", description=text, tags=[])
    for attempt in range(10):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=45.0
                ),
                timeout=45.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Text model error on attempt {attempt+1}: {e}")
            await asyncio.sleep(15)
    return {}

async def _call_text_llm(prompt: str, json_mode: bool = False) -> str:
    client = get_async_client(MODELS["text_extractor"]["provider"])
    model_name = MODELS["text_extractor"]["model"]
    
    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.1
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs["timeout"] = 45.0
    for attempt in range(10):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs),
                timeout=45.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Combiner/Caption LLM error on attempt {attempt+1}: {e}")
            await asyncio.sleep(15)
    return "{}"

async def analyze_garment_deep(
    image_b64: str,
    product_page_text: str,
    db_fabric_ids: set[str],
    db_embellishment_ids: list[str],
    designer_config: dict,
    gates: list,
    price: str = "",
) -> dict:

    # ── 1. Parse product page → ground truth ──────────────────────────────────
    print("Parsing product metadata...")
    gt = parse_product_metadata(product_page_text)

    # ── 2. Build anchor block (populates gt["unknown_fabrics"] as side effect) ─
    print("Building anchor block...")
    anchor = build_anchor_block(gt, db_fabric_ids)

    is_menswear = None
    if gt.get("fabric_map"):
        # if "pants" or "trouser" in component names, likely menswear
        components = list(gt["fabric_map"].keys())
        is_menswear = any(
            c in components for c in ["pants", "trouser", "jacket", "sherwani"]
        )

    # ── 4. Call dual vision models ─────────────────────────────────────────────
    print("Calling vision models with dual-model routing...")
    print(f"IMAGE B64 LENGTH: {len(image_b64) if image_b64 else 'NONE'}")
    vision_raw, model_audit = await call_dual_vision(image_b64, anchor)
    print("VISION AUDIT:", model_audit)
    print("VISION KEYS:", list(vision_raw.keys()))

    if vision_raw.get("_models_used", {}).get("fabric_colour") == "failed" \
       and vision_raw.get("_models_used", {}).get("structure") == "failed":
        return {
            "error": "all_vision_models_failed",
            "needs_manual_review": True,
            "confidence_notes": "Both vision models failed.",
            "fabric_vocabulary": {"confirmed": [], "vision_only": [], "text_only": []},
            "techniques": {"confirmed": [], "vision_only": [], "text_only": []},
            "new_fabric_candidates": [],
            "caption": {},
        }

    # ── 5. Call text extractor (product page text only, no image) ──────────────
    print("Extracting text attributes...")
    text_raw = await _extract_text_attributes(product_page_text)

    # ── 6. Build combiner prompt and reconcile ─────────────────────────────────
    print("Reconciling outputs...")
    expected_aesthetic = ""
    for g in gates:
        if g.get("designer_id") == designer_config["id"]:
            expected_aesthetic = g.get("expected_aesthetic", "")
            break

    combiner_prompt = build_combiner_prompt(
        vision_raw=vision_raw,
        text_raw=text_raw,
        anchor_gt=gt,
        known_fabric_ids=db_fabric_ids,
        expected_aesthetic=expected_aesthetic,
    )
    # The combiner prompt specifically asks for JSON output without fences, but since it's a 70B model it might fence it.
    combined_str = await _call_text_llm(combiner_prompt, json_mode=True)
    output = _parse_llm_json(combined_str, label="combiner")

    # ── TECHNIQUE CLEANING ──────────────────────────────────────────────────
    if "techniques" in output and "confirmed" in output["techniques"]:
        conf_techs = output["techniques"]["confirmed"]
        if isinstance(conf_techs, list):
            clean_techs = []
            for t in conf_techs:
                if isinstance(t, str): clean_techs.append(t)
                elif isinstance(t, dict): clean_techs.append(t.get("name") or t.get("technique") or str(t))
            lower_techs = [t.lower() for t in clean_techs]
            if "embroidery" in lower_techs and len(conf_techs) > 1:
                output["techniques"]["confirmed"] = [t for t, lt in zip(conf_techs, lower_techs) if lt != "embroidery"]
                
    # ── FABRIC CLEANING ─────────────────────────────────────────────────────
    if "fabric_vocabulary" in output and "confirmed" in output["fabric_vocabulary"]:
        for fab in output["fabric_vocabulary"]["confirmed"]:
            if isinstance(fab, dict):
                fab["db_matched"] = True

    # ── SEASONAL FALLBACK RULE ───────────────────────────────────────────────
    ci = output.get("colour_intelligence", {})
    if isinstance(ci, dict):
        sw = ci.get("season_wearability")
        if not sw or sw.lower() == "all-year":
            techs = []
            if "techniques" in output:
                techs.extend(output["techniques"].get("confirmed", []))
                techs.extend(output["techniques"].get("vision_only", []))
            lower_techs = [str(t).lower() for t in techs]
            
            has_heavy_tech = any(
                "mirror" in t or "sheesha" in t or "zardozi" in t 
                for t in lower_techs
            )
            
            if has_heavy_tech:
                fam = ci.get("family", "").lower()
                if fam in ["jewel", "dark"]:
                    output["colour_intelligence"]["season_wearability"] = "winter"
                elif fam in ["pastel", "neutral"]:
                    output["colour_intelligence"]["season_wearability"] = "transitional"

    # ── 7. Override color from controlled lookup ───────────────────────────────
    if gt.get("color_hex"):
        output["color_palette"] = [gt["color_hex"]]
    if gt.get("color_name"):
        color_lower = gt["color_name"].lower()
        ch = output.get("color_harmony", {})
        if not isinstance(ch, dict):
            ch = {"description": ch}
        ch["color_name_confirmed"] = gt["color_name"]
        output["color_harmony"] = ch
        if "pastel" in color_lower or "beige" in color_lower or "champagne" in color_lower:
            output["color_harmony"]["family"] = "pastel"
            output["color_harmony"]["temperature"] = "warm"

    # ── 7b. Enforce Aesthetic ──────────────────────────────────────────────────
    if expected_aesthetic:
        output["aesthetic_category"] = expected_aesthetic

    # ── 8. Candidate forcer — must run last ────────────────────────────────────
    print("Enforcing unknown fabric candidates...")
    output = enforce_unknown_fabric_candidates(output, gt.get("unknown_fabrics", []), db_fabric_ids)

    # ── 9. Generate caption ────────────────────────────────────────────────────
    print("Generating caption...")
    craft_region = "kutch" if any(
        "mirror" in x.lower() or "sheesha" in x.lower() 
        for x in gt.get("techniques", [])
    ) else "generic"

    caption_prompt = build_caption_prompt(
        vision_output=output,
        product_metadata=gt,
        price=price,
        craft_region=craft_region,
    )
    caption_str = await _call_text_llm(caption_prompt, json_mode=True)
    output["caption"] = _parse_llm_json(caption_str, label="caption")
    output["region_of_craft"] = craft_region

    # ── QUALITY GATE VALIDATION ──────────────────────────────────────────────
    output = validate_product(output, designer_config["id"], gates, anchor_gt=gt)

    return output
