# extractors/prompts/caption_prompt.py

# ── Banned vocabulary ──────────────────────────────────────────────────────────
# These words produce generic, low-trust captions. Never use them.
BANNED_WORDS = [
    "stunning", "beautiful", "gorgeous", "amazing", "exquisite",
    "luxurious", "opulent", "timeless", "effortlessly", "perfect",
    "elevate", "elevates", "masterpiece", "iconic", "heritage piece",
]

# ── Sheesha craft reference library ───────────────────────────────────────────
# Inject the relevant entry when sheesha/mirror work is confirmed.
SHEESHA_CRAFT_CONTEXT = {
    "kutch": (
        "Kutch mirror embroidery (sheesha work) originates from the Rabari and Ahir "
        "communities of Gujarat's Rann of Kutch. Traditionally, mica was used before "
        "glass mirrors became available. Each mirror is secured with a buttonhole stitch "
        "anchor ring called a 'tikki', with resham thread radiating outward. "
        "The scatter pattern varies by community — Rabari use irregular 'constellation' "
        "placement while Sodha Rajput work tends toward denser grid arrangements."
    ),
    "rajasthan": (
        "Rajasthani mirror work (sheesha) is practiced primarily in Barmer and Jaisalmer "
        "districts. The technique uses smaller mirrors than Kutch work, typically under 10mm, "
        "set in chain stitch frames with silk resham thread. Rajasthani sheesha is "
        "characteristically dense, covering large surface areas — the optical effect in "
        "candlelight is the traditional measure of quality."
    ),
    "generic": (
        "Sheesha (mirror) embroidery is one of the oldest living craft traditions in "
        "the Indian subcontinent, practiced across Gujarat and Rajasthan. Each mirror "
        "is individually hand-secured using a resham thread anchor stitch, making the "
        "craft inherently slow and labor-intensive — a heavily covered jacket of this "
        "scale represents hundreds of hours of artisan work."
    ),
}


def _safe_str_list(items):
    if not items:
        return ""
    res = []
    for x in items:
        if isinstance(x, str):
            res.append(x)
        elif isinstance(x, dict):
            res.append(str(x.get('name', x.get('technique', x.get('type', x)))))
        else:
            res.append(str(x))
    return ", ".join(res)


def build_caption_prompt(
    vision_output: dict,
    product_metadata: dict,
    price: str = "",
    craft_region: str = "generic",
) -> str:
    """
    Builds the caption generation prompt.

    vision_output: the assembled output dict from the vision pipeline
    product_metadata: parsed gt dict from source_anchor.parse_product_metadata()
    price: display price string e.g. "₹2,51,000"
    craft_region: "kutch" | "rajasthan" | "generic" — controls sheesha context injected
    """

    if price:
        try:
            price = f"₹{int(float(str(price).replace(',','').replace('₹',''))):,}"
        except ValueError:
            price = ""
    else:
        price = ""

    has_mirrors = _has_mirror_work(vision_output)
    has_velvet = _has_velvet(vision_output)

    sheesha_context = ""
    if has_mirrors:
        sheesha_context = SHEESHA_CRAFT_CONTEXT.get(craft_region, SHEESHA_CRAFT_CONTEXT["generic"])
    
    banned = ", ".join(f'"{w}"' for w in BANNED_WORDS)

    price_section = ""
    if price:
        price_section = f"""
PRICE INTELLIGENCE:
The garment is priced at {price}.
In the value_assessment field, justify this price with SPECIFIC reasoning:
- Name the techniques present and their labor intensity
- Reference the fabric grade (velvet pile weight, mirror count estimate)
- Cite the craft tradition and why it commands this tier
Do not say "reflects the quality" — say WHY specifically.
"""

    velvet_mirror_note = ""
    if has_mirrors and has_velvet:
        velvet_mirror_note = (
            "VELVET + MIRROR INTERACTION NOTE: "
            "Velvet pile creates a unique optical effect with mirrors — the pile surface "
            "absorbs and diffuses reflected light differently than a flat silk base would. "
            "Mirrors on velvet scatter light softly at oblique angles rather than creating "
            "sharp reflections. Reference this specific interaction in fabric_and_craft."
        )

    maala_note = ""
    if has_mirrors:
        maala_note = (
            "MAALA RULE: For this garment, the mirrors ARE the jewellery. "
            "The styling_context field MUST say to skip the maala/necklace. "
            "Do not suggest any neckpiece that would compete with mirror work on the chest."
        )

    return f"""You are the senior editorial voice for SHAARU, an AI fashion intelligence platform.
Your task: generate a structured caption object for a garment that will be shown to a user
seeking deep fashion knowledge — not flattery.

Your reader is educated, fashion-literate, and will distrust generic praise.
Teach them something specific. Give them information they couldn't get from looking at the photo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GARMENT FACTS (do not contradict these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fabrics confirmed: {_format_fabric_map(product_metadata)}
Techniques confirmed: {_safe_str_list(product_metadata.get('techniques', []))}
Color: {product_metadata.get('color_name', 'not specified')}
Garment type: {vision_output.get('garment_type', 'not specified')}
Components: {_safe_str_list(vision_output.get('components', []))}
Aesthetic category: {vision_output.get('aesthetic_category', '')}
Mirror scatter pattern: {_get_mirror_scatter(vision_output)}
Embroidery density: {vision_output.get('embroidery_density', '')}
{price_section}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRAFT TRADITION CONTEXT (use this knowledge, cite it specifically)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{sheesha_context}

{velvet_mirror_note}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STYLING RULES FOR THIS GARMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{maala_note}
Footwear: Never use the word "juttis". Instead, use "mojri". Recommend specific footwear that complements the garment's aesthetic and occasion without competing with its details (e.g., embroidered velvet mojri in champagne or ivory).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNED WORDS — NEVER USE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{banned}
If you find yourself reaching for one of these words, replace it with a
specific technical observation instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a JSON object with exactly these 5 keys.
No markdown fences. No preamble. Start with {{ and end with }}.

{{
  "fabric_and_craft": "<2-3 sentences. Describe the primary fabrics' texture, weight, and hand feel. Then specifically explain how the construction or embellishments interact with the fabric visually. Name any specific techniques or stitches detected.>",

  "cultural_significance": "<2 sentences. Give ONE piece of information the reader couldn't get from looking at the photo — historical, technical, or social context regarding the craft traditions or textiles used in this garment.>",

  "styling_context": "<2 sentences. Provide specific styling advice for this exact garment. First: what accessories to pair or skip based on the neckline and embellishment density. Second: exact footwear recommendation (e.g. style and color).>",

  "occasion_intelligence": "<1-2 sentences. Name the best occasion for this garment. Name ONE occasion it would be wrong for and explain the specific reason (not just 'too formal' — name the sensory conflict or aesthetic clash).>",

  "value_assessment": "<2 sentences. Justify the price point or luxury positioning with specific craft economics — technique labor hours, material grade, or artisan community.>"
}}

HARD RULES:
- Every sentence must contain at least one specific technical term or proper noun
- No JSON keys may appear in the prose
- No sentence may start with "This garment" more than once across all fields
- No field may use any banned word listed above
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_mirror_work(vision_output: dict) -> bool:
    techniques = vision_output.get("techniques", {})
    confirmed = [t.lower() for t in techniques.get("confirmed", []) if isinstance(t, str)]
    return any("mirror" in t or "sheesha" in t for t in confirmed)


def _has_velvet(vision_output: dict) -> bool:
    fab_vocab = vision_output.get("fabric_vocabulary", {})
    confirmed = [f.lower() for f in fab_vocab.get("confirmed", []) if isinstance(f, str)]
    vision_only = fab_vocab.get("vision_only", [])
    fabric_ids = [(v.get("fabric_id") or "").lower() for v in vision_only if isinstance(v, dict)]
    return any("velvet" in f for f in confirmed + fabric_ids)


def _get_mirror_scatter(vision_output: dict) -> str:
    embellishments = vision_output.get("embellishments", [])
    if isinstance(embellishments, dict):
        embellishments = embellishments.get("vision_only", [])
    for e in embellishments:
        details = e.get("mirror_details") or {}
        pattern = details.get("scatter_pattern")
        if pattern:
            return pattern
    return "not detected"


def _format_fabric_map(product_metadata: dict) -> str:
    fabric_map = product_metadata.get("fabric_map", {})
    if not fabric_map:
        return "not specified"
    return ", ".join(f"{comp}: {fabric}" for comp, fabric in fabric_map.items())
