from typing import Optional

# ── Mirror scatter vocab ──────────────────────────────────────────────────────
# "constellation" = irregular scattered mirrors, no grid, like stars
# "jaal"          = dense grid mesh, mirrors in a regular net pattern
# "boota"         = isolated single motif islands
# "border"        = mirrors concentrated at hem/cuff/neckline only
# "all-over"      = even coverage, no clear pattern logic
# "diagonal"      = mirrors follow a diagonal axis
MIRROR_SCATTER_PATTERNS = [
    "constellation",
    "jaal",
    "boota",
    "border",
    "all-over",
    "diagonal",
]

# ── Embroidery density vocab ──────────────────────────────────────────────────
EMBROIDERY_DENSITY_OPTIONS = [
    "all-over",        # covers the entire surface front + back
    "heavy-front",     # dense on front panel only
    "border-only",     # hem/cuffs/neckline only
    "scattered",       # isolated motifs with significant negative space
    "medium",          # moderate coverage, some negative space
    "light",           # sparse, decorative only
]

# ── Silhouette vocab ──────────────────────────────────────────────────────────
SILHOUETTE_OPTIONS = [
    "straight-cut-3-piece",
    "flared-achkan",
    "slim-fit-achkan",
    "kurta-palazzo-set",
    "kurta-dhoti-set",
    "indo-western-jacket",
    "bandgala-suit",
    "sherwani-with-dupatta",
]

# ── Garment type vocab ────────────────────────────────────────────────────────
GARMENT_TYPE_OPTIONS = [
    "achkan-jacket-set",
    "3-piece-sherwani-set",
    "indo-western-jacket-set",
    "bandgala-suit",
    "kurta-set",
    "lehenga-choli",
    "saree",
    "salwar-suit",
    "anarkali-set",
    "co-ord-set",
]


def build_vision_prompt(
    anchor_block: str,
    known_embellishment_ids: list[str],
    known_fabric_ids: list[str],
    is_menswear: Optional[bool] = None,
) -> str:
    """
    Assembles the full vision prompt with the source anchor injected at the top.

    anchor_block: output of build_anchor_block() from source_anchor.py
    known_embellishment_ids: list of valid embellishment_id strings from DB
    known_fabric_ids: list of valid fabric_id strings from DB
    is_menswear: if known from product page, pass True/False; else None
    """

    gender_context = ""
    if is_menswear is True:
        gender_context = (
            "This is MENSWEAR. Apply menswear body logic — "
            "broad shoulders, flat chest, straight torso, no bust dart logic. "
            "Do not use womenswear silhouette vocabulary."
        )
    elif is_menswear is False:
        gender_context = "This is womenswear."

    fabric_id_list = "\n".join(f"  - {fid}" for fid in sorted(known_fabric_ids)[:60])
    embellishment_id_list = "\n".join(
        f"  - {eid}" for eid in sorted(known_embellishment_ids)[:40]
    )
    mirror_options = ", ".join(f'"{p}"' for p in MIRROR_SCATTER_PATTERNS)
    density_options = ", ".join(f'"{d}"' for d in EMBROIDERY_DENSITY_OPTIONS)
    silhouette_options = ", ".join(f'"{s}"' for s in SILHOUETTE_OPTIONS)
    garment_type_options = ", ".join(f'"{g}"' for g in GARMENT_TYPE_OPTIONS)

    return f"""{anchor_block}

{gender_context}

You are a master Indian fashion technologist with deep expertise in:
- Indian craft traditions (zardozi, resham, sheesha/mirror work, gota, dabka, kantha)
- Fabric identification from visual cues (weave structure, pile, drape, sheen, opacity)
- Garment construction and silhouette vocabulary
- Regional craft communities (Kutch mirror workers, Lucknow chikankari, Varanasi zari)

Analyze this garment image with extreme precision. Your output must be valid JSON only —
no preamble, no markdown fences, no trailing prose.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 1 — FABRIC IDENTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EACH visible fabric:
1. Check against the GROUND TRUTH ANCHOR above first.
2. If the brand has confirmed a fabric for a component, use it — observe its visual properties,
   do not substitute a different fabric.
3. If brand says "Special Silk" and that string is NOT in the known fabric IDs below,
   set db_matched = false and is_new_candidate = true. Never invent a known substitute.

Known fabric_ids in our database (match against these only):
{fabric_id_list}
... (truncated for prompt length — match against full DB at runtime)

For each fabric, output:
{{
  "fabric_id": "<closest match from known fabric_ids, or null if no match>",
  "location_on_garment": "<component name>",
  "confidence": "confirmed" | "probable" | "uncertain",
  "is_new_candidate": true | false,
  "new_candidate_description": "<if is_new_candidate, describe fully: fiber, weave, hand feel, sheen, drape>",
  "visual_observations": {{
    "weave_structure": "<pile_weave | plain_weave | twill | satin_weave | null>",
    "surface_texture": "<pile_soft | smooth | matte | grainy | slubbed | null>",
    "sheen_level": "high" | "medium" | "low" | "none",
    "drape_observation": "<how does this fabric hang — stiff, fluid, structured, flowing>",
    "opacity": "opaque" | "semi-sheer" | "sheer"
  }},
  "db_matched": true | false
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 2 — EMBELLISHMENT IDENTIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Known embellishment_ids in our database:
{embellishment_id_list}

MIRROR WORK CRITICAL RULE:
If you see mirrors/sheesha work, you MUST:
1. Identify the scatter pattern from this exact list: {mirror_options}
   "constellation" = irregular, random scatter like stars in a sky. NO GRID.
   "jaal" = regular grid mesh. Only use if mirrors form a clear net pattern.
   Look carefully at the spacing and arrangement before deciding.
2. Identify if resham thread connects or outlines the mirrors (very common in sheesha craft).
3. Note the mirror size: micro (<5mm), small (5-15mm), medium (15-30mm), large (>30mm).

HALLUCINATION GUARD — CRITICAL:
The following embellishments are commonly hallucinated. Only report them if you
are genuinely certain they are present:
  - zardozi: requires visible GOLD METAL wire threadwork. If you see only resham
    thread, do NOT report zardozi.
  - dabka: requires visible coiled metal wire, usually gold. Distinct from resham.
  - gota: requires visible flat metallic ribbon strips.

For each embellishment:
{{
  "embellishment_id": "<from known list, or null>",
  "location_on_garment": "<component>",
  "confidence": "confirmed" | "probable" | "uncertain",
  "mirror_details": {{
    "scatter_pattern": "<one of: {mirror_options}, or null if no mirrors>",
    "mirror_size": "micro" | "small" | "medium" | "large" | null,
    "connecting_thread": "resham" | "zari" | "none" | null
  }},
  "is_new_candidate": true | false,
  "new_candidate_description": "<if new, describe the technique precisely>"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 3 — GARMENT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Embroidery density — choose from: {density_options}
This describes overall embellishment coverage across the entire garment.
"all-over" means both front AND back are heavily covered.

Garment type — choose from: {garment_type_options}
Be specific. "sherwani" alone is not acceptable — is it a 3-piece set?
An achkan jacket over an inner kurta? Name it precisely.

Silhouette — choose from: {silhouette_options}
Describe the actual cut, not the garment category.

Components — list ALL visible pieces:
e.g. ["jacket", "inner kurta", "straight pants"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 4 — COLOR INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: If the GROUND TRUTH ANCHOR above confirms a color name,
use ONLY that color family. Do NOT invent hex values —
output the confirmed color name and null for hex.
The color pipeline will resolve hex from the controlled lookup table.

{{
  "color_name_confirmed": "<from anchor, or null>",
  "color_family": "pastel" | "earth" | "jewel" | "neutral" | "dark" | "bright",
  "temperature": "warm" | "cool" | "neutral",
  "metallic_accent": "gold" | "silver" | "rose-gold" | "none",
  "contrast_level": "high" | "medium" | "low"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 5 — SURFACE PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Only report if there is a PRINTED or WOVEN pattern separate from embellishment.
If the garment's visual complexity comes entirely from embellishment (mirrors,
embroidery), set all fields to null.

{{
  "has_surface_print": true | false,
  "print_method": "digital" | "block" | "screen" | "woven-jacquard" | null,
  "motif_type": "<describe if present, else null>",
  "motif_placement": "all-over" | "border" | "scattered" | "panel" | null
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION 6 — STYLING DNA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "aesthetic_category": "<single best-fit label, e.g. 'Modern Minimalist' or 'Traditional Heritage'>",
  "aesthetic_justification": "<one sentence, cite specific visual evidence>",
  "occasion_suitability": ["<wedding>", "<formal event>"],
  "occasion_not_suitable_for": ["<describe occasions this garment would be wrong for>"],
  "trend_position": "classic" | "trending" | "avant-garde" | "legacy",
  "gender": "menswear" | "womenswear" | "unisex",
  "styling_observations": "<one sentence, specific to THIS garment>"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return a single JSON object with these top-level keys:
  fabrics, embellishments, embroidery_density, garment_type, silhouette,
  components, color_intelligence, surface_patterns, styling_dna

No other keys. No markdown. No preamble. Start your response with {{ and end with }}.
"""
