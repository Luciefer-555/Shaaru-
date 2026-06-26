"""
trend_sources.py — SHAARU Trend Intelligence Source Configuration

Master source list, search queries, and LLM prompts for the
autonomous trend ingestion pipeline. No hardcoded values —
everything in structured constants.
"""

# ══════════════════════════════════════════════════════════════════
#  Hashtag Searches
# ══════════════════════════════════════════════════════════════════

HASHTAG_SEARCHES = [
    # Indian street style
    "#IndianFashion",
    "#DesiStreetStyle",
    "#IndianOOTD",
    "#BengaluruFashion",
    "#MumbaiFashion",
    "#DelhiFashion",

    # Aesthetic categories
    "#QuietLuxuryIndia",
    "#IndoWestern",
    "#IndianMinimalism",
    "#SustainableIndianFashion",
    "#IndianEditorial",

    # Brand signals
    "#Nicobar",
    "#RareRabbit",
    "#Snitch",
    "#FableStreet",
    "#Bewakoof",

    # Trend signals
    "#LakhmeFashionWeek",
    "#FDCI",
    "#IndianDesigner",
    "#IndianCouture",
]

# ══════════════════════════════════════════════════════════════════
#  Tavily Web Search Queries
# ══════════════════════════════════════════════════════════════════

TAVILY_QUERIES = [
    "Indian fashion trends 2026 women",
    "Vogue India latest style guide",
    "indie Indian D2C brands trending 2026",
    "quiet luxury India fashion aesthetic",
    "indo-western fusion outfits 2026",
    "Indian street style influencers 2026",
    "Lakmé Fashion Week 2026 trends",
    "sustainable Indian fashion brands",
    "Indian minimalist fashion aesthetic",
    "body type styling guide Indian women",
    "warm undertone outfit guide India",
    "pear shape styling India fashion",
    "oval face styling guide fashion",
    "Indian capsule wardrobe 2026",
    "Are.na Indian fashion boards",
    "Cosmos fashion aesthetics 2026",
    "Savee fashion editorial looks",
]

# ══════════════════════════════════════════════════════════════════
#  Are.na Channels
# ══════════════════════════════════════════════════════════════════

ARE_NA_CHANNELS = [
    "indian-fashion-editorial",
    "quiet-luxury-aesthetic",
    "indo-western-fashion",
    "south-asian-style",
    "minimalist-fashion-india",
]

# ══════════════════════════════════════════════════════════════════
#  Quality Gate
# ══════════════════════════════════════════════════════════════════

QUALITY_THRESHOLD = 7.0  # minimum quality score to save

# ══════════════════════════════════════════════════════════════════
#  LLM Prompts
# ══════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """
You are extracting fashion intelligence for an Indian fashion AI.

From the content below, extract:
1. aesthetic_name: specific name (e.g. "Contemporary Indian Minimalism")
2. aesthetic_description: 2-3 sentences
3. key_silhouettes: list of 2-4 silhouette names
4. key_colors: list of 3-5 colors
5. fabrics: list of 2-3 fabrics
6. styling_rules_do: list of 3-5 actionable rules
7. styling_rules_dont: list of 2-3 things to avoid
8. body_compatibility: list from [pear, hourglass, rectangle, inverted_triangle, apple]
9. occasion: list from [college, work, brunch, casual, nights_out, festive, weddings, editorial]
10. indian_context: 1 sentence about India-specific relevance
11. is_duplicate_risk: true/false — is this too similar to common aesthetics?

Return ONLY valid JSON. No markdown. No explanation.
If content has no clear fashion aesthetic, return {{"skip": true}}.

Content:
{content}
"""

QUALITY_PROMPT = """
Score this styling guide for an Indian fashion AI assistant (1-10):

Aesthetic: {aesthetic_name}
Description: {description}
Indian context: {indian_context}

Scoring criteria:
- Specificity: is it actionable and specific? (not generic)
- Indian relevance: does it apply to Indian D2C market?
- Uniqueness: is it genuinely different from basic Western fashion?
- Body inclusivity: does it work across body types?
- Practicality: can a real person wear this in Bengaluru/Mumbai/Delhi?

Return ONLY a JSON object: {{"score": 8.2, "reason": "one sentence"}}
"""
