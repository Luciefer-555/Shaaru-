"""
Source map and query builders for Tavily-backed knowledge verification.

Keeps source selection explicit so knowledge records are verified against
technical fabric, construction, sourcing, and measurement references.
"""

FABRIC_SOURCE_DOMAINS = [
    "fibre2fashion.com",
    "textileschool.com",
    "fabric.com",
    "moodfabrics.com",
    "craftsvilla.com",
    "jaypore.com",
]

CONSTRUCTION_SOURCE_DOMAINS = [
    "threadsmagazine.com",
    "craftsy.com",
    "burdastyle.com",
    "utsavfashion.com",
    "kalkifashion.com",
]

SOURCING_SOURCE_DOMAINS = [
    "justdial.com",
    "indiamart.com",
]

MEASUREMENT_SOURCE_DOMAINS = [
    "nift.ac.in",
    "academia.edu",
    "sizeindia.org",
]

CITY_MARKETS = {
    "bengaluru": ["Chickpet", "Commercial Street"],
    "bangalore": ["Chickpet", "Commercial Street"],
    "mumbai": ["Dharavi fabric market", "Mangaldas Market"],
    "delhi": ["Chandni Chowk", "Gandhi Nagar"],
    "chennai": ["Pondy Bazaar", "Purasawalkam"],
    "hyderabad": ["Laad Bazaar", "Begum Bazaar"],
    "kolkata": ["Metiabruz", "Burrabazar"],
    "surat": ["Ring Road fabric market"],
}


def site_query(query: str, domains: list[str]) -> list[str]:
    return [f"{query} site:{domain}" for domain in domains]


def fabric_queries(candidate: dict) -> list[str]:
    fabric_id = candidate.get("fabric_id", "fabric").replace("_", " ")
    common_names = " ".join(candidate.get("common_names", []))
    composition = candidate.get("fiber_composition", "")
    base = common_names or fabric_id
    queries = [
        f"{base} GSM weave fiber composition fabric properties",
        f"{base} drape structure hand feel textile specifications",
        f"{composition} {base} fabric GSM weave",
    ]
    return [q for query in queries for q in site_query(query, FABRIC_SOURCE_DOMAINS)]


def construction_queries(candidate: dict) -> list[str]:
    garment = candidate.get("garment_id", "garment").replace("_", " ")
    tradition = candidate.get("tradition", "")
    queries = [
        f"{garment} sewing construction sequence seam allowance",
        f"{garment} pattern construction tutorial critical points",
        f"{tradition} {garment} tailor construction notes",
    ]
    return [q for query in queries for q in site_query(query, CONSTRUCTION_SOURCE_DOMAINS)]


def sourcing_queries(candidate: dict, city: str) -> list[str]:
    city_key = city.lower()
    markets = CITY_MARKETS.get(city_key, [])
    ask_for = candidate.get("ask_for") or candidate.get("fabric_id", "fabric").replace("_", " ")
    market_query = " OR ".join(markets) if markets else f"{city} fabric market"
    queries = [
        f"{ask_for} wholesale price per meter {city} fabric market",
        f"{city} {market_query} textile fabric wholesale",
    ]
    return [q for query in queries for q in site_query(query, SOURCING_SOURCE_DOMAINS)]


def measurement_queries(candidate: dict) -> list[str]:
    garment = candidate.get("garment", "garment").replace("_", " ")
    gender = candidate.get("gender", "")
    queries = [
        f"ASTM D5585 body measurement sizing table {gender} {garment}",
        f"NIFT Indian body measurement standards {gender} {garment}",
        f"Size India survey body measurement {gender} height",
    ]
    return [q for query in queries for q in site_query(query, MEASUREMENT_SOURCE_DOMAINS)]
