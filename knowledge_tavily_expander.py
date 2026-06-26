"""
Harvest source-backed knowledge evidence with Tavily.

This does not write final fabric/construction/measurement records directly.
It stores raw source snippets in MongoDB so they can be reviewed or promoted
through the existing verification/seeding pipeline.
"""

import argparse
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import UpdateOne
from tavily import TavilyClient

from knowledge_sources import (
    CITY_MARKETS,
    FABRIC_SOURCE_DOMAINS,
    CONSTRUCTION_SOURCE_DOMAINS,
    SOURCING_SOURCE_DOMAINS,
    MEASUREMENT_SOURCE_DOMAINS,
    site_query,
)
from shaaru_brain import _get_db

load_dotenv()

FABRIC_TOPICS = [
    "cotton poplin GSM weave fiber composition drape",
    "linen fabric GSM weave hand feel structure",
    "chanderi fabric cotton silk weave regional sourcing",
    "banarasi brocade fabric weave zari structure GSM",
    "khadi cotton fabric handspun weave texture GSM",
    "georgette fabric GSM drape fiber composition",
    "organza fabric GSM crisp structure weave",
    "modal jersey fabric GSM stretch drape",
]

FABRIC_NAMES = [
    "cotton poplin", "cotton lawn", "cotton cambric", "cotton voile", "cotton muslin",
    "cotton twill", "cotton drill", "cotton canvas", "cotton duck", "cotton sateen",
    "cotton jersey", "cotton rib knit", "cotton fleece", "cotton flannel", "cotton gauze",
    "cotton dobby", "cotton jacquard", "cotton denim", "cotton chambray", "cotton seersucker",
    "khadi cotton", "handloom cotton", "ikat cotton", "ajrakh cotton", "kalamkari cotton",
    "linen plain weave", "linen suiting", "linen cambric", "linen slub", "linen viscose blend",
    "ramie linen", "hemp fabric", "bamboo cotton", "bamboo jersey", "modal jersey",
    "tencel twill", "tencel lyocell", "cupro fabric", "rayon challis", "viscose crepe",
    "viscose georgette", "viscose satin", "viscose twill", "viscose linen", "poly viscose suiting",
    "wool suiting", "worsted wool", "wool crepe", "wool gabardine", "wool flannel",
    "wool tweed", "wool melton", "wool felt", "cashmere fabric", "merino wool knit",
    "silk charmeuse", "silk satin", "silk crepe de chine", "silk georgette", "silk chiffon",
    "silk organza", "silk dupion", "raw silk", "silk habotai", "silk noil",
    "silk velvet", "silk brocade", "silk taffeta", "silk faille", "silk shantung",
    "chanderi cotton silk", "maheshwari silk cotton", "banarasi brocade", "banarasi silk", "kanjivaram silk",
    "tussar silk", "matka silk", "muga silk", "eri silk", "paithani silk",
    "polyester georgette", "polyester chiffon", "polyester satin", "polyester crepe", "polyester organza",
    "polyester taffeta", "polyester twill", "polyester suiting", "polyester mesh", "polyester scuba",
    "polyester jersey", "polyester fleece", "polyester velvet", "microfiber polyester", "crepe knit",
    "nylon taffeta", "nylon ripstop", "nylon mesh", "nylon spandex", "power mesh",
    "spandex lycra", "stretch velvet", "stretch satin", "stretch crepe", "stretch denim",
    "ponte knit", "interlock knit", "single jersey knit", "double jersey knit", "french terry",
    "rib knit", "waffle knit", "thermal knit", "tricot knit", "warp knit mesh",
    "denim rigid", "denim stretch", "denim chambray", "bull denim", "corduroy",
    "gabardine", "twill suiting", "herringbone", "houndstooth", "pinstripe suiting",
    "jacquard fabric", "brocade zari", "damask", "matelasse", "cloque",
    "lace chantilly", "lace guipure", "lace corded", "lace embroidered", "lace net",
    "net embroidered", "tulle", "illusion tulle", "soft net", "can can net",
    "organza nylon", "organza polyester", "crystal organza", "glass organza", "tissue organza",
    "chiffon silk", "chiffon polyester", "georgette silk", "georgette faux", "gorgette satin",
    "crepe back satin", "moss crepe", "bubble crepe", "banana crepe", "heavy crepe",
    "satin duchess", "satin crepe", "satin silk", "satin lycra", "satin polyester",
    "velvet silk", "velvet polyester", "velvet crushed", "velvet burnout", "velour",
    "suede faux", "leather faux", "vegan leather", "patent leather faux", "nubuck faux",
    "sherpa fleece", "boucle", "terry cloth", "towelling", "quilted fabric",
    "interfacing woven", "fusible interfacing", "horsehair canvas", "buckram", "lining acetate",
    "lining polyester", "lining cotton voile", "lining satin", "lining habotai", "lining mesh",
    "mashru silk", "jamdani cotton", "jamdani silk", "phulkari fabric", "kantha cotton",
    "bandhani cotton", "bandhani georgette", "leheriya chiffon", "leheriya georgette", "bagru cotton",
    "dabu cotton", "ajrakh modal silk", "kota doria cotton", "kota silk", "mangalagiri cotton",
    "pochampally ikat", "sambalpuri ikat", "patola silk", "patola cotton", "narayanpet cotton",
    "gadwal silk", "venkatagiri cotton", "chikankari cotton", "chikankari georgette", "mukaish fabric",
    "zardozi net", "sequined mesh", "sequin georgette", "beaded lace", "embroidered organza",
    "embroidered tulle", "schiffli cotton", "eyelet cotton", "cutwork cotton", "applique fabric",
    "neoprene scuba", "bonded crepe", "laminated cotton", "metallic lame", "foil print jersey",
    "digital print crepe", "block print cotton", "screen print rayon", "hand painted silk", "batik cotton",
    "faux fur", "pile fleece", "minky fabric", "chenille", "upholstery jacquard",
    "taffeta silk", "taffeta polyester", "faille fabric", "ottoman rib fabric", "gazar silk",
    "peau de soie", "mikado silk", "bridal satin", "bridal net", "bridal lace",
]

CONSTRUCTION_TOPICS = [
    "wide leg trousers sewing construction sequence seam allowance",
    "straight trousers pattern construction darts inseam crotch seam",
    "kurta sewing construction side slits neckline facing",
    "anarkali kurta construction panels bodice waist seam",
    "saree blouse construction darts lining sleeve attachment",
    "single breasted blazer construction lapel lining sleeve head",
    "lehenga skirt construction panels lining waistband can can",
]

MEASUREMENT_TOPICS = [
    "ASTM D5585 body measurement sizing table",
    "NIFT Indian body measurement standards sizing chart",
    "Size India survey body measurements height gender",
]


def _client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def _save_results(domain: str, query: str, results: list[dict], db) -> int:
    operations = []
    now = datetime.now(timezone.utc).isoformat()
    for result in results:
        url = result.get("url")
        if not url:
            continue
        operations.append(
            UpdateOne(
                {"domain": domain, "url": url},
                {
                    "$set": {
                        "domain": domain,
                        "url": url,
                        "title": result.get("title"),
                        "content": result.get("content"),
                        "score": result.get("score"),
                        "query": query,
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        )
    if not operations:
        return 0
    return db["knowledge_source_evidence"].bulk_write(operations).upserted_count


def _harvest_domain(client: TavilyClient, domain: str, queries: list[str], db, max_results: int) -> int:
    inserted = 0
    for query in queries:
        print(f"[TAVILY] {domain}: {query}")
        result = client.search(query, max_results=max_results)
        inserted += _save_results(domain, query, result.get("results", []), db)
    return inserted


def build_queries() -> dict[str, list[str]]:
    fabric_seed_queries = [
        q
        for topic in FABRIC_TOPICS
        for q in site_query(topic, FABRIC_SOURCE_DOMAINS)
    ]
    fabric_name_queries = []
    for index, fabric_name in enumerate(FABRIC_NAMES):
        domain = FABRIC_SOURCE_DOMAINS[index % len(FABRIC_SOURCE_DOMAINS)]
        fabric_name_queries.append(
            f"{fabric_name} fabric GSM weave fiber composition drape structure site:{domain}"
        )
    fabric_queries = fabric_name_queries + fabric_seed_queries
    construction_queries = [
        q
        for topic in CONSTRUCTION_TOPICS
        for q in site_query(topic, CONSTRUCTION_SOURCE_DOMAINS)
    ]
    sourcing_topics = []
    for city, markets in CITY_MARKETS.items():
        for market in markets:
            sourcing_topics.append(f"{city} {market} fabric wholesale textile market price")
    sourcing_queries = [
        q
        for topic in sourcing_topics
        for q in site_query(topic, SOURCING_SOURCE_DOMAINS)
    ]
    measurement_queries = [
        q
        for topic in MEASUREMENT_TOPICS
        for q in site_query(topic, MEASUREMENT_SOURCE_DOMAINS)
    ]
    return {
        "fabric": fabric_queries,
        "construction": construction_queries,
        "sourcing": sourcing_queries,
        "measurement": measurement_queries,
    }


def harvest_all(
    max_queries_per_domain: int = 12,
    max_results: int = 3,
    domains: list[str] | None = None,
) -> dict:
    db = _get_db()
    if db is None:
        raise RuntimeError("Database unavailable")

    client = _client()
    query_map = build_queries()
    if domains:
        selected_domains = {domain.strip() for domain in domains}
        query_map = {domain: queries for domain, queries in query_map.items() if domain in selected_domains}

    summary = {}
    for domain, queries in query_map.items():
        selected = queries[:max_queries_per_domain]
        summary[domain] = {
            "queries": len(selected),
            "inserted": _harvest_domain(client, domain, selected, db, max_results),
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest Tavily knowledge evidence into MongoDB.")
    parser.add_argument("--max-queries-per-domain", type=int, default=12)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument(
        "--domains",
        default="fabric,construction,sourcing,measurement",
        help="Comma-separated domains: fabric,construction,sourcing,measurement",
    )
    args = parser.parse_args()

    print("[START] Tavily knowledge evidence harvest")
    domains = [domain.strip() for domain in args.domains.split(",") if domain.strip()]
    print(harvest_all(args.max_queries_per_domain, args.max_results, domains))
