"""
Scrapes Indian fashion editorial articles.
Stores article text, designer mentions,
technique vocabulary, and styling language.
"""

import os
import json
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
except Exception:
    tavily = None

_col = None
def _get_col():
    global _col
    if _col is None:
        client = MongoClient(os.environ["MONGODB_URI"])
        db = client[os.getenv('MONGODB_DB', 'shaaru_db')]
        _col = db["editorial"]
    return _col

EDITORIAL_SOURCES = [
    {"name": "Vogue India", "domain": "vogue.in", "weight": 1.0},
    {"name": "Harper's Bazaar India", "domain": "harpersbazaar.in", "weight": 0.9},
    {"name": "Elle India", "domain": "elle.in", "weight": 0.85},
    {"name": "The Voice of Fashion", "domain": "thevoiceoffashion.com", "weight": 0.95}
]

SCRAPE_QUERIES = [
    "Indian bridal fashion editorial",
    "lehenga styling guide India",
    "saree draping styles editorial",
    "Indian designer interview techniques",
    "festive wear India fashion guide",
    "Indian fashion week designer review",
    "Indian textile craft tradition editorial",
    "mirror work embroidery editorial",
    "zardozi handwork India editorial",
    "block print natural dye India editorial"
]


def scrape_editorial(max_articles: int = 100) -> list:
    """
    Scrapes editorial articles from all sources.
    Stores in MongoDB editorial collection.
    """
    articles = []
    if not tavily:
        return articles
    col = _get_col()
    
    for source in EDITORIAL_SOURCES:
        print(f"Scraping {source['name']}...")
        for query in SCRAPE_QUERIES:
            try:
                results = tavily.search(
                    query=query,
                    include_domains=[source["domain"]],
                    max_results=3,
                    search_depth="advanced"
                )
                
                for article in results.get("results", []):
                    if col.find_one({"url": article.get("url", "")}):
                        continue
                    
                    doc = {
                        "source": source["name"],
                        "source_weight": source["weight"],
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", ""),
                        "published_date": article.get("published_date", ""),
                        "scraped_at": datetime.utcnow().isoformat(),
                        "query": query,
                        "processed": False
                    }
                    
                    col.insert_one(doc)
                    articles.append(doc)
                    if len(articles) >= max_articles:
                        break
            except Exception as e:
                print(f"Scrape failed ({source['name']}/{query}): {e}")
            if len(articles) >= max_articles:
                break
        if len(articles) >= max_articles:
            break
            
    print(f"Scraped {len(articles)} new articles")
    return articles
