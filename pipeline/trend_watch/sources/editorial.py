"""
Scans Indian fashion editorial for trend signals.
Sources: Vogue India, Harper's Bazaar India,
         Elle India, The Voice of Fashion
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
except Exception:
    tavily = None

EDITORIAL_SOURCES = [
    {
        "name": "Vogue India",
        "domain": "vogue.in",
        "url": "https://www.vogue.in/fashion"
    },
    {
        "name": "Harper's Bazaar India",
        "domain": "harpersbazaar.in",
        "url": "https://www.harpersbazaar.in/fashion"
    },
    {
        "name": "Elle India",
        "domain": "elle.in",
        "url": "https://elle.in/fashion"
    },
    {
        "name": "The Voice of Fashion",
        "domain": "thevoiceoffashion.com",
        "url": "https://www.thevoiceoffashion.com"
    }
]

TREND_QUERIES = [
    "new Indian bridal fashion trends",
    "latest lehenga styles India",
    "trending saree draping styles",
    "new collection Indian designers",
    "Indian fashion week highlights",
    "trending embroidery techniques India",
    "new season Indian ethnic wear"
]


def scan_editorial() -> list:
    """
    Scans editorial sources for trend signals.
    Returns raw trend signals before scoring.
    """
    raw_signals = []
    if not tavily:
        return raw_signals
        
    for source in EDITORIAL_SOURCES:
        for query in TREND_QUERIES:
            try:
                results = tavily.search(
                    query=query,
                    include_domains=[source["domain"]],
                    max_results=5,
                    search_depth="advanced"
                )
                
                for article in results.get("results", []):
                    raw_signals.append({
                        "source": source["name"],
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", ""),
                        "published_date": article.get("published_date", ""),
                        "query": query
                    })
                    
            except Exception as e:
                print(f"Editorial scan failed ({source['name']}): {e}")
    
    return raw_signals
