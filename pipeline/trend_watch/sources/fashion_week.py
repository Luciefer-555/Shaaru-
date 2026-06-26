"""
Watches FDCI and Lakme Fashion Week coverage.
Fashion week = authoritative trend signal.
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
except Exception:
    tavily = None

FASHION_WEEK_SOURCES = [
    "fdci.org",
    "lakmefashionweek.co.in",
    "vogue.in",
    "harpersbazaar.in"
]


def scan_fashion_week() -> list:
    """
    Searches for recent fashion week coverage.
    Returns trend signals from runway data.
    """
    signals = []
    if not tavily:
        return signals
        
    queries = [
        "India Couture Week 2025 collection trends",
        "Lakme Fashion Week 2025 highlights",
        "FDCI 2025 designer showcases",
        "Indian fashion week runway trends 2025"
    ]
    
    for query in queries:
        try:
            results = tavily.search(
                query=query,
                include_domains=FASHION_WEEK_SOURCES,
                max_results=5,
                search_depth="advanced"
            )
            
            for article in results.get("results", []):
                signals.append({
                    "signal_type": "fashion_week",
                    "source": article.get("url", ""),
                    "title": article.get("title", ""),
                    "content": article.get("content", ""),
                    "confidence": 0.85
                })
                
        except Exception as e:
            print(f"Fashion week scan failed: {e}")
    
    return signals
