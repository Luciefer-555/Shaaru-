import os
from tavily import TavilyClient

EDITORIAL_SOURCES = [
    {"name": "Vogue India", "url": "https://www.vogue.in/fashion", "type": "magazine"},
    {"name": "Harper's Bazaar India", "url": "https://www.harpersbazaar.in/fashion", "type": "magazine"},
    {"name": "The Voice of Fashion", "url": "https://www.thevoiceoffashion.com", "type": "trade"},
    {"name": "Elle India", "url": "https://elle.in/fashion", "type": "magazine"},
]

def get_tavily_client():
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY not found in environment")
    return TavilyClient(api_key=key)

def scrape_editorial(query: str):
    """
    Scrape editorial pages using Tavily.
    For this first run, editorial scraping is marked active=False in config, 
    but the logic here is ready to be used.
    """
    client = get_tavily_client()
    # E.g. query = "Latest Indian fashion trends site:vogue.in/fashion"
    response = client.search(
        query=query, 
        search_depth="advanced",
        include_raw_content=True,
        include_images=True
    )
    return response.get("results", [])
