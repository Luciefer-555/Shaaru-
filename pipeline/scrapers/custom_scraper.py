import os
import time
import uuid
import datetime
from tavily import TavilyClient

def scrape_custom(config: dict):
    """
    Scraper for non-Shopify custom websites and multi-designer platforms using Tavily.
    """
    key = os.getenv("TAVILY_API_KEY", "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")
    if not key:
        print("Error: TAVILY_API_KEY not found in environment and fallback failed")
        return
    
    client = TavilyClient(api_key=key)
    
    from scrapers.silhouette_enforcer import SilhouetteEnforcer
    enforcer = SilhouetteEnforcer()
    
    base_url = config["url"]
    is_multi_designer = config.get("platform") == "multi_designer" or base_url in ["perniaspopupshop.com", "azafashions.com", "jaypore.com", "itokri.com"]
    
    # We use search to discover product pages. To get a good yield, we search with relevant keywords.
    queries = [
        f"site:{base_url} clothing",
        f"site:{base_url} lehenga sari kurta jacket",
        f"site:{base_url} new arrivals",
    ]
    
    seen_urls = set()
    
    for query in queries:
        time.sleep(2)  # Respect crawl delay rule
        
        try:
            response = client.search(
                query=query, 
                search_depth="advanced",
                include_raw_content=True,
                include_images=True,
                max_results=100  # Increased for production runs
            )
            
            results = response.get("results", [])
            for res in results:
                url = res.get("url", "")
                if not url or url in seen_urls:
                    continue
                if not any(x in url for x in ["/products/", "/p/", "/product/", "/collections/"]):
                    continue
                seen_urls.add(url)
                
                content = res.get("raw_content", "")
                title = res.get("title", "")
                
                # Check for original designer in multi-designer platforms
                original_designer = None
                if is_multi_designer:
                    # Simple heuristic: try to find 'By [Designer]' or just pass the text to the LLM later
                    # For now, we tag it so the pipeline knows it's multi-designer
                    original_designer = "PENDING_LLM_EXTRACTION"
                    
                # We need images. Tavily sometimes returns 'images' list or we extract from raw content.
                images = res.get("images", [])
                if not images:
                    # Fallback to a placeholder or skip if strict.
                    # Since we require vision, an image is mandatory. If Tavily didn't parse images, we skip.
                    continue
                    
                item_dict = {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, url)),
                    "title": title,
                    "handle": url.split("/")[-1],
                    "product_type": "custom_scrape",
                    "tags": [],
                    "raw_description": content,
                    "variants": [],
                    "images": images,
                    "created_at": datetime.datetime.now().isoformat(),
                    "source_url": url,
                    "original_designer": original_designer
                }
                if enforcer.should_accept(item_dict):
                    yield item_dict
                
        except Exception as e:
            print(f"Error scraping {base_url} with Tavily: {e}")
            
