import requests
import time
import re

def clean_html(raw_html):
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    # also replace &nbsp; etc if needed
    return cleantext.replace('&nbsp;', ' ').replace('\n', ' ').strip()

def scrape_shopify(store_url):
    """
    Yields parsed product dictionaries from a Shopify store's products.json endpoint.
    Pagination continues until < 250 products returned.
    """
    from scrapers.silhouette_enforcer import SilhouetteEnforcer
    enforcer = SilhouetteEnforcer()
    
    base_url = store_url if store_url.startswith("http") else f"https://{store_url}"
    page = 1
    limit = 250
    
    while True:
        url = f"{base_url}/products.json?limit={limit}&page={page}"
        print(f"Scraping Shopify page {page}: {url}")
        
        # Respect crawl delays
        time.sleep(3)
        
        retry_count = 0
        success = False
        data = {}
        
        while retry_count < 5:
            try:
                import urllib3
                urllib3.disable_warnings()
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/",
                }
                resp = requests.get(url, headers=headers, timeout=15, verify=False)
                if resp.status_code != 200:
                    print(f"Failed to fetch {url}: {resp.status_code}")
                    break

                
                data = resp.json()
                success = True
                break
            except Exception as e:
                retry_count += 1
                print(f"Network error scraping {store_url} page {page}: {e}. Retrying ({retry_count}/5) after 5 seconds...")
                time.sleep(5)
                
        if not success:
            break
            
        products = data.get("products", [])
        
        if not products:
            break
            
        for p in products:
            # Clean description
            desc = clean_html(p.get("body_html", ""))
            
            # Extract images
            images = [img.get("src") for img in p.get("images", []) if img.get("src")]
            
            # Extract variants
            variants = []
            for v in p.get("variants", []):
                variants.append({
                    "color": v.get("option1") or "", # Sometimes color is option1 or option2
                    "size": v.get("option2") or "",
                    "price": float(v.get("price", 0)),
                    "available": v.get("available", True)
                })
            
            item_dict = {
                "id": str(p.get("id")),
                "title": p.get("title"),
                "handle": p.get("handle"),
                "product_type": p.get("product_type"),
                "tags": p.get("tags", []),
                "raw_description": desc,
                "variants": variants,
                "images": images,
                "created_at": p.get("created_at"),
                "source_url": f"{base_url}/products/{p.get('handle')}"
            }
            if enforcer.should_accept(item_dict):
                yield item_dict
            
        if len(products) < limit:
            break
        page += 1


import asyncio

class ShopifyScraper:
    """Async wrapper class for Shopify scraping."""
    def __init__(self, designer_config: dict, balance_genders: bool = True):
        self.designer_config = designer_config
        self.url = designer_config.get("url", "")
        
    async def scrape(self, limit: int = 250):
        items = await asyncio.to_thread(lambda: list(scrape_shopify(self.url)))
        for item in items[:limit]:
            yield item
