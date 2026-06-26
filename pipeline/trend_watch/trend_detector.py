import os
import sys
import json
import asyncio
import datetime
import requests
import urllib3
urllib3.disable_warnings()

pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
root_dir = os.path.abspath(os.path.join(pipeline_dir, ".."))
if pipeline_dir not in sys.path: sys.path.append(pipeline_dir)
if root_dir not in sys.path: sys.path.append(root_dir)

try:
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception:
    tavily = None


class TrendDetector:
    """
    Runs on a schedule (daily/weekly).
    Finds emerging trends before users ask about them.
    Feeds the extraction pipeline proactively.
    """

    EDITORIAL_SOURCES = [
        "https://www.vogue.in/fashion",
        "https://www.harpersbazaar.in/fashion",
        "https://www.thevoiceoffashion.com",
        "https://www.elle.in/fashion"
    ]

    DESIGNER_URLS = {
        "abhinav_mishra": "abhinavmishraofficial.com",
        "sabyasachi":     "sabyasachi.com",
        "raw_mango":      "rawmango.in",
        "house_of_masaba":"houseofmasaba.com",
        "torani":         "torani.in",
        "anavila":        "anavila.in"
    }

    async def detect_trends(self) -> list[dict]:
        """
        Returns list of detected trends with confidence scores.
        
        Each trend looks like:
        {
            "trend_name": "mirror work lehenga",
            "confidence": 0.87,
            "signal_sources": ["vogue_india", "instagram"],
            "first_seen": "2025-10-01",
            "velocity": "rising",
            "related_aesthetics": ["Mirror Maximalism"],
            "related_techniques": ["sheesha", "resham"],
            "suggested_designers": ["abhinav_mishra", "torani"],
            "suggested_extraction_count": 10
        }
        """
        trends = []
        
        # Editorial signal
        editorial_trends = await self._scan_editorial()
        trends.extend(editorial_trends)
        
        # New arrivals signal
        arrivals_trends = await self._scan_new_arrivals()
        trends.extend(arrivals_trends)
        
        # Deduplicate and score
        return self._rank_and_deduplicate(trends)

    async def _scan_editorial(self) -> list:
        """
        Uses Tavily to scan fashion editorial pages.
        Extracts: what techniques/aesthetics/designers
        are being written about most.
        """
        trends = []
        if not tavily:
            return trends
            
        for url in self.EDITORIAL_SOURCES:
            try:
                domain = url.split("/")[2]
                result = await asyncio.to_thread(
                    tavily.search,
                    query="new Indian fashion trends lehenga saree",
                    include_domains=[domain],
                    max_results=5
                )
                for article in result.get("results", []):
                    content = article.get("content", "")
                    signals = await asyncio.to_thread(self._extract_trend_signals, content)
                    for s in signals:
                        if isinstance(s, dict) and s.get("trend_name"):
                            trends.append({
                                "trend_name": s["trend_name"],
                                "confidence": float(s.get("confidence", 0.7)),
                                "signal_sources": [domain],
                                "first_seen": datetime.datetime.now().strftime("%Y-%m-%d"),
                                "velocity": "rising",
                                "related_aesthetics": s.get("aesthetics_mentioned", []),
                                "related_techniques": s.get("techniques_mentioned", []),
                                "suggested_designers": s.get("designers_mentioned", []) or ["torani", "abhinav_mishra"],
                                "suggested_extraction_count": 5
                            })
            except Exception as e:
                print(f"[TrendDetector] Editorial scan error for {url}: {e}")
        return trends

    def _is_recent(self, created_at_str: str) -> bool:
        if not created_at_str: return False
        try:
            dt = datetime.datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now - dt).days <= 30
        except Exception:
            return False

    async def _scan_new_arrivals(self) -> list:
        """
        Checks new arrivals on each designer's Shopify.
        Products added in last 30 days = potential trend signal.
        """
        trends = []
        headers = {"User-Agent": "Mozilla/5.0"}
        for designer_id, url in self.DESIGNER_URLS.items():
            try:
                endpoint = f"https://{url}/products.json?limit=50"
                resp = await asyncio.to_thread(
                    requests.get, endpoint, headers=headers, timeout=15, verify=False
                )
                products = resp.json().get("products", [])
                
                recent = [
                    p for p in products
                    if self._is_recent(p.get("created_at", ""))
                ]
                
                if len(recent) > 5:
                    trends.append({
                        "trend_name": f"new {designer_id} collection",
                        "confidence": 0.75,
                        "signal_sources": [designer_id, "new_arrivals"],
                        "first_seen": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "velocity": "peak",
                        "related_aesthetics": [f"{designer_id.title()} Signature"],
                        "related_techniques": [],
                        "suggested_designers": [designer_id],
                        "suggested_extraction_count": min(len(recent), 10)
                    })
            except Exception as e:
                print(f"[TrendDetector] New arrivals error for {designer_id}: {e}")
                
        return trends

    def _extract_trend_signals(self, text: str) -> list:
        """
        Uses LLM to extract trend signals from article text.
        """
        if not text: return []
        prompt = f"""
        Read this Indian fashion article and extract trends.
        
        Return JSON array of trends found:
        [{{
            "trend_name": "",
            "techniques_mentioned": [],
            "aesthetics_mentioned": [],
            "designers_mentioned": [],
            "sentiment": "positive/neutral",
            "confidence": 0.0-1.0
        }}]
        
        Article: {text[:2000]}
        """
        try:
            from trend_ingestion import _call_nvidia, _parse_json_response
            resp = _call_nvidia(prompt, max_tokens=1024)
            parsed = _parse_json_response(resp)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "trends" in parsed:
                return parsed["trends"]
            elif isinstance(parsed, dict):
                return [parsed]
        except Exception as e:
            print(f"[TrendDetector] LLM extract error: {e}")
            
        return []

    def _rank_and_deduplicate(self, trends: list) -> list[dict]:
        seen = {}
        for t in trends:
            name = t.get("trend_name", "").lower().strip()
            if not name: continue
            if name not in seen:
                seen[name] = t
            else:
                existing = seen[name]
                existing["confidence"] = min(1.0, existing.get("confidence", 0.5) + 0.1)
                sources = set(existing.get("signal_sources", [])) | set(t.get("signal_sources", []))
                existing["signal_sources"] = list(sources)
                
        ranked = list(seen.values())
        ranked.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return ranked
