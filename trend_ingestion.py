"""
trend_ingestion.py
SHAARU — Autonomous Trend Intelligence + Knowledge Ingestion Pipeline

Fetches from web → extracts aesthetics → deduplicates →
quality gates → stores in MongoDB + Neo4j.

Run manually: python trend_ingestion.py
Scheduled:    runs via APScheduler every 24 hours
API trigger:  POST /api/trends/refresh
"""

import os
import re
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
import functools

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("shaaru.trends")
logging.basicConfig(level=logging.INFO)

# Force all print() to flush immediately (Windows buffering workaround)
print = functools.partial(print, flush=True)

# ── Imports from SHAARU stack ────────────────────────────────────
from shaaru_brain import _get_db, _get_client, MODEL_TEXT
try:
    from shaaru_brain import nvidia_call
except ImportError:
    nvidia_call = None

from knowledge_graph import get_kg

from trend_sources import (
    HASHTAG_SEARCHES,
    TAVILY_QUERIES,
    ARE_NA_CHANNELS,
    QUALITY_THRESHOLD,
    EXTRACTION_PROMPT,
    QUALITY_PROMPT,
)

# ── Tavily client (lazy init) ───────────────────────────────────
_tavily_client = None

def _get_tavily():
    """Lazy-init Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                log.warning("[TAVILY] No API key found")
                return None
            _tavily_client = TavilyClient(api_key=api_key)
        except ImportError:
            log.error("[TAVILY] tavily-python not installed")
            return None
    return _tavily_client


# ══════════════════════════════════════════════════════════════════
#  Fetch Functions
# ══════════════════════════════════════════════════════════════════

def fetch_from_tavily(query: str) -> list[str]:
    """
    Search Tavily for fashion content. Returns list of text snippets.
    """
    try:
        client = _get_tavily()
        if client is None:
            return []

        result = client.search(query, max_results=5)
        snippets = []
        for item in result.get("results", []):
            content = item.get("content", "")
            if content:
                # Cap at 500 chars per snippet
                snippets.append(content[:500])
            if len(snippets) >= 10:
                break

        print(f"[OK] Tavily: fetched {len(snippets)} snippets for '{query}'")
        return snippets

    except Exception as e:
        print(f"[FAIL] Tavily search '{query}': {e}")
        return []


def fetch_from_arena(channel: str) -> list[str]:
    """
    Fetch text content from Are.na channel (open API, no key needed).
    """
    try:
        import requests
        url = f"https://api.are.na/v2/channels/{channel}/contents"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 404:
            # Channel doesn't exist — skip silently
            return []
        resp.raise_for_status()

        data = resp.json()
        texts = []
        for item in data.get("contents", []):
            if item.get("class") == "Text":
                desc = item.get("content", "") or item.get("description", "")
                if desc:
                    texts.append(desc[:500])

        print(f"[OK] Are.na: fetched {len(texts)} texts from '{channel}'")
        return texts

    except Exception as e:
        print(f"[FAIL] Are.na '{channel}': {e}")
        return []


# ══════════════════════════════════════════════════════════════════
#  LLM Extraction & Scoring
# ══════════════════════════════════════════════════════════════════

def _call_nvidia(prompt: str, max_tokens: int = 1024, model: str | None = None) -> str:
    """Unified NVIDIA NIM call wrapper."""
    try:
        client = _get_client()
        messages = [{"role": "user", "content": prompt}]
        target_model = model or MODEL_TEXT

        if nvidia_call:
            return nvidia_call(client, target_model, messages,
                               max_tokens=max_tokens, temperature=0.4)
        else:
            resp = client.chat.completions.create(
                model=target_model, messages=messages,
                max_tokens=max_tokens, temperature=0.4
            )
            return resp.choices[0].message.content
    except Exception as e:
        log.error(f"[NVIDIA] Call failed: {e}")
        return ""


def _parse_json_response(text: str) -> dict | list | None:
    """Extract JSON from LLM response, handling markdown fences."""
    if not text:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try array first
        match_arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match_arr:
            try:
                return json.loads(match_arr.group())
            except json.JSONDecodeError:
                pass
        # Try object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


def extract_aesthetic(content: str) -> dict | None:
    """
    Call NVIDIA NIM to extract structured aesthetic data from content.
    Returns parsed dict or None if content has no clear aesthetic.
    """
    try:
        prompt = EXTRACTION_PROMPT.format(content=content[:1500])
        response = _call_nvidia(prompt, max_tokens=1024)
        parsed = _parse_json_response(response)

        if parsed is None:
            print(f"[FAIL] Extract: JSON parse failed")
            return None
        if parsed.get("skip"):
            return None
        if parsed.get("is_duplicate_risk"):
            return None

        name = parsed.get("aesthetic_name", "")
        if not name:
            return None

        print(f"[OK] Extracted: {name}")
        return parsed

    except Exception as e:
        print(f"[FAIL] Extract aesthetic: {e}")
        return None


def score_aesthetic(aesthetic: dict) -> float:
    """
    Call NVIDIA NIM to score aesthetic quality (1-10).
    Returns score float, or 0.0 on failure.
    """
    try:
        prompt = QUALITY_PROMPT.format(
            aesthetic_name=aesthetic.get("aesthetic_name", ""),
            description=aesthetic.get("aesthetic_description", ""),
            indian_context=aesthetic.get("indian_context", ""),
        )
        response = _call_nvidia(prompt, max_tokens=256)
        parsed = _parse_json_response(response)

        if parsed and "score" in parsed:
            score = float(parsed["score"])
            reason = parsed.get("reason", "")
            print(f"[OK] Quality score: {score:.1f} — {reason}")
            return score

        print(f"[FAIL] Score parse failed")
        return 0.0

    except Exception as e:
        print(f"[FAIL] Score aesthetic: {e}")
        return 0.0


# ══════════════════════════════════════════════════════════════════
#  Deduplication
# ══════════════════════════════════════════════════════════════════

STOPWORDS = {"the", "a", "an", "and", "or", "for", "with", "of", "in", "to", "is"}

def is_duplicate(aesthetic_name: str, description: str, db) -> bool:
    """
    Check MongoDB styling_guides for semantic duplicates.
    Step 1: exact name match (case insensitive)
    Step 2: keyword overlap (2+ shared content words)
    """
    try:
        # Step 1 — exact name match
        existing = db["styling_guides"].find_one(
            {"aesthetic": {"$regex": re.escape(aesthetic_name), "$options": "i"}}
        )
        if existing:
            print(f"[OK] Duplicate detected (exact): {aesthetic_name}")
            return True

        # Step 2 — keyword overlap
        words = {w.lower() for w in aesthetic_name.split()
                 if w.lower() not in STOPWORDS and len(w) > 2}
        if len(words) < 2:
            print(f"[OK] Unique: {aesthetic_name}")
            return False

        # Check existing guides for keyword overlap
        all_guides = list(db["styling_guides"].find({}, {"aesthetic": 1}))
        for guide in all_guides:
            existing_words = {w.lower() for w in guide.get("aesthetic", "").split()
                              if w.lower() not in STOPWORDS and len(w) > 2}
            overlap = words & existing_words
            if len(overlap) >= 2:
                print(f"[OK] Duplicate detected (keyword overlap): {aesthetic_name}")
                return True

        print(f"[OK] Unique: {aesthetic_name}")
        return False

    except Exception as e:
        log.warning(f"[DEDUP] Check failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
#  Storage
# ══════════════════════════════════════════════════════════════════

def save_styling_guide(aesthetic: dict, score: float,
                       source: str, db) -> str | None:
    """
    Save extracted aesthetic to MongoDB styling_guides collection.
    Returns inserted _id as string, or None on failure.
    """
    try:
        now = datetime.now(timezone.utc)
        doc = {
            "aesthetic": aesthetic.get("aesthetic_name", ""),
            "description": aesthetic.get("aesthetic_description", ""),
            "silhouettes": aesthetic.get("key_silhouettes", []),
            "colors": aesthetic.get("key_colors", []),
            "fabrics": aesthetic.get("fabrics", []),
            "rules": {
                "do": aesthetic.get("styling_rules_do", []),
                "dont": aesthetic.get("styling_rules_dont", []),
            },
            "occasion": aesthetic.get("occasion", []),
            "body_compatibility": aesthetic.get("body_compatibility", []),
            "indian_context": aesthetic.get("indian_context", ""),
            "source": source,
            "quality_score": score,
            "ingested_at": now,
            "embedding": [],
        }
        result = db["styling_guides"].insert_one(doc)
        doc_id = str(result.inserted_id)
        print(f"[OK] Saved styling guide: {doc['aesthetic']} (id={doc_id})")
        return doc_id

    except Exception as e:
        print(f"[FAIL] Save styling guide: {e}")
        return None


def save_aesthetic_to_neo4j(aesthetic: dict) -> bool:
    """
    Create/merge Aesthetic node in Neo4j with Occasion and BodyType relationships.
    """
    if not get_kg().is_connected:
        print("[FAIL] Neo4j not connected — skipping graph write")
        return False

    try:
        name = aesthetic.get("aesthetic_name", "")
        now = datetime.now(timezone.utc).isoformat()

        # Merge Aesthetic node
        get_kg().query(
            """
            MERGE (a:Aesthetic {name: $name})
            SET a.description = $description,
                a.indian_context = $indian_context,
                a.ingested_at = $now
            RETURN a
            """,
            {
                "name": name,
                "description": aesthetic.get("aesthetic_description", ""),
                "indian_context": aesthetic.get("indian_context", ""),
                "now": now,
            }
        )

        # Link to Occasion nodes
        for occasion in aesthetic.get("occasion", []):
            get_kg().query(
                """
                MERGE (a:Aesthetic {name: $name})
                MERGE (o:Occasion {name: $occasion})
                MERGE (a)-[:SUITS_OCCASION]->(o)
                """,
                {"name": name, "occasion": occasion}
            )

        # Link to BodyType nodes
        for body_type in aesthetic.get("body_compatibility", []):
            get_kg().query(
                """
                MERGE (a:Aesthetic {name: $name})
                MERGE (b:BodyType {name: $body_type})
                MERGE (a)-[:WORKS_FOR]->(b)
                """,
                {"name": name, "body_type": body_type}
            )

        print(f"[OK] Neo4j: merged Aesthetic '{name}' with relationships")
        return True

    except Exception as e:
        print(f"[FAIL] Neo4j save: {e}")
        return False


def update_trends_collection(saved_aesthetics: list[dict], db) -> None:
    """
    Write fresh trend snapshot to MongoDB trends collection.
    Inserts a new document — old ones expire naturally via expires_at.
    """
    try:
        now = datetime.now(timezone.utc)

        # Top 5 by quality score
        sorted_aesthetics = sorted(
            saved_aesthetics,
            key=lambda a: a.get("quality_score", 0),
            reverse=True
        )[:5]

        rising = [
            {"trend": a.get("aesthetic", ""), "score": a.get("quality_score", 0)}
            for a in sorted_aesthetics
        ]

        # Generate seasonal direction summary via NVIDIA
        seasonal_direction = "Light fabrics, earthy tones, Indian craft aesthetics."
        if saved_aesthetics:
            names = ", ".join(a.get("aesthetic", "") for a in saved_aesthetics[:10])
            prompt = (
                f"Summarize these Indian fashion trends into ONE sentence "
                f"describing the current seasonal direction: {names}"
            )
            try:
                summary = _call_nvidia(prompt, max_tokens=128)
                if summary and len(summary) > 10:
                    # Clean up — take first sentence only
                    seasonal_direction = summary.split(".")[0].strip() + "."
            except Exception:
                pass

        sources = ["tavily"]
        if any(a.get("source") == "are.na" for a in saved_aesthetics):
            sources.append("are.na")

        trend_doc = {
            "rising": rising,
            "declining": [],
            "seasonal_direction": seasonal_direction,
            "captured_at": now,
            "expires_at": now + timedelta(hours=24),
            "source": sources,
        }

        db["trends"].insert_one(trend_doc)
        print(f"[OK] Trends updated: {len(rising)} rising trends stored")

    except Exception as e:
        print(f"[FAIL] Update trends: {e}")


# ══════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════

def run_pipeline(max_queries: int = 0) -> dict:
    """
    Main orchestrator. Runs the full trend intelligence pipeline.
    Args:
        max_queries: limit Tavily queries (0 = all). Use 3 for quick test.
    Returns summary dict with counts.
    """
    print("\n" + "=" * 60)
    print("  SHAARU Trend Intelligence Pipeline")
    print("=" * 60 + "\n")

    # Init database
    db = _get_db()
    if db is None:
        print("[FAIL] Database unavailable — aborting pipeline")
        return {"status": "error", "message": "Database unavailable"}

    # Counters
    fetched = 0
    extracted = 0
    passed_quality = 0
    saved = 0
    saved_docs = []  # Track saved docs for trends update

    # ── Phase 1: Tavily Searches ─────────────────────────────────
    queries = TAVILY_QUERIES[:max_queries] if max_queries > 0 else TAVILY_QUERIES
    print(f"\n── Phase 1: Tavily Web Search ({len(queries)} queries) ──\n")
    for query in queries:
        content_list = fetch_from_tavily(query)
        for content in content_list:
            fetched += 1

            # Extract aesthetic
            aesthetic = extract_aesthetic(content)
            if aesthetic is None:
                continue
            extracted += 1

            # Deduplication check
            aname = aesthetic.get("aesthetic_name", "")
            adesc = aesthetic.get("aesthetic_description", "")
            if is_duplicate(aname, adesc, db):
                continue

            # Quality gate
            score = score_aesthetic(aesthetic)
            if score < QUALITY_THRESHOLD:
                print(f"[SKIP] Quality too low ({score:.1f}): {aname}")
                continue
            passed_quality += 1

            # Save to MongoDB
            doc_id = save_styling_guide(aesthetic, score, "tavily", db)
            if doc_id:
                # Save to Neo4j
                save_aesthetic_to_neo4j(aesthetic)
                saved += 1
                saved_docs.append({
                    "aesthetic": aname,
                    "quality_score": score,
                    "source": "tavily",
                })

            # Rate limit between NVIDIA calls
            time.sleep(0.5)

    # ── Phase 2: Are.na Channels ─────────────────────────────────
    print("\n── Phase 2: Are.na Channels ──\n")
    for channel in ARE_NA_CHANNELS:
        content_list = fetch_from_arena(channel)
        for content in content_list:
            fetched += 1

            aesthetic = extract_aesthetic(content)
            if aesthetic is None:
                continue
            extracted += 1

            aname = aesthetic.get("aesthetic_name", "")
            adesc = aesthetic.get("aesthetic_description", "")
            if is_duplicate(aname, adesc, db):
                continue

            score = score_aesthetic(aesthetic)
            if score < QUALITY_THRESHOLD:
                print(f"[SKIP] Quality too low ({score:.1f}): {aname}")
                continue
            passed_quality += 1

            doc_id = save_styling_guide(aesthetic, score, "are.na", db)
            if doc_id:
                save_aesthetic_to_neo4j(aesthetic)
                saved += 1
                saved_docs.append({
                    "aesthetic": aname,
                    "quality_score": score,
                    "source": "are.na",
                })

            time.sleep(0.5)

    # ── Phase 3: Update Trends ───────────────────────────────────
    print("\n── Phase 3: Updating Trends Collection ──\n")
    update_trends_collection(saved_docs, db)

    # ── Summary ──────────────────────────────────────────────────
    summary = {
        "status": "complete",
        "fetched": fetched,
        "extracted": extracted,
        "passed_quality": passed_quality,
        "saved": saved,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'=' * 60}")
    print(f"  [OK] Pipeline complete")
    print(f"  Fetched:         {fetched} content pieces")
    print(f"  Extracted:       {extracted} aesthetics")
    print(f"  Passed quality:  {passed_quality}")
    print(f"  Saved to DB:     {saved}")
    print(f"{'=' * 60}\n")

    return summary


# ══════════════════════════════════════════════════════════════════
#  Direct Execution
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    result = run_pipeline(max_queries=n)
