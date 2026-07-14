import os
import json
import re
import time
from datetime import datetime, timezone
from tavily import TavilyClient
from shaaru_brain import nvidia_call, _get_client, MODEL_TEXT
from knowledge_sources import (
    fabric_queries,
    construction_queries,
    sourcing_queries,
    measurement_queries,
)

# Initialize Tavily Client
client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
llm_client = _get_client()

def extract_json_from_response(text: str) -> dict:
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except:
        pass
    # Try extracting JSON block from markdown
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    # Try finding first { to last }
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}

class QuotaExceededError(Exception):
    """Raised when Tavily search quota is exhausted or returns 0 sources during quota throttling."""
    pass

def check_content_specificity(candidate: dict, sources: list[str], source_contents: list[str] = None) -> bool:
    """Enforce that at least 2 returned sources explicitly contain the target keywords or close synonyms."""
    if not sources or len(sources) < 2:
        return False
    item_id = str(candidate.get('garment_id') or candidate.get('fabric_id') or candidate.get('item_name') or '').lower()
    keywords = set()
    if item_id:
        keywords.add(item_id.replace('_', ' '))
        keywords.add(item_id.replace('_', ''))
        for part in item_id.split('_'):
            if len(part) > 2 and part not in ('set', 'wear', 'style', 'dress', 'shirt', 'pants', 'jacket', 'skirt', 'boots', 'shoes'):
                keywords.add(part)
    for cn in candidate.get('common_names', []):
        if isinstance(cn, str) and cn.strip():
            keywords.add(cn.lower().strip())
            for w in cn.split():
                if len(w) > 2:
                    keywords.add(w.lower())
    matching_sources_count = 0
    for idx, url in enumerate(sources):
        content = source_contents[idx] if (source_contents and idx < len(source_contents)) else ""
        text_to_check = (str(url) + " " + str(content)).lower()
        if any(kw in text_to_check for kw in keywords if len(kw) > 2):
            matching_sources_count += 1
    return matching_sources_count >= 2


_tavily_quota_exceeded = False

def _run_queries_and_get_sources(queries, max_sources=4):
    global _tavily_quota_exceeded
    if _tavily_quota_exceeded:
        return [], []
    sources = []
    source_contents = []
    seen_urls = set()
    for q in queries:
        if len(sources) >= max_sources:
            break
        retries = 3
        while retries > 0:
            try:
                print(f"Querying Tavily: {q}")
                result = client.search(q, max_results=2)
                print(f"Tavily returned {len(result.get('results', []))} results")
                for r in result.get('results', []):
                    url = r.get('url')
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    sources.append(url)
                    source_contents.append(r.get('content', ''))
                    if len(sources) >= max_sources:
                        break
                time.sleep(1.0) # Pace requests between queries
                break
            except Exception as e:
                err_str = str(e)
                print(f"Error querying Tavily for '{q}': {err_str}")
                if "exceeds your plan" in err_str.lower() or "usage limit" in err_str.lower() or "quota" in err_str.lower():
                    print("[TAVILY QUOTA] Plan limit exceeded. Fast-failing remaining queries for this run.")
                    _tavily_quota_exceeded = True
                    return sources, source_contents
                elif "blocked" in err_str.lower() or "excessive" in err_str.lower() or "rate" in err_str.lower() or "429" in err_str:
                    retries -= 1
                    if retries > 0:
                        print(f"[RATE LIMIT] Waiting 15s before retry ({retries} left)...")
                        time.sleep(15.0)
                    else:
                        break
                else:
                    break
    return sources, source_contents

def verify_fabric_record(candidate: dict) -> dict | None:
    candidate = {k: (str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v) for k, v in candidate.items() if k != '_id'}
    fabric_id = candidate.get('fabric_id', 'unknown')
    sources, source_contents = _run_queries_and_get_sources(fabric_queries(candidate))
    
    if not sources:
        if _tavily_quota_exceeded:
            raise QuotaExceededError(f"Tavily quota exceeded (0 sources) for '{fabric_id}'. Aborting.")
        print(f"[WARN] {fabric_id} - no web sources found. Using expert LLM domain verification...")
        prompt = f"""You are verifying a fabric specification record for a fashion AI system using expert textile engineering knowledge.

Candidate record:
{json.dumps(candidate, indent=2)}

Evaluate whether the specifications are technically sound, logical, and accurate according to textile science:
1. GSM range and weight classification: {candidate.get('gsm_range')}
2. Drape behavior and score: {candidate.get('drape_score')} out of 10
3. Fiber composition: {candidate.get('fiber_composition')}
4. Best use cases: {candidate.get('best_for')}

Return ONLY valid JSON:
{{
  "agreement_score": 0.0,
  "verified_fields": ["field1", "field2"],
  "conflicting_fields": [],
  "conflicts_detail": "",
  "verified": true
}}"""
    else:
        s1_content = source_contents[0] if len(source_contents) > 0 else ""
        s1_url = sources[0] if len(sources) > 0 else ""
        s2_content = source_contents[1] if len(source_contents) > 1 else ""
        s2_url = sources[1] if len(sources) > 1 else ""

        prompt = f"""You are verifying a fabric specification record for a fashion AI system.

Candidate record:
{json.dumps(candidate, indent=2)}

Source 1 content: {s1_content}
Source 1 URL: {s1_url}

Source 2 content: {s2_content}  
Source 2 URL: {s2_url}

Evaluate whether the sources support these specific claims:
1. GSM range: {candidate.get('gsm_range')}
2. Drape behavior: {candidate.get('drape_score')} out of 10
3. Fiber composition: {candidate.get('fiber_composition')}
4. Best use cases: {candidate.get('best_for')}

CRITICAL INSTRUCTION: You must strictly evaluate source content relevance.
If a source does not explicitly discuss or confirm the specific GSM range, drape, fiber composition, or technical attributes of THIS EXACT fabric target, score it as 0 (irrelevant / tangential domain match). Do NOT treat non-contradictory generic articles as supporting. Silence is not support.

Return ONLY valid JSON:
{{
  "agreement_score": 0.0,
  "verified_fields": ["field1", "field2"],
  "conflicting_fields": ["field3"],
  "conflicts_detail": "description of any conflicts",
  "verified": true
}}"""

    try:
        print(f"Calling LLM for fabric: {fabric_id}...")
        response = nvidia_call(llm_client, MODEL_TEXT, [{"role": "user", "content": prompt}])
        print(f"Got LLM response for fabric: {fabric_id}")
        data = extract_json_from_response(response)
        if not data:
            print(f"[REJECT] {fabric_id} - Invalid JSON response from LLM")
            return None
        
        agreement_score = data.get('agreement_score', 0.0)
        verified = data.get('verified', False)

        if agreement_score >= 0.60 and not check_content_specificity(candidate, sources, source_contents):
            print(f"[SPECIFICITY CAP] {fabric_id} - <2 sources contain target keywords. Capping agreement score at 0.45.")
            agreement_score = min(agreement_score, 0.45)
            verified = False

        if agreement_score >= 0.60:
            candidate['verified'] = True
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            candidate['verified_at'] = datetime.now(timezone.utc).isoformat()
            print(f"[ACCEPT] {fabric_id} with score {agreement_score}")
            return candidate
        elif 0.35 <= agreement_score < 0.60:
            candidate['verified'] = False
            candidate['flag'] = 'needs_review'
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            print(f"[FLAG] {fabric_id} with score {agreement_score}")
            return candidate
        else:
            print(f"[REJECT] {fabric_id} - score {agreement_score} < 0.35")
            return None

    except Exception as e:
        print(f"[REJECT] {fabric_id} - error during LLM verification: {e}")
        return None


def verify_construction_record(candidate: dict) -> dict | None:
    candidate = {k: (str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v) for k, v in candidate.items() if k != '_id'}
    garment_id = candidate.get('garment_id', 'unknown')
    sources, source_contents = _run_queries_and_get_sources(construction_queries(candidate))

    if not sources:
        if _tavily_quota_exceeded:
            raise QuotaExceededError(f"Tavily quota exceeded (0 sources) for '{garment_id}'. Aborting.")
        print(f"[WARN] {garment_id} - no web sources found. Using expert LLM domain verification...")
        prompt = f"""You are verifying a garment construction record for a fashion AI system using expert tailoring, pattern engineering, and apparel manufacturing knowledge.

Candidate record:
{json.dumps(candidate, indent=2)}

Evaluate whether the specifications are technically sound, industry-standard, and logical:
- Construction sequence logical order: {candidate.get('construction_sequence')}
- Seam allowances within industry standard: {candidate.get('seam_allowances')}
- Ease values reasonable for fit type: {candidate.get('ease_by_fit')}
- Critical points technically accurate: {candidate.get('critical_points')}

Return ONLY valid JSON:
{{
  "agreement_score": 0.0,
  "verified_fields": ["field1", "field2"],
  "conflicting_fields": [],
  "conflicts_detail": "",
  "verified": true
}}"""
    else:
        s1_content = source_contents[0] if len(source_contents) > 0 else ""
        s1_url = sources[0] if len(sources) > 0 else ""
        s2_content = source_contents[1] if len(source_contents) > 1 else ""
        s2_url = sources[1] if len(sources) > 1 else ""

        prompt = f"""You are verifying a garment construction record for a fashion AI system.

Candidate record:
{json.dumps(candidate, indent=2)}

Source 1 content: {s1_content}
Source 1 URL: {s1_url}

Source 2 content: {s2_content}
Source 2 URL: {s2_url}

Evaluate whether the sources support these claims:
- Construction sequence logical order: {candidate.get('construction_sequence')}
- Seam allowances within industry standard: {candidate.get('seam_allowances')}
- Ease values reasonable for fit type: {candidate.get('ease_by_fit')}
- Critical points technically accurate: {candidate.get('critical_points')}

CRITICAL INSTRUCTION: You must strictly evaluate source content relevance.
If a source does not explicitly discuss or confirm the specific construction sequence, seam allowance, ease values, or technical attributes of THIS EXACT garment silhouette, score it as 0 (irrelevant / tangential domain match). Do NOT treat non-contradictory generic articles as supporting. Silence is not support.

Return ONLY valid JSON:
{{
  "agreement_score": 0.0,
  "verified_fields": ["field1"],
  "conflicting_fields": ["field2"],
  "conflicts_detail": "description",
  "verified": true
}}"""

    try:
        print(f"Calling LLM for construction: {garment_id}...")
        response = nvidia_call(llm_client, MODEL_TEXT, [{"role": "user", "content": prompt}])
        print(f"Got LLM response for construction: {garment_id}")
        data = extract_json_from_response(response)
        if not data:
            print(f"[REJECT] {garment_id} - Invalid JSON from LLM")
            return None

        agreement_score = data.get('agreement_score', 0.0)
        verified = data.get('verified', False)

        if agreement_score >= 0.60 and not check_content_specificity(candidate, sources, source_contents):
            print(f"[SPECIFICITY CAP] {garment_id} - <2 sources contain target keywords. Capping agreement score at 0.45.")
            agreement_score = min(agreement_score, 0.45)
            verified = False

        if agreement_score >= 0.60:
            candidate['verified'] = True
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            candidate['verified_at'] = datetime.now(timezone.utc).isoformat()
            print(f"[ACCEPT] {garment_id} with score {agreement_score}")
            return candidate
        elif 0.35 <= agreement_score < 0.60:
            candidate['verified'] = False
            candidate['flag'] = 'needs_review'
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            print(f"[FLAG] {garment_id} with score {agreement_score}")
            return candidate
        else:
            print(f"[REJECT] {garment_id} - score {agreement_score} < 0.35")
            return None
    except Exception as e:
        print(f"[REJECT] {garment_id} - error: {e}")
        return None


def verify_sourcing_record(candidate: dict, city: str) -> dict | None:
    fabric_id = candidate.get('fabric_id', 'unknown')
    sources, source_contents = _run_queries_and_get_sources(sourcing_queries(candidate, city))

    if not sources:
        if _tavily_quota_exceeded:
            raise QuotaExceededError(f"Tavily quota exceeded (0 sources) for sourcing '{fabric_id}'. Aborting.")
        print(f"[REJECT] {fabric_id} - no sources found")
        return None

    s1_content = source_contents[0] if len(source_contents) > 0 else ""
    s1_url = sources[0] if len(sources) > 0 else ""
    s2_content = source_contents[1] if len(source_contents) > 1 else ""
    s2_url = sources[1] if len(sources) > 1 else ""

    prompt = f"""You are verifying a sourcing record for a fashion AI system.

City: {city}
Candidate record:
{json.dumps(candidate, indent=2)}

Source 1 content: {s1_content}
Source 1 URL: {s1_url}

Source 2 content: {s2_content}
Source 2 URL: {s2_url}

Evaluate:
- Market names exist in {city}
- Price range is realistic for Indian wholesale market: {candidate.get('price_inr_per_meter')}
- Fabric type available in that market

CRITICAL INSTRUCTION: You must strictly evaluate source content relevance.
If a source does not explicitly confirm the market and price for THIS EXACT fabric target, score 0. Do NOT treat non-contradictory generic articles as supporting.

Return ONLY valid JSON:
{{
  "agreement_score": 0.0,
  "verified_fields": [],
  "conflicting_fields": [],
  "conflicts_detail": "",
  "verified": true
}}"""

    try:
        response = nvidia_call(llm_client, MODEL_TEXT, [{"role": "user", "content": prompt}])
        data = extract_json_from_response(response)
        if not data:
            print(f"[REJECT] {fabric_id} - Invalid JSON")
            return None

        agreement_score = data.get('agreement_score', 0.0)
        verified = data.get('verified', False)

        if agreement_score >= 0.60 and not check_content_specificity(candidate, sources, source_contents):
            print(f"[SPECIFICITY CAP] {fabric_id} - <2 sources contain target keywords. Capping agreement score at 0.45.")
            agreement_score = min(agreement_score, 0.45)
            verified = False

        if agreement_score >= 0.60:
            candidate['verified'] = True
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            candidate['verified_at'] = datetime.now(timezone.utc).isoformat()
            print(f"[ACCEPT] {fabric_id} with score {agreement_score}")
            return candidate
        elif 0.35 <= agreement_score < 0.60:
            candidate['verified'] = False
            candidate['flag'] = 'needs_review'
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            print(f"[FLAG] {fabric_id} with score {agreement_score}")
            return candidate
        else:
            print(f"[REJECT] {fabric_id} - score {agreement_score} < 0.35")
            return None
    except Exception as e:
        print(f"[REJECT] {fabric_id} - error: {e}")
        return None


def verify_measurement_table(candidate: dict) -> dict | None:
    garment = candidate.get('garment', 'unknown')
    sources, source_contents = _run_queries_and_get_sources(measurement_queries(candidate))

    if not sources:
        if _tavily_quota_exceeded:
            raise QuotaExceededError(f"Tavily quota exceeded (0 sources) for measurement '{garment}'. Aborting.")
        print(f"[REJECT] {garment} - no sources found")
        return None

    s1_content = source_contents[0] if len(source_contents) > 0 else ""
    s1_url = sources[0] if len(sources) > 0 else ""
    s2_content = source_contents[1] if len(source_contents) > 1 else ""
    s2_url = sources[1] if len(sources) > 1 else ""

    prompt = f"""You are verifying a measurement table for a fashion AI system.

Candidate record:
{json.dumps(candidate, indent=2)}

Source 1 content: {s1_content}
Source 1 URL: {s1_url}

Source 2 content: {s2_content}
Source 2 URL: {s2_url}

Evaluate proportional accuracy of measurements against sources.
For a 6ft frame, inseam should be ~84-88cm, outseam ~112-116cm.
Flag any measurement more than 15% outside source-confirmed ranges.

CRITICAL INSTRUCTION: You must strictly evaluate source content relevance.
If a source does not explicitly discuss or confirm measurements for THIS EXACT garment target, score 0. Do NOT treat non-contradictory generic articles as supporting.

Return ONLY valid JSON:
{{
  "agreement_score": 0.0,
  "verified_fields": [],
  "conflicting_fields": [],
  "conflicts_detail": "",
  "verified": true
}}"""

    try:
        response = nvidia_call(llm_client, MODEL_TEXT, [{"role": "user", "content": prompt}])
        data = extract_json_from_response(response)
        if not data:
            print(f"[REJECT] {garment} - Invalid JSON")
            return None

        agreement_score = data.get('agreement_score', 0.0)
        verified = data.get('verified', False)

        if agreement_score >= 0.60 and not check_content_specificity(candidate, sources, source_contents):
            print(f"[SPECIFICITY CAP] {garment} - <2 sources contain target keywords. Capping agreement score at 0.45.")
            agreement_score = min(agreement_score, 0.45)
            verified = False

        if agreement_score >= 0.60:
            candidate['verified'] = True
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            candidate['verified_at'] = datetime.now(timezone.utc).isoformat()
            print(f"[ACCEPT] {garment} with score {agreement_score}")
            return candidate
        elif 0.35 <= agreement_score < 0.60:
            candidate['verified'] = False
            candidate['flag'] = 'needs_review'
            candidate['sources'] = sources[:4]
            candidate['agreement_score'] = agreement_score
            print(f"[FLAG] {garment} with score {agreement_score}")
            return candidate
        else:
            print(f"[REJECT] {garment} - score {agreement_score} < 0.35")
            return None
    except Exception as e:
        print(f"[REJECT] {garment} - error: {e}")
        return None


def run_verification_pipeline(domain: str, candidates: list) -> dict:
    accepted = []
    flagged = []
    rejected_count = 0

    for candidate in candidates:
        res = None
        print(f"Processing {domain} candidate {candidate.get('fabric_id', candidate.get('garment_id', 'unknown'))}...")
        if domain == 'fabric':
            res = verify_fabric_record(candidate)
        elif domain == 'construction':
            res = verify_construction_record(candidate)
        elif domain == 'sourcing':
            city = candidate.get('city', 'Bengaluru') # fallback
            res = verify_sourcing_record(candidate, city)
        elif domain == 'measurement':
            res = verify_measurement_table(candidate)
        else:
            print(f"Unknown domain: {domain}")
            continue

        if res:
            if res.get('verified'):
                accepted.append(res)
            else:
                flagged.append(res)
        else:
            rejected_count += 1

    stats = {
        'domain': domain,
        'total': len(candidates),
        'accepted': len(accepted),
        'flagged': len(flagged),
        'rejected': rejected_count,
        'accepted_records': accepted,
        'flagged_records': flagged,
        'run_at': datetime.now(timezone.utc).isoformat()
    }

    print(f"[PIPELINE COMPLETE] {domain} | Accepted: {stats['accepted']}, Flagged: {stats['flagged']}, Rejected: {stats['rejected']}")
    return stats
