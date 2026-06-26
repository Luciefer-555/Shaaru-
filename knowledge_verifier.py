import os
import json
import re
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

def _run_queries_and_get_sources(queries, max_sources=4):
    sources = []
    source_contents = []
    seen_urls = set()
    for q in queries:
        if len(sources) >= max_sources:
            break
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
        except Exception as e:
            print(f"Error querying Tavily for '{q}': {e}")
    return sources, source_contents

def verify_fabric_record(candidate: dict) -> dict | None:
    fabric_id = candidate.get('fabric_id', 'unknown')
    sources, source_contents = _run_queries_and_get_sources(fabric_queries(candidate))
    
    if not sources:
        print(f"[REJECT] {fabric_id} - no sources found")
        return None

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

Note: Sources may only partially confirm claims. 
If a source does not contradict a claim, treat it as supporting.

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
    garment_id = candidate.get('garment_id', 'unknown')
    sources, source_contents = _run_queries_and_get_sources(construction_queries(candidate))

    if not sources:
        print(f"[REJECT] {garment_id} - no sources found")
        return None

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

Note: Sources may only partially confirm claims. 
If a source does not contradict a claim, treat it as supporting.

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

Note: Sources may only partially confirm claims. 
If a source does not contradict a claim, treat it as supporting.

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

Note: Sources may only partially confirm claims. 
If a source does not contradict a claim, treat it as supporting.

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
