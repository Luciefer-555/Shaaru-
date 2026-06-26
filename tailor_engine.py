"""
tailor_engine.py
SHAARU — Universal Tailor Engine

Image → deep analysis → gap questions → spec sheet → FLUX sketch
"""

import os
import json
import base64
import requests
import re
from PIL import Image
import io
from bson import ObjectId

from shaaru_brain import _get_db, nvidia_call
from tavily import TavilyClient

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

def extract_city(text: str) -> str:
    # simple heuristic or regex
    cities = ["bengaluru", "mumbai", "delhi", "chennai", "kolkata", "hyderabad"]
    for city in cities:
        if city in text.lower():
            return city
    return ""

def extract_height_ft(text: str):
    import re
    match = re.search(r'(\d+)\s*(?:ft|feet|\')\s*(\d+)?\s*(?:in|inches|")?', text.lower())
    if match:
        ft = float(match.group(1))
        inches = float(match.group(2)) if match.group(2) else 0.0
        return round(ft + (inches / 12.0), 2)
    match2 = re.search(r'(\d+\.\d+)\s*(?:ft|feet)', text.lower())
    if match2:
        return float(match2.group(1))
    return None

def extract_primary_garment(garment_type: str) -> str:
    garment_type = garment_type.lower()
    if 'kurta' in garment_type:
        return 'kurta'
    if 'saree' in garment_type or 'sari' in garment_type:
        return 'saree'
    if 'lehenga' in garment_type:
        return 'lehenga'
    if 'blazer' in garment_type or 'jacket' in garment_type:
        return 'blazer'
    if 'shirt' in garment_type:
        return 'shirt'
    if 'pants' in garment_type or 'trouser' in garment_type:
        return 'trousers'
    return garment_type.split()[0] if garment_type else ""

def resize_image_b64(image_b64: str, max_size: int = 768) -> str:
    import base64
    from PIL import Image
    import io
    try:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buf, format='JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Image resize failed: {e}")
        return image_b64

def analyze_garment_deep(image_b64: str) -> dict:
    from shaaru_brain import nvidia_call, _get_client
    
    image_b64 = resize_image_b64(image_b64, max_size=768)
    
    prompt = """CONTEXT: This is a professional fashion garment image for tailoring analysis. 
Analyze only the clothing. Do not describe the person.

You are a master tailor with 40 years of experience.
Extract EVERY visible construction detail.
For each attribute: state the value AND confidence (high/medium/low/unclear).

Return ONLY valid JSON:
{
  "garment_type": "",
  "tradition": "",
  "occasion": "",
  "gender_expression": "",
  "silhouette": {
    "overall_shape": {"value": "", "confidence": ""},
    "fit": {"value": "", "confidence": ""},
    "length": {"value": "", "confidence": ""},
    "hem_shape": {"value": "", "confidence": ""}
  },
  "construction": {
    "shoulder_type": {"value": "", "confidence": ""},
    "chest_structure": {"value": "", "confidence": ""},
    "lining": {"value": "", "confidence": ""}
  },
  "front": {
    "closure_type": {"value": "", "confidence": ""},
    "collar_style": {"value": "", "confidence": ""},
    "neckline": {"value": "", "confidence": ""}
  },
  "sleeves": {
    "sleeve_type": {"value": "", "confidence": ""},
    "sleeve_length": {"value": "", "confidence": ""},
    "cuff_style": {"value": "", "confidence": ""}
  },
  "fabric": {
    "fiber_type": {"value": "", "confidence": ""},
    "weight": {"value": "", "confidence": ""},
    "primary_color": {"value": "", "confidence": ""},
    "pattern_type": {"value": "", "confidence": ""}
  },
  "embellishment": {
    "type": {"value": "", "confidence": ""},
    "placement": {"value": "", "confidence": ""}
  },
  "replication_complexity": "simple|intermediate|complex|master",
  "tailor_notes": ""
}

If attribute not visible: confidence = "unclear", value = ""
If section not applicable: all values = "N/A"
"""
    try:
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
        
        client = _get_client()
        raw_obj = client.chat.completions.create(
            model="meta/llama-3.2-90b-vision-instruct", 
            messages=vision_messages, 
            temperature=0.1, 
            timeout=120.0
        )
        response_text = raw_obj.choices[0].message.content
        print("RAW VISION MODEL TEXT RESPONSE:", repr(response_text))
        
        if not response_text or len(response_text) < 20:
            print("[VISION EMPTY] Full object:", raw_obj)
            
        if not response_text or "{" not in response_text:
            raw_obj2 = client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct", 
                messages=vision_messages, 
                temperature=0.1, 
                timeout=120.0
            )
            response_text = raw_obj2.choices[0].message.content
            print("RAW VISION MODEL TEXT RESPONSE 11B:", repr(response_text))
            
            if not response_text or len(response_text) < 20:
                print("[VISION EMPTY 11B] Full object:", raw_obj2)
            
        data = extract_json_from_response(response_text)
        
        if not data:
            print("[JSON RECOVERY] Vision model returned unstructured text. Attempting recovery with 8B...")
            recovery_prompt = f"""Convert this unstructured garment analysis into the exact requested JSON format.
            
Unstructured text:
{response_text}

Expected JSON schema:
{prompt}

Return ONLY valid JSON. No conversational text."""
            try:
                recovery_text = nvidia_call(client, "meta/llama-3.1-8b-instruct", [{"role": "user", "content": recovery_prompt}], temperature=0.1)
                data = extract_json_from_response(recovery_text)
            except Exception as e:
                print(f"[JSON RECOVERY FAILED] {e}")
                
        print(f"[OK] analyze_garment_deep: {data.get('garment_type')} | {data.get('tradition')}")
        return data
    except Exception as e:
        print(f"[FAIL] analyze_garment_deep: {e}")
        return {}

def extract_unclear_dimensions(analysis: dict) -> list:
    gaps = []
    skip_keys = {"tailor_notes", "replication_complexity", "garment_type", "tradition", "occasion", "gender_expression"}
    
    def _walk(d, path, section_name):
        if isinstance(d, dict):
            if "confidence" in d and "value" in d:
                if d["confidence"] in ["unclear", "low"] and not d["value"]:
                    gaps.append({
                        "path": path,
                        "label": path.split(".")[-1].replace("_", " ").title(),
                        "section": section_name.title()
                    })
            else:
                for k, v in d.items():
                    if k in skip_keys:
                        continue
                    new_path = f"{path}.{k}" if path else k
                    _walk(v, new_path, path if path else k)
                    
    _walk(analysis, "", "")
    return gaps

def generate_questions_for_gaps(gaps: list, garment_type: str, tradition: str) -> list:
    if not gaps:
        return []
        
    from shaaru_brain import nvidia_call, _get_client
    gaps_formatted = json.dumps(gaps, indent=2)
    
    prompt = f"""You are a master tailor. A customer wants to make a {garment_type} ({tradition}).

These details are unclear from the image and need clarification:
{gaps_formatted}

Generate a question for each unclear attribute in Shaaru's voice — direct, fashion-expert, casual.

Return ONLY a JSON array:
[
  {{
    "id": "snake_case_id",
    "label": "Short Label",
    "question": "Shaaru's question referencing the specific garment",
    "options": [
      {{"id": "option_id", "label": "Short", "desc": "One precise line"}}
    ]
  }}
]

Rules:
- 3-5 options per question
- Options specific to this garment type and tradition
- Max 10 questions total
- For Indian garments include tradition-specific options
"""
    try:
        client = _get_client()
        response_text = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": prompt}], temperature=0.5)
        questions = extract_json_from_response(response_text)
        return questions[:10]
    except Exception as e:
        print(f"[FAIL] generate_questions_for_gaps: {e}")
        return []

def generate_universal_brief(analysis: dict, user_answers: dict, user_profile: dict) -> dict:
    from shaaru_brain import nvidia_call, _get_client
    
    spec_list = []
    # Collect high/medium from analysis
    def _collect_spec(d, section_name):
        if isinstance(d, dict):
            if "confidence" in d and "value" in d:
                if d["confidence"] in ["high", "medium"] and d["value"]:
                    spec_list.append({
                        "source": "image",
                        "section": section_name.title() if section_name else "Details",
                        "value": d["value"],
                        "confidence": d["confidence"]
                    })
            else:
                for k, v in d.items():
                    if k not in ["tailor_notes", "replication_complexity", "garment_type", "tradition", "occasion", "gender_expression"]:
                        _collect_spec(v, k.replace("_", " "))
    
    _collect_spec(analysis, "")
    
    # Add user answers
    for q_id, ans in user_answers.items():
        spec_list.append({
            "source": "user",
            "section": q_id.replace("_", " ").title(),
            "value": ans,
            "confidence": "high"
        })
        
    measurements = user_profile.get("measurements", {"chest": "", "waist": "", "shoulder": "", "sleeve": "", "height": ""})
    
    # LLM for instructions and notes
    instr_prompt = "Write technical tailor instructions (under 300 words) for making this garment. Use professional tailoring language."
    shaaru_prompt = "Write 2-3 sentences of closing notes to the user in Shaaru's voice (casual, expert). Under 80 words."
    
    try:
        client = _get_client()
        tailor_instructions = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": f"{instr_prompt}\nGarment: {analysis.get('garment_type')}\nSpec: {spec_list}"}], temperature=0.3)
        shaaru_notes = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": f"{shaaru_prompt}\nGarment: {analysis.get('garment_type')}"}], temperature=0.7)
    except:
        tailor_instructions = ""
        shaaru_notes = "Let's get this made!"

    return {
        "garment_name": analysis.get("garment_type", "Garment"),
        "tradition": analysis.get("tradition", ""),
        "occasion": analysis.get("occasion", ""),
        "replication_complexity": analysis.get("replication_complexity", ""),
        "spec": spec_list,
        "measurements": measurements,
        "quality_signals": [],
        "tailor_instructions": tailor_instructions,
        "shaaru_notes": shaaru_notes,
        "sketch_url": None
    }

def generate_garment_sketch(brief: dict) -> str | None:
    prompt = (
        f"technical fashion flat sketch of a {brief.get('garment_name')}, "
        f"{brief.get('tradition')} construction, "
        f"pure white background, precise clean ink line art, "
        f"professional fashion technical illustration, "
        f"front view, no person, garment only, "
        f"with dashed measurement annotation lines, "
        f"high contrast black lines on white, "
        f"editorial atelier quality, fashion design spec sheet style"
    )
    try:
        response = requests.post(
            "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev",
            headers={"Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"},
            json={"prompt": prompt, "width": 768, "height": 1024},
            timeout=120
        )
        response.raise_for_status()
        b64 = response.json()["artifacts"][0]["base64"]
        print(f"[OK] generate_garment_sketch")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"[FAIL] generate_garment_sketch: {e}")
        return None

def generate_and_save_sketch_bg(session_id: str, brief: dict):
    from shaaru_brain import _get_db
    db = _get_db()
    sketch_url = generate_garment_sketch(brief)
    if sketch_url:
        db["tailor_sessions"].update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"brief.sketch_url": sketch_url}}
        )

def get_tailor_context(user_id: str) -> str:
    try:
        from shaaru_brain import _get_db
        db = _get_db()
        session = db["tailor_sessions"].find_one(
            {"user_id": user_id, "status": "active"},
            sort=[("created_at", -1)]
        )
        if session:
            ans = session.get("current_question_index", 0)
            tot = session.get("total_questions", 0)
            g_type = session.get("analysis", {}).get("garment_type", "Garment")
            return f"[CONTEXT] User is currently tailoring a {g_type} ({ans}/{tot} questions answered)."
        return ""
    except:
        return ""

def extract_style_dna(analysis: dict, product_details: dict = None) -> dict:
    from shaaru_brain import _get_client
    
    product_context = ""
    if product_details:
        product_context = f"\n\nBrand's Stated Details (Cross-Reference):\n{json.dumps(product_details, indent=2)}\n\nNOTE: If the brand's stated fabric or material conflicts with the vision analysis, trust the brand's stated details for the fabric base, and note the conflict."
    
    prompt = f"""You are a fashion intelligence system analyzing a garment.
From this vision analysis, extract the style DNA:

Analysis: {json.dumps(analysis, indent=2)}{product_context}

Identify:
1. What to KEEP if user wants to replicate this garment
2. What is REPLACEABLE (base fabric, color, fit)
3. Core embellishment style and technique
4. Silhouette classification

Return ONLY valid JSON:
{{
  "keep": ["element1", "element2"],
  "replaceable": ["fabric_base", "color", "fit_level"],
  "embellishment_style": "string",
  "embellishment_density": "high|medium|low|none",
  "silhouette": "string",
  "tradition": "western|indian|fusion",
  "construction_complexity": "simple|intermediate|complex|master",
  "key_details": ["string"]
}}"""
    try:
        client = _get_client()
        # Model Priority: 70b -> 8b
        try:
            response_text = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": prompt}], temperature=0.1)
        except:
            response_text = nvidia_call(client, "meta/llama-3.1-8b-instruct", [{"role": "user", "content": prompt}], temperature=0.1)
            
        data = extract_json_from_response(response_text)
        if not data:
            recovery_prompt = f"Convert this unstructured text to the requested JSON format:\n{response_text}\n\nExpected JSON:\n{{ 'keep': [], 'replaceable': [], 'embellishment_style': '', 'embellishment_density': '', 'silhouette': '', 'tradition': '', 'construction_complexity': '', 'key_details': [] }}"
            try:
                rec = nvidia_call(client, "meta/llama-3.1-8b-instruct", [{"role": "user", "content": recovery_prompt}], temperature=0.1)
                data = extract_json_from_response(rec)
            except:
                pass
            if not data:
                raise ValueError("Empty or invalid JSON")
        print("[OK] extract_style_dna")
        return data
    except Exception as e:
        print(f"[FAIL] extract_style_dna: {e}")
        return {}

def parse_modification_request(user_message: str, style_dna: dict, user_profile: dict) -> dict:
    from shaaru_brain import _get_client
    prompt = f"""User message: {user_message}
Style DNA of reference garment: {json.dumps(style_dna, indent=2)}
User profile: height={user_profile.get('height')}, city={user_profile.get('city')}, body_type={user_profile.get('body_type')}

Parse the user's modification request. Extract:
- What they want to keep from the reference
- What they want to change
- Their city (for sourcing)
- Their height in feet (for measurements)
- Their fit preference

Return ONLY valid JSON:
{{
  "keep_embellishment": true,
  "keep_silhouette": true,
  "base_change": "string or null",
  "color_change": "string or null",
  "fit_change": "string or null",
  "city": "string",
  "height_ft": 6.0,
  "gender": "string",
  "fit_preference": "baggy|loose|regular|fitted|tailored",
  "additional_requests": ["string"]
}}"""
    try:
        client = _get_client()
        try:
            response_text = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": prompt}], temperature=0.1)
        except:
            response_text = nvidia_call(client, "meta/llama-3.1-8b-instruct", [{"role": "user", "content": prompt}], temperature=0.1)
            
        data = extract_json_from_response(response_text)
        
        if not data:
            recovery_prompt = f"Convert this unstructured text to the requested JSON format:\n{response_text}\n\nExpected JSON format (keys: keep_embellishment, keep_silhouette, base_change, color_change, fit_change, city, height_ft, gender, fit_preference, additional_requests)"
            try:
                rec = nvidia_call(client, "meta/llama-3.1-8b-instruct", [{"role": "user", "content": recovery_prompt}], temperature=0.1)
                data = extract_json_from_response(rec)
            except:
                pass

        if not data:
            raise ValueError("Empty or invalid JSON")
            
        print("[OK] parse_modification_request")
        return data
    except Exception as e:
        print(f"[FAIL] parse_modification_request: {e}")
        return {
            "city": extract_city(user_message) or "",
            "height_ft": extract_height_ft(user_message),
            "gender": "male",
            "fit_preference": "regular"
        }

def query_fashion_intelligence(garment_type: str, fabric_need: str, city: str, height_ft: float, fit: str) -> dict:
    import re
    db = _get_db()
    
    # Safe coercion for LLM hallucinated dicts
    if isinstance(garment_type, dict):
        garment_type = garment_type.get('value') or garment_type.get('garment_type') or str(garment_type)
    if isinstance(fabric_need, dict):
        fabric_need = fabric_need.get('value') or str(fabric_need)
        
    if garment_type:
        garment_type = str(garment_type)
    if fabric_need:
        fabric_need = str(fabric_need)
    
    # Query 1 — Fabric
    fabric_record = None
    fabrics = []
    
    if fabric_need and fabric_need.lower() != "fabric":
        import re
        words = [w for w in re.split(r'[^a-zA-Z0-9]', fabric_need) if len(w) > 3]
        or_conditions = []
        for w in words:
            rx = re.compile(f".*{w}.*", re.IGNORECASE)
            or_conditions.append({"fiber_composition": rx})
            or_conditions.append({"common_names": rx})
            or_conditions.append({"fabric_id": rx})
            
        if or_conditions:
            query = {"$or": or_conditions}
            if fit and fit.lower() in ["baggy", "structured"]:
                query["structure_score"] = {"$gte": 6}
            print("DEBUG FABRIC SPECIFIC QUERY:", query)
            fabrics = list(db['fabric_intelligence'].find(query))

    if not fabrics and garment_type:
        rx = re.compile(f".*{garment_type.replace('_', ' ')}.*", re.IGNORECASE)
        query = {"best_for": rx}
        
        if fit and fit.lower() in ["baggy", "structured"]:
            query["structure_score"] = {"$gte": 6}
            
        print("DEBUG FABRIC GARMENT QUERY:", query)
        fabrics = list(db['fabric_intelligence'].find(query))
        
    if not fabrics:
        print("DEBUG FABRIC QUERY FALLBACK TO ALL")
        fabrics = list(db['fabric_intelligence'].find({}))
        
    for f in fabrics:
        if city and city.lower() in [k.lower() for k in f.get('sourcing', {}).keys()]:
            fabric_record = f
            break
    if not fabric_record and fabrics:
        fabric_record = fabrics[0]
            
    # Query 2 — Construction
    construction_record = None
    if garment_type:
        construction_record = db['garment_construction'].find_one({"garment_id": garment_type})
        if not construction_record:
            rx_garment = re.compile(f".*{garment_type}.*", re.IGNORECASE)
            construction_record = db['garment_construction'].find_one({"garment_id": rx_garment})
        
    # Query 3 — Measurements
    measurement_record = None
    if height_ft:
        meas_list = list(db['body_measurement_tables'].find({"garment": garment_type}))
        if not meas_list:
            rx_garment = re.compile(f".*{garment_type}.*", re.IGNORECASE)
            meas_list = list(db['body_measurement_tables'].find({"garment": rx_garment}))
        if meas_list:
            measurement_record = min(meas_list, key=lambda x: abs(x.get('height_ft', 0) - height_ft))
            
    # Query 4 — Tavily live sourcing
    live_sourcing = {"fabric": None, "embellishment": None}
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        if fabric_need and city:
            res_f = client.search(f"{fabric_need} wholesale price {city} fabric market", max_results=1)
            if res_f and res_f.get('results'):
                live_sourcing['fabric'] = res_f['results'][0]
    except Exception as e:
        print(f"[WARN] Tavily search failed: {e}")

    if fabric_record: fabric_record.pop('_id', None)
    if construction_record: construction_record.pop('_id', None)
    if measurement_record: measurement_record.pop('_id', None)
        
    # Query 5 — Embellishment Sourcing
    embellishment_record = None
    if city:
        embellishment_record = db['embellishment_sourcing'].find_one({f"sourcing.{city.lower()}": {"$exists": True}})
        if embellishment_record: embellishment_record.pop('_id', None)
            
    return {
        'fabric': fabric_record,
        'construction': construction_record,
        'measurements': measurement_record,
        'embellishment_sourcing': embellishment_record,
        'live_sourcing': live_sourcing
    }

def generate_construction_brief(
    fabric_rec: dict,
    measurement_rec: dict,
    construction_rec: dict,
    embellishment_rec: dict,
    modification: dict,
    vision_analysis: dict
) -> dict:
    from shaaru_brain import _get_client
    from datetime import datetime, timezone
    client = _get_client()

    context = {
        'garment_type': extract_primary_garment(
            vision_analysis.get('garment_type', '')
        ),
        'fabric_id': fabric_rec.get('fabric_id', ''),
        'fiber_composition': fabric_rec.get('fiber_composition', ''),
        'gsm': fabric_rec.get('gsm_range', {}),
        'drape_score': fabric_rec.get('drape_score', 5),
        'structure_score': fabric_rec.get('structure_score', 5),
        'weave': fabric_rec.get('weave', ''),
        'hand_feel': fabric_rec.get('hand_feel', ''),
        'embellishment_type': embellishment_rec.get('type', 'none'),
        'embellishment_technique': embellishment_rec.get('technique', ''),
        'height_ft': measurement_rec.get('height_ft', 5.5),
        'height_cm': measurement_rec.get('height_cm', 167),
        'gender': measurement_rec.get('gender', 'male'),
        'fit': modification.get('fit_preference', 'regular'),
        'measurements_cm': measurement_rec.get('measurements_cm', {}),
        'fabric_meters': measurement_rec.get('fabric_meters_needed', 2.5),
        'archetype_steps': construction_rec.get('construction_sequence', []),
        'archetype_critical': construction_rec.get('critical_points', []),
        'seam_allowances': construction_rec.get('seam_allowances', {}),
        'ease_by_fit': construction_rec.get('ease_by_fit', {}),
        'tradition': construction_rec.get('tradition', 'western'),
        'color': modification.get('color_change', ''),
        'embellishment_placement': vision_analysis.get('embellishment', {}).get('placement', {}).get('value', '')
    }

    context['garment_type'] = str(context.get('garment_type') or 'garment')
    context['tradition'] = str(context.get('tradition') or 'western')
    context['fiber_composition'] = str(context.get('fiber_composition') or 'unknown')
    context['weave'] = str(context.get('weave') or 'unknown')
    context['hand_feel'] = str(context.get('hand_feel') or 'unknown')
    context['embellishment_type'] = str(context.get('embellishment_type') or 'none')
    context['embellishment_technique'] = str(context.get('embellishment_technique') or 'none')
    context['embellishment_placement'] = str(context.get('embellishment_placement') or 'none')
    context['color'] = str(context.get('color') or 'as reference')
    context['gender'] = str(context.get('gender') or 'unisex')

    fit_label = (modification.get('fit_preference') or 'regular').upper()

    prompt = f"""You are a master tailor with 20 years experience in 
both Indian and Western garment construction. Generate a complete, 
professional tailor brief for the following garment.

GARMENT CONTEXT:
- Garment type: {context['garment_type']}
- Tradition: {context['tradition']}
- Fabric: {context['fabric_id']} ({context['fiber_composition']})
- GSM: {context['gsm']}
- Weave: {context['weave']}
- Hand feel: {context['hand_feel']}
- Drape score: {context['drape_score']}/10
- Structure score: {context['structure_score']}/10

BODY MEASUREMENTS:
- Height: {context['height_ft']}ft ({context['height_cm']}cm)
- Gender: {context['gender']}
- Fit preference: {context['fit']}
- Measurements (cm): {json.dumps(context['measurements_cm'])}
- Fabric needed: {context['fabric_meters']} meters

EMBELLISHMENT:
- Type: {context['embellishment_type']}
- Technique: {context['embellishment_technique']}
- Placement: {context['embellishment_placement']}

CONSTRUCTION ARCHETYPE (use as scaffold, expand with specifics):
{json.dumps(context['archetype_steps'], indent=2)}

CRITICAL POINTS FROM ARCHETYPE:
{json.dumps(context['archetype_critical'], indent=2)}

SEAM ALLOWANCES:
{json.dumps(context['seam_allowances'])}

EASE VALUES FOR {fit_label} FIT:
{json.dumps(context['ease_by_fit'].get(context['fit'], {}))}

Generate a complete tailor brief. Return ONLY valid JSON:
{{
  "construction_sequence": [
    "Step 1: [specific step with measurements and technique]",
    "Step 2: ...",
    ... minimum 15 steps, maximum 25 steps
  ],
  "critical_points": [
    "specific critical point with reason",
    ... minimum 5 points
  ],
  "pressing_sequence": [
    "Press instruction at specific construction stage",
    ... minimum 4 pressing instructions
  ],
  "grain_direction": "straight grain|bias|cross grain with reason",
  "interfacing_needed": true|false,
  "interfacing_spec": "where and what weight interfacing",
  "lining_needed": true|false,
  "lining_spec": "fabric type and coverage if needed",
  "embellishment_timing": "before_cutting|after_construction|during_construction with reason",
  "quality_checkpoints": [
    "checkpoint 1 with what to verify",
    "checkpoint 2",
    "checkpoint 3"
  ],
  "fabric_prep": "pre-wash/pre-shrink instructions based on fiber composition",
  "pressing_temperature": "iron temperature for this fiber",
  "estimated_construction_time": "X-Y hours for a skilled tailor"
}}

Rules:
- Every construction step must reference the actual measurements 
  provided — never say 'standard measurements'
- Seam allowances must be fabric-appropriate based on GSM and weave
- Embellishment timing must be enforced — heavy embroidery always 
  before cutting, sequins always after construction
- If drape_score > 7, mention bias grain consideration
- If structure_score < 4, specify interfacing requirements
- Steps must be in non-negotiable order — cannot be resequenced
- Use professional tailor terminology throughout
"""

    def validate_construction_output(result: dict) -> bool:
        if len(result.get('construction_sequence', [])) < 10:
            return False
        if len(result.get('critical_points', [])) < 3:
            return False
        if not result.get('grain_direction'):
            return False
        if not result.get('fabric_prep'):
            return False
        return True

    def get_fallback():
        fallback = dict(construction_rec)
        fallback['llm_generated'] = False
        return fallback

    try:
        raw_obj = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            timeout=45,
            max_tokens=2000
        )
        parsed = extract_json_from_response(raw_obj.choices[0].message.content)
        if not parsed or not validate_construction_output(parsed):
            raise Exception("Validation failed for 70b")
    except Exception as e:
        print(f"70b failed: {e}")
        try:
            raw_obj = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": prompt}],
                timeout=30,
                max_tokens=2000
            )
            parsed = extract_json_from_response(raw_obj.choices[0].message.content)
            if not parsed or not validate_construction_output(parsed):
                raise Exception("Validation failed for 8b")
        except Exception as e2:
            print(f"8b failed: {e2}")
            try:
                # Retry with temperature=0.2
                raw_obj = client.chat.completions.create(
                    model="meta/llama-3.1-8b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    timeout=30,
                    max_tokens=2000
                )
                parsed = extract_json_from_response(raw_obj.choices[0].message.content)
                if not parsed or not validate_construction_output(parsed):
                    return get_fallback()
            except:
                return get_fallback()

    parsed['llm_generated'] = True
    parsed['generated_at'] = datetime.now(timezone.utc).isoformat()
    return parsed

def assemble_complete_brief(vision_analysis: dict, style_dna: dict, modification: dict, knowledge: dict) -> dict:
    from shaaru_brain import _get_client
    client = _get_client()
    
    # LLM for tailor_instructions
    meas_json = json.dumps(knowledge.get('measurements', {}), indent=2, default=str)
    prompt_instr = f"""You are a master tailor. Write professional technical instructions for making this garment (under 400 words).
Cover the construction sequence and refer to these specific measurements:
{meas_json}
"""
    try:
        resp_instr = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": prompt_instr}], temperature=0.3)
    except:
        resp_instr = ""

    # LLM for shaaru_notes
    prompt_notes = "Write 2-3 sentences in Shaaru's voice (casual, expert) describing what this piece will feel like when done."
    try:
        resp_notes = nvidia_call(client, "meta/llama-3.1-70b-instruct", [{"role": "user", "content": prompt_notes}], temperature=0.7)
    except:
        resp_notes = ""

    print("Knowledge FABRIC:", knowledge.get('fabric'))
    print("Knowledge keys:", list(knowledge.keys()))  # debug — shows what's actually in knowledge dict
    
    city = modification.get('city', 'bengaluru')
    if isinstance(city, str):
        city = city.lower()

    # --- FIX 1: construction record — try both 'construction' and 'garment_construction' keys ---
    construction_rec = knowledge.get('construction') or knowledge.get('garment_construction') or {}
    
    construction_output = generate_construction_brief(
        fabric_rec=knowledge.get('fabric') or {},
        measurement_rec=knowledge.get('measurements') or {},
        construction_rec=construction_rec,
        embellishment_rec=knowledge.get('embellishment_sourcing') or knowledge.get('embellishment') or {},
        modification=modification,
        vision_analysis=vision_analysis
    )
    
    construction_sequence = construction_output.get('construction_sequence', [])
    critical_points = construction_output.get('critical_points', [])
    print(f"[DEBUG] construction_rec keys: {list(construction_rec.keys()) if construction_rec else 'EMPTY'}")
    print(f"[DEBUG] construction_sequence: {construction_sequence}")
    
    fabric_sourcing = (knowledge.get('fabric') or {}).get('sourcing', {}).get(city, {})
    if not fabric_sourcing:
        fabric_sourcing = (knowledge.get('fabric') or {}).get('sourcing', {}).get('bengaluru', {})

    # --- FIX 2: embellishment sourcing — try both 'embellishment_sourcing' and 'embellishment' keys ---
    emb_knowledge = knowledge.get('embellishment_sourcing') or knowledge.get('embellishment') or {}
    emb_sourcing = emb_knowledge.get('sourcing', {}).get(city, {})
    if not emb_sourcing:
        emb_sourcing = emb_knowledge.get('sourcing', {}).get('bengaluru', {})
    print(f"[DEBUG] emb_knowledge keys: {list(emb_knowledge.keys()) if emb_knowledge else 'EMPTY'}")
    print(f"[DEBUG] emb_sourcing ask_for: {emb_sourcing.get('ask_for', 'MISSING')}")

    fabric_markets = ", ".join(fabric_sourcing.get('markets', [])) or "Sourcing data unavailable"
    fabric_price_raw = fabric_sourcing.get('price_inr_per_meter', {})
    if isinstance(fabric_price_raw, dict) and 'min' in fabric_price_raw:
        fabric_price_min = fabric_price_raw['min']
        fabric_price_max = fabric_price_raw.get('max', fabric_price_min)
        fabric_price = f"₹{fabric_price_min}-{fabric_price_max}/meter"
    else:
        fabric_price_min = 0
        fabric_price_max = 0
        fabric_price = str(fabric_price_raw) if fabric_price_raw else 'Price data unavailable'
    
    emb_markets = ", ".join(emb_sourcing.get('markets', [])) or "Sourcing data unavailable"
    emb_price_raw = emb_sourcing.get('price_inr', {})
    if isinstance(emb_price_raw, dict) and 'min' in emb_price_raw:
        emb_price_min = emb_price_raw['min']
        emb_price_max = emb_price_raw.get('max', emb_price_min)
        emb_price = f"₹{emb_price_min}-{emb_price_max}"
    else:
        emb_price_min = 0
        emb_price_max = 0
        emb_price = str(emb_price_raw) if emb_price_raw else 'Price data unavailable'
        
    raw_gsm = (knowledge.get('fabric') or {}).get('gsm_range', '')
    if isinstance(raw_gsm, dict) and 'min' in raw_gsm and 'max' in raw_gsm:
        gsm_str = f"{raw_gsm['min']}-{raw_gsm['max']} GSM"
    else:
        gsm_str = str(raw_gsm) if raw_gsm else "Unknown GSM"
    
    garment_val = vision_analysis.get('garment_type', 'Custom Garment')
    if isinstance(garment_val, dict):
        garment_val = garment_val.get('value') or str(garment_val)

    # Resolve embellishment type
    vision_emb = vision_analysis.get('embellishment', {}).get('type', {})
    if isinstance(vision_emb, dict):
        vision_emb = vision_emb.get('value', '')
    dna_emb = style_dna.get('embellishment_style', '')
    
    final_emb_type = dna_emb
    if not final_emb_type or "buttonhole" in final_emb_type.lower():
        final_emb_type = vision_emb

    # Fetch embellishment details from DB
    emb_placement = ''
    emb_technique = ''
    emb_time = ''
    if final_emb_type:
        from shaaru_brain import _get_db
        import re
        db_inst = _get_db()
        kw = final_emb_type.lower()
        
        indian_terms = ["gold", "zari", "zardozi", "gota", "sequin", "mirror", "shisha", "thread", 
                        "metallic", "embroidery", "crystal", "pearl", "bead", "stone", "kundan"]
                        
        search_kw = None
        for term in indian_terms:
            if term in kw:
                search_kw = term
                break
                
        emb_rec = None
        
        if search_kw:
            rx_emb = re.compile(f".*{search_kw}.*", re.IGNORECASE)
            emb_rec = db_inst['embellishment_sourcing'].find_one({"type": rx_emb})
            
        if not emb_rec:
            words = [w for w in re.split(r'[^a-zA-Z0-9]', kw) if len(w) > 3]
            if words:
                or_conditions = [{"type": re.compile(f".*{w}.*", re.IGNORECASE)} for w in words]
                emb_rec = db_inst['embellishment_sourcing'].find_one({"$or": or_conditions})
                
        if not emb_rec:
            emb_rec = db_inst['embellishment_sourcing'].find_one({
                "technique": {"$regex": "hand-embroidery|hand-sewn", "$options": "i"}
            })

        if emb_rec:
            emb_technique = str(emb_rec.get('technique', ''))
            
            # also pull city-level ask_for from the fetched DB record (not just knowledge dict)
            db_emb_sourcing = emb_rec.get('sourcing', {}).get(city, {}) or emb_rec.get('sourcing', {}).get('bengaluru', {})
            if db_emb_sourcing and not emb_sourcing.get('ask_for'):
                emb_sourcing = db_emb_sourcing  # use DB record as fallback
                emb_markets = ", ".join(emb_sourcing.get('markets', [])) or emb_markets
                emb_price_raw2 = emb_sourcing.get('price_inr', {})
                if isinstance(emb_price_raw2, dict) and 'min' in emb_price_raw2:
                    emb_price_min = emb_price_raw2['min']
                    emb_price_max = emb_price_raw2.get('max', emb_price_min)
                    emb_price = f"₹{emb_price_min}-{emb_price_max}"
            
            coverage = emb_rec.get('coverage_estimate', {})
            units_needed = 0
            labor_hours = 0
            cov_desc = "high-density panel coverage"
            
            if isinstance(coverage, dict):
                for k, v in coverage.items():
                    if isinstance(v, dict):
                        cov_desc = k.replace('_', ' ')
                        units_needed = v.get('units_needed', 0)
                        labor_hours = v.get('labor_hours', 0)
                        break
                        
            if units_needed:
                t_lower = emb_technique.lower()
                type_lower = str(emb_rec.get('type', '')).lower()
                if 'hand-embroidery' in t_lower or any(x in type_lower for x in ['thread', 'zari', 'zardozi']):
                    target_placement = vision_analysis.get('embellishment', {}).get('placement', {}).get('value', 'the garment')
                    emb_placement = f"Hand embroidery work covering approximately {cov_desc}, concentrated on {target_placement}"
                else:
                    emb_placement = f"Approximately {units_needed} crystal/embellishment units across {cov_desc}"
            else:
                emb_placement = str(coverage)
                
            if labor_hours:
                emb_time = f"~{labor_hours} hours hand-embroidery work"
            else:
                emb_time = str(emb_rec.get('time_estimate', ''))

    # --- FIX 3: total_cost_estimate — calculate from fabric + embellishment ---
    meters_needed = float((knowledge.get('measurements') or {}).get('fabric_meters_needed', 2.5) or 2.5)
    tailor_labor_estimate = 800  # base labor flat rate INR, adjust later
    
    if fabric_price_min and fabric_price_max:
        fabric_cost_min = round(fabric_price_min * meters_needed)
        fabric_cost_max = round(fabric_price_max * meters_needed)
    else:
        fabric_cost_min = 0
        fabric_cost_max = 0

    total_min = fabric_cost_min + emb_price_min + tailor_labor_estimate
    total_max = fabric_cost_max + emb_price_max + tailor_labor_estimate

    if total_min > 0:
        total_cost_estimate = f"₹{total_min:,}–{total_max:,} (fabric ₹{fabric_cost_min}–{fabric_cost_max} + embellishment ₹{emb_price_min}–{emb_price_max} + tailor ₹{tailor_labor_estimate})"
    else:
        total_cost_estimate = "Estimate unavailable — sourcing data missing"

    brief = {
        'garment_name': garment_val,
        'reference_description': vision_analysis.get('tradition', ''),
        'modification_summary': str(modification.get('additional_requests', [])),
        'fabric_spec': {
            'fabric': (knowledge.get('fabric') or {}).get('fabric_id', ''),
            'gsm': gsm_str,
            'weave': (knowledge.get('fabric') or {}).get('weave', ''),
            'hand_feel': (knowledge.get('fabric') or {}).get('hand_feel', ''),
            'meters_needed': meters_needed,
            'color': modification.get('color_change', '')
        },
        'sourcing': {
            'fabric_market': fabric_markets,
            'fabric_ask_for': fabric_sourcing.get('ask_for', ''),
            'fabric_price_range': fabric_price,
            'embellishment_market': emb_markets,
            'embellishment_ask_for': emb_sourcing.get('ask_for', ''),
            'embellishment_price_range': emb_price,
            'total_cost_estimate': total_cost_estimate  # FIX 3
        },
        'measurements': knowledge.get('measurements') or {},
        'construction_sequence': construction_sequence,
        'critical_points': critical_points,
        'pressing_sequence': construction_output.get('pressing_sequence', []),
        'grain_direction': construction_output.get('grain_direction', ''),
        'interfacing_spec': construction_output.get('interfacing_spec', ''),
        'lining_spec': construction_output.get('lining_spec', ''),
        'embellishment_timing': construction_output.get('embellishment_timing', ''),
        'quality_checkpoints': construction_output.get('quality_checkpoints', []),
        'fabric_prep': construction_output.get('fabric_prep', ''),
        'pressing_temperature': construction_output.get('pressing_temperature', ''),
        'estimated_construction_time': construction_output.get('estimated_construction_time', ''),
        'llm_generated': construction_output.get('llm_generated', False),
        'embellishment_brief': {
            'type': final_emb_type,
            'placement': emb_placement,
            'technique': emb_technique,
            'time_estimate': emb_time
        },
        'tailor_instructions': resp_instr,
        'shaaru_notes': resp_notes,
        'sketch_url': None,
        'verified_sources': []
    }
    
    print("[OK] assemble_complete_brief")
    return brief

def fetch_product_page_details(url: str) -> dict:
    import requests
    from bs4 import BeautifulSoup
    import re
    from shaaru_brain import nvidia_call, _get_client
    
    result = {
        'product_name': '',
        'stated_fabric': None,
        'stated_details': '',
        'additional_image_urls': []
    }
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if resp.status_code != 200:
            return result
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        h1 = soup.find('h1')
        if h1:
            result['product_name'] = h1.get_text(strip=True)
            
        images = []
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src and ('jpg' in src.lower() or 'png' in src.lower() or 'webp' in src.lower()):
                if src.startswith('//'):
                    src = 'https:' + src
                if src not in images:
                    images.append(src)
            if len(images) >= 3:
                break
        result['additional_image_urls'] = images
        
        text_content = soup.get_text(separator=' ', strip=True)
        sentences = re.split(r'(?<=[.!?]) +', text_content)
        relevant_sentences = []
        keywords = ['fabric', 'composition', 'material', 'care', 'wash', 'cotton', 'silk', 'polyester', 'blend', 'made of']
        for s in sentences:
            if any(k in s.lower() for k in keywords):
                relevant_sentences.append(s)
                
        context_text = " ".join(relevant_sentences)[:2000]
        if not context_text:
            context_text = text_content[:2000]
            
        client = _get_client()
        prompt = f"""Extract factual product details from this webpage text.
Text snippet: {context_text}

Extract ONLY factual details:
1. Fabric composition or material if stated
2. Brief factual summary (no marketing fluff, just plain facts like color, style, care instructions)

Return ONLY valid JSON:
{{
  "stated_fabric": "string or null",
  "stated_details": "string"
}}"""
        try:
            llm_text = nvidia_call(client, "meta/llama-3.1-8b-instruct", [{"role": "user", "content": prompt}], temperature=0.1)
            data = extract_json_from_response(llm_text)
            if data:
                result['stated_fabric'] = data.get('stated_fabric')
                result['stated_details'] = data.get('stated_details', '')
        except:
            pass
            
    except Exception as e:
        print(f"[FAIL] fetch_product_page_details: {e}")
        
    return result

def process_reference_with_modification(image_b64: str, user_message: str, user_profile: dict, product_url: str = None) -> dict:
    print("Starting process_reference_with_modification pipeline...")
    analysis = analyze_garment_deep(image_b64)
    print("RAW VISION ANALYSIS:", json.dumps(analysis, indent=2))
    
    product_details = None
    if product_url:
        print(f"Fetching product details from {product_url}...")
        product_details = fetch_product_page_details(product_url)
        print("Product Details:", product_details)
        if product_details.get('stated_fabric'):
            # Override vision fabric if brand explicitly states it
            if 'tailor_notes' not in analysis:
                analysis['tailor_notes'] = ""
            analysis['tailor_notes'] += f"\nNote: Vision model identified fabric as {analysis.get('fabric', {}).get('fiber_type', {}).get('value')}, but brand explicitly states: {product_details['stated_fabric']}."
            
    style_dna = extract_style_dna(analysis, product_details)
    modification = parse_modification_request(user_message, style_dna, user_profile)
    
    garment_type = analysis.get("garment_type", "")
    primary_garment = extract_primary_garment(garment_type)
    
    fabric_need = modification.get("base_change")
    if not fabric_need or fabric_need.lower() == "fabric":
        if product_details and product_details.get('stated_fabric'):
            fabric_need = product_details['stated_fabric']
        else:
            fabric_need = "fabric"
    city = modification.get("city", "")
    height_ft = modification.get("height_ft")
    fit = modification.get("fit_preference", "regular")
    
    # query DB using primary_garment, and fallback height if None so the DB query doesn't crash
    knowledge = query_fashion_intelligence(primary_garment, fabric_need, city, height_ft if height_ft is not None else 5.5, fit)
    
    brief = assemble_complete_brief(analysis, style_dna, modification, knowledge)
    
    # Restore the original compound garment_type name for the brief display
    brief['garment_name'] = garment_type if garment_type else primary_garment
    
    if height_ft is None:
        brief['measurements'] = None
        brief['measurements_status'] = 'needs_user_height'
        
    print("[OK] process_reference_with_modification complete")
    return brief
