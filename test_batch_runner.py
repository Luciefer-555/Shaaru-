import requests
import base64
import json
import time
import concurrent.futures
from typing import List, Dict, Optional
import os
import re

# Load environment to get token creation util
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from auth import create_access_token

TEST_CASES = [
    {
        'name': 'sage_kurta_palazzo',
        'image_path': r'C:\Users\saipr\Downloads\Sage Green Ethnic Kurta with Palazzo _ Minimal Elegant Indian Outfit.jpg',
        'user_message': 'hey shaaru i found this kurta i love for college can u let me where can i get this kind of piece or can u let me where to source the materials and make this piece can u help me make this piece',
        'product_url': None
    },
    {
        'name': 'red_sheer_top_mumbai',
        'image_path': r'C:\Users\saipr\Downloads\585116176632119147.jpg',
        'user_message': 'hey shaaru i found this kurta i love for college can u let me where can i get this kind of piece or can u let me where to source the materials and make this piece can u help me make this piece i stay in mumbai',
        'product_url': None
    },
    {
        'name': 'chirag_kurta_wedding',
        'image_path': r'C:\Users\saipr\Downloads\Chirag Khanna Diwali Look _ Modern Kurta Style _ Festive Menswear Fashion Inspo.jpg',
        'user_message': 'hey shaaru i wanna make this kurta and as a guy for wedding i love this deep cut in the collar can u help me get the fabric and get the suggestions right for this kurta i wanna give it to a tailor with right measurements and everything',
        'product_url': None
    },
    {
        'name': 'double_breasted_suit_blr',
        'image_path': r'C:\Users\saipr\Downloads\Double-Breasted Jacket and Wide Pants.jpg',
        'user_message': 'i love this double breasted suit would mind telling me where to source these materials and whos a good tailor i can give this to and help me get a really good fabric i stay in bangalore',
        'product_url': None
    },
    {
        'name': 'ripped_thigh_jeans',
        'image_path': r'C:\Users\saipr\Downloads\Грег _ Greg.jpg',
        'user_message': 'hi shaaru i came across these pants where the thigh part is cut open how can i make this and sew it',
        'product_url': None
    },
    {
        'name': 'denim_skirt_from_pants',
        'image_path': r'C:\Users\saipr\Downloads\trendy_college_outfit_denim_skirt.jpg',
        'user_message': 'hi shaaru i came across this denim skirt and i have denim pants i could repurpose can you tell me how to make this and sew it',
        'product_url': None
    },
]

def flag_suspicious_output(brief: dict, user_message: str) -> List[str]:
    flags = []
    
    if not brief:
        return ["Brief is entirely empty"]
        
    garment_name = brief.get('garment_name', '')
    if garment_name in ('Custom Garment', 'Jacket') and 'jacket' not in user_message.lower():
        flags.append(f"Suspicious garment_name: {garment_name}")
        
    # Check for raw dict representations
    brief_str = json.dumps(brief)
    if "{'" in brief_str or "': '" in brief_str:
        flags.append("Raw Python dict representation leaked into JSON")
        
    emb = brief.get('embellishment_brief', {})
    emb_type = str(emb.get('type', '')).lower()
    has_embellishment = emb_type not in ['none', '', 'minimal', 'n/a', 'none.']

    sourcing = brief.get('sourcing', {})
    for k, v in sourcing.items():
        if v == "":
            if k == 'embellishment_ask_for' and not has_embellishment:
                continue
            flags.append(f"Empty sourcing field: {k}")
            
    if has_embellishment:
        if not emb.get('placement') or not emb.get('technique'):
            flags.append("Embellishment placement or technique is empty")
        
    # Check height extraction logic
    meas = brief.get('measurements', {})
    if meas:
        height_ft = meas.get('height_ft')
        if height_ft is not None:
            # simple check if user_message mentioned a height that differs widely
            match = re.search(r'(\d+)ft\s*(\d+)in', user_message)
            if match:
                ft = int(match.group(1))
                inch = int(match.group(2))
                expected_ht = ft + inch/12.0
                if abs(height_ft - expected_ht) > 0.3:
                    flags.append(f"Measurements height_ft ({height_ft}) diverges from expected ({expected_ht})")
                    
    seq = brief.get('construction_sequence', [])
    if not seq or len(seq) == 0:
        flags.append("Empty construction sequence")
        
    return flags

def run_single_test(test_case: Dict) -> Dict:
    start_time = time.time()
    result = {
        'name': test_case['name'],
        'status': 'error',
        'time_seconds': 0.0,
        'brief': None,
        'error': None,
        'flags': [],
        'construction_steps': 0,
        'llm_generated': False,
        'fabric_matched': '',
        'city_sourcing': ''
    }
    
    try:
        if not os.path.exists(test_case['image_path']):
            raise FileNotFoundError(f"Image not found: {test_case['image_path']}")
            
        with open(test_case['image_path'], 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
            
        token = create_access_token({'sub': 'batch_user', 'user_id': 'batch_user'})
        headers = {'Authorization': f'Bearer {token}'}
        
        payload = {
            'user_id': 'batch_user',
            'project_id': 'batch_project',
            'image_b64': image_b64,
            'user_message': test_case['user_message']
        }
        if test_case.get('product_url'):
            payload['product_url'] = test_case['product_url']
            
        resp = requests.post(
            'http://localhost:8000/api/tailor/reference',
            headers=headers,
            json=payload,
            timeout=500
        )
        
        if resp.status_code != 200:
            result['error'] = f"HTTP {resp.status_code}: {resp.text}"
        else:
            data = resp.json()
            brief = data.get('brief')
            result['brief'] = brief
            result['status'] = 'success'
            result['flags'] = flag_suspicious_output(brief, test_case['user_message'])
            if brief:
                result['construction_steps'] = len(brief.get('construction_sequence', []))
                result['llm_generated'] = brief.get('llm_generated', False)
                result['fabric_matched'] = brief.get('fabric_spec', {}).get('fabric', '')
                result['city_sourcing'] = brief.get('sourcing', {}).get('fabric_market', '')
            
    except Exception as e:
        result['error'] = str(e)
        
    result['time_seconds'] = round(time.time() - start_time, 2)
    return result

def run_batch(max_concurrent: int = 1) -> List[Dict]:
    results = []
    print(f"Running batch with {len(TEST_CASES)} cases (max_concurrent={max_concurrent})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_case = {executor.submit(run_single_test, case): case for case in TEST_CASES}
        for future in concurrent.futures.as_completed(future_to_case):
            results.append(future.result())
    return results

def print_report(results: List[Dict]):
    print("\n" + "="*135)
    print("BATCH TEST REPORT")
    print("="*135)
    print(f"{'TEST NAME':<26} | {'STATUS':<6} | {'TIME(s)':<7} | {'STEPS':<5} | {'LLM':<5} | {'FABRIC':<30} | {'SOURCING':<30}")
    print("-" * 135)
    
    for r in results:
        status = r['status'].upper()
        if r['error']:
            status = 'ERR'
        steps = r.get('construction_steps', 0)
        llm = str(r.get('llm_generated', False))
        fabric = str(r.get('fabric_matched', ''))[:30]
        sourcing = str(r.get('city_sourcing', ''))[:30]
        print(f"{r['name']:<26} | {status:<6} | {r['time_seconds']:<7.2f} | {steps:<5} | {llm:<5} | {fabric:<30} | {sourcing:<30}")
        
    print("="*135)
    
    # Print full details for anything that failed or had flags
    for r in results:
        if r['error'] or r['flags']:
            print(f"\n[!] Details for: {r['name']}")
            if r['error']:
                print(f"    ERROR: {r['error']}")
            for flag in r['flags']:
                print(f"    FLAG : {flag}")

if __name__ == '__main__':
    results = run_batch(max_concurrent=1)
    print_report(results)
