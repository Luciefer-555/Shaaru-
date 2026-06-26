"""
SHAARU Bulk Knowledge Seeder
Feeds 200 fabric names + 50 construction types into the verified pipeline.
Run overnight: python bulk_seeder.py
"""

import os, json, time, logging
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient
from knowledge_verifier import run_verification_pipeline, extract_json_from_response
from shaaru_brain import _get_db, nvidia_call, _get_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('bulk_seeder')

tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
db = _get_db()
llm_client = _get_client()

# ============================================================
# FABRIC CANDIDATES (180 NEW)
# ============================================================
FABRIC_NAMES = [
    # INDIAN (80)
    "banarasi_silk", "kanjivaram_silk", "tussar_silk", "muga_silk", "eri_silk",
    "patola_silk", "maheshwari_silk", "sambalpuri_silk", "baluchari_silk",
    "paithani_silk", "pochampally_ikat", "bagru_print_cotton",
    "ajrakh_cotton", "kalamkari_cotton", "block_print_cotton",
    "bandhani_cotton", "chikankari_georgette", "lucknowi_cotton_muslin",
    "kashmiri_pashmina", "kullu_shawl_wool", "kerala_cotton_kasavu",
    "chettinad_cotton", "kota_doria", "jamdani_cotton", "tant_cotton",
    "gadwal_silk", "narayanpet_cotton", "ilkal_saree_silk",
    "molakalmuru_silk", "dharwad_cotton", "khun_fabric", "uppada_silk",
    "venkatagiri_cotton", "mangalagiri_cotton", "handloom_cotton",
    "handloom_silk", "zari_fabric", "kinkhab_brocade", "himroo_brocade",
    "mashru_cotton_silk", "tanchoi_silk", "georgette_chiffon_blend",
    "crepe_silk", "tissue_silk", "cotton_silk_blend", "linen_silk_blend",
    "wool_silk_blend", "cotton_mulmul", "voile_cotton",
    "net_embroidered_silk", "raw_silk_plain", "art_silk",
    "viscose_georgette", "polyester_georgette_heavy",
    "cotton_jacquard", "silk_jacquard", "polyester_jacquard",
    "velvet_silk", "velvet_cotton", "velvet_polyester_crush",
    "net_plain", "net_sequined", "shimmer_fabric", "mirror_work_fabric",
    "gota_patti_fabric", "threadwork_base_fabric", "cutwork_fabric",
    "sequin_fabric_heavy", "sequin_fabric_light", "embroidered_net",
    "embroidered_georgette", "embroidered_chiffon",
    "khadi_silk", "khadi_wool", "modal_cotton_blend_india",
    "bamboo_cotton_india", "organic_cotton_india", "slub_cotton",
    "dobby_cotton", "honeycomb_cotton", "waffle_cotton",
    # GLOBAL WOVENS (60)
    "wool_crepe", "wool_gabardine", "wool_flannel", "wool_melton",
    "cashmere_plain", "alpaca_blend", "mohair_blend",
    "belgian_linen", "irish_linen", "ramie_linen_blend",
    "hemp_cotton_blend", "cotton_broadcloth", "cotton_twill",
    "cotton_sateen", "cotton_chambray", "cotton_madras",
    "cotton_seersucker", "denim_raw_selvedge",
    "japanese_selvedge_denim", "canvas_cotton", "duck_cotton",
    "ripstop_nylon", "nylon_taffeta", "polyester_taffeta",
    "silk_taffeta", "silk_charmeuse", "silk_habotai",
    "silk_crepe_de_chine", "silk_velvet", "rayon_challis",
    "rayon_twill", "modal_woven", "tencel_woven", "bamboo_woven",
    "cupro_woven", "acetate_satin", "microfiber_woven",
    "polyester_satin_heavy", "polyester_crepe", "neoprene_fabric",
    "scuba_fabric", "ponte_roma", "bengaline_fabric",
    "faille_fabric", "grosgrain_fabric", "shantung_silk",
    "dupioni_polyester", "brocade_polyester", "jacquard_wool",
    "tartan_wool", "houndstooth_wool", "herringbone_wool",
    "tweed_wool", "bouclé_wool", "corduroy_cotton",
    "velveteen_cotton", "moleskin_fabric", "suede_fabric",
    "faux_leather", "faux_fur", "technical_mesh",
    # GLOBAL KNITS (40)
    "modal_jersey", "bamboo_jersey", "tencel_jersey",
    "viscose_jersey", "polyester_jersey", "nylon_jersey",
    "spandex_blend_jersey", "cotton_interlock", "cotton_rib_knit",
    "cotton_pique", "french_terry", "loopback_terry",
    "fleece_fabric", "polar_fleece", "sherpa_fleece",
    "velour_knit", "burnout_velvet_knit", "athletic_mesh",
    "power_mesh", "compression_fabric", "supplex_nylon",
    "lycra_blend", "neoprene_knit", "pointelle_knit",
    "cable_knit_fabric", "ribbed_knit_heavy", "waffle_knit",
    "thermal_knit", "double_knit", "jacquard_knit",
    "sweater_knit_chunky", "merino_knit", "lambswool_knit",
    "angora_blend_knit", "acrylic_knit", "acrylic_wool_blend",
    "techno_fabric", "bonded_fabric", "quilted_fabric",
    "double_faced_fabric"
]

def auto_build_fabric_candidate(fabric_id: str) -> dict:
    prompt = f"""Generate a complete fabric specification record 
for {fabric_id.replace('_', ' ')} fabric used in fashion/clothing.

Return ONLY valid JSON with these exact fields:
{{
  "fabric_id": "{fabric_id}",
  "common_names": ["list of common names for this fabric"],
  "fiber_composition": "exact fiber % composition",
  "gsm_range": {{"min": 100, "max": 200}},
  "weave": "weave or construction type",
  "drape_score": 5,
  "structure_score": 5,
  "hand_feel": "tactile description in 1 line",
  "best_for": ["list of garment types this fabric suits"],
  "avoid_for": ["list of garment types to avoid"],
  "embellishment_compatibility": {{
    "heavy_crystal_pearl": "high|medium|low",
    "embroidery": "high|medium|low",
    "sequins": "high|medium|low",
    "reason": "one line reason"
  }},
  "seasonal": ["list of suitable seasons"],
  "sourcing": {{
    "bengaluru": {{"markets": ["Chickpet"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}},
    "mumbai": {{"markets": ["Mangaldas Market"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}},
    "delhi": {{"markets": ["Chandni Chowk"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}},
    "chennai": {{"markets": ["Pondy Bazaar"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}},
    "hyderabad": {{"markets": ["Laad Bazaar"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}},
    "kolkata": {{"markets": ["Burrabazar"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}},
    "surat": {{"markets": ["Ring Road fabric market"], "ask_for": "what to say to shopkeeper", "price_inr_per_meter": {{"min": 100, "max": 500}}, "quality_check": "what to check"}}
  }}
}}"""
    
    try:
        response = llm_client.chat.completions.create(model="meta/llama-3.1-8b-instruct", messages=[{"role": "user", "content": prompt}], max_tokens=1000, timeout=25)
        return extract_json_from_response(response.choices[0].message.content)
    except Exception as e:
        log.info(f"[SKIP] {fabric_id}: {e}")
        return None

# ============================================================
# CONSTRUCTION CANDIDATES (40 NEW)
# ============================================================
GARMENT_TYPES = [
    "sherwani", "nehru_jacket", "bandhgala_suit", "jodhpuri_suit",
    "dhoti_pants", "sharara_set", "gharara_set", "lehenga_choli",
    "saree_draping_guide", "half_saree", "salwar_kameez_straight",
    "salwar_kameez_patiala", "churidar_pants", "angrakha_kurta",
    "indo_western_jacket", "cape_kurta", "asymmetric_kurta",
    "shirt_kurta", "longline_kurta", "mirror_work_blouse",
    "backless_blouse", "halter_blouse", "crop_top_indian",
    "cold_shoulder_top", "peplum_top", "wrap_dress",
    "shirt_dress", "maxi_dress", "midi_skirt", "mini_skirt",
    "cargo_pants", "jogger_pants", "chino_pants", "shorts_bermuda",
    "bomber_jacket", "trench_coat", "structured_blazer_double",
    "cape_coat", "jumpsuit_wide_leg", "romper"
]

def auto_build_construction_candidate(garment: str) -> dict:
    prompt = f"""Generate a detailed construction sequence for {garment.replace('_', ' ')} garment. Return ONLY valid JSON matching this schema:
{{
  "garment_id": "{garment}",
  "category": "tops|bottoms|dresses|outerwear",
  "tradition": "indian|western|fusion",
  "construction_sequence": ["step 1", "step 2", "..."],
  "critical_points": ["point 1", "point 2"],
  "seam_allowances": {{"standard": "1.5cm"}},
  "ease_by_fit": {{"regular": {{"chest_ease_cm": 10, "hip_ease_cm": 10}}}},
  "measurements_needed": ["chest", "waist", "length"],
  "embellishment_notes": {{}},
  "recommended_fabrics": ["fabric1", "fabric2"]
}}"""
    
    try:
        response = llm_client.chat.completions.create(model="meta/llama-3.1-8b-instruct", messages=[{"role": "user", "content": prompt}], max_tokens=1000, timeout=25)
        return extract_json_from_response(response.choices[0].message.content)
    except Exception as e:
        log.info(f"[SKIP] {garment}: {e}")
        return None

# ============================================================
# SEED FUNCTIONS
# ============================================================
def bulk_seed_fabrics():
    log.info("Starting Fabric Bulk Seed")
    count = 0
    batch = []
    
    for i, fid in enumerate(FABRIC_NAMES):
        cand = auto_build_fabric_candidate(fid)
        if cand:
            batch.append(cand)
            
        if len(batch) >= 10 or i == len(FABRIC_NAMES) - 1:
            for candidate in batch:
                if candidate and candidate.get('fabric_id'):
                    candidate['verified'] = False
                    candidate['auto_generated'] = True
                    candidate['generated_at'] = datetime.now(timezone.utc).isoformat()
                    db['fabric_intelligence'].update_one(
                        {'fabric_id': candidate['fabric_id']},
                        {'$set': candidate},
                        upsert=True
                    )
                    count += 1
            batch = []
            log.info(f"[PROGRESS] {i+1}/{len(FABRIC_NAMES)} fabrics seeded")
            time.sleep(2)
            
    return count

def bulk_seed_constructions():
    log.info("Starting Construction Bulk Seed")
    count = 0
    batch = []
    
    for i, gid in enumerate(GARMENT_TYPES):
        cand = auto_build_construction_candidate(gid)
        if cand:
            batch.append(cand)
            
        if len(batch) >= 10 or i == len(GARMENT_TYPES) - 1:
            for candidate in batch:
                if candidate and candidate.get('garment_id'):
                    candidate['verified'] = False
                    candidate['auto_generated'] = True
                    candidate['generated_at'] = datetime.now(timezone.utc).isoformat()
                    db['garment_construction'].update_one(
                        {'garment_id': candidate['garment_id']},
                        {'$set': candidate},
                        upsert=True
                    )
                    count += 1
            batch = []
            log.info(f"[PROGRESS] {i+1}/{len(GARMENT_TYPES)} constructions seeded")
            time.sleep(2)
            
    return count

def bulk_seed_measurements():
    log.info("Starting Measurements Bulk Seed")
    count = 0
    
    garments = GARMENT_TYPES + ["kurta", "lehenga", "sherwani", "blouse", "trouser", "skirt"]
    
    def generate_measurements(gender, height_in):
        height_cm = round(height_in * 2.54)
        
        inseam = round(height_cm * 0.47)
        outseam = inseam + 24
        rise_front = round(inseam * 0.32)
        rise_back = rise_front + 4
        
        if gender == "male":
            thigh_circ = round(50 + ((height_in - 58) / 18) * 16)
            chest = round(84 + ((height_in - 58) / 18) * 24)
        else:
            thigh_circ = round(52 + ((height_in - 58) / 18) * 10)
            chest = round(80 + ((height_in - 58) / 18) * 20)
            
        kurta_length = round(height_cm * 0.38)
        sleeve_length = round(height_cm * 0.33)
        
        return {
            "inseam": inseam,
            "outseam": outseam,
            "rise_front": rise_front,
            "rise_back": rise_back,
            "thigh_circumference": thigh_circ,
            "chest_circumference": chest,
            "kurta_length": kurta_length,
            "sleeve_length": sleeve_length
        }
        
    for garment in garments:
        for height_in in range(58, 77, 2):
            height_ft = round(height_in / 12, 2)
            for gender in ["male", "female"]:
                meas = generate_measurements(gender, height_in)
                doc = {
                    "garment": garment,
                    "height_ft": height_ft,
                    "gender": gender,
                    "measurements": meas,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                db['body_measurement_tables'].update_one(
                    {"garment": garment, "height_ft": height_ft, "gender": gender},
                    {"$set": doc},
                    upsert=True
                )
                count += 1
                
    log.info(f"Seeded {count} measurement tables")
    return count

def run_bulk_seed_all():
    log.info("=== STARTING SHAARU BULK KNOWLEDGE SEED ===")
    
    m_count = bulk_seed_measurements()
    f_count = bulk_seed_fabrics()
    c_count = bulk_seed_constructions()
    
    log.info("=== BULK SEED COMPLETE ===")
    log.info(f"Fabrics: {f_count}")
    log.info(f"Constructions: {c_count}")
    log.info(f"Measurements: {m_count}")

if __name__ == "__main__":
    run_bulk_seed_all()
