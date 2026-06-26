import math
from pymongo import UpdateOne
from knowledge_verifier import run_verification_pipeline
from shaaru_brain import _get_db

FABRIC_CANDIDATES = [
    {
        "fabric_id": "poly_viscose_suiting_twill",
        "common_names": ["PV suiting", "Twill suiting"],
        "fiber_composition": "65% polyester 35% viscose",
        "gsm_range": {"min": 280, "max": 340},
        "weave": "Twill",
        "drape_score": 3,
        "structure_score": 8,
        "hand_feel": "structured, smooth",
        "best_for": ["trousers", "blazers"],
        "avoid_for": ["draped dresses"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "high", "embroidery": "high", "sequins": "high", "reason": "strong structure"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "PV suiting", "price_inr_per_meter": {"min": 200, "max": 400}, "quality_check": "drape"}}
    },
    {
        "fabric_id": "cotton_poplin",
        "common_names": ["Poplin"],
        "fiber_composition": "100% cotton",
        "gsm_range": {"min": 90, "max": 120},
        "weave": "Plain",
        "drape_score": 4,
        "structure_score": 6,
        "hand_feel": "crisp, smooth",
        "best_for": ["shirts", "kurtas", "blouses"],
        "avoid_for": ["heavy outerwear"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "low", "reason": "lightweight"},
        "seasonal": ["summer"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "cotton poplin", "price_inr_per_meter": {"min": 80, "max": 150}, "quality_check": "smoothness"}}
    },
    {
        "fabric_id": "georgette_polyester",
        "common_names": ["Faux Georgette"],
        "fiber_composition": "100% polyester",
        "gsm_range": {"min": 60, "max": 80},
        "weave": "Crepe",
        "drape_score": 9,
        "structure_score": 2,
        "hand_feel": "grainy, flowy",
        "best_for": ["sarees", "dupattas", "dresses"],
        "avoid_for": ["structured suits"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "medium", "reason": "needs backing"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Commercial Street"], "ask_for": "poly georgette", "price_inr_per_meter": {"min": 50, "max": 120}, "quality_check": "drape"}}
    },
    {
        "fabric_id": "raw_silk_dupion",
        "common_names": ["Dupion Silk"],
        "fiber_composition": "100% silk",
        "gsm_range": {"min": 120, "max": 150},
        "weave": "Plain",
        "drape_score": 3,
        "structure_score": 8,
        "hand_feel": "textured sheen, crisp",
        "best_for": ["ethnic wear", "lehengas"],
        "avoid_for": ["flowy skirts"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "high", "embroidery": "high", "sequins": "high", "reason": "excellent structure"},
        "seasonal": ["winter", "festive"],
        "sourcing": {"bengaluru": {"markets": ["Avenue Road"], "ask_for": "raw silk", "price_inr_per_meter": {"min": 500, "max": 1500}, "quality_check": "slub pattern"}}
    },
    {
        "fabric_id": "chanderi_cotton_silk",
        "common_names": ["Chanderi"],
        "fiber_composition": "Cotton Silk blend",
        "gsm_range": {"min": 80, "max": 100},
        "weave": "Plain",
        "drape_score": 6,
        "structure_score": 5,
        "hand_feel": "lightweight sheen, crisp",
        "best_for": ["kurtas", "sarees"],
        "avoid_for": ["trousers"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "low", "reason": "sheer and delicate"},
        "seasonal": ["summer", "festive"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "chanderi silk", "price_inr_per_meter": {"min": 150, "max": 400}, "quality_check": "sheen"}}
    },
    {
        "fabric_id": "linen_plain_weave",
        "common_names": ["Linen"],
        "fiber_composition": "100% linen",
        "gsm_range": {"min": 140, "max": 180},
        "weave": "Plain",
        "drape_score": 4,
        "structure_score": 6,
        "hand_feel": "textured, breathable",
        "best_for": ["summer kurtas", "pants"],
        "avoid_for": ["evening gowns"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "low", "reason": "casual look"},
        "seasonal": ["summer"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "pure linen", "price_inr_per_meter": {"min": 300, "max": 800}, "quality_check": "creasing"}}
    },
    {
        "fabric_id": "crepe_de_chine",
        "common_names": ["Silk Crepe"],
        "fiber_composition": "100% silk or poly blend",
        "gsm_range": {"min": 75, "max": 100},
        "weave": "Crepe",
        "drape_score": 9,
        "structure_score": 2,
        "hand_feel": "fluid, slightly pebbled",
        "best_for": ["blouses", "draped silhouettes"],
        "avoid_for": ["structured coats"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "medium", "embroidery": "medium", "sequins": "medium", "reason": "delicate but resilient"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Commercial Street"], "ask_for": "crepe de chine", "price_inr_per_meter": {"min": 400, "max": 1200}, "quality_check": "drape"}}
    },
    {
        "fabric_id": "velvet_stretch",
        "common_names": ["Stretch Velvet"],
        "fiber_composition": "Polyester Spandex blend",
        "gsm_range": {"min": 280, "max": 320},
        "weave": "Pile",
        "drape_score": 6,
        "structure_score": 4,
        "hand_feel": "soft pile, stretchy",
        "best_for": ["evening wear", "blazers"],
        "avoid_for": ["loose summer wear"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "medium", "embroidery": "high", "sequins": "high", "reason": "pile holds thread well"},
        "seasonal": ["winter"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "stretch velvet", "price_inr_per_meter": {"min": 200, "max": 500}, "quality_check": "stretch recovery"}}
    },
    {
        "fabric_id": "cotton_lawn",
        "common_names": ["Lawn"],
        "fiber_composition": "100% cotton",
        "gsm_range": {"min": 60, "max": 80},
        "weave": "Plain",
        "drape_score": 7,
        "structure_score": 3,
        "hand_feel": "ultra-soft, sheer",
        "best_for": ["summer blouses", "linings"],
        "avoid_for": ["trousers"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "low", "sequins": "low", "reason": "too lightweight"},
        "seasonal": ["summer"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "cotton lawn", "price_inr_per_meter": {"min": 60, "max": 120}, "quality_check": "softness"}}
    },
    {
        "fabric_id": "brocade_zari",
        "common_names": ["Zari Brocade"],
        "fiber_composition": "Silk / Metallic blend",
        "gsm_range": {"min": 200, "max": 280},
        "weave": "Jacquard",
        "drape_score": 2,
        "structure_score": 9,
        "hand_feel": "stiff, metallic texture",
        "best_for": ["lehenga panels", "blouse"],
        "avoid_for": ["flowy skirts"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "high", "embroidery": "high", "sequins": "low", "reason": "already embellished via weave"},
        "seasonal": ["festive"],
        "sourcing": {"bengaluru": {"markets": ["Avenue Road"], "ask_for": "brocade fabric", "price_inr_per_meter": {"min": 300, "max": 1000}, "quality_check": "metallic thread quality"}}
    },
    {
        "fabric_id": "chiffon_polyester",
        "common_names": ["Faux Chiffon"],
        "fiber_composition": "100% polyester",
        "gsm_range": {"min": 40, "max": 60},
        "weave": "Plain",
        "drape_score": 10,
        "structure_score": 1,
        "hand_feel": "sheer float, slippery",
        "best_for": ["layers", "dupattas"],
        "avoid_for": ["structured wear"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "low", "sequins": "medium", "reason": "too delicate"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Commercial Street"], "ask_for": "poly chiffon", "price_inr_per_meter": {"min": 40, "max": 100}, "quality_check": "sheerness"}}
    },
    {
        "fabric_id": "denim_rigid",
        "common_names": ["Raw Denim"],
        "fiber_composition": "100% cotton",
        "gsm_range": {"min": 280, "max": 380},
        "weave": "Twill",
        "drape_score": 2,
        "structure_score": 9,
        "hand_feel": "structured, stiff",
        "best_for": ["jeans", "jackets"],
        "avoid_for": ["flowy tops"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "high", "embroidery": "high", "sequins": "medium", "reason": "very strong fabric"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "rigid denim", "price_inr_per_meter": {"min": 150, "max": 400}, "quality_check": "weight"}}
    },
    {
        "fabric_id": "modal_jersey",
        "common_names": ["Modal Knit"],
        "fiber_composition": "Modal Spandex blend",
        "gsm_range": {"min": 150, "max": 180},
        "weave": "Knit",
        "drape_score": 9,
        "structure_score": 2,
        "hand_feel": "soft drape, stretchy",
        "best_for": ["casual dresses", "tops"],
        "avoid_for": ["tailored suits"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "low", "sequins": "low", "reason": "knit structure sags"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "modal jersey", "price_inr_per_meter": {"min": 150, "max": 300}, "quality_check": "stretch recovery"}}
    },
    {
        "fabric_id": "organza_silk",
        "common_names": ["Silk Organza"],
        "fiber_composition": "100% silk",
        "gsm_range": {"min": 50, "max": 70},
        "weave": "Plain",
        "drape_score": 3,
        "structure_score": 7,
        "hand_feel": "stiff sheer, crisp",
        "best_for": ["overlay", "lehenga flare"],
        "avoid_for": ["fitted trousers"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "medium", "embroidery": "high", "sequins": "high", "reason": "strong despite being sheer"},
        "seasonal": ["festive"],
        "sourcing": {"bengaluru": {"markets": ["Avenue Road"], "ask_for": "silk organza", "price_inr_per_meter": {"min": 400, "max": 900}, "quality_check": "stiffness"}}
    },
    {
        "fabric_id": "muslin_cotton",
        "common_names": ["Muslin"],
        "fiber_composition": "100% cotton",
        "gsm_range": {"min": 80, "max": 120},
        "weave": "Plain",
        "drape_score": 6,
        "structure_score": 4,
        "hand_feel": "soft loose weave",
        "best_for": ["toile", "kurtas"],
        "avoid_for": ["heavy outerwear"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "low", "sequins": "low", "reason": "too loose"},
        "seasonal": ["summer"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "cotton muslin", "price_inr_per_meter": {"min": 40, "max": 100}, "quality_check": "weave density"}}
    },
    {
        "fabric_id": "satin_polyester",
        "common_names": ["Poly Satin"],
        "fiber_composition": "100% polyester",
        "gsm_range": {"min": 100, "max": 130},
        "weave": "Satin",
        "drape_score": 8,
        "structure_score": 3,
        "hand_feel": "slippery sheen, smooth",
        "best_for": ["lining", "evening wear"],
        "avoid_for": ["casual wear"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "medium", "reason": "slippery surface"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Commercial Street"], "ask_for": "poly satin", "price_inr_per_meter": {"min": 60, "max": 150}, "quality_check": "sheen"}}
    },
    {
        "fabric_id": "khadi_cotton",
        "common_names": ["Khadi"],
        "fiber_composition": "100% cotton",
        "gsm_range": {"min": 100, "max": 160},
        "weave": "Plain",
        "drape_score": 5,
        "structure_score": 5,
        "hand_feel": "handspun texture, matte",
        "best_for": ["kurtas", "ethnic"],
        "avoid_for": ["evening gowns"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "low", "reason": "rustic look"},
        "seasonal": ["all-year"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "khadi cotton", "price_inr_per_meter": {"min": 100, "max": 250}, "quality_check": "slub texture"}}
    },
    {
        "fabric_id": "net_embroidered",
        "common_names": ["Embroidered Net"],
        "fiber_composition": "Nylon/Polyester",
        "gsm_range": {"min": 80, "max": 120},
        "weave": "Knit",
        "drape_score": 7,
        "structure_score": 4,
        "hand_feel": "openwork, textured",
        "best_for": ["overlay", "dupatta"],
        "avoid_for": ["pants"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "high", "embroidery": "high", "sequins": "high", "reason": "used as base for heavy work"},
        "seasonal": ["festive"],
        "sourcing": {"bengaluru": {"markets": ["Commercial Street"], "ask_for": "heavy net", "price_inr_per_meter": {"min": 150, "max": 600}, "quality_check": "embroidery finish"}}
    },
    {
        "fabric_id": "wool_suiting",
        "common_names": ["Worsted Wool"],
        "fiber_composition": "100% wool",
        "gsm_range": {"min": 240, "max": 320},
        "weave": "Twill",
        "drape_score": 4,
        "structure_score": 8,
        "hand_feel": "warm structure, smooth",
        "best_for": ["winter blazers", "trousers"],
        "avoid_for": ["summer wear"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "medium", "embroidery": "high", "sequins": "low", "reason": "strong fabric but formal"},
        "seasonal": ["winter"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "wool suiting", "price_inr_per_meter": {"min": 600, "max": 1500}, "quality_check": "hand feel"}}
    },
    {
        "fabric_id": "cotton_cambric",
        "common_names": ["Cambric"],
        "fiber_composition": "100% cotton",
        "gsm_range": {"min": 80, "max": 100},
        "weave": "Plain",
        "drape_score": 6,
        "structure_score": 4,
        "hand_feel": "fine plain weave, smooth",
        "best_for": ["summer shirts", "kurtas"],
        "avoid_for": ["heavy coats"],
        "embellishment_compatibility": {"heavy_crystal_pearl": "low", "embroidery": "medium", "sequins": "low", "reason": "lightweight"},
        "seasonal": ["summer"],
        "sourcing": {"bengaluru": {"markets": ["Chickpet"], "ask_for": "cambric cotton", "price_inr_per_meter": {"min": 70, "max": 140}, "quality_check": "weave tightness"}}
    }
]

CONSTRUCTION_CANDIDATES = [
    {
        "garment_id": "wide_leg_baggy_trousers",
        "category": "bottoms",
        "tradition": "western",
        "construction_sequence": ["Draft pattern", "Cut panels", "Sew inseam", "Sew crotch", "Join side seams", "Attach waistband", "Hem"],
        "critical_points": ["Crotch curve depth"],
        "seam_allowances": {"standard": "1.5cm", "crotch": "1cm"},
        "ease_by_fit": {"baggy": {"hip_ease_cm": 15, "thigh_ease_cm": 10}},
        "measurements_needed": ["waist", "hip", "inseam", "outseam"],
        "embellishment_notes": {"pockets": {"placement": "side", "sequence": "before side seams", "technique": "inset"}},
        "recommended_fabrics": ["poly_viscose_suiting_twill", "denim_rigid"]
    },
    {
        "garment_id": "straight_leg_trousers",
        "category": "bottoms",
        "tradition": "western",
        "construction_sequence": ["Draft pattern", "Cut panels", "Sew darts", "Sew inseam", "Sew crotch", "Join side seams", "Attach waistband", "Hem"],
        "critical_points": ["Dart placement"],
        "seam_allowances": {"standard": "1.5cm", "crotch": "1cm"},
        "ease_by_fit": {"regular": {"hip_ease_cm": 5, "thigh_ease_cm": 4}},
        "measurements_needed": ["waist", "hip", "inseam", "outseam", "thigh"],
        "embellishment_notes": {"none": {"placement": "none", "sequence": "none", "technique": "none"}},
        "recommended_fabrics": ["poly_viscose_suiting_twill", "wool_suiting"]
    },
    {
        "garment_id": "kurta_straight_cut",
        "category": "tops",
        "tradition": "indian",
        "construction_sequence": ["Draft pattern", "Cut front and back", "Sew neckline facing", "Join shoulders", "Attach sleeves", "Sew side seams leaving slits", "Hem slits and bottom"],
        "critical_points": ["Slit finishing", "Neckline flatness"],
        "seam_allowances": {"standard": "1.5cm", "neckline": "1cm", "slits": "2cm"},
        "ease_by_fit": {"regular": {"chest_ease_cm": 10, "hip_ease_cm": 10}},
        "measurements_needed": ["chest", "waist", "hip", "shoulder", "length", "sleeve_length"],
        "embellishment_notes": {"embroidery": {"placement": "neckline", "sequence": "before construction", "technique": "hoop"}},
        "recommended_fabrics": ["cotton_poplin", "linen_plain_weave", "khadi_cotton"]
    },
    {
        "garment_id": "kurta_anarkali_flared",
        "category": "dresses",
        "tradition": "indian",
        "construction_sequence": ["Draft bodice and panels", "Sew bodice darts", "Join skirt panels", "Attach skirt to bodice", "Join shoulders", "Attach sleeves", "Side seams", "Hem"],
        "critical_points": ["Waist joining", "Panel alignment"],
        "seam_allowances": {"standard": "1.5cm", "waist": "1.5cm"},
        "ease_by_fit": {"fitted_bodice": {"chest_ease_cm": 5, "waist_ease_cm": 4}},
        "measurements_needed": ["chest", "waist", "shoulder", "length", "sleeve_length"],
        "embellishment_notes": {"zari": {"placement": "hem", "sequence": "after panels joined", "technique": "border attachment"}},
        "recommended_fabrics": ["georgette_polyester", "chanderi_cotton_silk", "raw_silk_dupion"]
    },
    {
        "garment_id": "saree_blouse_standard",
        "category": "tops",
        "tradition": "indian",
        "construction_sequence": ["Draft pattern with darts", "Cut fabric and lining", "Sew darts on both", "Join lining to fabric at neck", "Turn inside out", "Shoulder seams", "Side seams", "Attach sleeves", "Hem and closures"],
        "critical_points": ["Dart alignment", "Snug fit at bust and underbust"],
        "seam_allowances": {"standard": "1.5cm", "side": "2.5cm for alterations"},
        "ease_by_fit": {"tight": {"chest_ease_cm": 0, "underbust_ease_cm": 0}},
        "measurements_needed": ["chest", "underbust", "shoulder", "front_neck_depth", "back_neck_depth", "length", "sleeve_length", "sleeve_girth"],
        "embellishment_notes": {"heavy_work": {"placement": "all over", "sequence": "before cutting", "technique": "maggam"}},
        "recommended_fabrics": ["brocade_zari", "raw_silk_dupion", "cotton_poplin"]
    },
    {
        "garment_id": "shirt_formal_collar",
        "category": "tops",
        "tradition": "western",
        "construction_sequence": ["Draft pattern", "Apply interfacing to collar and cuffs", "Sew yoke to back", "Join shoulders", "Construct collar and attach", "Attach sleeves flat", "Sew side and sleeve seams", "Attach cuffs", "Hem", "Buttonholes"],
        "critical_points": ["Collar stand attachment", "Cuff pleats"],
        "seam_allowances": {"standard": "1.5cm", "collar": "1cm"},
        "ease_by_fit": {"regular": {"chest_ease_cm": 12, "waist_ease_cm": 10}},
        "measurements_needed": ["chest", "waist", "shoulder", "neck", "sleeve_length", "shirt_length"],
        "embellishment_notes": {"none": {"placement": "none", "sequence": "none", "technique": "none"}},
        "recommended_fabrics": ["cotton_poplin", "cotton_cambric"]
    },
    {
        "garment_id": "lehenga_skirt_flared",
        "category": "bottoms",
        "tradition": "indian",
        "construction_sequence": ["Draft kalis (panels)", "Cut main fabric and lining", "Join kalis", "Join lining panels", "Attach lining at waist", "Attach waistband and zip", "Hem with can-can if required"],
        "critical_points": ["Hem leveling", "Waistband finish"],
        "seam_allowances": {"standard": "1.5cm"},
        "ease_by_fit": {"fitted_waist": {"waist_ease_cm": 2}},
        "measurements_needed": ["waist", "length"],
        "embellishment_notes": {"borders": {"placement": "hem", "sequence": "before final side seam", "technique": "stitching"}},
        "recommended_fabrics": ["raw_silk_dupion", "brocade_zari", "organza_silk"]
    },
    {
        "garment_id": "blazer_single_breasted",
        "category": "outerwear",
        "tradition": "western",
        "construction_sequence": ["Draft pattern", "Apply extensive interfacing", "Construct welt pockets", "Join front and back", "Construct lapel and collar", "Set in sleeves", "Construct lining", "Bag out blazer", "Finish hems and buttons"],
        "critical_points": ["Lapel roll", "Sleeve head ease", "Shoulder pad insertion"],
        "seam_allowances": {"standard": "1.5cm", "neckline": "1cm", "hem": "4cm"},
        "ease_by_fit": {"tailored": {"chest_ease_cm": 10, "waist_ease_cm": 8}},
        "measurements_needed": ["chest", "waist", "shoulder", "sleeve_length", "length", "bicep"],
        "embellishment_notes": {"none": {"placement": "none", "sequence": "none", "technique": "none"}},
        "recommended_fabrics": ["wool_suiting", "poly_viscose_suiting_twill", "velvet_stretch"]
    },
    {
        "garment_id": "co_ord_set_top",
        "category": "tops",
        "tradition": "fusion",
        "construction_sequence": ["Draft pattern", "Join shoulders", "Finish neckline", "Attach sleeves or finish armhole", "Side seams", "Hem"],
        "critical_points": ["Pattern matching with bottoms"],
        "seam_allowances": {"standard": "1.5cm"},
        "ease_by_fit": {"relaxed": {"chest_ease_cm": 15}},
        "measurements_needed": ["chest", "shoulder", "length"],
        "embellishment_notes": {"none": {"placement": "none", "sequence": "none", "technique": "none"}},
        "recommended_fabrics": ["crepe_de_chine", "modal_jersey", "cotton_poplin"]
    },
    {
        "garment_id": "palazzo_pants",
        "category": "bottoms",
        "tradition": "fusion",
        "construction_sequence": ["Draft pattern", "Cut panels", "Sew inseams", "Sew crotch seam", "Join side seams", "Attach elastic waistband", "Hem"],
        "critical_points": ["Elastic distribution", "Crotch ease"],
        "seam_allowances": {"standard": "1.5cm"},
        "ease_by_fit": {"loose": {"hip_ease_cm": 20, "thigh_ease_cm": 15}},
        "measurements_needed": ["waist", "hip", "length", "rise"],
        "embellishment_notes": {"none": {"placement": "none", "sequence": "none", "technique": "none"}},
        "recommended_fabrics": ["georgette_polyester", "crepe_de_chine", "modal_jersey"]
    }
]

MEASUREMENT_TABLES = []
garments = ['wide_leg_baggy_trousers', 'kurta_straight_cut']
genders = ['male', 'female']

for height_in in range(60, 75, 2):  # 5ft (60in) to 6ft2 (74in)
    height_cm = int(height_in * 2.54)
    height_ft = round(height_in / 12, 2)
    
    for gender in genders:
        for garment in garments:
            # Proportions
            inseam = int(height_cm * 0.47)
            rise_front = int(inseam * 0.32)
            rise_back = rise_front + 4
            
            # Thigh scaling
            min_height_cm = 60 * 2.54
            max_height_cm = 74 * 2.54
            scale_factor = (height_cm - min_height_cm) / (max_height_cm - min_height_cm)
            base_thigh = 52 if gender == 'male' else 50
            max_thigh = 64 if gender == 'male' else 62
            thigh_circ = int(base_thigh + scale_factor * (max_thigh - base_thigh))
            
            # Outseam
            outseam = inseam + rise_front
            
            fabric_meters = 2.5 if height_ft > 5.5 else 2.0
            
            table = {
                'height_ft': height_ft,
                'height_cm': height_cm,
                'gender': gender,
                'garment': garment,
                'fit': 'standard',
                'measurements_cm': {
                    'inseam': inseam,
                    'outseam': outseam,
                    'rise_front': rise_front,
                    'rise_back': rise_back,
                    'thigh_circumference': thigh_circ,
                    'knee_circumference': int(thigh_circ * 0.8),
                    'leg_opening': int(thigh_circ * 0.7)
                },
                'fabric_meters_needed': fabric_meters,
                'notes': 'Derived from industry standards'
            }
            MEASUREMENT_TABLES.append(table)

EMBELLISHMENT_CANDIDATES = [
    {
        'embellishment_id': 'crystal_rhinestone_sewon',
        'type': 'clear rhinestones, sew-on',
        'technique': 'hand-sewn',
        'sizes_mm': [4, 5, 6, 7, 8],
        'best_for': ['bridal wear', 'lehengas', 'evening gowns'],
        'avoid_for': ['casual daywear', 'lightweight flowing fabrics'],
        'coverage_estimate': {
            'high_density_full_panel': {
                'units_needed': 1500,
                'labor_hours': 24
            }
        },
        'sourcing': {
            'bengaluru': {
                'markets': ['Commercial Street', 'Chickpet'],
                'ask_for': 'sew on crystal stones',
                'price_inr': {'min': 150, 'max': 350},
                'quality_check': 'check for glass clarity, no plastic backing'
            }
        }
    },
    {
        'embellishment_id': 'pearl_ivory_sewon',
        'type': 'ivory pearls, sew-on',
        'technique': 'hand-sewn',
        'sizes_mm': [4, 6, 8, 10],
        'best_for': ['bridal blouses', 'sarees', 'kurtas'],
        'avoid_for': ['activewear'],
        'coverage_estimate': {
            'high_density_full_panel': {
                'units_needed': 1200,
                'labor_hours': 20
            }
        },
        'sourcing': {
            'bengaluru': {
                'markets': ['Chickpet', 'Shivajinagar'],
                'ask_for': 'ivory moti, sew on',
                'price_inr': {'min': 100, 'max': 250},
                'quality_check': 'check coating durability (scratch test)'
            }
        }
    },
    {
        'embellishment_id': 'sequins_flat_polyester',
        'type': 'flat sequins',
        'technique': 'hand-embroidery',
        'sizes_mm': [6, 8, 10, 12],
        'best_for': ['party wear', 'dresses', 'sarees'],
        'avoid_for': ['office wear'],
        'coverage_estimate': {
            'high_density_full_panel': {
                'units_needed': 5000,
                'labor_hours': 30
            }
        },
        'sourcing': {
            'bengaluru': {
                'markets': ['Chickpet', 'Avenue Road'],
                'ask_for': 'flat sitara, sequins',
                'price_inr': {'min': 50, 'max': 150},
                'quality_check': 'color bleeding test in water'
            }
        }
    },
    {
        'embellishment_id': 'zardozi_thread_gold',
        'type': 'gold zardozi embroidery thread',
        'technique': 'hand-embroidery',
        'sizes_mm': [1],
        'best_for': ['heavy bridal wear', 'lehengas', 'sherwanis'],
        'avoid_for': ['casual cottons'],
        'coverage_estimate': {
            'high_density_full_panel': {
                'units_needed': 50, # grams
                'labor_hours': 40
            }
        },
        'sourcing': {
            'bengaluru': {
                'markets': ['Commercial Street', 'Shivajinagar'],
                'ask_for': 'zari thread, dabka',
                'price_inr': {'min': 500, 'max': 1200},
                'quality_check': 'check for oxidation resistance'
            }
        }
    },
    {
        'embellishment_id': 'mirror_work_shisha',
        'type': 'traditional shisha mirror pieces',
        'technique': 'hand-embroidery',
        'sizes_mm': [10, 12, 15, 20],
        'best_for': ['ethnic kurtas', 'gujarati chaniya choli', 'dupattas'],
        'avoid_for': ['stretchy fabrics', 'sheer nets'],
        'coverage_estimate': {
            'high_density_full_panel': {
                'units_needed': 400,
                'labor_hours': 35
            }
        },
        'sourcing': {
            'bengaluru': {
                'markets': ['Chickpet', 'Malleswaram'],
                'ask_for': 'abla, mirror work pieces',
                'price_inr': {'min': 100, 'max': 300},
                'quality_check': 'smooth edges, genuine glass vs plastic'
            }
        }
    },
    {
        'embellishment_id': 'bugle_beads_silver',
        'type': 'silver tube beads',
        'technique': 'hand-sewn',
        'sizes_mm': [6, 7, 8, 9],
        'best_for': ['evening wear', 'fringe details', 'gowns'],
        'avoid_for': ['thick denims'],
        'coverage_estimate': {
            'high_density_full_panel': {
                'units_needed': 3000,
                'labor_hours': 25
            }
        },
        'sourcing': {
            'bengaluru': {
                'markets': ['Commercial Street', 'Chickpet'],
                'ask_for': 'cut dana, tube beads',
                'price_inr': {'min': 200, 'max': 450},
                'quality_check': 'uniform size, color fastness'
            }
        }
    }
]

def seed_fabrics():
    print("Seeding Fabrics...")
    db = _get_db()
    collection = db['fabric_intelligence']
    stats = run_verification_pipeline('fabric', FABRIC_CANDIDATES)
    
    operations = []
    for record in stats['accepted_records']:
        operations.append(UpdateOne({'fabric_id': record['fabric_id']}, {'$set': record}, upsert=True))
    
    if operations:
        collection.bulk_write(operations)
        
    return {'seeded': len(operations), 'flagged': stats['flagged'], 'rejected': stats['rejected']}

def seed_constructions():
    print("Seeding Constructions...")
    db = _get_db()
    collection = db['garment_construction']
    stats = run_verification_pipeline('construction', CONSTRUCTION_CANDIDATES)
    
    operations = []
    for record in stats['accepted_records']:
        operations.append(UpdateOne({'garment_id': record['garment_id']}, {'$set': record}, upsert=True))
    
    if operations:
        collection.bulk_write(operations)
        
    return {'seeded': len(operations), 'flagged': stats['flagged'], 'rejected': stats['rejected']}

def seed_measurements():
    print("Seeding Measurements...")
    db = _get_db()
    collection = db['body_measurement_tables']
    operations = []
    for record in MEASUREMENT_TABLES:
        # unique key can be height, gender, garment
        query = {'height_cm': record['height_cm'], 'gender': record['gender'], 'garment': record['garment']}
        operations.append(UpdateOne(query, {'$set': record}, upsert=True))
    
    if operations:
        collection.bulk_write(operations)
        
    return {'seeded': len(operations)}

def seed_embellishments():
    print("Seeding Embellishments...")
    db = _get_db()
    collection = db['embellishment_sourcing']
    operations = []
    for record in EMBELLISHMENT_CANDIDATES:
        record['verified'] = False
        operations.append(UpdateOne({'embellishment_id': record['embellishment_id']}, {'$set': record}, upsert=True))
    
    if operations:
        collection.bulk_write(operations)
        
    return {'seeded': len(operations)}

def seed_all():
    fab_stats = seed_fabrics()
    con_stats = seed_constructions()
    mea_stats = seed_measurements()
    emb_stats = seed_embellishments()
    
    return {
        'fabrics': fab_stats,
        'constructions': con_stats,
        'measurements': mea_stats,
        'embellishments': emb_stats
    }
