// ─── Request ──────────────────────────────────────────────────────────────────
export interface TailorRequestPayload {
  user_id: string;
  project_id: string;
  image_b64?: string;
  user_message: string;
  product_url?: string | null;
}

// ─── API Response shape — matches FastAPI tailor_router.py exactly ─────────────
export interface ApiFabricSpec {
  fabric?: string;         // "PV Suiting"
  gsm?: string;            // "280 GSM"
  weave?: string;          // "Twill 2/2"
  hand_feel?: string;
  meters_needed?: number;
  color?: string;
}

export interface ApiSourcingInfo {
  fabric_market?: string;
  fabric_ask_for?: string;
  fabric_price_range?: string;
  embellishment_market?: string;
  embellishment_ask_for?: string;
  embellishment_price_range?: string;
  total_cost_estimate?: string;
}

export interface ApiMeasurementsCm {
  inseam?: number;
  outseam?: number;
  rise_front?: number;
  rise_back?: number;
  thigh_circumference?: number;
  knee_circumference?: number;
  leg_opening?: number;
  [key: string]: number | undefined;
}

export interface ApiMeasurements {
  height_ft?: number;
  height_cm?: number;
  gender?: string;
  measurements_cm?: ApiMeasurementsCm;
  fabric_meters_needed?: number;
}

export interface ApiEmbellishmentBrief {
  type?: string;
  placement?: string;
  technique?: string;
  time_estimate?: string;
}

// ─── The full brief returned by the backend ────────────────────────────────────
export interface TailorBrief {
  // Garment
  garment_name?: string;
  reference_description?: string;
  modification_summary?: string;

  // Fabric
  fabric_spec?: ApiFabricSpec;

  // Construction
  construction_sequence?: string[];   // ["Step 1: ...", "Step 2: ..."]
  critical_points?: string[];
  pressing_sequence?: string[];

  // Additional specs (flat on brief)
  grain_direction?: string;
  interfacing_spec?: string;
  lining_spec?: string;
  embellishment_timing?: string;
  fabric_prep?: string;
  pressing_temperature?: string;
  estimated_construction_time?: string;

  // Embellishment
  embellishment_brief?: ApiEmbellishmentBrief;

  // Sourcing
  sourcing?: ApiSourcingInfo;

  // Measurements
  measurements?: ApiMeasurements;

  // Quality
  quality_checkpoints?: string[];     // array of plain strings

  // Meta
  llm_generated?: boolean;
  tailor_instructions?: string;
  shaaru_notes?: string;            // used as opening_message

  // Kept for legacy / demo compatibility
  opening_message?: string;
}

// ─── API envelope ─────────────────────────────────────────────────────────────
export interface TailorAPIResponse {
  brief: TailorBrief;
  message: string;
}

// ─── Demo brief ───────────────────────────────────────────────────────────────
export const DEMO_BRIEF: TailorBrief = {
  garment_name: "Double-Breasted Peak Lapel Suit",
  reference_description:
    "A structured double-breasted suit with peak lapels, wide shoulders, and tapered trousers. Inspired by Italian suiting tradition.",
  modification_summary:
    "Slim tapered trouser leg, extended jacket length to mid-hip, pick-stitched lapels.",
  shaaru_notes:
    "Okay so I've gone through your reference and honestly this is a clean pick — the silhouette reads very strongly and I can already see exactly how this comes together. Here's your full brief:",
  fabric_spec: {
    fabric: "PV Suiting (Polyester-Viscose Blend)",
    gsm: "280 GSM",
    weave: "Twill 2/2",
    hand_feel: "Smooth, medium drape, slight sheen",
    meters_needed: 3.5,
    color: "Charcoal Grey",
  },
  construction_sequence: [
    "Pre-shrink and steam press fabric before cutting.",
    "Interface jacket front, collar, and cuffs with medium-weight woven interfacing.",
    "Stitch shoulder seams first; set sleeves with ease.",
    "Construct peak lapels: pad stitch by hand for structure and roll.",
    "Set double-breasted front overlap; mark button positions.",
    "Construct trousers: set crease line before assembling, press sharply.",
    "Attach lining. Slip stitch sleeve lining at cuff.",
    "Final pressing: use tailor's ham for chest and lapels.",
  ],
  critical_points: [
    "Pad stitching on lapels is non-negotiable for structure — do not skip.",
    "Press each seam before crossing another seam.",
    "Match grain on all pattern pieces before cutting.",
  ],
  pressing_sequence: [
    "Press shoulder seams over a tailor's ham.",
    "Press side seams open on a pressing board.",
    "Final press lapels with a damp cloth — never iron directly.",
  ],
  grain_direction: "Straight grain on jacket front and back",
  pressing_temperature: "150–160°C (wool/poly setting), damp pressing cloth",
  fabric_prep: "Steam press, no washing. Hang overnight before cutting.",
  interfacing_spec: "Medium woven fusible (Vilene G700 or equivalent)",
  lining_spec: "Acetate or bemberg lining — cut on the bias for ease",
  embellishment_timing: "Add pick stitching after all structural seams are complete.",
  estimated_construction_time: "14–18 hours",
  embellishment_brief: {
    type: "Pick stitching",
    placement: "Lapel edges and pocket flaps",
    technique: "Hand pick stitch in contrast or matching thread",
    time_estimate: "2–3 hours",
  },
  sourcing: {
    fabric_market: "Chickpet, Bengaluru",
    fabric_ask_for: '"PV suiting 280 GSM, twill weave, charcoal"',
    fabric_price_range: "₹180–250 per metre",
    embellishment_market: "S.P. Road, Bengaluru",
    embellishment_ask_for: '"Pick stitch thread, merino wool"',
    embellishment_price_range: "₹80–150",
    total_cost_estimate: "₹700–950 for fabric + trims",
  },
  measurements: {
    height_ft: 5.9,
    height_cm: 175,
    gender: "Male",
    measurements_cm: {
      inseam: 81,
      outseam: 107,
      rise_front: 28,
      rise_back: 34,
      thigh_circumference: 58,
      knee_circumference: 42,
      leg_opening: 38,
    },
    fabric_meters_needed: 3.5,
  },
  quality_checkpoints: [
    "Lapel roll falls naturally without pressing flat",
    "Double-breasted overlap is even — both button rows align",
    "Trouser crease is sharp from hip to hem, does not twist",
  ],
  llm_generated: false,
};

// ─── Fetch wrapper ────────────────────────────────────────────────────────────
export async function submitTailorRequest(
  payload: TailorRequestPayload
): Promise<TailorAPIResponse> {
  const response = await fetch("/api/tailor/reference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    credentials: "include",
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}
