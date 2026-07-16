"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { X, Loader2, Mic, Zap } from "lucide-react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { useVoiceInput } from "@/hooks/useVoiceInput";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type BBox = { x: number; y: number; w: number; h: number };

export type ScannedItem = {
  id: string;
  label: string;
  description: string;
  category: string;
  color: string;
  aesthetic: string;
  bbox: BBox;
  confidence: number;
  pixel_boxes?: { id: string; xyxy: number[] }[];
  track_id?: string;
  state?: "new" | "confirmed" | "coasting";
  fabric_type?: string;
  corrected?: boolean;
  pillBounds?: { x: number; y: number; w: number; h: number };
};

export type TrackedBox = ScannedItem & {
  track_id: string;
  state: "new" | "confirmed" | "coasting";
  currentBbox: BBox;
  targetBbox: BBox;
  opacity: number;
  targetOpacity: number;
};

const TAXONOMY_CATEGORIES = ["top", "bottom", "outerwear", "footwear", "accessory", "dress", "set"];
const TAXONOMY_FABRICS = [
  "poplin", "denim", "linen", "canvas", "corduroy", "twill", "chambray",
  "khadi cotton", "handloom cotton",
  "ribbed knit", "jersey knit", "cable knit", "waffle knit",
  "genuine leather", "faux/PU leather", "suede",
  "fine wool", "wool-blend", "tweed", "cashmere",
  "silk satin", "crepe de chine", "charmeuse", "georgette",
  "organza", "chanderi silk", "chiffon", "mulmul",
  "raw silk dupion", "banarasi brocade", "jacquard", "ikat",
  "velvet", "chikankari", "zardozi"
];

const TAXONOMY_CONSTRUCTIONS = [
  "shirt_blouse", "top_t_shirt_sweatshirt", "sweater", "cardigan", "kurta",
  "pants", "shorts", "skirt",
  "jacket", "vest", "coat", "cape",
  "dress", "jumpsuit", "saree", "anarkali_dress",
  "co_ord_set", "lehenga_set", "salwar_kameez_set", "sharara_set",
  "shoe", "bag_wallet", "belt", "scarf", "dupatta", "glasses", "hat"
];

const CONSTRUCTION_TO_CATEGORY: Record<string, string> = {
  shirt_blouse: "top", top_t_shirt_sweatshirt: "top", sweater: "top", cardigan: "top", kurta: "top",
  pants: "bottom", shorts: "bottom", skirt: "bottom",
  jacket: "outerwear", vest: "outerwear", coat: "outerwear", cape: "outerwear",
  dress: "dress", jumpsuit: "dress", saree: "dress", anarkali_dress: "dress",
  co_ord_set: "set", lehenga_set: "set", salwar_kameez_set: "set", sharara_set: "set",
  shoe: "footwear", bag_wallet: "accessory", belt: "accessory", scarf: "accessory", dupatta: "accessory", glasses: "accessory", hat: "accessory"
};

type MissingPiece = { role: string; find: string };

export type StyleCombo = {
  id: string;
  name: string;
  vibe: string;
  items_used: string[];
  directions: string;
  missing: MissingPiece[];
  reference_images?: string[];
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  top:       "#39FF14",
  bottom:    "#E040FB",
  outerwear: "#FF6D00",
  footwear:  "#00E5FF",
  accessory: "#FFD700",
  dress:     "#FF4081",
  set:       "#FF4081",
  default:   "#A855F7",
};

const _COLOR_HEX: Record<string, string> = {
  black:"#111",white:"#f5f5f0",red:"#e53935",blue:"#1e88e5",
  navy:"#1a237e",green:"#43a047",yellow:"#fdd835",orange:"#fb8c00",
  pink:"#e91e63",purple:"#8e24aa",grey:"#757575",gray:"#757575",
  brown:"#6d4c41",beige:"#d7ccc8",cream:"#fffde7",indigo:"#3949ab",
  teal:"#00897b",ivory:"#fffff0",olive:"#827717",khaki:"#c0a060",
  maroon:"#880e4f",coral:"#ff7043",mint:"#a5d6a7",lavender:"#ce93d8",
};

function catColor(category: string): string {
  return CATEGORY_COLORS[(category ?? "").toLowerCase()] ?? CATEGORY_COLORS.default;
}

function toHex(colorName: string): string {
  if (!colorName) return "#888";
  if (colorName.startsWith("#")) return colorName;
  return _COLOR_HEX[colorName.toLowerCase().split(/[\s-]/)[0]] ?? "#888";
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas drawing helpers
// ─────────────────────────────────────────────────────────────────────────────

interface PlacedLabel {
  x: number;
  y: number;
  w: number;
  h: number;
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  trackedMap: Map<string, TrackedBox>,
  canvasW: number,
  canvasH: number
) {
  ctx.clearRect(0, 0, canvasW, canvasH);

  // Scanline vignette — subtle
  ctx.fillStyle = "rgba(0,0,0,0.025)";
  for (let i = 0; i < canvasH; i += 4) ctx.fillRect(0, i, canvasW, 1);

  const placedLabels: PlacedLabel[] = [];

  trackedMap.forEach((item) => {
    const bbox = item.currentBbox || item.bbox;
    if (!bbox) return;

    const x  = bbox.x * canvasW;
    const y  = bbox.y * canvasH;
    const bw = bbox.w * canvasW;
    const bh = bbox.h * canvasH;

    if (bw < 16 || bh < 16) return;

    ctx.save();
    ctx.globalAlpha = Math.max(0, Math.min(1, item.opacity ?? 1.0));

    const color = catColor(item.category);
    const bracketSize = Math.min(bw, bh) * 0.2;

    // ── Ghost fill ──────────────────────────────────────────────────────────
    ctx.fillStyle = `${color}08`;
    ctx.fillRect(x, y, bw, bh);

    // ── Ghost border ────────────────────────────────────────────────────────
    ctx.strokeStyle = `${color}33`;
    ctx.lineWidth = 1;
    ctx.shadowBlur = 0;
    ctx.strokeRect(x, y, bw, bh);

    // ── L-bracket corners ───────────────────────────────────────────────────
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.lineCap = "square";
    ctx.shadowColor = color;
    ctx.shadowBlur = 7;

    const corners: [number, number][][] = [
      [[x, y + bracketSize], [x, y], [x + bracketSize, y]],
      [[x + bw - bracketSize, y], [x + bw, y], [x + bw, y + bracketSize]],
      [[x, y + bh - bracketSize], [x, y + bh], [x + bracketSize, y + bh]],
      [[x + bw - bracketSize, y + bh], [x + bw, y + bh], [x + bw, y + bh - bracketSize]],
    ];

    corners.forEach(([[ax, ay], [bx2, by2], [cx2, cy2]]) => {
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx2, by2);
      ctx.lineTo(cx2, cy2);
      ctx.stroke();
    });
    ctx.shadowBlur = 0;

    // ── Confidence % — top-right of box ─────────────────────────────────────
    const conf = Math.round((item.confidence ?? 0) * 100);
    ctx.font = '10px "Courier New", monospace';
    ctx.fillStyle = color;
    ctx.shadowBlur = 0;
    ctx.fillText(`${conf}%`, x + bw - 30, y - 4);

    // ── Label pill ──────────────────────────────────────────────────────────
    const fabricText = (!item.fabric_type || item.fabric_type === "pending")
      ? "detecting fabric..."
      : item.fabric_type;
    const rawLabel = `${item.corrected ? "✓ " : ""}${(item.label ?? "item").toLowerCase().trim()} • ${fabricText}`;
    ctx.font = 'bold 13px "Courier New", monospace';
    const maxTextW = canvasW * 0.40;
    const padX = 8;
    const padY = 4;

    const words = rawLabel.split(/\s+/);
    const lines: string[] = [];
    let currentLine = "";

    for (const word of words) {
      let w = word;
      if (ctx.measureText(w).width > maxTextW) {
        while (w.length > 1 && ctx.measureText(w + "…").width > maxTextW) {
          w = w.slice(0, -1);
        }
        w = w + "…";
      }

      const testLine = currentLine ? `${currentLine} ${w}` : w;
      if (ctx.measureText(testLine).width <= maxTextW) {
        currentLine = testLine;
      } else {
        if (currentLine) lines.push(currentLine);
        currentLine = w;
        if (lines.length === 2) {
          let second = lines.pop()!;
          while (second.length > 1 && ctx.measureText(second + "…").width > maxTextW) {
            second = second.slice(0, -1);
          }
          lines.push(second + "…");
          currentLine = "";
          break;
        }
      }
    }
    if (currentLine && lines.length < 2) {
      lines.push(currentLine);
    }
    if (lines.length === 0) lines.push("item");

    const isTwoLines = lines.length > 1;
    const maxLineW = Math.max(...lines.map((l) => ctx.measureText(l).width), 20);
    const labelW = maxLineW + padX * 2 + 14; // 14 for dot
    const labelH = isTwoLines ? 38 : 22;

    // Position: below box if box centre is in top half, else above
    const boxCentreY = y + bh / 2;
    let labelX = x;
    let labelY = boxCentreY < canvasH / 2
      ? y + bh + 6        // below
      : y - labelH - 6;   // above

    // Clamp to canvas edges
    labelX = Math.max(2, Math.min(labelX, canvasW - labelW - 2));
    labelY = Math.max(2, Math.min(labelY, canvasH - labelH - 2));

    // Overlap resolution — nudge lower-priority labels
    let attempts = 0;
    while (attempts < 6) {
      const pl = placedLabels.find(
        (pl) =>
          labelX < pl.x + pl.w &&
          labelX + labelW > pl.x &&
          labelY < pl.y + pl.h &&
          labelY + labelH > pl.y
      );
      if (!pl) break;

      const spaceBelow = canvasH - (pl.y + pl.h + 6 + labelH);
      const spaceAbove = pl.y - 6 - labelH;
      const maxVertSpace = Math.max(spaceBelow, spaceAbove);

      const spaceRight = canvasW - (pl.x + pl.w + 6 + labelW);
      const spaceLeft  = pl.x - 6 - labelW;
      const maxHorizSpace = Math.max(spaceRight, spaceLeft);

      if (maxHorizSpace > maxVertSpace && maxHorizSpace >= 0) {
        if (spaceRight >= spaceLeft) {
          labelX = pl.x + pl.w + 6;
        } else {
          labelX = pl.x - labelW - 6;
        }
      } else if (maxVertSpace >= 0) {
        if (spaceBelow >= spaceAbove) {
          labelY = pl.y + pl.h + 6;
        } else {
          labelY = pl.y - labelH - 6;
        }
      } else {
        labelY += labelH + 6;
      }

      // Re-clamp after nudge
      labelX = Math.max(2, Math.min(labelX, canvasW - labelW - 2));
      labelY = Math.max(2, Math.min(labelY, canvasH - labelH - 2));
      attempts++;
    }

    placedLabels.push({ x: labelX, y: labelY, w: labelW, h: labelH });
    item.pillBounds = { x: labelX, y: labelY, w: labelW, h: labelH };

    // Pill background
    ctx.fillStyle = "rgba(0,0,0,0.80)";
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(labelX, labelY, labelW, labelH, 5);
    } else {
      ctx.rect(labelX, labelY, labelW, labelH);
    }
    ctx.fill();

    // Pill border
    ctx.strokeStyle = item.corrected ? "#00E5FF" : `${color}55`;
    ctx.lineWidth = item.corrected ? 1.5 : 0.8;
    ctx.stroke();

    // Color dot
    ctx.beginPath();
    ctx.arc(labelX + 9, labelY + labelH / 2, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Label text
    ctx.fillStyle = "#ffffff";
    ctx.font = 'bold 13px "Courier New", monospace';
    if (isTwoLines) {
      ctx.fillText(lines[0] || "", labelX + 18, labelY + 16);
      ctx.fillText(lines[1] || "", labelX + 18, labelY + 31);
    } else {
      ctx.fillText(lines[0] || "", labelX + 18, labelY + labelH - padY - 1);
    }

    ctx.restore();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// HUD top/bottom bars — drawn on overlay canvas
// ─────────────────────────────────────────────────────────────────────────────

function drawHUD(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  isScanning: boolean
) {
  // Top bar
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(0, 0, w, 24);

  // Timestamp
  ctx.fillStyle = "#39FF14";
  ctx.font = 'bold 8px "Courier New", monospace';
  ctx.fillText(new Date().toTimeString().substring(0, 8), 8, 15);

  // LIVE indicator — pulse ring when scanning
  const dotX = w - 46;
  const dotY = 12;
  if (isScanning) {
    ctx.beginPath();
    ctx.arc(dotX, dotY, 7, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,34,34,0.25)";
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
  ctx.fillStyle = isScanning ? "#ff4444" : "#ff2222";
  ctx.fill();
  ctx.fillStyle = "#ff2222";
  ctx.font = 'bold 8px "Courier New", monospace';
  ctx.fillText("LIVE", dotX + 8, 15);

  // Corner bracket HUD marks
  const S = 14;
  ctx.strokeStyle = "rgba(57,255,20,0.28)";
  ctx.lineWidth = 1.2;
  const hudCorners: number[][][] = [
    [[0, S + 24], [0, 24], [S, 24]],
    [[w - S, 24], [w, 24], [w, S + 24]],
    [[0, h - 2 - S], [0, h - 2], [S, h - 2]],
    [[w - S, h - 2], [w, h - 2], [w, h - 2 - S]],
  ];
  hudCorners.forEach(([[ax, ay], [bx, by], [cx, cy]]) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.stroke();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Voice intent parsing — occluded item detection gate
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Two-tier intent gate before firing /api/cv/targeted-scan.
 *
 * Tier 1 (REQUIRED): The transcript must contain at least one phrase that
 * signals the user is actively pointing something out — not just speaking
 * about fashion in general.
 *
 * Examples that PASS Tier 1:
 *   "can you see the jacket behind that top"
 *   "there's a beige trouser hiding behind the rack"
 *   "what's that under the pile"
 *   "show me that thing folded on the shelf"
 *   "that kurta inside the bag"
 *
 * Examples that FAIL Tier 1 (no scan triggered):
 *   "I love how denim looks in summer"
 *   "what fabric is this shirt"
 *   "tell me about this look"
 *
 * Tier 2: Extract garment noun + optional color from the transcript.
 * Only runs if Tier 1 passed.
 */
function parseVoiceForOccludedIntent(
  transcript: string,
  currentLabels: string[],
): { shouldScan: boolean; targetDescription: string | null } {

  const t = transcript.toLowerCase().trim();

  // ── Tier 1: Pointing-intent phrase patterns ──────────────────────────────
  // These patterns signal the user is spatially referencing something in the
  // camera frame — occluded, stacked, or only partially visible.
  const INTENT_PATTERNS = [
    /can you see\b/,
    /do you see\b/,
    /there['']?s a\b/,
    /there is a\b/,
    /i see a\b/,
    /i can see\b/,
    /what['']?s that\b/,
    /what is that\b/,
    /that .{2,30} (behind|under|inside|beneath|below|next to|beside|on top of)/,
    /\b(behind|under|beneath|below|inside|folded|stacked|hidden|tucked)\b.{0,40}\b(jacket|shirt|top|jeans|trouser|pant|skirt|dress|kurta|shawl|coat|blazer|hoodie|sweater|jumper|cardigan|tee|blouse|lehenga|saree|sari|dupatta|vest|shorts|cargo|chino|suit|boot|sneaker|slipper|sandal|bag|belt|scarf)\b/,
    /\b(show|spot|point|find)\b.{0,20}\b(item|garment|piece|thing|that|it)\b/,
  ];

  const hasIntent = INTENT_PATTERNS.some((re) => re.test(t));
  if (!hasIntent) {
    return { shouldScan: false, targetDescription: null };
  }

  // ── Tier 2: Garment noun + color extraction ──────────────────────────────
  const GARMENT_NOUNS = [
    "jacket", "blazer", "coat", "outerwear", "hoodie", "sweater", "jumper", "cardigan",
    "shirt", "tee", "t-shirt", "blouse", "top", "kurta", "kurti",
    "jeans", "denim", "trousers", "pants", "pant", "chinos", "cargo", "shorts", "skirt",
    "dress", "lehenga", "sari", "saree", "dupatta", "shawl",
    "sneakers", "boots", "sandals", "slippers", "shoes", "footwear",
    "bag", "handbag", "clutch", "tote", "backpack",
    "belt", "scarf", "cap", "hat", "sunglasses",
  ];

  const COLOR_WORDS = [
    "black", "white", "grey", "gray", "navy", "blue", "red", "green", "yellow",
    "orange", "pink", "purple", "brown", "beige", "cream", "olive", "khaki",
    "maroon", "burgundy", "mustard", "lavender", "coral", "teal", "rust",
    "charcoal", "indigo", "camel", "tan", "sand", "off-white", "ivory",
  ];

  // Find first garment noun in transcript
  const foundNoun = GARMENT_NOUNS.find((noun) => t.includes(noun));
  if (!foundNoun) {
    return { shouldScan: false, targetDescription: null };
  }

  // Skip if this garment type is already tracked (avoid redundant scans)
  const alreadyTracked = currentLabels.some(
    (label) => label.toLowerCase().includes(foundNoun)
  );
  if (alreadyTracked) {
    return { shouldScan: false, targetDescription: null };
  }

  // Extract color if present near the noun
  const nounIdx = t.indexOf(foundNoun);
  const contextWindow = t.slice(Math.max(0, nounIdx - 25), nounIdx + foundNoun.length + 15);
  const foundColor = COLOR_WORDS.find((c) => contextWindow.includes(c));

  // Build description string for the targeted scan prompt
  const targetDescription = foundColor
    ? `${foundColor} ${foundNoun}`
    : foundNoun;

  return { shouldScan: true, targetDescription };
}

// ─────────────────────────────────────────────────────────────────────────────
// Props
// ─────────────────────────────────────────────────────────────────────────────


interface CameraScannerProps {
  onClose: () => void;
  onItemSelected: (item: ScannedItem) => void;
  userId?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onAnalysisComplete: (result: any) => void;
  onItemTouched?: (data: {
    label: string;
    comment: string;
    bbox: BBox;
    color: string;
  }) => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export function CameraScanner({
  onClose,
  onItemSelected,
  userId,
  onAnalysisComplete,
  onItemTouched,
}: CameraScannerProps) {
  // ── Camera state ──────────────────────────────────────────────────────────
  const [isStreaming, setIsStreaming]     = useState(false);
  const [cameraError, setCameraError]    = useState<string | null>(null);

  // ── Scan state ────────────────────────────────────────────────────────────
  const [, setTrackedVersion]                 = useState(0);
  const [isLiveScanning, setIsLiveScanning]   = useState(false);

  // ── UI state ──────────────────────────────────────────────────────────────
  const [selectedItem, setSelectedItem]   = useState<ScannedItem | null>(null);
  const [showItemSheet, setShowItemSheet] = useState(false);
  const [showCombosSheet, setShowCombosSheet] = useState(false);
  const [showCorrectionSheet, setShowCorrectionSheet] = useState(false);
  const [correctingItem, setCorrectingItem] = useState<TrackedBox | null>(null);
  const [editLabel, setEditLabel]         = useState("");
  const [editCategory, setEditCategory]   = useState("top");
  const [editFabric, setEditFabric]       = useState("poplin");
  const [combos, setCombos]               = useState<StyleCombo[]>([]);
  const [loadingCombos, setLoadingCombos] = useState(false);
  const [analyzing, setAnalyzing]         = useState(false);

  // ── Voice state ───────────────────────────────────────────────────────────
  const [userTranscript, setUserTranscript]           = useState<string | null>(null);
  const [voiceReply, setVoiceReply]                   = useState<string | null>(null);
  const [isFadingOut, setIsFadingOut]                 = useState(false);

  const {
    isRecording: isRecordingVoice,
    isTranscribing: isTranscribingVoice,
    toggleRecording: toggleVoice,
  } = useVoiceInput({
    onResult: (res) => {
      if (res.transcribed_text) setUserTranscript(res.transcribed_text);
      if (res.reply) setVoiceReply(res.reply);

      // ── Voice-triggered occluded item detection ──────────────────────
      // Only fire a targeted re-scan when the transcript shows clear
      // POINTING INTENT — not just any sentence with a garment word.
      if (res.transcribed_text && lastCapturedB64.current) {
        const intent = parseVoiceForOccludedIntent(
          res.transcribed_text,
          Array.from(trackedItemsRef.current.values()).map((i) => i.label),
        );
        if (intent.shouldScan && intent.targetDescription) {
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
          const currentLabels = Array.from(trackedItemsRef.current.values()).map((i) => i.label);
          fetch(`${apiUrl}/api/cv/targeted-scan`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_b64: lastCapturedB64.current,
              target_description: intent.targetDescription,
              current_labels: currentLabels,
              user_id: userId || "default",
            }),
          })
            .then((r) => r.json())
            .then((data) => {
              if (data.found && data.item && data.item.bbox) {
                const item = data.item;
                const tid = item.track_id || `voice_${Date.now()}`;
                const bboxCopy = { ...item.bbox };
                trackedItemsRef.current.set(tid, {
                  ...item,
                  id: tid,
                  track_id: tid,
                  state: "new" as const,
                  currentBbox: bboxCopy,
                  targetBbox: { ...item.bbox },
                  bbox: bboxCopy,
                  opacity: 0.0,
                  targetOpacity: 1.0,
                  corrected: false,
                });
                setTrackedVersion((v) => v + 1);
                console.log("[Voice Targeted] Injected item:", item.label);
              }
            })
            .catch((e) => console.warn("[Voice Targeted] Failed:", e));
        }
      }
    },
    onError: (err) => {
      console.error("[CameraScanner] voice error:", err);
    }
  });

  // ── Refs ──────────────────────────────────────────────────────────────────
  const dismissTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isRecordingRef    = useRef<boolean>(false);
  const videoRef          = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef  = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef   = useRef<HTMLCanvasElement>(null);
  const cameraStreamRef   = useRef<MediaStream | null>(null);
  const lastCapturedB64   = useRef<string | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const dwellTimerRef     = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dwellItemRef      = useRef<string | null>(null);
  const commentedItemsRef = useRef<Map<string, number>>(new Map());
  const isSpeakingRef     = useRef<boolean>(false);
  const rafRef            = useRef<number>(0);

  // ── Persistent tracking map keyed by track_id ──
  const trackedItemsRef = useRef<Map<string, TrackedBox>>(new Map());
  const enrichingSetRef = useRef<Set<string>>(new Set());

  // ─────────────────────────────────────────────────────────────────────────
  // 1 — Camera init
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    let mounted = true;

    const initCamera = async () => {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width:  { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }

      if (!mounted) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      cameraStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try { await videoRef.current.play(); } catch { /* ignore */ }
        setIsStreaming(true);
      }
    };

    initCamera().catch((err) => {
      console.error("[Camera] init error:", err);
      if (mounted) setCameraError("Camera access denied");
    });

    return () => {
      mounted = false;
      cameraStreamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // 2 — MediaPipe hand tracking init
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const init = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const hl = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.6,
          minHandPresenceConfidence: 0.6,
          minTrackingConfidence: 0.5,
        });
        handLandmarkerRef.current = hl;
        console.log("[MediaPipe] Hand tracking ready");
      } catch (e) {
        console.warn("[MediaPipe] Init failed:", e, "— continuing without hand tracking");
      }
    };
    init();
    return () => {
      if (dwellTimerRef.current) clearTimeout(dwellTimerRef.current);
    };
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // 3 — Hand tracking helper
  // ─────────────────────────────────────────────────────────────────────────

  const processHandTracking = useCallback(
    (videoEl: HTMLVideoElement, timestamp: number, frameB64: string, items: ScannedItem[]) => {
      if (!handLandmarkerRef.current || !items.length) return;
      try {
        const results = handLandmarkerRef.current.detectForVideo(videoEl, timestamp);
        if (!results?.landmarks?.length) {
          if (dwellTimerRef.current) {
            clearTimeout(dwellTimerRef.current);
            dwellTimerRef.current = null;
            dwellItemRef.current  = null;
          }
          return;
        }

        const tip = results.landmarks[0][8]; // index fingertip
        const fx  = tip.x;
        const fy  = tip.y;

        let touched: ScannedItem | null = null;
        for (const item of items) {
          const b = item.bbox;
          if (!b) continue;
          if (fx >= b.x && fx <= b.x + b.w && fy >= b.y && fy <= b.y + b.h) {
            touched = item;
            break;
          }
        }

        if (!touched) {
          if (dwellTimerRef.current) {
            clearTimeout(dwellTimerRef.current);
            dwellTimerRef.current = null;
            dwellItemRef.current  = null;
          }
          return;
        }

        const lbl       = touched.label;
        const lastTime  = commentedItemsRef.current.get(lbl) ?? 0;
        if (Date.now() - lastTime < 60000) return;
        if (dwellItemRef.current === lbl) return;

        if (dwellTimerRef.current) clearTimeout(dwellTimerRef.current);
        dwellItemRef.current = lbl;

        dwellTimerRef.current = setTimeout(async () => {
          if (isSpeakingRef.current) return;
          try {
            isSpeakingRef.current = true;
            commentedItemsRef.current.set(lbl, Date.now());

            const resp = await fetch("/api/cv/touch", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                item_label:    lbl,
                item_color:    touched!.color ?? "",
                item_category: touched!.category ?? "",
                item_aesthetic:touched!.aesthetic ?? "",
                all_items: items.map((i) => i.label),
                user_id:   userId ?? "default",
              }),
            });

            const data = await resp.json();
            if (data.comment) {
              onItemTouched?.({
                label:   lbl,
                comment: data.comment,
                bbox:    touched!.bbox,
                color:   touched!.color,
              });

              const ttsResp = await fetch("/api/voice/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: data.comment, voice: "nova" }),
              });
              const ttsData = await ttsResp.json();
              if (ttsData.audio_b64) {
                const audio = new Audio("data:audio/mp3;base64," + ttsData.audio_b64);
                audio.onended = () => { isSpeakingRef.current = false; };
                audio.play();
              } else {
                isSpeakingRef.current = false;
              }
            } else {
              isSpeakingRef.current = false;
            }
          } catch (e) {
            console.warn("[Touch] Failed:", e);
            isSpeakingRef.current = false;
          }
          dwellTimerRef.current = null;
          dwellItemRef.current  = null;
        }, 1500);
      } catch (e) {
        console.warn("[MediaPipe] Frame error:", e);
      }
    },
    [userId, onItemTouched]
  );

  // ─────────────────────────────────────────────────────────────────────────
  // 4 — Auto-scan loop (every 4 s)
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!isStreaming) return;

    let scanInterval: ReturnType<typeof setInterval>;
    let isScanning = false;

    const runScan = async () => {
      if (isScanning) return;
      if (!videoRef.current) return;

      isScanning = true;
      setIsLiveScanning(true);

      try {
        const tmpCanvas = document.createElement("canvas");
        tmpCanvas.width  = 640;
        tmpCanvas.height = 640;
        const tmpCtx = tmpCanvas.getContext("2d");
        if (!tmpCtx || !videoRef.current) { isScanning = false; setIsLiveScanning(false); return; }

        tmpCtx.drawImage(videoRef.current, 0, 0, 640, 640);
        const frameB64 = tmpCanvas.toDataURL("image/jpeg", 0.8).split(",")[1];
        lastCapturedB64.current = frameB64;

        // Hand tracking on the live frame
        const activeItems = Array.from(trackedItemsRef.current.values()).filter((i) => i.targetOpacity > 0);
        processHandTracking(videoRef.current, performance.now(), frameB64, activeItems);

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
        const resp = await fetch(`${apiUrl}/api/cv/scan`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_b64: frameB64,
            user_id:   userId || "default",
            run_combos: false,
          }),
          signal: AbortSignal.timeout(30000),
        });

        const data = await resp.json();

        const incomingItems: ScannedItem[] = data.items || [];
        const CONFIDENCE_FLOOR = 0.55;
        const filtered = incomingItems.filter(
          (i) => (i.confidence ?? 0) >= CONFIDENCE_FLOOR || i.state === "coasting"
        );
        const top4 = [...filtered]
          .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
          .slice(0, 4);

        const incomingTrackIds = new Set<string>();
        top4.forEach((item) => {
          const tid = item.track_id || item.id;
          if (!tid || !item.bbox) return;
          incomingTrackIds.add(tid);

          const existing = trackedItemsRef.current.get(tid);
          if (existing) {
            existing.targetBbox = { ...item.bbox };
            existing.confidence = item.confidence ?? existing.confidence;
            existing.state = item.state || "confirmed";
            if (item.pixel_boxes) existing.pixel_boxes = item.pixel_boxes;

            if (!existing.corrected) {
              existing.label = item.label || existing.label;
              existing.description = item.description || existing.description;
              existing.category = item.category || existing.category;
              existing.color = item.color || existing.color;
              existing.aesthetic = item.aesthetic || existing.aesthetic;
              if (item.fabric_type && item.fabric_type !== "pending") {
                existing.fabric_type = item.fabric_type;
              } else if (!existing.fabric_type) {
                existing.fabric_type = item.fabric_type || "pending";
              }
            }

            existing.targetOpacity = existing.state === "coasting" ? 0.7 : 1.0;
          } else {
            const bboxCopy = { ...item.bbox };
            trackedItemsRef.current.set(tid, {
              ...item,
              id: item.id || tid,
              track_id: tid,
              state: item.state || "new",
              currentBbox: bboxCopy,
              targetBbox: { ...item.bbox },
              bbox: bboxCopy,
              opacity: 0.0,
              targetOpacity: item.state === "coasting" ? 0.7 : 1.0,
            });
          }
        });

        trackedItemsRef.current.forEach((tracked, tid) => {
          if (!incomingTrackIds.has(tid)) {
            tracked.targetOpacity = 0.0;
          } else if (
            (tracked.fabric_type === "pending" || !tracked.fabric_type) &&
            !tracked.corrected &&
            tracked.state !== "coasting" &&
            !enrichingSetRef.current.has(tid)
          ) {
            enrichingSetRef.current.add(tid);
            fetch(`${apiUrl}/api/cv/enrich-fabric`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                image_b64: frameB64,
                track_id: tid,
                bbox: tracked.bbox,
                label: tracked.label || "garment",
                category: tracked.category || "top",
                user_id: userId || "default",
              }),
            })
              .then((res) => res.json())
              .then((resData) => {
                const currentTrack = trackedItemsRef.current.get(tid);
                if (currentTrack && !currentTrack.corrected && resData?.fabric_type && resData.fabric_type !== "pending") {
                  currentTrack.fabric_type = resData.fabric_type;
                  setTrackedVersion((v) => v + 1);
                }
              })
              .catch((e) => console.warn("[Enrich Fabric] Failed:", e))
              .finally(() => {
                enrichingSetRef.current.delete(tid);
              });
          }
        });

        setTrackedVersion((v) => v + 1);
      } catch (err) {
        console.warn("[Scan] Frame failed:", err);
        // Keep last good result — no state change needed
      } finally {
        isScanning = false;
        setIsLiveScanning(false);
      }
    };

    runScan(); // immediate first scan
    scanInterval = setInterval(runScan, 4000);

    return () => {
      clearInterval(scanInterval);
    };
  }, [isStreaming, userId, processHandTracking]);

  // ─────────────────────────────────────────────────────────────────────────
  // 5 — Canvas overlay rendering (redraw on item change + rAF loop for HUD)
  // ─────────────────────────────────────────────────────────────────────────

  // Items to actually display/interact with: active tracked items that are not fading out
  const displayItems = Array.from(trackedItemsRef.current.values()).filter(
    (item) => item.targetOpacity > 0
  );

  const redrawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = video.clientWidth;
    const h = video.clientHeight;
    if (w === 0 || h === 0) return;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width  = w;
      canvas.height = h;
    }

    // Draw item overlays from persistent tracking map
    drawOverlay(ctx, trackedItemsRef.current, w, h);
    // Draw HUD on top
    drawHUD(ctx, w, h, isLiveScanning);
  }, [isLiveScanning]);

  // rAF loop for per-frame bbox/opacity interpolation and live HUD
  useEffect(() => {
    let running = true;
    const loop = () => {
      if (!running) return;

      let pruned = false;
      trackedItemsRef.current.forEach((item, tid) => {
        // LERP bbox toward targetBbox
        item.currentBbox.x += (item.targetBbox.x - item.currentBbox.x) * 0.15;
        item.currentBbox.y += (item.targetBbox.y - item.currentBbox.y) * 0.15;
        item.currentBbox.w += (item.targetBbox.w - item.currentBbox.w) * 0.15;
        item.currentBbox.h += (item.targetBbox.h - item.currentBbox.h) * 0.15;

        // Keep item.bbox synced with currentBbox for touch / combos / hand tracking
        item.bbox.x = item.currentBbox.x;
        item.bbox.y = item.currentBbox.y;
        item.bbox.w = item.currentBbox.w;
        item.bbox.h = item.currentBbox.h;

        // LERP opacity toward targetOpacity (faster decay multiplier 0.35 for vanishing items per Phase 1 report)
        const opacityLerp = item.targetOpacity < item.opacity ? 0.35 : 0.15;
        item.opacity += (item.targetOpacity - item.opacity) * opacityLerp;

        // Remove from map once fade-out completes
        if (item.targetOpacity === 0.0 && item.opacity < 0.02) {
          trackedItemsRef.current.delete(tid);
          pruned = true;
        }
      });

      if (pruned) {
        setTrackedVersion((v) => v + 1);
      }

      redrawOverlay();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, [redrawOverlay]);

  // ─────────────────────────────────────────────────────────────────────────
  // 6 — Tap detection on camera view
  // ─────────────────────────────────────────────────────────────────────────

  const handleConfirmCorrection = async () => {
    if (!correctingItem) return;
    const tid = correctingItem.track_id || correctingItem.id;
    const existing = trackedItemsRef.current.get(tid);
    if (!existing) return;

    const orig = {
      label: existing.label,
      category: existing.category,
      fabric_type: existing.fabric_type || "uncertain",
      bbox: existing.bbox || existing.currentBbox,
    };

    const corr = {
      label: editLabel.trim() || existing.label,
      category: editCategory,
      fabric_type: editFabric,
      bbox: existing.bbox || existing.currentBbox,
    };

    existing.label = corr.label;
    existing.category = corr.category;
    existing.fabric_type = corr.fabric_type;
    existing.corrected = true;

    setShowCorrectionSheet(false);
    setCorrectingItem(null);
    setTrackedVersion((v) => v + 1);

    // Attempt to capture crop b64 from videoRef if possible
    let imageCropB64: string | undefined = undefined;
    try {
      const video = videoRef.current;
      const box = existing.bbox || existing.currentBbox;
      if (video && box) {
        const tempCanvas = document.createElement("canvas");
        const ctx = tempCanvas.getContext("2d");
        if (ctx && video.videoWidth > 0 && video.videoHeight > 0) {
          const w = video.videoWidth;
          const h = video.videoHeight;
          const xMin = Math.max(0, Math.floor((box.x_min ?? 0) * w));
          const yMin = Math.max(0, Math.floor((box.y_min ?? 0) * h));
          const xMax = Math.min(w, Math.ceil((box.x_max ?? 1) * w));
          const yMax = Math.min(h, Math.ceil((box.y_max ?? 1) * h));
          const cropW = Math.max(1, xMax - xMin);
          const cropH = Math.max(1, yMax - yMin);
          tempCanvas.width = cropW;
          tempCanvas.height = cropH;
          ctx.drawImage(video, xMin, yMin, cropW, cropH, 0, 0, cropW, cropH);
          imageCropB64 = tempCanvas.toDataURL("image/jpeg", 0.85);
        }
      }
    } catch (cropErr) {
      console.warn("[Correction] Crop capture failed:", cropErr);
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
      await fetch(`${apiUrl}/api/cv/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_id: tid,
          original: orig,
          corrected: corr,
          confidence: existing.confidence ?? 0.0,
          user_id: userId || "default",
          bbox: existing.bbox || existing.currentBbox,
          image_crop_b64: imageCropB64,
        }),
      });
    } catch (err) {
      console.error("[Correction] Log failed:", err);
    }
  };

  const handleCameraTap = (e: React.TouchEvent | React.MouseEvent) => {
    // Don't dismiss sheet if tapping the sheet itself
    if (showCombosSheet) { setShowCombosSheet(false); return; }
    if (showCorrectionSheet) { setShowCorrectionSheet(false); return; }

    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    const rect    = canvas.getBoundingClientRect();
    const clientX = "touches" in e
      ? (e.touches[0]?.clientX ?? (e as React.TouchEvent).changedTouches[0].clientX)
      : (e as React.MouseEvent).clientX;
    const clientY = "touches" in e
      ? (e.touches[0]?.clientY ?? (e as React.TouchEvent).changedTouches[0].clientY)
      : (e as React.MouseEvent).clientY;

    const tapPxX = clientX - rect.left;
    const tapPxY = clientY - rect.top;

    // 1. Check if tap hits any confirmed item's label pill
    const tappedPill = displayItems.find((item) => {
      if (item.state !== "confirmed") return false;
      const pb = item.pillBounds;
      if (!pb) return false;
      return tapPxX >= pb.x && tapPxX <= pb.x + pb.w && tapPxY >= pb.y && tapPxY <= pb.y + pb.h;
    });

    if (tappedPill) {
      setCorrectingItem(tappedPill);
      setEditLabel(tappedPill.label || "");
      setEditCategory(tappedPill.category || "top");
      setEditFabric(tappedPill.fabric_type || "poplin");
      setShowCorrectionSheet(true);
      setShowItemSheet(false);
      setSelectedItem(null);
      return;
    }

    // 2. Otherwise check if tap hits any item's bounding box
    const tapX = tapPxX / rect.width;
    const tapY = tapPxY / rect.height;

    const tapped = displayItems.find((item) => {
      const b = item.bbox;
      if (!b) return false;
      return tapX >= b.x && tapX <= b.x + b.w && tapY >= b.y && tapY <= b.y + b.h;
    });

    if (tapped) {
      setSelectedItem(tapped);
      setShowItemSheet(true);
      setShowCorrectionSheet(false);
    } else {
      setShowItemSheet(false);
      setShowCorrectionSheet(false);
      setSelectedItem(null);
      setCorrectingItem(null);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // 7 — "Style this" — full analyze
  // ─────────────────────────────────────────────────────────────────────────

  const handleStyleThis = async (item: ScannedItem) => {
    if (!lastCapturedB64.current) return;
    setAnalyzing(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
      const response = await fetch(`${apiUrl}/api/cv/analyze`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64:  lastCapturedB64.current,
          item_id:    item.id,
          item_label: item.label,
          user_id:    userId,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Analyze failed");
      onItemSelected(item);
      onAnalysisComplete(result);
      onClose();
    } catch (err: unknown) {
      console.error("[StyleThis] Error:", err);
      setAnalyzing(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // 8 — Combos
  // ─────────────────────────────────────────────────────────────────────────

  const handleGetCombos = async () => {
    if (!displayItems.length) return;
    setLoadingCombos(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
      const res  = await fetch(`${apiUrl}/api/cv/style-combos`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: displayItems, user_id: userId }),
      });
      const data = await res.json();
      setCombos(data.combos || []);
    } catch (err) {
      console.error("[CameraScanner] combos error:", err);
      setCombos([]);
    } finally {
      setLoadingCombos(false);
    }
  };

  const openCombos = () => {
    setShowItemSheet(false);
    setShowCombosSheet(true);
    if (!combos.length && !loadingCombos) handleGetCombos();
  };

  // ─────────────────────────────────────────────────────────────────────────
  // 9 — Voice & Auto-Dismiss Timer
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    isRecordingRef.current = isRecordingVoice;
  }, [isRecordingVoice]);

  useEffect(() => {
    // Clear any existing timer when dependencies change (resets timer on new pairs)
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }

    // If no text overlay active, reset fading state and return
    if (!userTranscript && !voiceReply) {
      setIsFadingOut(false);
      return;
    }

    // Ensure panel is visible when new content arrives
    setIsFadingOut(false);

    // If actively recording or transcribing, pause countdown so it doesn't dismiss mid-question
    if (isRecordingVoice || isTranscribingVoice) {
      return;
    }

    // Start 10-second inactivity countdown
    dismissTimerRef.current = setTimeout(() => {
      // Safety check: if user started recording right as timer fired
      if (isRecordingRef.current) return;
      
      setIsFadingOut(true);
      setTimeout(() => {
        setUserTranscript(null);
        setVoiceReply(null);
        setIsFadingOut(false);
      }, 300);
    }, 10000);

    return () => {
      if (dismissTimerRef.current) {
        clearTimeout(dismissTimerRef.current);
      }
    };
  }, [userTranscript, voiceReply, isRecordingVoice, isTranscribingVoice]);

  const toggleLiveVoice = () => {
    const activeItems = Array.from(trackedItemsRef.current.values())
      .filter((i) => (i.targetOpacity ?? 0) > 0 || (i.opacity ?? 0) > 0)
      .map((i) => `${i.color || ""} ${i.label} (${i.category || "item"}, ${i.aesthetic || "casual"} vibe, fabric: ${i.fabric_type || "unknown"})`.trim());
    
    const scanContextStr = activeItems.length > 0 
      ? activeItems.join("; ") 
      : "No items currently detected in camera view.";

    toggleVoice(scanContextStr, userId || "default");
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-black overflow-hidden">

      {/* Hidden canvas for frame capture */}
      <canvas ref={hiddenCanvasRef} className="hidden" />

      {/* ── Camera error state ───────────────────────────────────────────── */}
      {cameraError && (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
          <p className="text-red-400 mb-4 font-mono text-sm">{cameraError}</p>
          <button
            onClick={onClose}
            className="px-6 py-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors font-mono text-sm"
          >
            Close
          </button>
        </div>
      )}

      {/* ── Live camera view ─────────────────────────────────────────────── */}
      {!cameraError && (
        <div className="relative flex-1 overflow-hidden">

          {/* Video — always live */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onClick={handleCameraTap}
            onTouchEnd={handleCameraTap}
            className="absolute inset-0 w-full h-full object-cover cursor-pointer"
          />

          {/* Canvas overlay — same dimensions, drawn via rAF */}
          <canvas
            ref={overlayCanvasRef}
            onClick={handleCameraTap}
            onTouchEnd={handleCameraTap}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ pointerEvents: "none" }}
          />

          {/* Close button — top right, above canvas */}
          <button
            onClick={onClose}
            className="absolute top-8 right-4 z-20 p-2 bg-black/50 hover:bg-black/70 rounded-full border border-white/15 transition-colors backdrop-blur-sm"
          >
            <X className="w-5 h-5 text-white" />
          </button>

          {/* Voice transcript & reply panel */}
          {(userTranscript || voiceReply) && (
            <div
              className={`absolute top-16 left-4 right-4 z-20 max-w-md mx-auto pointer-events-auto transition-all duration-300 ${
                isFadingOut
                  ? "opacity-0 -translate-y-2"
                  : "opacity-100 translate-y-0 animate-in fade-in slide-in-from-top-2 duration-300"
              }`}
            >
              <div className="bg-black/85 border border-[#A855F7]/40 rounded-xl p-4 shadow-xl backdrop-blur-md relative">
                <button
                  onClick={() => {
                    setUserTranscript(null);
                    setVoiceReply(null);
                    setIsFadingOut(false);
                  }}
                  className="absolute top-2.5 right-2.5 p-1 text-white/50 hover:text-white transition-colors"
                  title="Dismiss"
                >
                  <X className="w-4 h-4" />
                </button>

                {userTranscript && (
                  <div className="mb-3 pr-6">
                    <p className="font-mono text-[10px] text-[#A855F7] tracking-wider mb-0.5 uppercase">YOU</p>
                    <p className="font-sans text-xs text-white/90 leading-relaxed italic">&ldquo;{userTranscript}&rdquo;</p>
                  </div>
                )}

                {voiceReply && (
                  <div>
                    <p className="font-mono text-[10px] text-[#39FF14] tracking-wider mb-0.5 uppercase flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#39FF14] animate-pulse"></span>
                      RILEY
                    </p>
                    <p className="font-sans text-sm text-white font-medium leading-relaxed">{voiceReply}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Bottom UI bar ─────────────────────────────────────────────── */}
          <div className="absolute bottom-0 left-0 right-0 z-20 pb-8">
            {/* Hint text */}
            <p
              className="text-center font-mono text-[10px] tracking-[0.2em] mb-3"
              style={{ color: "rgba(57,255,20,0.55)" }}
            >
              Shaaru can see and hear you
            </p>

            {/* Button row */}
            <div className="flex items-center justify-center gap-6 px-8">

              {/* COMBOS button */}
              <button
                onClick={openCombos}
                disabled={displayItems.length === 0}
                className="flex items-center gap-2 px-5 py-2.5 rounded-full border border-[#A855F7]/50 bg-black/60 hover:bg-[#A855F7]/15 transition-all font-mono text-xs tracking-[0.15em] text-[#A855F7] disabled:opacity-30 backdrop-blur-sm"
              >
                <Zap className="w-3.5 h-3.5" />
                COMBOS
              </button>

              {/* Mic button */}
              <button
                onClick={toggleLiveVoice}
                disabled={isTranscribingVoice}
                className={[
                  "w-14 h-14 rounded-full flex items-center justify-center border transition-all backdrop-blur-sm",
                  isRecordingVoice
                    ? "bg-red-500/30 border-red-500 text-red-400 animate-pulse"
                    : isTranscribingVoice
                    ? "bg-white/10 border-white/20 text-[#A855F7]"
                    : "bg-black/60 border-white/20 text-white hover:bg-white/10",
                ].join(" ")}
                title="Speak to Riley"
              >
                {isTranscribingVoice
                  ? <Loader2 className="w-6 h-6 animate-spin" />
                  : <Mic className="w-6 h-6" />
                }
              </button>

            </div>
          </div>

        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════
          Item detail bottom sheet
          ════════════════════════════════════════════════════════════════════ */}
      <div
        className="fixed inset-0 z-40 pointer-events-none"
        style={{ display: showItemSheet ? "block" : "none" }}
      >
        {/* Backdrop */}
        <div
          className="absolute inset-0 bg-black/40 pointer-events-auto"
          onClick={() => { setShowItemSheet(false); setSelectedItem(null); }}
        />

        {/* Sheet */}
        <div
          className="absolute bottom-0 left-0 right-0 pointer-events-auto"
          style={{
            transform: showItemSheet ? "translateY(0)" : "translateY(100%)",
            transition: "transform 0.3s cubic-bezier(0.32, 0.72, 0, 1)",
          }}
        >
          {selectedItem && (
            <div className="bg-[#0a0a0a] border-t border-[#39FF14]/25 rounded-t-2xl px-5 pt-4 pb-10">
              {/* Pull handle */}
              <div className="w-10 h-1 rounded-full bg-white/20 mx-auto mb-4" />

              {/* Category + color dot */}
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ background: catColor(selectedItem.category) }}
                />
                <span
                  className="font-mono text-[10px] tracking-[0.2em] uppercase"
                  style={{ color: catColor(selectedItem.category) }}
                >
                  {selectedItem.category}
                </span>
              </div>

              {/* Label */}
              <h2 className="font-mono text-lg font-bold text-white mb-1 capitalize">
                {selectedItem.label}
              </h2>

              {/* Color */}
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-4 h-4 rounded-sm border border-white/15 flex-shrink-0"
                  style={{ background: toHex(selectedItem.color) }}
                />
                <span className="font-mono text-xs text-gray-400 capitalize">{selectedItem.color}</span>
                <span className="text-gray-600 mx-1">•</span>
                <span
                  className="font-mono text-xs font-bold"
                  style={{ color: catColor(selectedItem.category) }}
                >
                  {Math.round((selectedItem.confidence ?? 0) * 100)}% confidence
                </span>
              </div>

              {/* Description */}
              {selectedItem.description && (
                <p className="font-mono text-sm text-gray-300 leading-relaxed mb-5 border-l-2 border-[#39FF14]/30 pl-3">
                  {selectedItem.description}
                </p>
              )}

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={() => handleStyleThis(selectedItem)}
                  disabled={analyzing}
                  className="flex-1 py-3 rounded-xl bg-[#39FF14] text-black font-mono font-bold text-sm tracking-wider hover:brightness-110 transition-all disabled:opacity-50"
                >
                  {analyzing ? "Analyzing…" : "Style this →"}
                </button>
                <button
                  onClick={() => { setShowItemSheet(false); setSelectedItem(null); }}
                  className="px-4 py-3 rounded-xl border border-white/15 font-mono text-sm text-gray-400 hover:bg-white/5 transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          Combos bottom sheet
          ════════════════════════════════════════════════════════════════════ */}
      <div
        className="fixed inset-0 z-40 pointer-events-none"
        style={{ display: showCombosSheet ? "block" : "none" }}
      >
        {/* Backdrop */}
        <div
          className="absolute inset-0 bg-black/50 pointer-events-auto"
          onClick={() => setShowCombosSheet(false)}
        />

        {/* Sheet */}
        <div
          className="absolute bottom-0 left-0 right-0 pointer-events-auto max-h-[75vh] flex flex-col"
          style={{
            transform: showCombosSheet ? "translateY(0)" : "translateY(100%)",
            transition: "transform 0.3s cubic-bezier(0.32, 0.72, 0, 1)",
          }}
        >
          <div className="bg-[#0a0a0a] border-t border-[#A855F7]/30 rounded-t-2xl flex flex-col min-h-0">
            {/* Header */}
            <div className="px-5 pt-4 pb-3 flex-shrink-0">
              <div className="w-10 h-1 rounded-full bg-white/20 mx-auto mb-4" />
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-mono font-bold text-base text-white tracking-wider">COMBOS</h2>
                  <p className="font-mono text-[10px] text-[#A855F7] tracking-[0.15em] mt-0.5">
                    Riley&apos;s looks for your scan
                  </p>
                </div>
                <button
                  onClick={() => setShowCombosSheet(false)}
                  className="p-1.5 rounded-full hover:bg-white/10 transition-colors"
                >
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto px-4 pb-10 flex flex-col gap-3">
              {loadingCombos ? (
                <div className="flex flex-col items-center justify-center py-12 gap-3">
                  <Loader2 className="w-7 h-7 animate-spin text-[#A855F7]" />
                  <p className="font-mono text-xs text-gray-400 tracking-wider">Riley is building looks…</p>
                </div>
              ) : combos.length === 0 ? (
                <p className="text-center font-mono text-sm text-gray-500 py-8">No combos generated.</p>
              ) : (
                combos.map((combo) => (
                  <div key={combo.id} className="rounded-xl bg-white/5 border border-white/10 overflow-hidden">
                    <div className="p-4 border-b border-white/8">
                      <p className="font-mono font-bold text-sm text-white tracking-wide">{combo.name}</p>
                      <p className="font-mono text-xs text-[#A855F7] mt-0.5">{combo.vibe}</p>

                      {/* ── Reference images horizontal scroll ──────────── */}
                      {combo.reference_images && combo.reference_images.length > 0 && (
                        <div className="mt-3 -mx-1">
                          <div className="flex gap-2 overflow-x-auto pb-1" style={{ scrollbarWidth: "none" }}>
                            {combo.reference_images.map((url, imgIdx) => (
                              <div
                                key={imgIdx}
                                className="flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border border-white/10 bg-white/5"
                              >
                                <img
                                  src={url}
                                  alt={`${combo.name} reference ${imgIdx + 1}`}
                                  className="w-full h-full object-cover"
                                  loading="lazy"
                                  onError={(e) => {
                                    (e.currentTarget as HTMLImageElement).parentElement!.style.display = "none";
                                  }}
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <div className="px-4 pt-3 flex flex-wrap gap-1.5">
                      {combo.items_used.map((id) => {
                        const item = displayItems.find((i) => i.id === id || i.track_id === id);
                        if (!item) return null;
                        const color = catColor(item.category);
                        return (
                          <span
                            key={id}
                            className="font-mono text-xs px-2.5 py-1 rounded-full border"
                            style={{ borderColor: `${color}55`, color }}
                          >
                            {item.label}
                          </span>
                        );
                      })}
                    </div>

                    <p className="px-4 pt-3 pb-3 font-mono text-xs text-gray-300 leading-relaxed">
                      {combo.directions}
                    </p>

                    {combo.missing.length > 0 && (
                      <div className="px-4 pb-4 flex flex-col gap-2">
                        <p className="font-mono text-[9px] text-gray-500 uppercase tracking-[0.18em]">
                          Find these
                        </p>
                        {combo.missing.map((m, i) => (
                          <div
                            key={i}
                            className="p-3 rounded-lg bg-[#A855F7]/8 border border-[#A855F7]/20"
                          >
                            <p className="font-mono text-[9px] text-[#A855F7] uppercase tracking-wider mb-1">{m.role}</p>
                            <p className="font-mono text-xs text-gray-200 leading-snug">{m.find}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          Correction bottom sheet
          ════════════════════════════════════════════════════════════════════ */}
      <div
        className="fixed inset-0 z-40 pointer-events-none"
        style={{ display: showCorrectionSheet ? "block" : "none" }}
      >
        <div
          className="absolute inset-0 bg-black/40 pointer-events-auto"
          onClick={() => { setShowCorrectionSheet(false); setCorrectingItem(null); }}
        />

        <div
          className="absolute bottom-0 left-0 right-0 pointer-events-auto max-h-[85vh] flex flex-col"
          style={{
            transform: showCorrectionSheet ? "translateY(0)" : "translateY(100%)",
            transition: "transform 0.3s cubic-bezier(0.32, 0.72, 0, 1)",
          }}
        >
          <div className="bg-[#0a0a0a] border-t border-[#00E5FF]/30 rounded-t-2xl flex flex-col min-h-0">
            {/* Header */}
            <div className="px-5 pt-4 pb-3 flex-shrink-0 border-b border-white/10">
              <div className="w-10 h-1 rounded-full bg-white/20 mx-auto mb-4" />
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-mono font-bold text-base text-white tracking-wider flex items-center gap-2">
                    <span className="text-[#00E5FF]">✓</span> CORRECT ITEM
                  </h2>
                  <p className="font-mono text-[10px] text-[#00E5FF] tracking-[0.15em] mt-0.5">
                    Override label, category & fabric
                  </p>
                </div>
                <button
                  onClick={() => { setShowCorrectionSheet(false); setCorrectingItem(null); }}
                  className="p-1.5 rounded-full hover:bg-white/10 transition-colors"
                >
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              </div>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-5">
              {/* Label Edit */}
              <div>
                <label className="block font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                  Displayed Label
                </label>
                <input
                  type="text"
                  value={editLabel}
                  onChange={(e) => setEditLabel(e.target.value)}
                  className="w-full bg-white/5 border border-white/15 rounded-lg px-3 py-2 font-mono text-sm text-white focus:outline-none focus:border-[#00E5FF] transition-colors"
                  placeholder="e.g. denim jacket"
                />
              </div>

              {/* Category Picker */}
              <div>
                <label className="block font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                  Category
                </label>
                <div className="flex flex-wrap gap-2">
                  {TAXONOMY_CATEGORIES.map((cat) => {
                    const isSelected = editCategory === cat;
                    return (
                      <button
                        key={cat}
                        onClick={() => setEditCategory(cat)}
                        className={`font-mono text-xs px-3 py-1.5 rounded-lg border transition-all ${
                          isSelected
                            ? "bg-[#00E5FF]/20 border-[#00E5FF] text-[#00E5FF] font-bold shadow-[0_0_10px_rgba(0,229,255,0.2)]"
                            : "bg-white/5 border-white/10 text-gray-300 hover:border-white/30"
                        }`}
                      >
                        {cat}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Construction / Silhouette Picker */}
              <div>
                <label className="block font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                  Specific Silhouette / Construction
                </label>
                <div className="flex flex-wrap gap-2 max-h-36 overflow-y-auto pr-1">
                  {TAXONOMY_CONSTRUCTIONS.map((con) => {
                    const isSelected = editLabel.toLowerCase().includes(con);
                    return (
                      <button
                        key={con}
                        onClick={() => {
                          setEditLabel(con);
                          if (CONSTRUCTION_TO_CATEGORY[con]) {
                            setEditCategory(CONSTRUCTION_TO_CATEGORY[con]);
                          }
                        }}
                        className={`font-mono text-xs px-2.5 py-1 rounded-lg border transition-all ${
                          isSelected
                            ? "bg-[#39FF14]/20 border-[#39FF14] text-[#39FF14] font-bold shadow-[0_0_10px_rgba(57,255,20,0.2)]"
                            : "bg-white/5 border-white/10 text-gray-300 hover:border-white/30"
                        }`}
                      >
                        {con}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Fabric Picker */}
              <div>
                <label className="block font-mono text-[10px] text-gray-400 uppercase tracking-wider mb-2">
                  Fabric Type
                </label>
                <div className="flex flex-wrap gap-2">
                  {TAXONOMY_FABRICS.map((fab) => {
                    const isSelected = editFabric === fab;
                    return (
                      <button
                        key={fab}
                        onClick={() => setEditFabric(fab)}
                        className={`font-mono text-xs px-3 py-1.5 rounded-lg border transition-all ${
                          isSelected
                            ? "bg-[#00E5FF]/20 border-[#00E5FF] text-[#00E5FF] font-bold shadow-[0_0_10px_rgba(0,229,255,0.2)]"
                            : "bg-white/5 border-white/10 text-gray-300 hover:border-white/30"
                        }`}
                      >
                        {fab}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Footer / Confirm */}
            <div className="p-5 border-t border-white/10 flex-shrink-0">
              <button
                onClick={handleConfirmCorrection}
                className="w-full py-3 rounded-xl bg-[#00E5FF] text-black font-mono font-bold text-sm tracking-wider hover:bg-[#00c8e0] transition-colors shadow-[0_0_15px_rgba(0,229,255,0.3)]"
              >
                CONFIRM CORRECTION
              </button>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
