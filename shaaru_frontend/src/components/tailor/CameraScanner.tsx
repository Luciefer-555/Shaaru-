"use client";

import { useEffect, useRef, useState } from "react";
import { X, Loader2, Mic } from "lucide-react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

async function fetchWithTimeout(resource: RequestInfo | URL, options: RequestInit = {}): Promise<Response> {
  const { timeout = 90000 } = options as any;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(resource, { ...options, signal: controller.signal });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    throw error;
  }
}

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
};

type MissingPiece = { role: string; find: string };

export type StyleCombo = {
  id: string;
  name: string;
  vibe: string;
  items_used: string[];
  directions: string;
  missing: MissingPiece[];
};

// ── Surveillance canvas helpers ──────────────────────────────────

const CAT_STYLES: Record<string, { color: string; label: string }> = {
  top:       { color: "#39FF14", label: "TOP" },
  bottom:    { color: "#E040FB", label: "BOTTOM" },
  outerwear: { color: "#FF6D00", label: "OUTERWEAR" },
  footwear:  { color: "#00E5FF", label: "FOOTWEAR" },
  dress:     { color: "#FF4081", label: "DRESS" },
  set:       { color: "#FF4081", label: "CO-ORD SET" },
  accessory: { color: "#FFD700", label: "ACCESSORY" },
};
const _DEFAULT_STYLE = { color: "#A855F7", label: "GARMENT" };

function catStyle(category: string) {
  return CAT_STYLES[category?.toLowerCase()] ?? _DEFAULT_STYLE;
}

const _COLOR_HEX: Record<string, string> = {
  black:"#111",white:"#f5f5f0",red:"#e53935",blue:"#1e88e5",
  navy:"#1a237e",green:"#43a047",yellow:"#fdd835",orange:"#fb8c00",
  pink:"#e91e63",purple:"#8e24aa",grey:"#757575",gray:"#757575",
  brown:"#6d4c41",beige:"#d7ccc8",cream:"#fffde7",indigo:"#3949ab",
  teal:"#00897b",ivory:"#fffff0",olive:"#827717",khaki:"#c0a060",
  maroon:"#880e4f",coral:"#ff7043",mint:"#a5d6a7",lavender:"#ce93d8",
};
function toHex(colorName: string): string {
  if (!colorName) return "#888";
  if (colorName.startsWith("#")) return colorName;
  return _COLOR_HEX[colorName.toLowerCase().split(/[\s-]/)[0]] ?? "#888";
}

function drawBrackets(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, color: string
) {
  const cx = Math.min(w * 0.18, 22);
  const cy = Math.min(h * 0.18, 22);
  ctx.strokeStyle = "#39FF14";
  ctx.lineWidth = 2;
  ctx.lineCap = "square";
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  ([
    [[x, y + cy], [x, y], [x + cx, y]],
    [[x + w - cx, y], [x + w, y], [x + w, y + cy]],
    [[x, y + h - cy], [x, y + h], [x + cx, y + h]],
    [[x + w - cx, y + h], [x + w, y + h], [x + w, y + h - cy]],
  ] as [number, number][][]).forEach(([[ax, ay], [bx, by], [ccx, ccy]]) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(ccx, ccy);
    ctx.stroke();
  });
  ctx.shadowBlur = 0;
}

function drawScanLabel(
  ctx: CanvasRenderingContext2D,
  item: ScannedItem,
  rendW: number,
  rendH: number,
  offX: number,
  offY: number
) {
  const { color: c, label: catLabel } = catStyle(item.category);
  const bx = offX + item.bbox.x * rendW;
  const by = offY + item.bbox.y * rendH;
  const bw = item.bbox.w * rendW;
  const bh = item.bbox.h * rendH;
  if (bw < 20 || bh < 20) return;

  // Ghost box
  ctx.strokeStyle = `${c}44`;
  ctx.lineWidth = 1;
  ctx.shadowBlur = 0;
  ctx.strokeRect(bx, by, bw, bh);
  drawBrackets(ctx, bx, by, bw, bh, c);

  // Card sizing + positioning
  const CW = 180, CH = 66, GAP = 10;
  let cardX = bx + bw + GAP;
  let cardY = by;
  if (cardX + CW > ctx.canvas.width - 4) cardX = bx - CW - GAP;
  cardY = Math.max(4, Math.min(cardY, ctx.canvas.height - CH - 4));

  // Connector line
  const lx0 = cardX < bx ? bx : bx + bw;
  const ly0 = by + bh * 0.42;
  const lx1 = cardX < bx ? cardX + CW : cardX;
  const ly1 = cardY + CH * 0.42;
  ctx.beginPath();
  ctx.strokeStyle = `${c}55`;
  ctx.lineWidth = 0.8;
  ctx.setLineDash([3, 4]);
  ctx.moveTo(lx0, ly0); ctx.lineTo(lx1, ly1);
  ctx.stroke();
  ctx.setLineDash([]);

  // Card background + border
  ctx.fillStyle = "rgba(3,3,8,0.91)";
  ctx.fillRect(cardX, cardY, CW, CH);
  ctx.strokeStyle = `${c}44`;
  ctx.lineWidth = 0.8;
  ctx.strokeRect(cardX, cardY, CW, CH);

  // Top color bar
  ctx.fillStyle = c;
  ctx.fillRect(cardX, cardY, CW, 2.5);

  // Category label
  ctx.fillStyle = c;
  ctx.font = 'bold 8.5px "Courier New", Consolas, monospace';
  ctx.fillText(`// ${catLabel}`, cardX + 7, cardY + 17);

  // Description text (2 lines max)
  const desc = (item.description || item.label || "").substring(0, 72);
  ctx.fillStyle = "rgba(222,222,222,0.88)";
  ctx.font = '7.5px "Courier New", Consolas, monospace';
  const LINE = 28;
  ctx.fillText(desc.substring(0, LINE), cardX + 7, cardY + 32);
  if (desc.length > LINE) {
    const l2 = desc.substring(LINE, LINE * 2) + (desc.length > LINE * 2 ? "…" : "");
    ctx.fillText(l2, cardX + 7, cardY + 43);
  }

  // Color swatch
  ctx.fillStyle = toHex(item.color);
  ctx.fillRect(cardX + 7, cardY + 51, 9, 9);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 0.5;
  ctx.strokeRect(cardX + 7, cardY + 51, 9, 9);

  // Confidence
  ctx.fillStyle = `${c}77`;
  ctx.font = '6.5px "Courier New", monospace';
  ctx.fillText(
    `${Math.round(item.confidence * 100)}% ✓`,
    cardX + CW - 50, cardY + 62
  );
}

function drawHUDStatic(
  ctx: CanvasRenderingContext2D,
  w: number, h: number, count: number
) {
  ctx.fillStyle = "rgba(0,0,0,0.022)";
  for (let i = 0; i < h; i += 4) ctx.fillRect(0, i, w, 1);

  ctx.fillStyle = "rgba(0,0,0,0.70)";
  ctx.fillRect(0, 0, w, 22);
  ctx.fillStyle = "#39FF14";
  ctx.font = 'bold 8px "Courier New", Consolas, monospace';
  ctx.fillText(new Date().toTimeString().substring(0, 8), 8, 14);
  ctx.fillStyle = "#ff2222";
  ctx.fillText("● LIVE", w - 45, 14);

  ctx.fillStyle = "rgba(0,0,0,0.70)";
  ctx.fillRect(0, h - 22, w, 22);
  ctx.fillStyle = "#39FF14";
  ctx.font = '7.5px "Courier New", Consolas, monospace';
  const msg = "Shaaru can see and hear you";
  ctx.fillText(msg, (w - ctx.measureText(msg).width) / 2, h - 8);

  const S = 16;
  ctx.strokeStyle = "rgba(57,255,20,0.32)";
  ctx.lineWidth = 1.2;
  ([
    [[0, S + 22], [0, 22], [S, 22]],
    [[w - S, 22], [w, 22], [w, S + 22]],
    [[0, h - 20 - S], [0, h - 20], [S, h - 20]],
    [[w - S, h - 20], [w, h - 20], [w, h - 20 - S]],
  ] as number[][][]).forEach(([[ax, ay], [bx, by], [cx, cy]]) => {
    ctx.beginPath();
    ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineTo(cx, cy);
    ctx.stroke();
  });
}
// ── End canvas helpers ───────────────────────────────────────────

interface CameraScannerProps {
  onClose: () => void;
  onItemSelected: (item: ScannedItem) => void;
  userId?: string;
  onAnalysisComplete: (result: any) => void;
  onItemTouched?: (data: {
    label: string;
    comment: string;
    bbox: { x: number; y: number; w: number; h: number };
    color: string;
  }) => void;
}

export function CameraScanner({
  onClose,
  onItemSelected,
  userId,
  onAnalysisComplete,
  onItemTouched,
}: CameraScannerProps) {
  const [scanning, setScanning] = useState(false);
  const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null);
  const [detectedItems, setDetectedItems] = useState<ScannedItem[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [guidance, setGuidance] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);

  const [combos, setCombos]               = useState<StyleCombo[]>([]);
  const [loadingCombos, setLoadingCombos] = useState(false);
  const [view, setView]                   = useState<"items" | "combos">("items");

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const dwellTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dwellItemRef = useRef<string | null>(null);
  const commentedItemsRef = useRef<Map<string, number>>(new Map());
  const isSpeakingRef = useRef<boolean>(false);

  // Need to track the last captured base64 for analyze step
  const lastCapturedB64 = useRef<string | null>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const resultImgRef = useRef<HTMLImageElement>(null);
  const [capturedSrc, setCapturedSrc] = useState<string | null>(null);

  const [isRecordingVoice, setIsRecordingVoice] = useState(false);
  const [isTranscribingVoice, setIsTranscribingVoice] = useState(false);
  const voiceRecorderRef = useRef<MediaRecorder | null>(null);
  const voiceChunksRef = useRef<Blob[]>([]);

  const toggleLiveVoice = async () => {
    if (isRecordingVoice) {
      voiceRecorderRef.current?.stop();
      setIsRecordingVoice(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      voiceChunksRef.current = [];
      const rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) voiceChunksRef.current.push(e.data);
      };
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(voiceChunksRef.current, { type: "audio/webm" });
        setIsTranscribingVoice(true);
        try {
          const formData = new FormData();
          formData.append("file", blob, "live_voice.webm");
          formData.append("user_id", userId || "default");
          formData.append("enable_tts", "true");
          if (lastCapturedB64.current) {
            formData.append("image_base64", lastCapturedB64.current);
          }
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
          const res = await fetch(`${apiUrl}/api/voice/stt`, { method: "POST", body: formData });
          if (res.ok) {
            const data = await res.json();
            if (data.reply) setGuidance(data.reply);
            if (data.audio_base64) {
              const audio = new Audio(`data:audio/mp3;base64,${data.audio_base64}`);
              audio.play().catch(() => {});
            }
          }
        } catch (err) {
          console.error("[LIVE VOICE] error:", err);
        } finally {
          setIsTranscribingVoice(false);
        }
      };
      voiceRecorderRef.current = rec;
      rec.start();
      setIsRecordingVoice(true);
    } catch (err) {
      console.error("[LIVE VOICE] mic error:", err);
    }
  };

  useEffect(() => {
    const initCamera = async () => {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: { ideal: "environment" }, 
                   width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
      } catch {
        // fallback — any available camera (works on desktop)
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: true 
        });
      }
      
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try { await videoRef.current.play(); } catch {}
      }
    };

    initCamera().catch((err) => {
      console.error("Camera access error:", err);
      setError("Camera access denied");
    });

    return () => {
      // Cleanup stream tracks
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []); // Run only once

  // Separate effect to handle unmount cleanup cleanly
  useEffect(() => {
    return () => {
      setCameraStream((currentStream) => {
        if (currentStream) {
          currentStream.getTracks().forEach((track) => track.stop());
        }
        return null;
      });
    };
  }, []);

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
        console.warn(
          "[MediaPipe] Init failed:",
          e,
          "— continuing without hand tracking"
        );
      }
    };
    init();
    return () => {
      if (dwellTimerRef.current) clearTimeout(dwellTimerRef.current);
    };
  }, []);

  const processHandTracking = (
    videoEl: HTMLVideoElement,
    timestamp: number,
    currentFrameB64: string,
    currentItems: any[]
  ) => {
    if (!handLandmarkerRef.current || !currentItems?.length) return;

    try {
      const results = handLandmarkerRef.current.detectForVideo(videoEl, timestamp);

      if (!results?.landmarks?.length) {
        // No hand — clear dwell
        if (dwellTimerRef.current) {
          clearTimeout(dwellTimerRef.current);
          dwellTimerRef.current = null;
          dwellItemRef.current = null;
        }
        return;
      }

      // Index fingertip = landmark 8
      const tip = results.landmarks[0][8];
      const fx = tip.x; // normalized 0-1
      const fy = tip.y;

      // Find which item bbox contains fingertip
      let touched: any = null;
      for (const item of currentItems) {
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
          dwellItemRef.current = null;
        }
        return;
      }

      const label = touched.label as string;

      // 60s dedup — already commented on this item?
      const lastTime = commentedItemsRef.current.get(label) ?? 0;
      if (Date.now() - lastTime < 60000) return;

      // Already timing this item
      if (dwellItemRef.current === label) return;

      // New item — start 1.5s dwell timer
      if (dwellTimerRef.current) clearTimeout(dwellTimerRef.current);
      dwellItemRef.current = label;

      dwellTimerRef.current = setTimeout(async () => {
        if (isSpeakingRef.current) return;

        try {
          isSpeakingRef.current = true;
          commentedItemsRef.current.set(label, Date.now());

          // Call touch endpoint
          const resp = await fetch("/api/cv/touch", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              item_label: label,
              item_color: touched.color ?? "",
              item_category: touched.category ?? "",
              item_aesthetic: touched.aesthetic ?? "",
              all_items: currentItems.map((i: any) => i.label),
              user_id: userId ?? "default",
            }),
          });

          const data = await resp.json();

          if (data.comment) {
            // Show visual overlay
            onItemTouched?.({
              label,
              comment: data.comment,
              bbox: touched.bbox,
              color: touched.color,
            });

            // Speak via Nova TTS
            const ttsResp = await fetch("/api/voice/tts", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                text: data.comment,
                voice: "nova",
              }),
            });
            const ttsData = await ttsResp.json();
            if (ttsData.audio_b64) {
              const audio = new Audio(
                "data:audio/mp3;base64," + ttsData.audio_b64
              );
              audio.onended = () => {
                isSpeakingRef.current = false;
              };
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
        dwellItemRef.current = null;
      }, 1500);
    } catch (e) {
      console.warn("[MediaPipe] Frame error:", e);
    }
  };

  useEffect(() => {
    if (!annotatedFrame || !detectedItems.length) return;
    const img = resultImgRef.current;
    const canvas = overlayRef.current;
    if (!img || !canvas) return;

    const draw = () => {
      const cw = img.clientWidth;
      const ch = img.clientHeight;
      if (cw === 0 || ch === 0) return;

      // Account for object-contain letterboxing
      const natW = img.naturalWidth || cw;
      const natH = img.naturalHeight || ch;
      const scale = Math.min(cw / natW, ch / natH);
      const rendW = natW * scale;
      const rendH = natH * scale;
      const offX = (cw - rendW) / 2;
      const offY = (ch - rendH) / 2;

      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, cw, ch);
      detectedItems.forEach((item) =>
        drawScanLabel(ctx, item, rendW, rendH, offX, offY)
      );
      drawHUDStatic(ctx, cw, ch, detectedItems.length);
    };

    let rafId: number;
    if (img.complete && img.naturalWidth > 0) {
      rafId = requestAnimationFrame(draw);
    } else {
      img.onload = () => { rafId = requestAnimationFrame(draw); };
    }
    return () => {
      cancelAnimationFrame(rafId);
      img.onload = null;
    };
  }, [detectedItems, annotatedFrame]);

  const handleCapture = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    setScanning(true);
    setGuidance(null);
    setError(null);

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      setScanning(false);
      return;
    }

    canvas.width = 1280;
    canvas.height = 720;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    const image_b64 = dataUrl.split(",")[1]; // strip data:image/jpeg;base64,
    lastCapturedB64.current = image_b64;
    processHandTracking(video, performance.now(), image_b64, detectedItems);
    setCapturedSrc(dataUrl);

    try {
      const response = await fetchWithTimeout("/api/cv/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64, user_id: userId }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Scan failed");
      }

      if (data.frame_quality === "poor") {
        setGuidance(data.guidance || "Poor quality frame, try again.");
        setDetectedItems([]);
        setAnnotatedFrame(null);
      } else {
        setAnnotatedFrame(data.annotated_frame_b64 || null);
        setDetectedItems(data.items || []);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred during scan");
    } finally {
      setScanning(false);
    }
  };

  const handleGetCombos = async () => {
    if (!detectedItems.length) return;
    setLoadingCombos(true);
    setView("combos");
    try {
      const res = await fetchWithTimeout("/api/cv/style-combos", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: detectedItems, user_id: userId }),
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

  const handleItemTap = async (item: ScannedItem) => {
    if (!lastCapturedB64.current) return;

    setAnalyzing(true);
    try {
      const response = await fetchWithTimeout("/api/cv/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64: lastCapturedB64.current,
          item_id: item.id,
          item_label: item.label,
          user_id: userId,
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Analyze failed");
      }

      onItemSelected(item);
      onAnalysisComplete(result);
      onClose();
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis");
      setAnalyzing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-[#1A1A1A] text-gray-100">
      {/* Top Bar */}
      <div className="flex items-center justify-between p-4 z-10">
        <h1 className="text-sm font-bold tracking-[0.2em]">RILEY IS LOOKING</h1>
        <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full transition-colors">
          <X className="w-6 h-6" />
        </button>
      </div>

      <div className="flex-1 relative flex flex-col items-center justify-center overflow-hidden">
        {/* Hidden Canvas for capture */}
        <canvas ref={canvasRef} className="hidden" />

        {error ? (
          <div className="flex flex-col items-center justify-center p-6 text-center">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={() => {
                setError(null);
                setScanning(false);
              }}
              className="px-6 py-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
            >
              Retry
            </button>
          </div>
        ) : !annotatedFrame ? (
          // LIVE VIDEO FEED
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover"
            />
            {guidance && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <p className="bg-black/60 px-6 py-3 rounded-full text-white font-medium text-lg text-center mx-4">
                  {guidance}
                </p>
              </div>
            )}
            <div className="absolute bottom-10 left-0 right-0 flex justify-center items-center gap-6">
              <button
                onClick={handleCapture}
                disabled={scanning}
                className="w-20 h-20 rounded-full bg-gradient-to-tr from-[#A855F7] to-purple-400 flex items-center justify-center hover:opacity-90 transition-opacity disabled:opacity-50 shadow-lg shadow-purple-500/20"
              >
                {scanning ? <Loader2 className="w-8 h-8 animate-spin text-white" /> : <div className="w-16 h-16 rounded-full border-4 border-white" />}
              </button>

              <button
                onClick={toggleLiveVoice}
                disabled={isTranscribingVoice}
                className={`w-14 h-14 rounded-full flex items-center justify-center border transition-all ${
                  isRecordingVoice
                    ? "bg-red-500/30 border-red-500 text-red-400 animate-pulse"
                    : isTranscribingVoice
                    ? "bg-white/10 border-white/20 text-[#A855F7]"
                    : "bg-black/50 border-white/20 text-white hover:bg-white/10"
                }`}
                title="Speak to Riley"
              >
                {isTranscribingVoice ? <Loader2 className="w-6 h-6 animate-spin" /> : <Mic className="w-6 h-6" />}
              </button>
            </div>
          </>
        ) : (
          // ANNOTATED RESULT
          <div className="flex flex-col w-full h-full">
            <div className="relative w-full max-h-[60vh] flex-shrink-0 bg-black flex items-center justify-center">
              <img
                ref={resultImgRef}
                src={capturedSrc ?? ""}
                alt="Captured frame"
                className="max-w-full max-h-[60vh] object-contain block"
              />
              <canvas
                ref={overlayRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
              />
              {analyzing && (
                <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center backdrop-blur-sm z-20">
                  <Loader2 className="w-12 h-12 animate-spin text-[#A855F7] mb-4" />
                  <p className="text-lg font-medium tracking-wide">Riley is reading this...</p>
                </div>
              )}
            </div>
            
            <div className="flex-1 overflow-y-auto flex flex-col min-h-0">

              {/* Tab bar */}
              <div className="flex border-b border-white/10 px-4 pt-2 gap-1 flex-shrink-0">
                <button
                  onClick={() => setView("items")}
                  className={`px-4 py-2 text-sm font-medium tracking-wide transition-colors border-b-2 -mb-px ${
                    view === "items"
                      ? "border-[#A855F7] text-white"
                      : "border-transparent text-gray-500 hover:text-gray-300"
                  }`}
                >
                  ITEMS ({detectedItems.length})
                </button>
                <button
                  onClick={() => {
                    if (!combos.length && !loadingCombos) handleGetCombos();
                    else setView("combos");
                  }}
                  className={`px-4 py-2 text-sm font-medium tracking-wide transition-colors border-b-2 -mb-px ${
                    view === "combos"
                      ? "border-[#A855F7] text-white"
                      : "border-transparent text-gray-500 hover:text-gray-300"
                  }`}
                >
                  COMBOS
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">

                {/* ITEMS VIEW */}
                {view === "items" && (
                  <>
                    {detectedItems.length === 0 && (
                      <p className="text-center text-gray-400 py-4">No items detected.</p>
                    )}
                    {detectedItems.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => handleItemTap(item)}
                        disabled={analyzing}
                        className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-left disabled:opacity-50 disabled:pointer-events-none"
                      >
                        <div>
                          <div className="flex items-center gap-2 mb-0.5">
                            <span
                              className="w-2 h-2 rounded-full flex-shrink-0"
                              style={{ background: catStyle(item.category).color }}
                            />
                            <p className="font-medium">{item.label}</p>
                          </div>
                          <p className="text-sm text-gray-400 capitalize pl-4">
                            {item.color} • {item.description}
                          </p>
                        </div>
                        <span className="text-xs font-bold text-[#A855F7] bg-[#A855F7]/10 px-2 py-1 rounded-full ml-3 flex-shrink-0">
                          {Math.round(item.confidence * 100)}%
                        </span>
                      </button>
                    ))}
                  </>
                )}

                {/* COMBOS VIEW */}
                {view === "combos" && (
                  <>
                    {loadingCombos ? (
                      <div className="flex flex-col items-center justify-center py-12 gap-3">
                        <Loader2 className="w-8 h-8 animate-spin text-[#A855F7]" />
                        <p className="text-sm text-gray-400 tracking-wide">Riley is building looks...</p>
                      </div>
                    ) : combos.length === 0 ? (
                      <p className="text-center text-gray-400 py-4">No combos generated.</p>
                    ) : (
                      combos.map((combo) => (
                        <div
                          key={combo.id}
                          className="rounded-xl bg-white/5 border border-white/10 overflow-hidden"
                        >
                          {/* Combo header */}
                          <div className="p-4 border-b border-white/8">
                            <p className="font-bold text-base tracking-wide">{combo.name}</p>
                            <p className="text-sm text-[#A855F7] mt-0.5">{combo.vibe}</p>
                          </div>

                          {/* Items used as colored tags */}
                          <div className="px-4 pt-3 flex flex-wrap gap-1.5">
                            {combo.items_used.map((id) => {
                              const item = detectedItems.find((i) => i.id === id);
                              if (!item) return null;
                              const { color } = catStyle(item.category);
                              return (
                                <span
                                  key={id}
                                  className="text-xs px-2.5 py-1 rounded-full font-mono border"
                                  style={{ borderColor: `${color}66`, color }}
                                >
                                  {item.label}
                                </span>
                              );
                            })}
                          </div>

                          {/* Riley's directions */}
                          <p className="px-4 pt-3 pb-3 text-sm text-gray-300 leading-relaxed">
                            {combo.directions}
                          </p>

                          {/* Missing pieces */}
                          {combo.missing.length > 0 && (
                            <div className="px-4 pb-4 flex flex-col gap-2">
                              <p className="text-xs text-gray-500 uppercase tracking-[0.15em]">
                                Find these
                              </p>
                              {combo.missing.map((m, i) => (
                                <div
                                  key={i}
                                  className="p-3 rounded-lg bg-[#A855F7]/8 border border-[#A855F7]/20"
                                >
                                  <p className="text-xs text-[#A855F7] uppercase tracking-wider mb-1">
                                    {m.role}
                                  </p>
                                  <p className="text-sm text-gray-200 leading-snug">{m.find}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </>
                )}

                {/* Scan Again */}
                <div className="mt-2 flex justify-center pb-2">
                  <button
                    onClick={() => {
                      setAnnotatedFrame(null);
                      setDetectedItems([]);
                      setCombos([]);
                      setView("items");
                    }}
                    disabled={analyzing}
                    className="px-6 py-3 rounded-full border border-white/20 hover:bg-white/10 transition-colors font-medium disabled:opacity-50"
                  >
                    Scan Again
                  </button>
                </div>

              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
