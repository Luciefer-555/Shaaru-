"use client"

import { useState, useRef, useEffect } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { ArrowUp, Paperclip, Camera, X, FileText } from "lucide-react"

// ─── SHAARU LOGO ───────────────────────────────────────
function ShaaruLogo({ size = 40 }: { size?: number }) {
  return (
    <svg viewBox="0 0 200 200" width={size} height={size}>
      <defs>
        <ellipse id="p" cx="100" cy="100" rx="90" ry="22" />
      </defs>
      <g fill="#8B1A1A" fillRule="evenodd">
        <use href="#p" transform="rotate(0 100 100)" />
        <use href="#p" transform="rotate(45 100 100)" />
        <use href="#p" transform="rotate(90 100 100)" />
        <use href="#p" transform="rotate(135 100 100)" />
      </g>
    </svg>
  )
}

// ─── TYPES ─────────────────────────────────────────────
interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  streaming?: boolean
}

interface ScannedItem {
  id: string
  label: string
  description: string
  category: string
  color: string
  aesthetic: string
  bbox: { x: number; y: number; w: number; h: number }
  confidence: number
}

interface PixelBox {
  id: string
  xyxy: [number, number, number, number]
}

interface AnalyzeResult {
  item_label: string
  garment_analysis: Record<string, unknown>
  fabric_intelligence: Record<string, unknown>
  profile_compatibility: { compatible: boolean; reason: string }
  tailor_available: boolean
}

// ─── WORD STREAMING HOOK ───────────────────────────────
function useWordStream(text: string, active: boolean, onDone: () => void) {
  const [displayed, setDisplayed] = useState("")
  const ref = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (!active || !text) return
    setDisplayed("")
    const words = text.split(" ")
    let i = 0
    const next = () => {
      if (i >= words.length) { onDone(); return }
      setDisplayed(prev => prev + (i === 0 ? "" : " ") + words[i])
      i++
      ref.current = setTimeout(next, 35)
    }
    next()
    return () => { if (ref.current) clearTimeout(ref.current) }
  }, [text, active])

  return displayed
}

// ─── STREAMING MESSAGE ────────────────────────────────
function StreamingMessage({ content, onDone }: { content: string; onDone: () => void }) {
  const text = useWordStream(content, true, onDone)
  return <span>{text}</span>
}

// ─── IDLE VIEW ─────────────────────────────────────────
function IdleView({ onSend }: { onSend: (msg: string, img?: string) => void }) {
  const [input, setInput] = useState("")
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const textRef = useRef<HTMLTextAreaElement>(null)

  const hour = new Date().getHours()
  const greeting = hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening"

  const send = async () => {
    if (!input.trim()) return
    onSend(input.trim())
    setInput("")
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send() }
  }

  const handleFile = (file: File) => {
    const reader = new FileReader()
    reader.onload = () => onSend(input.trim() || "make this for me", reader.result as string)
    reader.readAsDataURL(file)
  }

  useEffect(() => {
    if (textRef.current) {
      textRef.current.style.height = "auto"
      textRef.current.style.height = Math.min(textRef.current.scrollHeight, 200) + "px"
    }
  }, [input])

  return (
    <div style={{
      minHeight: "100svh",
      backgroundColor: "#1A1A1A",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "0 16px",
      fontFamily: "system-ui, sans-serif"
    }}>
      {/* Logo */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{ marginBottom: 20 }}
      >
        <ShaaruLogo size={48} />
      </motion.div>

      {/* Greeting */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        style={{ marginBottom: 32, textAlign: "center" }}
      >
        <h1 style={{
          fontSize: 32,
          fontWeight: 300,
          color: "#ECECEC",
          fontFamily: "Georgia, serif",
          lineHeight: 1.2,
          margin: 0
        }}>
          {greeting},{" "}
          <span style={{ position: "relative", display: "inline-block" }}>
            TasteMaxer
            <svg
              viewBox="0 0 160 12"
              style={{
                position: "absolute",
                bottom: -6,
                left: 0,
                width: "100%",
                height: 10
              }}
            >
              <path
                d="M4 8 Q 80 0, 156 8"
                stroke="#8B1A1A"
                strokeWidth="2.5"
                fill="none"
                strokeLinecap="round"
              />
            </svg>
          </span>
        </h1>
      </motion.div>

      {/* Input box */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
        style={{ width: "100%", maxWidth: 680 }}
      >
        {/* Suggestion pills */}
        <div style={{
          display: "flex",
          gap: 8,
          justifyContent: "center",
          marginBottom: 12,
          flexWrap: "wrap"
        }}>
          {[
            "Make from reference",
            "Describe a garment",
            "Repurpose old clothes",
            "Wedding outfit"
          ].map(label => (
            <button
              key={label}
              onClick={() => setInput(label.toLowerCase())}
              style={{
                padding: "6px 14px",
                borderRadius: 999,
                border: "1px solid #333330",
                backgroundColor: "transparent",
                color: "#888884",
                fontSize: 13,
                cursor: "pointer"
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Input container */}
        <div
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => {
            e.preventDefault()
            setDragging(false)
            const f = e.dataTransfer.files[0]
            if (f) handleFile(f)
          }}
          style={{
            backgroundColor: "#242422",
            border: `1px solid ${dragging ? "#8B1A1A" : "#333330"}`,
            borderRadius: 16,
            padding: "14px 14px 10px 14px",
            transition: "border-color 0.2s"
          }}
        >
          <textarea
            ref={textRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="show me what you want to make..."
            rows={1}
            style={{
              width: "100%",
              background: "transparent",
              border: "none",
              outline: "none",
              color: "#ECECEC",
              fontSize: 15,
              resize: "none",
              fontFamily: "inherit",
              lineHeight: 1.5,
              minHeight: 24,
              boxSizing: "border-box"
            }}
          />
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 8
          }}>
            <button
              onClick={() => fileRef.current?.click()}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                color: "#666663",
                display: "flex",
                padding: 4
              }}
            >
              <Paperclip size={18} />
            </button>
            <button
              onClick={send}
              disabled={!input.trim()}
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                border: "none",
                backgroundColor: input.trim() ? "#8B1A1A" : "#333330",
                color: "white",
                cursor: input.trim() ? "pointer" : "default",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "background-color 0.2s"
              }}
            >
              <ArrowUp size={16} />
            </button>
          </div>
        </div>

        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
        />
      </motion.div>
    </div>
  )
}

// ─── CHAT VIEW ─────────────────────────────────────────
function ChatView({
  initialMessage,
  onBack,
  onCameraOpen,
  onTailorFlow
}: {
  initialMessage: string
  onBack: () => void
  onCameraOpen?: () => void
  onTailorFlow?: () => void
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [streamContent, setStreamContent] = useState("")
  const [streamId, setStreamId] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isFocused, setIsFocused] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const didInit = useRef(false)

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => { scrollToBottom() }, [messages, streamContent])

  const handleFile = (f: File) => {
    setFile(f)
    if (f.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = () => setPreviewUrl(reader.result as string)
      reader.readAsDataURL(f)
    } else {
      setPreviewUrl(null)
    }
  }

  const sendMessage = async (text: string) => {
    if ((!text.trim() && !file) || loading) return

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: text.trim() || "[Attachment]"
    }

    setMessages(prev => [...prev, userMsg])
    setLoading(true)
    setInput("")
    setFile(null)
    setPreviewUrl(null)

    try {
      const res = await fetch("/api/chat/message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim(), history: [] })
      })
      if (!res.ok) throw new Error("Backend unavailable")
      const data = await res.json()
      const reply = data.reply || "something went wrong"
      if (data.tailor_flow) {
        onTailorFlow?.()
        return
      }

      const assistantId = (Date.now() + 1).toString()
      setStreamId(assistantId)
      setStreamContent(reply)

    } catch {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: "assistant",
        content: "something went wrong, try again"
      }])
      setLoading(false)
    }
  }

  const handleStreamDone = () => {
    if (!streamId || !streamContent) return
    setMessages(prev => [...prev, {
      id: streamId,
      role: "assistant",
      content: streamContent
    }])
    setStreamId(null)
    setStreamContent("")
    setLoading(false)
  }

  useEffect(() => {
    if (!didInit.current && initialMessage) {
      didInit.current = true
      sendMessage(initialMessage)
    }
  }, [])

  useEffect(() => {
    if (textRef.current) {
      textRef.current.style.height = "auto"
      textRef.current.style.height = Math.min(textRef.current.scrollHeight, 200) + "px"
    }
  }, [input])

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  return (
    <div style={{
      height: "100svh",
      backgroundColor: "#1A1A1A",
      display: "flex",
      flexDirection: "column",
      fontFamily: "system-ui, sans-serif"
    }}>
      {/* Header */}
      <div style={{
        height: 48,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        backgroundColor: "#1A1A1A",
        zIndex: 10,
        borderBottom: "1px solid #222220"
      }}>
        <ShaaruLogo size={22} />
        <span style={{
          color: "#ECECEC",
          fontSize: 15,
          fontWeight: 500,
          marginLeft: 8,
          letterSpacing: "0.05em"
        }}>
          SHAARU
        </span>
        <button
          onClick={onBack}
          style={{
            position: "absolute",
            right: 16,
            background: "none",
            border: "none",
            color: "#666663",
            cursor: "pointer",
            fontSize: 20,
            lineHeight: 1
          }}
        >
          ✎
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        paddingTop: 64,
        paddingBottom: 120
      }}>
        <div style={{
          maxWidth: 680,
          margin: "0 auto",
          padding: "0 16px"
        }}>
          {messages.map(msg => (
            <div
              key={msg.id}
              style={{
                display: "flex",
                justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                marginBottom: msg.role === "user" ? 16 : 24
              }}
            >
              {msg.role === "user" ? (
                <div style={{
                  backgroundColor: "#2A2A28",
                  color: "#ECECEC",
                  padding: "10px 16px",
                  borderRadius: "18px 18px 4px 18px",
                  fontSize: 14,
                  lineHeight: 1.6,
                  maxWidth: "75%"
                }}>
                  {msg.content}
                </div>
              ) : (
                <div style={{
                  display: "flex",
                  gap: 12,
                  maxWidth: "85%"
                }}>
                  <div style={{ flexShrink: 0, marginTop: 2 }}>
                    <ShaaruLogo size={20} />
                  </div>
                  <div style={{
                    color: "#ECECEC",
                    fontSize: 14,
                    lineHeight: 1.7
                  }}>
                    {msg.content}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Streaming message */}
          {streamId && (
            <div style={{
              display: "flex",
              gap: 12,
              marginBottom: 24,
              maxWidth: "85%"
            }}>
              <div style={{ flexShrink: 0, marginTop: 2 }}>
                <ShaaruLogo size={20} />
              </div>
              <div style={{ color: "#ECECEC", fontSize: 14, lineHeight: 1.7 }}>
                <StreamingMessage content={streamContent} onDone={handleStreamDone} />
              </div>
            </div>
          )}

          {/* Loading dots */}
          {loading && !streamId && (
            <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
              <div style={{ flexShrink: 0 }}>
                <ShaaruLogo size={20} />
              </div>
              <div style={{ display: "flex", gap: 4, alignItems: "center", paddingTop: 4 }}>
                {[0, 150, 300].map(delay => (
                  <div
                    key={delay}
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      backgroundColor: "#8B1A1A",
                      animation: `bounce 1s infinite ${delay}ms`
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input - fixed bottom */}
      <div style={{
        position: "fixed",
        bottom: 0,
        left: 0,
        right: 0,
        padding: "12px 16px 20px",
        background: "linear-gradient(to top, #1A1A1A 80%, transparent)"
      }}>
        <div style={{
          maxWidth: 680,
          margin: "0 auto",
          backgroundColor: "#242422",
          border: `1px solid ${isFocused ? "#4A4A48" : "#333330"}`,
          borderRadius: 16,
          padding: "12px 14px 10px 14px",
          transition: "border-color 0.2s ease"
        }}>
          {file && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.2 }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
                padding: "8px 12px",
                backgroundColor: "#1A1A1A",
                borderRadius: 12,
                width: "max-content",
                position: "relative"
              }}
            >
              {previewUrl ? (
                <img src={previewUrl} alt="preview" style={{ width: 40, height: 40, borderRadius: 8, objectFit: "cover" }} />
              ) : (
                <div style={{ display: "flex", alignItems: "center", gap: 8, color: "#ECECEC" }}>
                  <FileText size={20} color="#888884" />
                  <span style={{ fontSize: 13, maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {file.name}
                  </span>
                </div>
              )}
              <button
                onClick={() => { setFile(null); setPreviewUrl(null) }}
                style={{
                  position: "absolute",
                  top: -6,
                  right: -6,
                  width: 20,
                  height: 20,
                  borderRadius: "50%",
                  backgroundColor: "#333330",
                  color: "#ECECEC",
                  border: "none",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  cursor: "pointer"
                }}
              >
                <X size={12} />
              </button>
            </motion.div>
          )}

          <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
            <div style={{ display: "flex", gap: 4, paddingBottom: 4 }}>
              <button
                onClick={() => fileRef.current?.click()}
                onMouseEnter={e => e.currentTarget.style.color = "#ECECEC"}
                onMouseLeave={e => e.currentTarget.style.color = "#666663"}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "#666663",
                  display: "flex",
                  padding: 4,
                  transition: "color 0.2s"
                }}
              >
                <Paperclip size={20} />
              </button>
              <button
                onClick={onCameraOpen}
                onMouseEnter={e => e.currentTarget.style.color = "#ECECEC"}
                onMouseLeave={e => e.currentTarget.style.color = "#666663"}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "#666663",
                  display: "flex",
                  padding: 4,
                  transition: "color 0.2s"
                }}
              >
                <Camera size={20} />
              </button>
            </div>

            <textarea
              ref={textRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="message SHAARU..."
              rows={1}
              disabled={loading}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: "#ECECEC",
                fontSize: 15,
                resize: "none",
                fontFamily: "inherit",
                lineHeight: 1.5,
                minHeight: 24,
                maxHeight: 160,
                overflowY: "auto",
                boxSizing: "border-box",
                padding: "4px 0",
                transition: "height 0.15s ease"
              }}
            />

            <div style={{ paddingBottom: 2 }}>
              <button
                onClick={() => sendMessage(input)}
                disabled={(!input.trim() && !file) || loading}
                onMouseDown={e => e.currentTarget.style.transform = "scale(0.96)"}
                onMouseUp={e => e.currentTarget.style.transform = "scale(1)"}
                onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 12,
                  border: "none",
                  backgroundColor: (input.trim() || file) && !loading ? "#8B1A1A" : "#333330",
                  color: "white",
                  cursor: (input.trim() || file) && !loading ? "pointer" : "default",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "background-color 0.2s, transform 0.1s"
                }}
              >
                <ArrowUp size={16} />
              </button>
            </div>
          </div>

          <input
            ref={fileRef}
            type="file"
            accept="image/*,application/pdf,.doc,.docx"
            style={{ display: "none" }}
            onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
          />
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-4px); }
        }
      `}</style>
    </div>
  )
}

// ─── CAMERA VIEW ──────────────────────────────────────
function CameraView({
  onClose,
  onBriefReady,
}: {
  onClose: () => void
  onBriefReady: (brief: AnalyzeResult) => void
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const originalFrameRef = useRef<string | null>(null)

  const [phase, setPhase] = useState<"live" | "scanning" | "items" | "analyzing">("live")
  const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null)
  const [items, setItems] = useState<ScannedItem[]>([])
  const [pixelBoxes, setPixelBoxes] = useState<PixelBox[]>([])
  const [error, setError] = useState<string | null>(null)

  const startCamera = () => {
    navigator.mediaDevices.enumerateDevices().then(devices => {
      const videoDevices = devices.filter(d => d.kind === "videoinput")
      // skip Link to Windows virtual camera
      const realCam = videoDevices.find(d =>
        !d.label.toLowerCase().includes("link to windows") &&
        !d.label.toLowerCase().includes("virtual") &&
        !d.label.toLowerCase().includes("droidcam")
      ) || videoDevices[0]

      const isMobile = /Mobi|Android/i.test(navigator.userAgent)
      const constraints: MediaStreamConstraints = isMobile
        ? { video: { facingMode: "environment" } }
        : realCam
          ? { video: { deviceId: { exact: realCam.deviceId } } }
          : { video: true }

      navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
          streamRef.current = stream
          if (videoRef.current) {
            videoRef.current.srcObject = stream
            videoRef.current.play().catch(() => {})
          }
        })
        .catch(() => setError("Camera access denied — allow camera permissions and retry"))
    })
  }

  useEffect(() => {
    startCamera()
    return () => { streamRef.current?.getTracks().forEach(t => t.stop()) }
  }, [])

  const capture = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    // wait for video to have real dimensions
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setError("Camera not ready — wait a moment and try again")
      return
    }
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext("2d")!.drawImage(video, 0, 0)
    const b64 = canvas.toDataURL("image/jpeg", 0.85).split(",")[1]
    originalFrameRef.current = b64
    streamRef.current?.getTracks().forEach(t => t.stop())
    runScan(b64)
  }

  const runScan = async (b64: string) => {
    setPhase("scanning")
    setError(null)
    try {
      const res = await fetch("/api/cv/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_b64: b64, user_id: "guest" }),
      })
      if (!res.ok) throw new Error("scan failed")
      const data = await res.json()

      if (data.frame_quality === "poor") {
        setError(data.guidance || "Move closer and try again")
        setPhase("live")
        startCamera()
        return
      }

      setAnnotatedFrame(data.annotated_frame_b64 || null)
      setItems(data.items || [])
      setPixelBoxes(data.pixel_boxes || [])
      setPhase("items")
    } catch {
      setError("Scan failed — try again")
      setPhase("live")
      startCamera()
    }
  }

  const handleItemTap = async (item: ScannedItem) => {
    setPhase("analyzing")
    setError(null)

    let imageToSend = originalFrameRef.current || ""

    // Crop to bbox if coordinates are valid
    const box = pixelBoxes.find(b => b.id === item.id)
    if (box && originalFrameRef.current) {
      const [x1, y1, x2, y2] = box.xyxy
      const w = x2 - x1
      const h = y2 - y1
      if (w > 10 && h > 10) {
        try {
          const img = new Image()
          await new Promise<void>((res, rej) => {
            img.onload = () => res()
            img.onerror = rej
            img.src = `data:image/jpeg;base64,${originalFrameRef.current}`
          })
          const crop = document.createElement("canvas")
          crop.width = w
          crop.height = h
          crop.getContext("2d")!.drawImage(img, x1, y1, w, h, 0, 0, w, h)
          imageToSend = crop.toDataURL("image/jpeg", 0.9).split(",")[1]
        } catch {
          // fallback to full frame
        }
      }
    }

    try {
      const res = await fetch("/api/cv/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64: imageToSend,
          item_label: item.label,
          item_id: item.id,
          user_id: "guest",
        }),
      })
      if (!res.ok) throw new Error("analyze failed")
      const brief = await res.json()
      onBriefReady(brief as AnalyzeResult)
    } catch {
      setError("Analysis failed — try again")
      setPhase("items")
    }
  }

  return (
    <div style={{
      position: "fixed", inset: 0,
      backgroundColor: "#000",
      display: "flex", flexDirection: "column",
      zIndex: 100, fontFamily: "system-ui, sans-serif"
    }}>
      {/* Header */}
      <div style={{
        height: 48, display: "flex", alignItems: "center",
        justifyContent: "space-between", padding: "0 16px"
      }}>
        <span style={{ color: "#ECECEC", fontSize: 12, letterSpacing: "0.1em" }}>
          SHAARU LENS
        </span>
        <button onClick={onClose} style={{
          background: "none", border: "none",
          color: "#ECECEC", cursor: "pointer", display: "flex", padding: 4
        }}>
          <X size={20} />
        </button>
      </div>

      {/* Viewport */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden", backgroundColor: "#000" }}>
        <video
          ref={videoRef}
          autoPlay playsInline muted
          onLoadedMetadata={() => videoRef.current?.play()}
          style={{
            width: "100%", height: "100%", objectFit: "cover",
            display: phase === "live" || phase === "scanning" ? "block" : "none"
          }}
        />

        {(phase === "items" || phase === "analyzing") && annotatedFrame && (
          <img
            src={`data:image/png;base64,${annotatedFrame}`}
            alt="Scan result"
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
          />
        )}

        {phase === "scanning" && (
          <div style={{
            position: "absolute", inset: 0,
            backgroundColor: "rgba(0,0,0,0.65)",
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 16
          }}>
            <div style={{
              width: 44, height: 44,
              border: "2px solid #8B1A1A", borderTopColor: "transparent",
              borderRadius: "50%", animation: "lens-spin 0.8s linear infinite"
            }} />
            <span style={{ color: "#ECECEC", fontSize: 13 }}>reading the rack...</span>
            <span style={{ color: "#555552", fontSize: 11 }}>vision model — takes ~30s</span>
          </div>
        )}

        {phase === "analyzing" && (
          <div style={{
            position: "absolute", inset: 0,
            backgroundColor: "rgba(0,0,0,0.7)",
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 16
          }}>
            <div style={{
              width: 44, height: 44,
              border: "2px solid #8B1A1A", borderTopColor: "transparent",
              borderRadius: "50%", animation: "lens-spin 0.8s linear infinite"
            }} />
            <span style={{ color: "#ECECEC", fontSize: 13 }}>analyzing garment...</span>
          </div>
        )}

        {error && (
          <div style={{
            position: "absolute", bottom: 12, left: 16, right: 16,
            backgroundColor: "rgba(139,26,26,0.92)", borderRadius: 12,
            padding: "10px 16px", color: "#ECECEC", fontSize: 13, textAlign: "center"
          }}>
            {error}
          </div>
        )}
      </div>

      {/* Item chips */}
      {phase === "items" && items.length > 0 && (
        <div style={{
          padding: "12px 16px 20px",
          backgroundColor: "rgba(10,10,10,0.95)",
          borderTop: "1px solid #222220"
        }}>
          <p style={{
            color: "#555552", fontSize: 10, letterSpacing: "0.1em",
            textTransform: "uppercase", margin: "0 0 10px"
          }}>
            tap to analyze
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {items.map(item => (
              <button
                key={item.id}
                onClick={() => handleItemTap(item)}
                style={{
                  padding: "7px 14px", borderRadius: 999,
                  border: "1px solid #333330", backgroundColor: "transparent",
                  color: "#ECECEC", fontSize: 12, cursor: "pointer",
                  maxWidth: 240, overflow: "hidden",
                  textOverflow: "ellipsis", whiteSpace: "nowrap"
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Capture button */}
      {phase === "live" && (
        <div style={{
          padding: "20px 16px 36px",
          display: "flex", justifyContent: "center",
          backgroundColor: "rgba(0,0,0,0.4)"
        }}>
          <button
            onClick={capture}
            style={{
              width: 68, height: 68, borderRadius: "50%",
              border: "3px solid #ECECEC",
              backgroundColor: "rgba(255,255,255,0.1)",
              cursor: "pointer", display: "flex",
              alignItems: "center", justifyContent: "center"
            }}
          >
            <div style={{
              width: 52, height: 52, borderRadius: "50%",
              backgroundColor: "#ECECEC"
            }} />
          </button>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: "none" }} />

      <style>{`
        @keyframes lens-spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

// ─── BRIEF HELPERS ────────────────────────────────────
function dig(obj: unknown, ...keys: string[]): string {
  let cur: unknown = obj
  for (const k of keys) {
    if (cur == null || typeof cur !== "object") return "—"
    cur = (cur as Record<string, unknown>)[k]
  }
  if (cur == null) return "—"
  if (typeof cur === "object") {
    const v = (cur as Record<string, unknown>).value
    return v != null ? String(v) : "—"
  }
  return String(cur)
}

function BriefSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <p style={{
        color: "#555552", fontSize: 10, letterSpacing: "0.1em",
        textTransform: "uppercase", margin: "0 0 10px"
      }}>
        {title}
      </p>
      <div style={{
        backgroundColor: "#242422", borderRadius: 12,
        border: "1px solid #2E2E2C", overflow: "hidden"
      }}>
        {children}
      </div>
    </div>
  )
}

function BriefRow({ label, value }: { label: string; value: string }) {
  if (!value || value === "—") return null
  return (
    <div style={{
      display: "flex", justifyContent: "space-between",
      alignItems: "flex-start", padding: "11px 16px",
      borderBottom: "1px solid #2A2A28"
    }}>
      <span style={{ color: "#666663", fontSize: 13, flexShrink: 0, marginRight: 16 }}>
        {label}
      </span>
      <span style={{ color: "#ECECEC", fontSize: 13, textAlign: "right", lineHeight: 1.5 }}>
        {value}
      </span>
    </div>
  )
}

// ─── BRIEF VIEW ────────────────────────────────────────
function BriefView({ brief, onBack }: { brief: AnalyzeResult; onBack: () => void }) {
  const ga = brief.garment_analysis
  const fi = brief.fabric_intelligence

  const constructionSteps: string[] = Array.isArray((ga as Record<string, unknown>)?.construction_sequence)
    ? (ga as Record<string, string[]>).construction_sequence
    : Array.isArray((fi as Record<string, unknown>)?.construction_sequence)
    ? (fi as Record<string, string[]>).construction_sequence
    : []

  const sourcing: string[] = (() => {
    const s = (fi as Record<string, unknown>)?.sourcing
    if (!s || typeof s !== "object") return []
    const src = s as Record<string, unknown>
    return Array.isArray(src.bengaluru_specific)
      ? src.bengaluru_specific as string[]
      : Array.isArray(src.markets)
      ? src.markets as string[]
      : []
  })()

  const compat = brief.profile_compatibility

  return (
    <div style={{
      minHeight: "100svh", backgroundColor: "#1A1A1A",
      fontFamily: "system-ui, sans-serif", overflowY: "auto"
    }}>
      {/* Header */}
      <div style={{
        height: 48, display: "flex", alignItems: "center",
        padding: "0 16px", position: "sticky", top: 0,
        backgroundColor: "#1A1A1A", borderBottom: "1px solid #222220", zIndex: 10
      }}>
        <button onClick={onBack} style={{
          background: "none", border: "none",
          color: "#666663", cursor: "pointer", fontSize: 18, padding: 4
        }}>
          ←
        </button>
        <span style={{
          flex: 1, textAlign: "center",
          color: "#ECECEC", fontSize: 12, letterSpacing: "0.1em"
        }}>
          TAILOR BRIEF
        </span>
        <div style={{ width: 28 }} />
      </div>

      <div style={{ maxWidth: 640, margin: "0 auto", padding: "24px 16px 80px" }}>
        {/* Title */}
        <h1 style={{
          fontSize: 20, fontWeight: 400, color: "#ECECEC",
          fontFamily: "Georgia, serif", margin: "0 0 4px", lineHeight: 1.4
        }}>
          {brief.item_label}
        </h1>
        <p style={{ color: "#555552", fontSize: 13, margin: "0 0 28px" }}>
          {dig(ga, "garment_type")}
        </p>

        {/* Compatibility */}
        <div style={{
          padding: "13px 16px", borderRadius: 12, marginBottom: 28,
          backgroundColor: "#242422",
          border: "1px solid #2E2E2C"
        }}>
          <p style={{ color: "#ECECEC", fontSize: 13, margin: 0, lineHeight: 1.65 }}>
            {compat.compatible ? "✓ " : "✗ "}{compat.reason}
          </p>
        </div>

        {/* Fabric */}
        <BriefSection title="Fabric">
          <BriefRow label="Fiber" value={dig(ga, "fabric", "fiber_type")} />
          <BriefRow label="Weight" value={
            dig(ga, "fabric", "gsm") !== "—"
              ? `${dig(ga, "fabric", "gsm")} GSM`
              : "—"
          } />
          <BriefRow label="Weave" value={dig(ga, "fabric", "weave")} />
          <BriefRow label="Finish" value={dig(ga, "fabric", "finish")} />
          <BriefRow label="Primary color" value={dig(ga, "fabric", "primary_color")} />
        </BriefSection>

        {/* Silhouette */}
        <BriefSection title="Silhouette">
          <BriefRow label="Fit" value={dig(ga, "silhouette", "fit")} />
          <BriefRow label="Shape" value={dig(ga, "silhouette", "overall_shape")} />
          <BriefRow label="Hem" value={dig(ga, "silhouette", "hem_line")} />
        </BriefSection>

        {/* Sourcing */}
        {sourcing.length > 0 && (
          <BriefSection title="Where to source in Bengaluru">
            {sourcing.map((s, i) => (
              <BriefRow key={i} label={`Option ${i + 1}`} value={s} />
            ))}
          </BriefSection>
        )}

        {/* Construction steps */}
        {constructionSteps.length > 0 && (
          <BriefSection title={`Construction (${constructionSteps.length} steps)`}>
            <div style={{ padding: "12px 16px" }}>
              {constructionSteps.map((step, i) => (
                <div key={i} style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                  <span style={{
                    width: 20, height: 20, borderRadius: "50%",
                    backgroundColor: "#1A1A1A", border: "1px solid #333330",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 10, color: "#666663", flexShrink: 0, marginTop: 2
                  }}>
                    {i + 1}
                  </span>
                  <p style={{ color: "#ECECEC", fontSize: 13, lineHeight: 1.65, margin: 0 }}>
                    {step}
                  </p>
                </div>
              ))}
            </div>
          </BriefSection>
        )}

        {/* CTA */}
        <button style={{
          width: "100%", padding: "14px", borderRadius: 12,
          border: "none", backgroundColor: "#8B1A1A",
          color: "#ECECEC", fontSize: 14, cursor: "pointer",
          letterSpacing: "0.03em", marginTop: 8
        }}>
          Get this made →
        </button>
      </div>
    </div>
  )
}

// ─── MAIN PAGE ─────────────────────────────────────────
export default function TailorPage() {
  const [state, setState] = useState<"idle" | "chat" | "camera" | "brief">("idle")
  const [firstMessage, setFirstMessage] = useState("")
  const [briefData, setBriefData] = useState<AnalyzeResult | null>(null)

  const handleFirstSend = (msg: string, _img?: string) => {
    setFirstMessage(msg)
    setState("chat")
  }

  const handleBriefReady = (brief: AnalyzeResult) => {
    setBriefData(brief)
    setState("brief")
  }

  return (
    <>
      <AnimatePresence mode="wait">
        {state === "idle" && (
          <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.25 }}>
            <IdleView onSend={handleFirstSend} />
          </motion.div>
        )}
        {state === "chat" && (
          <motion.div key="chat" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.25 }} style={{ height: "100svh" }}>
            <ChatView
              initialMessage={firstMessage}
              onBack={() => setState("idle")}
              onCameraOpen={() => setState("camera")}
              onTailorFlow={() => setState("camera")}
            />
          </motion.div>
        )}
        {state === "camera" && (
          <motion.div key="camera" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}>
            <CameraView
              onClose={() => setState("chat")}
              onBriefReady={handleBriefReady}
            />
          </motion.div>
        )}
        {state === "brief" && briefData && (
          <motion.div key="brief" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}>
            <BriefView
              brief={briefData}
              onBack={() => setState("chat")}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
