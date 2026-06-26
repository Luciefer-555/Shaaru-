"use client";

import {
  useRef,
  useState,
  useCallback,
  type ChangeEvent,
  type KeyboardEvent,
} from "react";
import { Paperclip, ArrowUp, X, Camera, Pencil, Scissors, Sparkles } from "lucide-react";
import { cn } from "@/lib/cn";

interface Suggestion {
  id: string;
  label: string;
  icon: React.ReactNode;
  starter: string;
}

const SUGGESTIONS: Suggestion[] = [
  {
    id: "1",
    label: "Make from reference",
    icon: <Camera className="w-3.5 h-3.5" />,
    starter: "I have a reference image I'd like to recreate. ",
  },
  {
    id: "2",
    label: "Describe a garment",
    icon: <Pencil className="w-3.5 h-3.5" />,
    starter: "I want to make a ",
  },
  {
    id: "3",
    label: "Repurpose old clothes",
    icon: <Scissors className="w-3.5 h-3.5" />,
    starter: "I have some old clothes I'd like to repurpose into ",
  },
  {
    id: "4",
    label: "Wedding outfit",
    icon: <Sparkles className="w-3.5 h-3.5" />,
    starter: "I need a wedding outfit — ",
  },
];

interface TailorInputProps {
  onSend: (message: string, imageBase64?: string) => void;
  disabled?: boolean;
  hideSuggestions?: boolean;
  placeholder?: string;
  onCameraOpen?: () => void;
}

export function TailorInput({ onSend, disabled, hideSuggestions, placeholder, onCameraOpen }: TailorInputProps) {
  const [message, setMessage] = useState("");
  const [imageBase64, setImageBase64] = useState<string | undefined>();
  const [imagePreview, setImagePreview] = useState<string | undefined>();
  const [imageName, setImageName] = useState<string | undefined>();
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, []);

  const handleFileChange = useCallback((file: File) => {
    if (!file.type.match(/image\/(jpeg|png|webp)/)) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setImagePreview(result);
      setImageBase64(result.split(",")[1]);
      setImageName(file.name);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileChange(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileChange(file);
  };

  const handleSend = () => {
    if (!message.trim() && !imageBase64) return;
    onSend(message.trim(), imageBase64);
    setMessage("");
    setImageBase64(undefined);
    setImagePreview(undefined);
    setImageName(undefined);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (starter: string) => {
    setMessage(starter);
    textareaRef.current?.focus();
    setTimeout(adjustHeight, 10);
  };

  const canSend = (message.trim().length > 0 || !!imageBase64) && !disabled;

  return (
    <div className="w-full max-w-2xl flex flex-col gap-3">
      {/* Suggestion Pills */}
      {!hideSuggestions && (
        <div className="flex flex-wrap gap-2 justify-center">
          {SUGGESTIONS.map((s) => (
            <button
              key={s.id}
              onClick={() => handleSuggestion(s.starter)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300",
                "border border-glass-border hover:border-shaaru-crimson/80",
                "bg-glass-bg hover:bg-shaaru-crimson/10 backdrop-blur-md",
                "text-text-secondary-dark hover:text-white",
              )}
            >
              {s.icon}
              {s.label}
            </button>
          ))}
        </div>
      )}

      {/* Input Box */}
      <div
        className={cn(
          "relative rounded-2xl transition-all duration-300 shadow-2xl",
          "bg-glass-bg backdrop-blur-xl",
          "border border-glass-border",
          isDragging && "border-shaaru-crimson ring-2 ring-shaaru-crimson/30",
          disabled && "opacity-60 pointer-events-none",
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {/* Image Preview */}
        {imagePreview && (
          <div className="flex items-center gap-2 px-4 pt-3">
            <div className="relative group">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imagePreview}
                alt="Upload preview"
                className="w-14 h-14 rounded-lg object-cover border border-white/10"
              />
              <button
                onClick={() => {
                  setImageBase64(undefined);
                  setImagePreview(undefined);
                  setImageName(undefined);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-surface-dark border border-white/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <X className="w-3 h-3 text-text-secondary-dark" />
              </button>
            </div>
            {imageName && (
              <span className="text-xs text-text-secondary-dark truncate max-w-[140px]">
                {imageName}
              </span>
            )}
          </div>
        )}

        <div className="flex items-end px-4 py-3 gap-3">
          {/* Attach Button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-text-secondary-dark hover:text-text-primary-dark hover:bg-white/8 transition-all duration-150 mb-0.5"
            title="Attach image"
          >
            <Paperclip className="w-4 h-4" />
          </button>

          {/* Camera Button */}
          <button
            onClick={() => onCameraOpen?.()}
            className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-text-secondary-dark hover:text-text-primary-dark hover:bg-white/8 transition-all duration-150 mb-0.5"
            title="Take photo"
          >
            <Camera className="w-4 h-4" />
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={handleInputChange}
          />

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => {
              setMessage(e.target.value);
              adjustHeight();
            }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder || "show me what you want to make..."}
            rows={1}
            className={cn(
              "flex-1 resize-none bg-transparent outline-none",
              "text-sm text-text-primary-dark dark:text-text-primary-dark",
              "placeholder:text-text-secondary-dark/60",
              "min-h-[28px] max-h-[200px] leading-relaxed",
              "font-['DM_Sans',_sans-serif]",
            )}
          />

          {/* Send Button */}
          <button
            onClick={handleSend}
            disabled={!canSend}
            className={cn(
              "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 mb-0.5",
              canSend
                ? "bg-gradient-to-r from-shaaru-crimson to-rose-500 hover:from-shaaru-crimson-hover hover:to-rose-600 text-white shadow-[0_0_15px_rgba(255,26,64,0.4)]"
                : "bg-white/5 text-text-secondary-dark/40 cursor-not-allowed",
            )}
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
