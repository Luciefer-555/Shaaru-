"use client";

import { useState, useRef, useCallback, type KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, X } from "lucide-react";
import { cn } from "@/lib/cn";

interface ChatContinuationProps {
  open: boolean;
  onClose: () => void;
  onSend: (message: string) => void;
  disabled?: boolean;
}

export function ChatContinuation({
  open,
  onClose,
  onSend,
  disabled,
}: ChatContinuationProps) {
  const [message, setMessage] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  const handleSend = () => {
    if (!message.trim() || disabled) return;
    onSend(message.trim());
    setMessage("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="fixed bottom-6 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 z-50"
        >
          <div className="relative rounded-2xl bg-surface-dark border border-white/12 shadow-2xl shadow-black/40">
            {/* Dismiss */}
            <button
              onClick={onClose}
              className="absolute top-3 right-3 w-7 h-7 rounded-lg flex items-center justify-center text-text-secondary-dark/60 hover:text-text-secondary-dark hover:bg-white/8 transition-all"
            >
              <X className="w-3.5 h-3.5" />
            </button>

            <div className="px-4 pt-3 pb-1">
              <p className="text-xs text-text-secondary-dark/50 font-['DM_Sans',_sans-serif]">
                Refine your brief · changes update in place
              </p>
            </div>

            <div className="flex items-end px-4 pb-3 gap-3">
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => {
                  setMessage(e.target.value);
                  adjustHeight();
                }}
                onKeyDown={handleKeyDown}
                placeholder='e.g. "make it a mandarin collar instead"'
                rows={1}
                autoFocus
                className={cn(
                  "flex-1 resize-none bg-transparent outline-none",
                  "text-sm text-text-primary-dark",
                  "placeholder:text-text-secondary-dark/50",
                  "min-h-[28px] max-h-[160px] leading-relaxed",
                  "font-['DM_Sans',_sans-serif]",
                )}
              />
              <button
                onClick={handleSend}
                disabled={!message.trim() || disabled}
                className={cn(
                  "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center transition-all mb-0.5",
                  message.trim() && !disabled
                    ? "bg-shaaru-crimson hover:bg-shaaru-crimson-hover text-white"
                    : "bg-white/8 text-text-secondary-dark/40 cursor-not-allowed",
                )}
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
