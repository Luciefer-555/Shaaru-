"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SharuLogo } from "@/components/ui/SharuLogo";

const SHAARU_LOADING_MESSAGES = [
  "ohhh lovely, let me cook on this real quick ✨",
  "ur cooking and so am i, hold on bestie",
  "scanning the fabric matrix rn don't move",
  "okay okay okay i see the vision",
  "pulling from 194 fabrics for you specifically",
  "asking my tailor brain to work overtime",
  "constructing your blueprint, this is the fun part",
  "cross-referencing markets in bengaluru rn",
  "the brief is taking shape, you're gonna love this",
  "almost there, just getting the measurements perfect",
];

export function TailorLoading() {
  const [msgIndex, setMsgIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const cycle = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setMsgIndex((i) => (i + 1) % SHAARU_LOADING_MESSAGES.length);
        setVisible(true);
      }, 350);
    }, 3000);

    return () => clearInterval(cycle);
  }, []);

  return (
    <div className="flex flex-col items-center gap-8">
      {/* Logo */}
      <SharuLogo size={72} animate />

      {/* Breadcrumb / message */}
      <div className="flex flex-col items-center gap-4">
        {/* Dots */}
        <div className="flex items-center gap-1.5">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-shaaru-crimson"
              style={{
                animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
              }}
            />
          ))}
        </div>

        {/* Cycling message */}
        <div className="h-7 flex items-center justify-center">
          <AnimatePresence mode="wait">
            {visible && (
              <motion.p
                key={msgIndex}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="text-sm text-text-secondary-dark text-center font-['DM_Sans',_sans-serif] tracking-wide"
              >
                {SHAARU_LOADING_MESSAGES[msgIndex]}
              </motion.p>
            )}
          </AnimatePresence>
        </div>

        {/* Subtle label */}
        <p className="text-xs text-text-secondary-dark/40 font-['DM_Sans',_sans-serif] tracking-widest uppercase">
          SHAARU is thinking
        </p>
      </div>
    </div>
  );
}
