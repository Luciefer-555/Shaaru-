"use client";

import { useState } from "react";
import { CheckSquare, Square, ClipboardCheck } from "lucide-react";
import { cn } from "@/lib/cn";

// quality_checkpoints from the backend is string[] (plain labels, no id objects)
interface QualityCheckpointsProps {
  checkpoints: string[];
}

export function QualityCheckpoints({ checkpoints }: QualityCheckpointsProps) {
  const [checked, setChecked] = useState<Record<number, boolean>>({});

  const toggle = (idx: number) =>
    setChecked((prev) => ({ ...prev, [idx]: !prev[idx] }));

  const allDone =
    checkpoints.length > 0 && checkpoints.every((_, i) => checked[i]);

  return (
    <div className="rounded-2xl bg-surface-dark border border-white/8 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ClipboardCheck className="w-4 h-4 text-shaaru-crimson" />
            <h3 className="text-sm font-semibold text-text-primary-dark font-['DM_Sans',_sans-serif] tracking-wide uppercase">
              Quality Checkpoints
            </h3>
          </div>
          {allDone && (
            <span className="text-xs text-emerald-400 font-['DM_Sans',_sans-serif]">
              All clear ✓
            </span>
          )}
        </div>
      </div>

      <div className="px-5 py-4 flex flex-col gap-3">
        {checkpoints.map((label, idx) => (
          <button
            key={idx}
            onClick={() => toggle(idx)}
            className="flex items-start gap-3 text-left group transition-colors"
          >
            {checked[idx] ? (
              <CheckSquare className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5 transition-colors" />
            ) : (
              <Square className="w-4 h-4 text-text-secondary-dark/40 flex-shrink-0 mt-0.5 group-hover:text-text-secondary-dark/70 transition-colors" />
            )}
            <span
              className={cn(
                "text-sm font-['DM_Sans',_sans-serif] transition-colors leading-relaxed",
                checked[idx]
                  ? "text-text-secondary-dark/50 line-through"
                  : "text-text-secondary-dark",
              )}
            >
              {label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
