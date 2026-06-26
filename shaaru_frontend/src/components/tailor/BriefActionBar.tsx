"use client";

import { Download, RotateCcw, MessageSquare } from "lucide-react";
import { cn } from "@/lib/cn";

interface BriefActionBarProps {
  onStartOver: () => void;
  onAskShaaru: () => void;
  onDownload?: () => void;
}

export function BriefActionBar({
  onStartOver,
  onAskShaaru,
  onDownload,
}: BriefActionBarProps) {
  return (
    <div className="flex items-center justify-center gap-3 flex-wrap">
      <button
        onClick={onDownload}
        className={cn(
          "flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-150",
          "border border-white/10 bg-white/5 hover:bg-white/10 text-text-secondary-dark hover:text-text-primary-dark",
          "font-['DM_Sans',_sans-serif]",
        )}
      >
        <Download className="w-4 h-4" />
        Download Brief
      </button>

      <button
        onClick={onStartOver}
        className={cn(
          "flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-150",
          "border border-white/10 bg-white/5 hover:bg-white/10 text-text-secondary-dark hover:text-text-primary-dark",
          "font-['DM_Sans',_sans-serif]",
        )}
      >
        <RotateCcw className="w-4 h-4" />
        Start Over
      </button>

      <button
        onClick={onAskShaaru}
        className={cn(
          "flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-150",
          "bg-shaaru-crimson hover:bg-shaaru-crimson-hover text-white shadow-sm",
          "font-['DM_Sans',_sans-serif]",
        )}
      >
        <MessageSquare className="w-4 h-4" />
        Ask SHAARU
      </button>
    </div>
  );
}
