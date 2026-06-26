"use client";

import { MapPin, ShoppingBag, DollarSign } from "lucide-react";
import { type ApiSourcingInfo } from "@/lib/tailor-api";

interface SourcingCardProps {
  sourcing: ApiSourcingInfo;
}

export function SourcingCard({ sourcing }: SourcingCardProps) {
  return (
    <div className="rounded-2xl bg-surface-dark border border-white/8 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/6">
        <div className="flex items-center gap-2">
          <ShoppingBag className="w-4 h-4 text-shaaru-crimson" />
          <h3 className="text-sm font-semibold text-text-primary-dark font-['DM_Sans',_sans-serif] tracking-wide uppercase">
            Where to Get It
          </h3>
        </div>
      </div>

      <div className="px-5 py-4 flex flex-col gap-4">
        {/* Fabric Sourcing */}
        {(sourcing.fabric_market || sourcing.fabric_ask_for) && (
          <div className="flex flex-col gap-2">
            {sourcing.fabric_market && (
              <div className="flex items-center gap-1.5 text-text-secondary-dark text-xs uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                <MapPin className="w-3 h-3 flex-shrink-0" />
                {sourcing.fabric_market}
              </div>
            )}
            {sourcing.fabric_ask_for && (
              <p className="text-xs text-text-secondary-dark/70 font-['DM_Sans',_sans-serif]">
                Ask for:{" "}
                <span className="text-shaaru-crimson font-medium">
                  {sourcing.fabric_ask_for}
                </span>
              </p>
            )}
            {sourcing.fabric_price_range && (
              <span className="px-2 py-0.5 rounded-md bg-shaaru-crimson/10 text-shaaru-crimson text-xs font-medium font-['JetBrains_Mono',_monospace] inline-block w-fit">
                {sourcing.fabric_price_range}
              </span>
            )}
          </div>
        )}

        {/* Embellishment Sourcing */}
        {(sourcing.embellishment_market || sourcing.embellishment_ask_for) && (
          <>
            <div className="border-t border-white/6" />
            <div className="flex flex-col gap-2">
              {sourcing.embellishment_market && (
                <div className="flex items-center gap-1.5 text-text-secondary-dark text-xs uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                  <MapPin className="w-3 h-3 flex-shrink-0" />
                  {sourcing.embellishment_market}
                </div>
              )}
              {sourcing.embellishment_ask_for && (
                <p className="text-xs text-text-secondary-dark/70 font-['DM_Sans',_sans-serif]">
                  Ask for:{" "}
                  <span className="text-shaaru-crimson font-medium">
                    {sourcing.embellishment_ask_for}
                  </span>
                </p>
              )}
              {sourcing.embellishment_price_range && (
                <span className="px-2 py-0.5 rounded-md bg-shaaru-crimson/10 text-shaaru-crimson text-xs font-medium font-['JetBrains_Mono',_monospace] inline-block w-fit">
                  {sourcing.embellishment_price_range}
                </span>
              )}
            </div>
          </>
        )}

        {/* Total */}
        {sourcing.total_cost_estimate && (
          <>
            <div className="border-t border-white/6" />
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-text-secondary-dark text-xs font-['DM_Sans',_sans-serif] uppercase tracking-wider">
                <DollarSign className="w-3 h-3" />
                Total Estimate
              </div>
              <span className="text-base font-semibold text-text-primary-dark font-['JetBrains_Mono',_monospace]">
                {sourcing.total_cost_estimate}
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
