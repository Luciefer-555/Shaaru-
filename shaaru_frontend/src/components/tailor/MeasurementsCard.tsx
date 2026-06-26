"use client";

import { Ruler } from "lucide-react";
import { type ApiMeasurements } from "@/lib/tailor-api";

const MEASUREMENT_KEYS: { key: string; label: string }[] = [
  { key: "inseam", label: "Inseam" },
  { key: "outseam", label: "Outseam" },
  { key: "rise_front", label: "Rise Front" },
  { key: "rise_back", label: "Rise Back" },
  { key: "thigh_circumference", label: "Thigh" },
  { key: "knee_circumference", label: "Knee" },
  { key: "leg_opening", label: "Leg Opening" },
];

interface MeasurementsCardProps {
  measurements: ApiMeasurements;
}

export function MeasurementsCard({ measurements }: MeasurementsCardProps) {
  const cm = measurements.measurements_cm ?? {};

  // Build height string from height_cm or height_ft
  const heightStr =
    measurements.height_cm
      ? `${measurements.height_cm} cm`
      : measurements.height_ft
      ? `${measurements.height_ft} ft`
      : null;

  return (
    <div className="rounded-2xl bg-glass-bg border border-glass-border backdrop-blur-xl shadow-2xl overflow-hidden relative">
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
      {/* Header */}
      <div className="px-5 py-4 border-b border-white/6">
        <div className="flex items-center gap-2">
          <Ruler className="w-4 h-4 text-shaaru-crimson" />
          <h3 className="text-sm font-semibold text-text-primary-dark font-['DM_Sans',_sans-serif] tracking-wide uppercase">
            Measurements
          </h3>
        </div>
      </div>

      <div className="px-5 py-4 flex flex-col gap-4">
        {/* Profile row */}
        {(heightStr || measurements.gender || measurements.fabric_meters_needed) && (
          <div className="flex items-center gap-4 pb-3 border-b border-white/6 flex-wrap">
            {heightStr && (
              <div className="flex flex-col">
                <span className="text-xs text-text-secondary-dark/60 font-['DM_Sans',_sans-serif] uppercase tracking-wider">
                  Height
                </span>
                <span className="text-sm font-medium text-text-primary-dark font-['JetBrains_Mono',_monospace]">
                  {heightStr}
                </span>
              </div>
            )}
            {measurements.gender && (
              <div className="flex flex-col">
                <span className="text-xs text-text-secondary-dark/60 font-['DM_Sans',_sans-serif] uppercase tracking-wider">
                  Build
                </span>
                <span className="text-sm font-medium text-text-primary-dark font-['JetBrains_Mono',_monospace]">
                  {measurements.gender}
                </span>
              </div>
            )}
            {measurements.fabric_meters_needed != null && (
              <div className="flex flex-col">
                <span className="text-xs text-text-secondary-dark/60 font-['DM_Sans',_sans-serif] uppercase tracking-wider">
                  Fabric
                </span>
                <span className="text-sm font-medium text-text-primary-dark font-['JetBrains_Mono',_monospace]">
                  {measurements.fabric_meters_needed}m
                </span>
              </div>
            )}
          </div>
        )}

        {/* Measurement grid from measurements_cm */}
        {Object.keys(cm).length > 0 && (
          <div className="grid grid-cols-2 gap-x-4 gap-y-3">
            {MEASUREMENT_KEYS.map(({ key, label }) => {
              const value = cm[key];
              if (value == null) return null;
              return (
                <div key={key} className="flex flex-col gap-0.5">
                  <span className="text-xs text-text-secondary-dark/60 font-['DM_Sans',_sans-serif] uppercase tracking-wider">
                    {label}
                  </span>
                  <span className="text-sm font-medium text-text-primary-dark font-['JetBrains_Mono',_monospace]">
                    {value} cm
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
