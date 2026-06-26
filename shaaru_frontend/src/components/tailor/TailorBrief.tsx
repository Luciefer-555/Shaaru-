"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, Gem } from "lucide-react";
import { cn } from "@/lib/cn";
import { useAnimatedText } from "@/lib/use-animated-text";
import { type TailorBrief } from "@/lib/tailor-api";
import { SourcingCard } from "./SourcingCard";
import { MeasurementsCard } from "./MeasurementsCard";
import { QualityCheckpoints } from "./QualityCheckpoints";

// ─── Animated Section Wrapper ─────────────────────────────────────────────────
function Section({
  children,
  delay = 0,
  className,
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold text-text-secondary-dark/50 uppercase tracking-widest font-['DM_Sans',_sans-serif] mb-3">
      {children}
    </p>
  );
}

// ─── Opening Message (streams char-by-char) ───────────────────────────────────
function OpeningMessage({
  text,
  onComplete,
}: {
  text: string;
  onComplete: () => void;
}) {
  const { displayedText, isComplete } = useAnimatedText(text, {
    speed: 40,
    onComplete,
  });

  return (
    <div className="mb-8 p-4 rounded-r-xl bg-gradient-to-r from-shaaru-crimson/10 to-transparent border-l-2 border-shaaru-crimson shadow-lg backdrop-blur-sm">
      <p className="text-base text-text-primary-dark leading-relaxed font-['DM_Sans',_sans-serif] italic">
        {displayedText}
        {!isComplete && (
          <span className="inline-block w-0.5 h-4 bg-shaaru-crimson ml-0.5 animate-pulse align-middle shadow-[0_0_8px_rgba(255,26,64,0.8)]" />
        )}
      </p>
    </div>
  );
}

// ─── Garment Section ──────────────────────────────────────────────────────────
function GarmentSection({ brief }: { brief: TailorBrief }) {
  return (
    <Section delay={0.1}>
      <SectionLabel>Garment</SectionLabel>
      <h1 className="text-2xl md:text-3xl font-light text-text-primary-dark font-['Playfair_Display',_serif] mb-3 leading-tight">
        {brief.garment_name ?? "Unnamed Garment"}
      </h1>
      {brief.reference_description && (
        <p className="text-sm text-text-secondary-dark leading-relaxed font-['DM_Sans',_sans-serif] mb-2">
          {brief.reference_description}
        </p>
      )}
      {brief.modification_summary && (
        <div className="flex items-start gap-2 mt-3 p-3 rounded-lg bg-glass-bg border border-glass-border backdrop-blur-md shadow-sm">
          <span className="text-shaaru-crimson text-xs mt-0.5">✦</span>
          <p className="text-xs text-text-secondary-dark/90 font-['DM_Sans',_sans-serif] leading-relaxed">
            {brief.modification_summary}
          </p>
        </div>
      )}
      {/* Tailor instructions banner */}
      {brief.tailor_instructions && (
        <div className="mt-3 p-3 rounded-lg bg-glass-bg border border-glass-border backdrop-blur-md shadow-sm">
          <p className="text-xs text-text-secondary-dark/80 font-['DM_Sans',_sans-serif] leading-relaxed">
            <span className="text-text-primary-dark font-medium">For your tailor: </span>
            {brief.tailor_instructions}
          </p>
        </div>
      )}
    </Section>
  );
}

// ─── Fabric Spec Section ──────────────────────────────────────────────────────
function FabricSection({ brief }: { brief: TailorBrief }) {
  const spec = brief.fabric_spec;
  if (!spec) return null;

  // Normalize — backend uses `fabric` not `fabric_name`
  const fabricName = spec.fabric;
  const gsmRaw = spec.gsm;
  const weave = spec.weave;

  if (!fabricName && !gsmRaw && !weave) return null;

  return (
    <Section delay={0.2}>
      <SectionLabel>Fabric Specification</SectionLabel>
      <div className="rounded-xl bg-glass-bg border border-glass-border backdrop-blur-md shadow-lg p-4 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-shaaru-crimson to-rose-500" />
        <div className="flex items-start justify-between gap-3 mb-3">
          <h3 className="text-base font-semibold text-text-primary-dark font-['DM_Sans',_sans-serif]">
            {fabricName}
          </h3>
          {gsmRaw && (
            <span className="flex-shrink-0 px-2 py-0.5 rounded-md bg-shaaru-crimson/15 text-shaaru-crimson text-xs font-['JetBrains_Mono',_monospace] font-medium whitespace-nowrap">
              {gsmRaw}
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2.5">
          {weave && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Weave
              </p>
              <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif]">
                {weave}
              </p>
            </div>
          )}
          {spec.hand_feel && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Hand Feel
              </p>
              <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif]">
                {spec.hand_feel}
              </p>
            </div>
          )}
          {spec.meters_needed != null && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Metres Needed
              </p>
              <p className="text-sm text-text-primary-dark font-['JetBrains_Mono',_monospace] font-medium">
                {spec.meters_needed}m
              </p>
            </div>
          )}
          {spec.color && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Color
              </p>
              <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif]">
                {spec.color}
              </p>
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}

// ─── Construction Sequence ────────────────────────────────────────────────────
// Backend sends construction_sequence as plain string[], not ConstructionStep[]
function ConstructionSection({
  brief,
  onComplete,
}: {
  brief: TailorBrief;
  onComplete?: () => void;
}) {
  // Prefer construction_sequence (new API), fall back to nothing
  const steps: string[] = brief.construction_sequence ?? [];
  const [visibleCount, setVisibleCount] = useState(0);

  useEffect(() => {
    if (visibleCount < steps.length) {
      const t = setTimeout(() => setVisibleCount((n) => n + 1), 150);
      return () => clearTimeout(t);
    } else {
      onComplete?.();
    }
  }, [visibleCount, steps.length, onComplete]);

  if (steps.length === 0) return null;

  return (
    <Section delay={0.3}>
      <SectionLabel>Construction Sequence</SectionLabel>
      <div className="flex flex-col gap-2.5">
        {steps.slice(0, visibleCount).map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="flex items-start gap-3 p-3 rounded-lg bg-glass-bg border border-glass-border backdrop-blur-sm shadow-sm"
          >
            <span className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold font-['JetBrains_Mono',_monospace] bg-gradient-to-br from-shaaru-crimson to-rose-500 text-white shadow-[0_0_10px_rgba(255,26,64,0.3)]">
              {i + 1}
            </span>
            <p className="text-sm text-text-secondary-dark leading-relaxed font-['DM_Sans',_sans-serif] flex-1">
              {step}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Critical Points */}
      {brief.critical_points && brief.critical_points.length > 0 && (
        <div className="mt-4 p-4 rounded-lg bg-amber-500/6 border border-amber-500/20">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
            <p className="text-xs font-semibold text-amber-400 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
              Critical Points
            </p>
          </div>
          <ul className="flex flex-col gap-1.5">
            {brief.critical_points.map((pt, i) => (
              <li
                key={i}
                className="text-xs text-amber-400/70 font-['DM_Sans',_sans-serif] leading-relaxed flex items-start gap-2"
              >
                <span className="flex-shrink-0 text-amber-500 mt-0.5">·</span>
                {pt}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Pressing Sequence */}
      {brief.pressing_sequence && brief.pressing_sequence.length > 0 && (
        <div className="mt-4 p-4 rounded-lg bg-glass-bg border border-glass-border backdrop-blur-sm">
          <p className="text-xs font-semibold text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif] mb-2">
            Pressing Sequence
          </p>
          <ol className="flex flex-col gap-1.5">
            {brief.pressing_sequence.map((step, i) => (
              <li
                key={i}
                className="text-xs text-text-secondary-dark/70 font-['DM_Sans',_sans-serif] leading-relaxed flex items-start gap-2"
              >
                <span className="flex-shrink-0 text-text-secondary-dark/40 font-['JetBrains_Mono',_monospace] text-[11px] mt-0.5">
                  {i + 1}.
                </span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}
    </Section>
  );
}

// ─── Additional Specs (all flat fields on brief) ──────────────────────────────
function AdditionalSpecsSection({ brief }: { brief: TailorBrief }) {
  const rows: { label: string; value: string }[] = [
    brief.grain_direction
      ? { label: "Grain Direction", value: brief.grain_direction }
      : null,
    brief.pressing_temperature
      ? { label: "Pressing Temp", value: brief.pressing_temperature }
      : null,
    brief.fabric_prep
      ? { label: "Fabric Prep", value: brief.fabric_prep }
      : null,
    brief.interfacing_spec
      ? { label: "Interfacing", value: brief.interfacing_spec }
      : null,
    brief.lining_spec
      ? { label: "Lining", value: brief.lining_spec }
      : null,
    brief.embellishment_timing
      ? { label: "Embellishment Timing", value: brief.embellishment_timing }
      : null,
    brief.estimated_construction_time
      ? { label: "Est. Time", value: brief.estimated_construction_time }
      : null,
  ].filter(Boolean) as { label: string; value: string }[];

  if (rows.length === 0) return null;

  return (
    <Section delay={0.4}>
      <SectionLabel>Additional Specs</SectionLabel>
      <div className="rounded-xl bg-glass-bg border border-glass-border backdrop-blur-md shadow-lg overflow-hidden">
        {rows.map((row, i) => (
          <div
            key={row.label}
            className={cn(
              "flex items-start gap-4 px-4 py-3",
              i < rows.length - 1 && "border-b border-white/5",
            )}
          >
            <p className="w-36 flex-shrink-0 text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif] pt-0.5">
              {row.label}
            </p>
            <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif] leading-relaxed">
              {row.value}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ─── Embellishment (from embellishment_brief) ─────────────────────────────────
function EmbellishmentSection({ brief }: { brief: TailorBrief }) {
  const emb = brief.embellishment_brief;
  if (!emb || (!emb.type && !emb.placement)) return null;

  return (
    <Section delay={0.5}>
      <SectionLabel>Embellishment</SectionLabel>
      <div className="rounded-xl bg-glass-bg border border-glass-border backdrop-blur-md ring-1 ring-shaaru-crimson/20 shadow-lg p-4 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-shaaru-crimson/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none" />
        <div className="flex items-center gap-2 mb-3">
          <Gem className="w-4 h-4 text-shaaru-crimson" />
          <span className="text-sm font-medium text-shaaru-crimson font-['DM_Sans',_sans-serif]">
            {emb.type}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-2">
          {emb.placement && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Placement
              </p>
              <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif]">
                {emb.placement}
              </p>
            </div>
          )}
          {emb.technique && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Technique
              </p>
              <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif]">
                {emb.technique}
              </p>
            </div>
          )}
          {emb.time_estimate && (
            <div>
              <p className="text-xs text-text-secondary-dark/50 uppercase tracking-wider font-['DM_Sans',_sans-serif]">
                Time
              </p>
              <p className="text-sm text-text-secondary-dark font-['DM_Sans',_sans-serif]">
                {emb.time_estimate}
              </p>
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}

// ─── Main TailorBrief Component ───────────────────────────────────────────────
interface TailorBriefProps {
  brief: TailorBrief;
  onStartOver: () => void;
  onAskShaaru: () => void;
}

export function TailorBrief({ brief }: TailorBriefProps) {
  const [openingDone, setOpeningDone] = useState(false);

  // shaaru_notes is the "opening message" from the backend
  const openingText =
    brief.shaaru_notes ??
    brief.opening_message ??
    "Here's your complete tailor brief — ready to hand to your tailor.";

  return (
    <div className="w-full max-w-5xl mx-auto">
      <div className="flex flex-col lg:flex-row gap-8 items-start">
        {/* ── LEFT COLUMN (60%) ── */}
        <div className="flex-1 flex flex-col gap-8 min-w-0">
          {/* Opening message streams first */}
          <OpeningMessage
            text={openingText}
            onComplete={() => setOpeningDone(true)}
          />

          <AnimatePresence>
            {openingDone && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex flex-col gap-8"
              >
                <GarmentSection brief={brief} />
                <FabricSection brief={brief} />
                <ConstructionSection brief={brief} />
                <AdditionalSpecsSection brief={brief} />
                <EmbellishmentSection brief={brief} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── RIGHT COLUMN (40%) ── */}
        <div className="w-full lg:w-80 xl:w-96 flex-shrink-0 flex flex-col gap-5 lg:sticky lg:top-8">
          <AnimatePresence>
            {openingDone && (
              <motion.div
                initial={{ opacity: 0, x: 16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.4 }}
                className="flex flex-col gap-5"
              >
                {brief.sourcing && (
                  <SourcingCard sourcing={brief.sourcing} />
                )}
                {brief.measurements && (
                  <MeasurementsCard measurements={brief.measurements} />
                )}
                {brief.quality_checkpoints &&
                  brief.quality_checkpoints.length > 0 && (
                    <QualityCheckpoints
                      checkpoints={brief.quality_checkpoints}
                    />
                  )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
