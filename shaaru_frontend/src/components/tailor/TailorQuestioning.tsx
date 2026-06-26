"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";
import { SharuLogo } from "@/components/ui/SharuLogo";

export interface QuestionOption {
  id: string;
  label: string;
  emoji?: string;
  description?: string;
}

export interface TailorQuestion {
  id: string;
  question: string;
  options: QuestionOption[];
}

export const DEFAULT_QUESTIONS: TailorQuestion[] = [
  {
    id: "collar",
    question: "what kind of collar are you thinking?",
    options: [
      { id: "notch", label: "Notch Lapel", emoji: "🔺", description: "Classic, versatile" },
      { id: "peak", label: "Peak Lapel", emoji: "⬆️", description: "Bold, structured" },
      { id: "shawl", label: "Shawl Lapel", emoji: "〰️", description: "Sleek, formal" },
      { id: "mandarin", label: "Mandarin", emoji: "⭕", description: "Minimal, modern" },
    ],
  },
  {
    id: "fit",
    question: "what fit are we going for?",
    options: [
      { id: "baggy", label: "Baggy", emoji: "🌊", description: "Relaxed, oversized" },
      { id: "regular", label: "Regular", emoji: "📐", description: "Classic proportion" },
      { id: "slim", label: "Slim", emoji: "✏️", description: "Closer cut" },
      { id: "tailored", label: "Tailored", emoji: "🎯", description: "Precision fit" },
    ],
  },
  {
    id: "length",
    question: "how long should it be?",
    options: [
      { id: "cropped", label: "Cropped", emoji: "✂️", description: "Above the hip" },
      { id: "standard", label: "Standard", emoji: "📏", description: "Hip length" },
      { id: "longline", label: "Longline", emoji: "⬇️", description: "Below the hip" },
    ],
  },
  {
    id: "closure",
    question: "how does it close?",
    options: [
      { id: "buttons", label: "Buttons", emoji: "🔘", description: "Classic finish" },
      { id: "zip", label: "Concealed Zip", emoji: "🤐", description: "Clean look" },
      { id: "hook", label: "Hook & Eye", emoji: "🪝", description: "Couture closure" },
      { id: "open", label: "Open Front", emoji: "🔓", description: "No closure" },
    ],
  },
  {
    id: "embellishment",
    question: "any embellishment?",
    options: [
      { id: "none", label: "None", emoji: "⬜", description: "Clean & minimal" },
      { id: "minimal", label: "Subtle", emoji: "✨", description: "Light detail" },
      { id: "embroidered", label: "Embroidered", emoji: "🌸", description: "Hand work" },
      { id: "crystal", label: "Crystal Work", emoji: "💎", description: "Statement pieces" },
    ],
  },
];

interface TailorQuestioningProps {
  questions?: TailorQuestion[];
  onComplete: (answers: Record<string, string>) => void;
}

export function TailorQuestioning({
  questions = DEFAULT_QUESTIONS,
  onComplete,
}: TailorQuestioningProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [direction, setDirection] = useState(1);

  const currentQuestion = questions[currentIndex];
  const isLast = currentIndex === questions.length - 1;

  const handleSelect = (optionId: string) => {
    setSelected(optionId);
  };

  const handleNext = useCallback(() => {
    if (!selected) return;
    const newAnswers = { ...answers, [currentQuestion.id]: selected };
    setAnswers(newAnswers);

    if (isLast) {
      onComplete(newAnswers);
      return;
    }

    setDirection(1);
    setSelected(null);
    setCurrentIndex((i) => i + 1);
  }, [selected, answers, currentQuestion.id, isLast, onComplete]);

  const handleBack = () => {
    if (currentIndex === 0) return;
    setDirection(-1);
    setSelected(answers[questions[currentIndex - 1].id] ?? null);
    const newAnswers = { ...answers };
    delete newAnswers[currentQuestion.id];
    setAnswers(newAnswers);
    setCurrentIndex((i) => i - 1);
  };

  return (
    <div className="w-full max-w-2xl flex flex-col gap-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <SharuLogo size={36} animate className="opacity-80" />
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-secondary-dark/60 font-['DM_Sans',_sans-serif]">
            {currentIndex + 1} of {questions.length}
          </span>
          <div className="flex gap-1">
            {questions.map((_, i) => (
              <div
                key={i}
                className={cn(
                  "h-1 rounded-full transition-all duration-300",
                  i === currentIndex
                    ? "w-6 bg-shaaru-crimson"
                    : i < currentIndex
                    ? "w-2 bg-shaaru-crimson/50"
                    : "w-2 bg-white/15",
                )}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Question */}
      <AnimatePresence mode="wait" custom={direction}>
        <motion.div
          key={currentQuestion.id}
          custom={direction}
          initial={{ opacity: 0, x: direction * 40 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: direction * -40 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="flex flex-col gap-6"
        >
          <h2 className="text-2xl md:text-3xl font-light text-text-primary-dark font-['Playfair_Display',_serif]">
            {currentQuestion.question}
          </h2>

          {/* Option Cards Grid */}
          <div className="grid grid-cols-2 gap-3">
            {currentQuestion.options.map((option, i) => {
              const isSelected = selected === option.id;
              return (
                <motion.button
                  key={option.id}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.06, duration: 0.25 }}
                  onClick={() => handleSelect(option.id)}
                  className={cn(
                    "relative text-left p-4 rounded-xl transition-all duration-200",
                    "border",
                    isSelected
                      ? "border-shaaru-crimson bg-shaaru-crimson/10 shadow-[0_0_0_1px_#8B1A1A]"
                      : "border-white/10 bg-white/4 hover:border-white/20 hover:bg-white/7",
                  )}
                >
                  {/* Checkmark */}
                  {isSelected && (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="absolute top-2.5 right-2.5 w-5 h-5 rounded-full bg-shaaru-crimson flex items-center justify-center"
                    >
                      <Check className="w-3 h-3 text-white" strokeWidth={3} />
                    </motion.div>
                  )}

                  <div className="flex flex-col gap-1.5">
                    {option.emoji && (
                      <span className="text-xl">{option.emoji}</span>
                    )}
                    <span
                      className={cn(
                        "text-sm font-medium font-['DM_Sans',_sans-serif]",
                        isSelected
                          ? "text-text-primary-dark"
                          : "text-text-primary-dark/90",
                      )}
                    >
                      {option.label}
                    </span>
                    {option.description && (
                      <span className="text-xs text-text-secondary-dark/70">
                        {option.description}
                      </span>
                    )}
                  </div>
                </motion.button>
              );
            })}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={handleBack}
          disabled={currentIndex === 0}
          className="text-sm text-text-secondary-dark/60 hover:text-text-secondary-dark disabled:opacity-30 disabled:pointer-events-none transition-colors font-['DM_Sans',_sans-serif]"
        >
          ← back
        </button>

        <button
          onClick={handleNext}
          disabled={!selected}
          className={cn(
            "flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200",
            "font-['DM_Sans',_sans-serif]",
            selected
              ? "bg-shaaru-crimson hover:bg-shaaru-crimson-hover text-white shadow-sm"
              : "bg-white/8 text-text-secondary-dark/40 cursor-not-allowed",
          )}
        >
          {isLast ? "Generate Brief" : "Next"}
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
