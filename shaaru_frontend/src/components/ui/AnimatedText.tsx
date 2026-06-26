"use client";

import { useAnimatedText } from "@/lib/use-animated-text";
import { cn } from "@/lib/cn";

interface AnimatedTextProps {
  text: string;
  speed?: number;
  className?: string;
  onComplete?: () => void;
  as?: keyof React.JSX.IntrinsicElements;
}

export function AnimatedText({
  text,
  speed = 30,
  className,
  onComplete,
  as: Tag = "span",
}: AnimatedTextProps) {
  const { displayedText } = useAnimatedText(text, { speed, onComplete });

  return (
    <Tag className={cn("whitespace-pre-wrap", className)}>
      {displayedText}
      <span className="animate-pulse text-shaaru-crimson">|</span>
    </Tag>
  );
}
