"use client";

import { useState, useEffect, useRef } from "react";

export function useAnimatedText(
  text: string,
  {
    speed = 18,
    onComplete,
  }: {
    speed?: number; // characters per second
    onComplete?: () => void;
  } = {}
) {
  const [displayedText, setDisplayedText] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const indexRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Reset when text changes
    setDisplayedText("");
    setIsComplete(false);
    indexRef.current = 0;

    if (!text) {
      setIsComplete(true);
      return;
    }

    const interval = 1000 / speed;

    const tick = () => {
      if (indexRef.current < text.length) {
        indexRef.current += 1;
        setDisplayedText(text.slice(0, indexRef.current));
        timerRef.current = setTimeout(tick, interval);
      } else {
        setIsComplete(true);
        onComplete?.();
      }
    };

    timerRef.current = setTimeout(tick, interval);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [text, speed]); // eslint-disable-line react-hooks/exhaustive-deps

  return { displayedText, isComplete };
}
