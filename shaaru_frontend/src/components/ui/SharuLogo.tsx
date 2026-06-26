"use client";

import { cn } from "@/lib/cn";

interface SharuLogoProps {
  size?: number;
  className?: string;
  animate?: boolean;
}

export function SharuLogo({
  size = 80,
  className,
  animate = true,
}: SharuLogoProps) {
  return (
    <div
      className={cn("inline-flex items-center justify-center", className)}
      style={
        animate
          ? {
              animation: "float 4s ease-in-out infinite",
            }
          : undefined
      }
    >
      <svg
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        aria-label="SHAARU logo"
      >
        <defs>
          <ellipse id="petal-pair" cx="100" cy="100" rx="90" ry="22" />
        </defs>
        <g fill="#8B1A1A" fillRule="evenodd">
          <use href="#petal-pair" transform="rotate(0 100 100)" />
          <use href="#petal-pair" transform="rotate(45 100 100)" />
          <use href="#petal-pair" transform="rotate(90 100 100)" />
          <use href="#petal-pair" transform="rotate(135 100 100)" />
        </g>
      </svg>
    </div>
  );
}
