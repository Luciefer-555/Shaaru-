import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Silence the "multiple lockfiles" workspace root warning
  outputFileTracingRoot: path.join(__dirname),
  devIndicators: {
    position: 'bottom-right',
  },
};

export default nextConfig;
