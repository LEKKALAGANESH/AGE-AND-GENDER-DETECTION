import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: "#0a0a0a",
          surface: "#18181b",
          border: "#27272a",
          text: {
            primary: "#ededed",
            secondary: "#a1a1aa",
            muted: "#71717a",
          },
          male: "#3b82f6",
          female: "#ec4899",
          success: "#10b981",
          error: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};

export default config;
