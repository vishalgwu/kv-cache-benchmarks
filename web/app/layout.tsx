import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Astera Mission Control — Engineering Diagnostics Copilot",
  description: "Independent portfolio demo: multi-agent diagnostics mission control with approval gates, citations, and evaluation. Synthetic data only.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
