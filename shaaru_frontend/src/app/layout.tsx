import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SHAARU Tailor Studio",
  description:
    "AI-powered tailor briefs crafted for Indian garment construction. Upload a reference, answer a few questions, and get a complete brief ready for your tailor.",
  keywords: ["tailor", "Indian fashion", "garment construction", "SHAARU", "AI tailor"],
  openGraph: {
    title: "SHAARU Tailor Studio",
    description: "Your AI tailor brief generator",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="bg-bg-dark text-text-primary-dark antialiased">
        {children}
      </body>
    </html>
  );
}
