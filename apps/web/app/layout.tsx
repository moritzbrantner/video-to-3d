import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video to 3D",
  description: "Browser-local sparse 3D reconstruction powered by Rust and WebAssembly.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <nav
          aria-label="Video to 3D tools"
          style={{
            display: "flex",
            gap: 18,
            padding: "12px 24px",
            borderBottom: "1px solid rgba(127, 127, 127, 0.28)",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          <Link href="/">Reconstruction</Link>
          <Link href="/feature-lab/">Feature matching lab</Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
