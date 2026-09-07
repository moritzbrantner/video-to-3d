import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video to 3D",
  description: "Browser-local sparse 3D reconstruction powered by Rust and WebAssembly.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
