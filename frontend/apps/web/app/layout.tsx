import type { Metadata } from "next";
import localFont from "next/font/local";
import { ConsoleShell } from "@/components/mission/shell";
import { MissionProvider } from "@/lib/ground/mission-context";
import "./globals.css";
import "./workspace.css";
import "./header.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  title: "Artemis-X Mission Control",
  description: "Autonomous lunar rover command and telemetry interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" data-scroll-behavior="smooth">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {/* one mission session (selection + WebSocket) for every page of the console */}
        <MissionProvider>
          <ConsoleShell>{children}</ConsoleShell>
        </MissionProvider>
      </body>
    </html>
  );
}
