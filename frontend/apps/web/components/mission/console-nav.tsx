"use client";

// Navigation model shared by the sidebar, the unified command center and the clean workspace pages.
import { createContext, useContext } from "react";
import {
  FileText,
  History,
  Home as HomeIcon,
  Map as MapIcon,
  Crosshair,
  RadioTower,
  Settings,
  Signal,
  Bot,
  type LucideIcon,
} from "lucide-react";

export type ViewName =
  | "Overview"
  | "Rover"
  | "Mapping"
  | "Detection"
  | "Telemetry"
  | "Communication"
  | "Missions"
  | "Logs"
  | "Settings";

/** route: the clean-workspace page; target: the panel id in the unified command center. */
export const NAVIGATION: { label: ViewName; icon: LucideIcon; route: string; target?: string; sidebar?: boolean }[] = [
  { label: "Overview", icon: HomeIcon, route: "/" },
  { label: "Rover", icon: Bot, route: "/rover", target: "rover" },
  { label: "Mapping", icon: MapIcon, route: "/map", target: "surface" },
  { label: "Detection", icon: Crosshair, route: "/detection", target: "detection" },
  { label: "Telemetry", icon: Signal, route: "/telemetry", target: "telemetry" },
  { label: "Communication", icon: RadioTower, route: "/network", target: "network" },
  { label: "Missions", icon: History, route: "/missions", target: "history", sidebar: false },
  { label: "Logs", icon: FileText, route: "/logs", target: "logs" },
  { label: "Settings", icon: Settings, route: "/settings" },
];

export const UNIFIED_TARGETS = NAVIGATION.filter((n) => n.target).map((n) => n.target!);

export type ConsoleNav = {
  highlighted: string | null;
  setHighlighted: (target: string | null) => void;
  /** unified view: panel currently in view (scroll spy) */
  setSpy: (target: string | null) => void;
};

export const ConsoleNavContext = createContext<ConsoleNav>({
  highlighted: null,
  setHighlighted: () => {},
  setSpy: () => {},
});

export const useConsoleNav = () => useContext(ConsoleNavContext);
