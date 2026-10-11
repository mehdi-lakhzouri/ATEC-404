"use client";

// Console shell shared by both workspaces: the Mission Control header and sidebar around every page.
// Clean workspace: sidebar items are links to dedicated pages. Unified: on "/", items scroll to the command-center panels.
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { PanelLeftClose, PanelLeftOpen, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMission } from "@/lib/ground/mission-context";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { HomeGlyph, RoverNavigationGlyph } from "./shared";
import { MissionHeader } from "./header";
import { ConsoleNavContext, NAVIGATION, type ViewName } from "./console-nav";

const SIDEBAR_COLLAPSED_KEY = "artemis-x.sidebar-collapsed";

function Sidebar({
  open,
  collapsed,
  active,
  onClose,
  onToggleCollapsed,
  itemFor,
}: {
  open: boolean;
  collapsed: boolean;
  active: ViewName;
  onClose: () => void;
  onToggleCollapsed: () => void;
  itemFor: (label: ViewName) => { href?: string; onClick?: () => void };
}) {
  // below 1024px the sidebar is a drawer with full labels: the icon rail (and its tooltips) is desktop only
  const [desktop, setDesktop] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(min-width: 1024px)");
    const sync = () => setDesktop(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);
  const rail = collapsed && desktop;
  const toggleLabel = collapsed ? "Expand sidebar" : "Collapse sidebar";

  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.button
            className="navigation-backdrop"
            aria-label="Close navigation"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
        )}
      </AnimatePresence>
      <aside
        id="mission-navigation"
        className={cn("mission-sidebar", open && "is-open", collapsed && "is-collapsed")}
        aria-label="Mission navigation"
      >
        <button className="sidebar-close icon-button" aria-label="Close navigation" onClick={onClose}>
          <X />
        </button>
        <TooltipProvider delay={120}>
          <nav className="sidebar-navigation">
            {NAVIGATION.filter((item) => item.sidebar !== false).map(({ label, icon: Icon }) => {
              const body = (
                <>
                  {label === "Overview" ? <HomeGlyph /> : label === "Rover" ? <RoverNavigationGlyph /> : <Icon className="navigation-icon" strokeWidth={1.8} fill={label === "Settings" ? "currentColor" : "none"} />}
                  <span className="navigation-label">{label}</span>
                </>
              );
              const { href, onClick } = itemFor(label);
              const props = {
                className: cn("navigation-item", active === label && "is-active"),
                "aria-current": active === label ? ("page" as const) : undefined,
                onClick,
              };
              return (
                <Tooltip key={label} disabled={!rail}>
                  <TooltipTrigger render={href ? <Link href={href} /> : <button type="button" />} {...props}>
                    {body}
                  </TooltipTrigger>
                  <TooltipContent side="right" sideOffset={10}>
                    {label}
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </nav>
          <div className="sidebar-landscape" aria-hidden="true" />
          <p className="sidebar-motto">
            EXPLORE
            <br />
            AUTONOMOUSLY
            <br />
            BUILD FURTHER
          </p>
          <Tooltip>
            <TooltipTrigger
              render={<button type="button" />}
              className="sidebar-toggle"
              aria-label={toggleLabel}
              aria-expanded={!collapsed}
              onClick={onToggleCollapsed}
            >
              {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
            </TooltipTrigger>
            <TooltipContent side="right" sideOffset={10}>
              {toggleLabel}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </aside>
    </>
  );
}

export function ConsoleShell({ children }: { children: ReactNode }) {
  const m = useMission();
  const pathname = usePathname();
  const reducedMotion = useReducedMotion();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarAnimated, setSidebarAnimated] = useState(false);
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const [spy, setSpy] = useState<string | null>(null);
  const unifiedHome = m.workspace === "unified" && pathname === "/";

  // restored after mount so the server render and first client render match
  useEffect(() => {
    try {
      setSidebarCollapsed(window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1");
    } catch {}
  }, []);

  const toggleSidebarCollapsed = () => {
    const next = !sidebarCollapsed;
    setSidebarAnimated(true);
    setSidebarCollapsed(next);
    try {
      window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0");
    } catch {}
  };

  useEffect(() => {
    if (!sidebarOpen) return;
    const escape = (event: KeyboardEvent) => event.key === "Escape" && setSidebarOpen(false);
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [sidebarOpen]);

  const routeLabel = NAVIGATION.find((n) => n.route === pathname)?.label ?? "Overview";
  const active: ViewName = unifiedHome
    ? (NAVIGATION.find((n) => n.target && n.target === spy)?.label ?? "Overview")
    : routeLabel;

  const scrollTo = useCallback(
    (target: string | null) => {
      setHighlighted(target);
      if (target)
        document.getElementById(target)?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "nearest" });
      else window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" });
    },
    [reducedMotion],
  );

  const itemFor = (label: ViewName) => {
    const item = NAVIGATION.find((n) => n.label === label)!;
    const close = () => setSidebarOpen(false);
    if (m.workspace !== "unified" || label === "Settings") return { href: m.href(item.route), onClick: close };
    // unified workspace: the command center's panels
    if (unifiedHome)
      return {
        onClick: () => {
          close();
          setSpy(item.target ?? null);
          scrollTo(item.target ?? null);
        },
      };
    return { href: `${m.href("/")}${item.target ? `#${item.target}` : ""}`, onClick: close };
  };

  // arriving on the unified view with #panel (from another page): scroll there once it is rendered
  useEffect(() => {
    if (!unifiedHome) return;
    const target = window.location.hash.slice(1);
    if (!target) return;
    const t = window.setTimeout(() => scrollTo(target), 120);
    return () => window.clearTimeout(t);
  }, [unifiedHome, scrollTo]);

  // a new page starts at the top, without a stale highlight
  useEffect(() => {
    setHighlighted(null);
    setSidebarOpen(false);
  }, [pathname]);

  const nav = useMemo(() => ({ highlighted, setHighlighted, setSpy }), [highlighted]);
  return (
    <ConsoleNavContext.Provider value={nav}>
      <div
        id="overview"
        className={cn(
          "mission-app",
          (!m.motionEnabled || reducedMotion) && "reduce-motion",
          sidebarCollapsed && "sidebar-collapsed",
          sidebarAnimated && "sidebar-animated",
        )}
        data-freshness={m.fresh}
        data-workspace={m.workspace ?? "pending"}
      >
        <MissionHeader onMenu={() => setSidebarOpen(true)} menuOpen={sidebarOpen} />
        <Sidebar
          open={sidebarOpen}
          collapsed={sidebarCollapsed}
          active={active}
          onClose={() => setSidebarOpen(false)}
          onToggleCollapsed={toggleSidebarCollapsed}
          itemFor={itemFor}
        />
        <main className="mission-main" aria-label="Artemis-X mission dashboard">
          {children}
        </main>
      </div>
    </ConsoleNavContext.Provider>
  );
}
