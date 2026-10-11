"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, ChevronDown, Clock3, History, LayoutDashboard, Menu, PanelsTopLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverDescription, PopoverHeader, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { formatObsTime } from "@/lib/ground/freshness";
import { useMission } from "@/lib/ground/mission-context";
import { ArtemisLogo, FreshnessPill, Scene, scenes, utcDate, utcTime } from "./shared";

function MissionClocks({ now, onNavigate }: { now: Date | null; onNavigate?: () => void }) {
  const m = useMission();
  const mission = m.state?.mission;
  const id = mission?.mission_id ?? m.missionId;
  return (
    <div className="mission-context">
      <div className="context-utc">
        <span>UTC</span>
        <p>
          <span className="context-value">{now ? utcDate.format(now) : "—"}</span>
          <time dateTime={now?.toISOString()}>{now ? utcTime.format(now) : "—"}</time>
        </p>
      </div>
      <div className="context-day">
        <span>Mission</span>
        <p title={id ?? undefined}>
          <Link href={m.href("/missions")} className="context-mission-link" aria-label="Open mission history" onClick={onNavigate}>
            <span className="context-value">{id ? `#${id}` : "Mission archive"}</span>
            <small>{mission?.status ?? (id ? "awaiting ground data" : "browse sessions")} <ArrowUpRight aria-hidden="true" /></small>
          </Link>
        </p>
      </div>
      <div className="context-location">
        <span title="Rover simulation clock at the last received pose">Rover clock · sim</span>
        <p>
          <span className="context-value">{formatObsTime(m.latest?.t_obs_s)}</span>
          <small>Last pose · Webots / odom</small>
        </p>
      </div>
    </div>
  );
}

/** The shell stays one session; only the presentation changes with available CSS-pixel space. */
export function MissionHeader({ onMenu, menuOpen }: { onMenu: () => void; menuOpen: boolean }) {
  const m = useMission();
  const headerRef = useRef<HTMLElement>(null);
  const [now, setNow] = useState<Date | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const mode = m.workspace === "unified" ? "Unified console" : "Clean workspace";
  const ModeIcon = m.workspace === "unified" ? LayoutDashboard : PanelsTopLeft;

  useEffect(() => {
    const update = () => setNow(new Date());
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const header = headerRef.current;
    const shell = header?.closest<HTMLElement>(".mission-app");
    if (!header || !shell) return;
    // Real height, including wrapped text, keeps the desktop sidebar below the sticky header at every zoom.
    const update = () => shell.style.setProperty("--mission-header-height", `${Math.ceil(header.getBoundingClientRect().height)}px`);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(header);
    return () => { observer.disconnect(); shell.style.removeProperty("--mission-header-height"); };
  }, []);

  useEffect(() => {
    if (!detailsOpen) return;
    const query = window.matchMedia("(min-width: 1440px)");
    const closeWhenExpanded = () => query.matches && setDetailsOpen(false);
    query.addEventListener("change", closeWhenExpanded);
    return () => query.removeEventListener("change", closeWhenExpanded);
  }, [detailsOpen]);

  return (
    <header ref={headerRef} className="mission-header console-header">
      <Link className="brand-lockup" href={m.href("/")} aria-label="Artemis-X Mission Control home">
        <ArtemisLogo />
        <span className="brand-type"><strong>ARTEMIS-X</strong><span>MISSION CONTROL</span></span>
      </Link>

      <Button variant="ghost" size="icon" className="mobile-menu" aria-label="Open navigation"
        aria-controls="mission-navigation" aria-expanded={menuOpen}
        onClick={() => { setDetailsOpen(false); onMenu(); }}>
        <Menu aria-hidden="true" />
      </Button>

      <div className="header-heading">
        <div className="header-title-row"><h1><span className="header-title-brand">Artemis-X</span> Mission Control</h1></div>
        <p className="mission-disciplines" aria-label="Lunar robotics, autonomy, mapping, detection and exploration">
          <span>LUNAR ROBOTICS</span><i aria-hidden="true">·</i><span>AUTONOMY</span><i aria-hidden="true">·</i><span>MAPPING</span><i aria-hidden="true">·</i><span>DETECTION</span><i aria-hidden="true">·</i><span>EXPLORATION</span>
        </p>
      </div>

      <div className="header-controls" aria-label="Console status and display mode">
        <FreshnessPill value={m.fresh} className="simulation-status" title={m.freshLabel} />
        {m.workspace && <Link href={m.href("/settings")} className="workspace-chip"
          aria-label={`Display mode: ${mode}. Open settings.`} title="Change display mode in Settings">
          <ModeIcon aria-hidden="true" /><span>{mode}</span>
        </Link>}
      </div>

      <div className="header-context-region">
        <MissionClocks now={now} />
        <Popover open={detailsOpen} onOpenChange={setDetailsOpen}>
          <PopoverTrigger render={<Button variant="outline" className="header-context-trigger" />}
            aria-label="Show mission and clock details">
            <Clock3 aria-hidden="true" />
            <span className="header-context-caption">Mission details</span>
            <ChevronDown className="header-context-chevron" aria-hidden="true" />
          </PopoverTrigger>
          <PopoverContent align="end" sideOffset={12} className="header-context-popover">
            <PopoverHeader><PopoverTitle>Mission & clocks</PopoverTitle>
              <PopoverDescription>Current UTC and the last received rover simulation time. These clocks are independent.</PopoverDescription>
            </PopoverHeader>
            <FreshnessPill value={m.fresh} title={m.freshLabel} />
            <MissionClocks now={now} onNavigate={() => setDetailsOpen(false)} />
            <Link href={m.href("/missions")} className="header-archive-link" onClick={() => setDetailsOpen(false)}>
              <History aria-hidden="true" /> Browse mission archive <ArrowUpRight aria-hidden="true" />
            </Link>
          </PopoverContent>
        </Popover>
      </div>

      <div className="header-artwork" aria-hidden="true"><Scene crop={scenes.moon} label="The Moon" className="header-moon" /></div>
    </header>
  );
}
