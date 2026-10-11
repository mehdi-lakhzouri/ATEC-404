"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Switch } from "@/components/ui/switch";
import { describeError, groundApi } from "@/lib/ground/api";
import { GROUND_API, GROUND_WS } from "@/lib/ground/config";
import { useMission } from "@/lib/ground/mission-context";
import type { Health } from "@/lib/ground/types";
import { cn } from "@/lib/utils";
import { ErrorNote, Facts, Page, PageHeader, Section } from "./primitives";

function SettingRow({
  id,
  label,
  description,
  checked,
  onChange,
  disabled,
}: {
  id: string;
  label: string;
  description: ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <div className="cw-setting" data-setting={id}>
      <div>
        {/* the visible control is the switch itself (base-ui keeps a hidden form input), so label it by id */}
        <span className="cw-setting-label" id={`${id}-label`} onClick={() => !disabled && onChange(!checked)}>
          {label}
        </span>
        <p id={`${id}-desc`}>{description}</p>
      </div>
      <Switch
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
        aria-labelledby={`${id}-label`}
        aria-describedby={`${id}-desc`}
      />
    </div>
  );
}

export function SettingsPage() {
  const m = useMission();
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  useEffect(() => {
    const abort = new AbortController();
    groundApi
      .health(abort.signal)
      .then((env) => setHealth(env.data))
      .catch((e: unknown) => !abort.signal.aborted && setHealthError(describeError(e)));
    return () => abort.abort();
  }, []);

  return (
    <Page id="settings">
      <PageHeader title="Settings" description="How the console looks and which mission it follows. Preferences are kept in this browser." />
      <div className="cw-grid">
        <div className="span-8 cw-stack">
          <Section title="Workspace & Display Preferences">
            <SettingRow
              id="unified-mode"
              label="Unified Mission Control"
              description="Display all mission systems in a single command center instead of separate focused pages."
              checked={m.workspace === "unified"}
              disabled={m.workspace === null}
              onChange={(v) => m.setWorkspace(v ? "unified" : "clean")}
            />
            <SettingRow
              id="motion"
              label="Motion effects"
              description="Rover locator pulse, live signal animation and interface transitions. Your system's reduced-motion setting always wins."
              checked={m.motionEnabled}
              onChange={m.setMotionEnabled}
            />
          </Section>
          <Section title="Mission selection">
            <SettingRow
              id="follow-latest"
              label="Follow the newest mission"
              description="Switch automatically when a new rover session starts sending telemetry. Picking a mission in Missions turns this off."
              checked={m.followLatest}
              onChange={(v) => (v ? m.followNewest() : m.stopFollowing())}
            />
          </Section>
        </div>
        <Section title="Ground station" className="span-4">
          {healthError && <ErrorNote message={healthError} />}
          <Facts
            items={[
              ["REST API", <span key="r" className="cw-mono cw-break">{GROUND_API}/api/v1</span>],
              ["WebSocket", <span key="w" className="cw-mono cw-break">{GROUND_WS}/ws/v1/missions/{"{id}"}</span>],
              ["Service", health ? <span key="s" className={cn(health.status === "ok" ? "cw-ok" : "cw-warn")}>{health.status}</span> : "—"],
              ["MongoDB", health ? <span key="d" className={cn(health.mongodb === "up" ? "cw-ok" : "cw-bad")}>{health.mongodb}</span> : "—"],
              ["WebSocket clients", <span key="c" className="cw-mono">{health?.websocket_clients ?? "—"}</span>],
              ["Active display mode", m.workspace === "unified" ? "Unified Mission Control" : m.workspace === "clean" ? "Clean workspace" : "—"],
            ]}
          />
        </Section>
      </div>
    </Page>
  );
}
