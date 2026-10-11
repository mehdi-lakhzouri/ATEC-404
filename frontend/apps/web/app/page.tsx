"use client";

import { UnifiedDashboard } from "@/components/mission/unified-dashboard";
import { OverviewPage } from "@/components/workspace/overview";
import { useMission } from "@/lib/ground/mission-context";

// "/" is the Overview: the clean workspace overview, or the unified command center when Settings asks for it.
// Until the stored preference is read (hydration), nothing mode-specific renders, so neither mode flashes.
export default function Home() {
  const { workspace } = useMission();
  if (workspace === null) return <div className="cw-pending" aria-busy="true" />;
  return workspace === "unified" ? <UnifiedDashboard /> : <OverviewPage />;
}
