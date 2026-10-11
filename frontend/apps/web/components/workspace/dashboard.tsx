"use client";

import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export type DashboardTone = "signal" | "nominal" | "caution" | "fault" | "muted";

export function DashboardCard({ title, description, action, children, className, flush = false }: {
  title: ReactNode; description?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; flush?: boolean;
}) {
  return <Card className={cn("dashboard-card", flush && "dashboard-card-flush", className)}>
    <CardHeader><CardTitle>{title}</CardTitle>{description && <CardDescription>{description}</CardDescription>}{action && <CardAction>{action}</CardAction>}</CardHeader>
    {flush ? children : <CardContent>{children}</CardContent>}
  </Card>;
}

export function DashboardMetric({ label, value, detail, icon: Icon, tone = "signal" }: {
  label: string; value: ReactNode; detail: ReactNode; icon: LucideIcon; tone?: DashboardTone;
}) {
  return <Card className={cn("dashboard-card dashboard-metric", "is-" + tone)}>
    <CardHeader><CardDescription>{label}</CardDescription><CardAction><Icon aria-hidden="true" /></CardAction></CardHeader>
    <CardContent><strong>{value}</strong><p>{detail}</p></CardContent>
  </Card>;
}

export function DashboardBadge({ children, tone = "signal", className }: { children: ReactNode; tone?: DashboardTone; className?: string }) {
  return <Badge variant="outline" className={cn("dashboard-badge", "is-" + tone, className)}>{children}</Badge>;
}

export function DashboardSelect({ label, value, onChange, options }: {
  label: string; value: string; onChange: (value: string) => void; options: readonly (readonly [string, string])[];
}) {
  return <Select value={value} onValueChange={(next) => next !== null && onChange(next)}>
    <SelectTrigger size="sm" aria-label={label}><SelectValue>{options.find(([key]) => key === value)?.[1]}</SelectValue></SelectTrigger>
    <SelectContent>{options.map(([key, text]) => <SelectItem value={key} key={key}>{text}</SelectItem>)}</SelectContent>
  </Select>;
}

export function DashboardSheet({ open, onClose, eyebrow, title, description, tags, children, footer }: {
  open: boolean; onClose: () => void; eyebrow: ReactNode; title: ReactNode; description: ReactNode;
  tags?: ReactNode; children: ReactNode; footer?: ReactNode;
}) {
  return <Sheet open={open} onOpenChange={(next) => !next && onClose()}>
    <SheetContent className="detection-sheet dashboard-detail-sheet">
      <SheetHeader><div className="detection-sheet-eyebrow">{eyebrow}</div><SheetTitle className="detection-sheet-title">{title}</SheetTitle><SheetDescription>{description}</SheetDescription>{tags && <div className="detection-sheet-tags">{tags}</div>}</SheetHeader>
      <ScrollArea className="detection-sheet-scroll"><div className="detection-sheet-body">{children}</div></ScrollArea>
      <div className="detection-sheet-footer">{footer}<Button variant="outline" onClick={onClose}>Close inspection</Button></div>
    </SheetContent>
  </Sheet>;
}
