// Typed REST client for the ground station API v1. Every response is {data, meta}; errors are {error, meta}.
import { GROUND_API } from "./config";
import type {
  Envelope,
  ErrorEnvelope,
  Health,
  LinkMetric,
  Mission,
  MissionOverview,
  MissionStats,
  MissionStatus,
  NetworkStatus,
  Obstacle,
  Position,
  Preset,
  TelemetryRecord,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId: string | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type Query = Record<string, string | number | boolean | null | undefined>;

function url(path: string, query?: Query) {
  const u = new URL(`${GROUND_API}/api/v1${path}`);
  for (const [k, v] of Object.entries(query ?? {}))
    if (v !== undefined && v !== null) u.searchParams.set(k, String(v));
  return u.toString();
}

async function request<T>(
  path: string,
  init?: RequestInit & { query?: Query },
): Promise<Envelope<T>> {
  let response: Response;
  try {
    response = await fetch(url(path, init?.query), {
      cache: "no-store",
      ...init,
      headers: { Accept: "application/json", ...init?.headers },
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "network_error", `ground station unreachable at ${GROUND_API}`);
  }
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // non-JSON body (proxy error page, empty 502): reported below with the HTTP status
  }
  if (!response.ok) {
    const e = (body as ErrorEnvelope | null)?.error;
    throw new ApiError(
      response.status,
      e?.code ?? `http_${response.status}`,
      e?.message ?? response.statusText,
      (body as ErrorEnvelope | null)?.meta?.request_id ?? response.headers.get("X-Request-ID"),
    );
  }
  const env = body as Envelope<T> | null;
  if (!env || typeof env !== "object" || !("data" in env))
    throw new ApiError(response.status, "invalid_envelope", `unexpected response for ${path}`);
  return env;
}

const get = <T>(path: string, query?: Query, signal?: AbortSignal) =>
  request<T>(path, { query, signal });

/** Follows `meta.pagination.next_offset` until exhausted or `max` items were collected. */
export async function getAllPages<T>(
  path: string,
  query: Query,
  { max = 5000, pageSize = 1000, signal }: { max?: number; pageSize?: number; signal?: AbortSignal } = {},
): Promise<{ items: T[]; total: number; truncated: boolean }> {
  const items: T[] = [];
  let offset: number | null = 0;
  let total = 0;
  while (offset !== null && items.length < max) {
    const limit = Math.min(pageSize, max - items.length);
    const env: Envelope<T[]> = await get<T[]>(path, { ...query, limit, offset }, signal);
    items.push(...env.data);
    total = env.meta.pagination?.total ?? items.length;
    offset = env.meta.pagination?.next_offset ?? null;
  }
  return { items, total, truncated: items.length < total };
}

const m = (id: string) => `/missions/${encodeURIComponent(id)}`;

export const groundApi = {
  health: (signal?: AbortSignal) => get<Health>("/health", undefined, signal),
  missions: (q: { limit?: number; offset?: number; status?: MissionStatus } = {}, signal?: AbortSignal) =>
    get<Mission[]>("/missions", q, signal),
  mission: (id: string, signal?: AbortSignal) => get<Mission>(m(id), undefined, signal),
  overview: (id: string, signal?: AbortSignal) => get<MissionOverview>(`${m(id)}/overview`, undefined, signal),
  positions: (
    id: string,
    q: { limit?: number; offset?: number; since_t_obs_s?: number; order?: "asc" | "desc" } = {},
    signal?: AbortSignal,
  ) => get<Position[]>(`${m(id)}/positions`, q, signal),
  allPositions: (id: string, q: { since_t_obs_s?: number } = {}, opts?: { max?: number; signal?: AbortSignal }) =>
    getAllPages<Position>(`${m(id)}/positions`, { ...q, order: "asc" }, opts),
  obstacles: (
    id: string,
    q: { present?: boolean; source?: string; type?: string } = {},
    opts?: { max?: number; signal?: AbortSignal },
  ) => getAllPages<Obstacle>(`${m(id)}/obstacles`, { present: true, ...q }, opts),
  telemetry: (
    id: string,
    q: { limit?: number; offset?: number; packet_type?: "pose" | "obstacles" | "event" } = {},
    signal?: AbortSignal,
  ) => get<TelemetryRecord[]>(`${m(id)}/telemetry`, q, signal),
  events: (
    id: string,
    q: { limit?: number; offset?: number; kind?: "health" | "map_reset" | "inference" } = {},
    signal?: AbortSignal,
  ) => get<TelemetryRecord[]>(`${m(id)}/events`, q, signal),
  link: (id: string, q: { limit?: number; offset?: number } = {}, signal?: AbortSignal) =>
    get<LinkMetric[]>(`${m(id)}/link`, q, signal),
  stats: (id: string, signal?: AbortSignal) => get<MissionStats>(`${m(id)}/stats`, undefined, signal),
  networkStatus: (signal?: AbortSignal) => get<NetworkStatus>("/network/status", undefined, signal),
  applyPreset: (preset: Preset) =>
    request<NetworkStatus>("/network/preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset }),
    }),
};

export function describeError(error: unknown): string {
  if (error instanceof ApiError)
    return error.status === 0 ? error.message : `${error.code} (${error.status}): ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}
