// Ground station endpoints. NEXT_PUBLIC_* values are inlined at build time (see apps/web/.env.example).
export const GROUND_API = (
  process.env.NEXT_PUBLIC_GROUND_API ?? "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export const GROUND_WS = (
  process.env.NEXT_PUBLIC_GROUND_WS ?? GROUND_API.replace(/^http/, "ws")
).replace(/\/+$/, "");
