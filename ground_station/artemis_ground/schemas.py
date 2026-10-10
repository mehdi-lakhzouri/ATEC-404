"""Versioned response contracts. Every REST response is an envelope:

    success  {"data": <payload>, "meta": {"api_version": "v1", "request_id": "...", "generated_utc": "...",
                                          "pagination": {"limit", "offset", "total", "next_offset"} | null}}
    error    {"error": {"code": "...", "message": "...", "details": ...}, "meta": {...}}

Clock domains: ``*_t_obs_s`` / ``t_obs_s`` = rover observation time in SIMULATION seconds; ``*_utc`` = ground reception
time, UTC wall clock. Freshness fields are computed from UTC only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Pagination(BaseModel):
    limit: int
    offset: int
    total: int
    next_offset: int | None


class Meta(BaseModel):
    api_version: str = "v1"
    request_id: str | None = None
    generated_utc: datetime
    pagination: Pagination | None = None


class Envelope(BaseModel, Generic[T]):
    data: T
    meta: Meta


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
    meta: Meta


class _Doc(BaseModel):
    model_config = ConfigDict(extra="allow")


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    mongodb: Literal["up", "down"]
    receiver: dict
    ingestion: dict
    websocket_clients: int


class Mission(_Doc):
    mission_id: str
    session: int
    status: Literal["active", "idle"]
    created_utc: datetime
    last_rx_utc: datetime
    first_t_obs_s: float | None = None
    last_t_obs_s: float | None = None
    last_rx_age_s: float = Field(description="UTC now - last ground reception (wall clock only)")
    counters: dict = {}
    health: dict | None = None
    config: dict = {}


class Position(_Doc):
    seq: int
    t_obs_s: float
    rx_utc: datetime
    frame: Literal["odom"]
    x_m: float
    y_m: float
    yaw_rad: float
    heading_deg: float
    sigma_xy_m: float
    sigma_xy_saturated: bool


class Obstacle(_Doc):
    obstacle_id: str
    type: str
    frame: Literal["odom"]
    x_m: float
    y_m: float
    radius_m: float
    confidence: float
    uncertainty_m: float
    source: str
    status: str
    present: bool
    persistent_id: bool
    obs_timestamp_s: float
    last_rx_utc: datetime


class TelemetryRecord(_Doc):
    seq: int
    packet_type: str
    kind: str
    t_obs_s: float
    rx_utc: datetime
    seq_flag: str
    frame: str


class LinkMetric(_Doc):
    rx_utc: datetime
    receiver: dict
    link_up: bool
    emulator: dict | None = None


class PresetRequest(BaseModel):
    preset: Literal["NORMAL", "DEGRADED", "SEVERE", "BLACKOUT", "RESTORE"]
