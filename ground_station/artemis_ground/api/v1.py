"""REST API v1 (mounted under /api/v1). Data comes ONLY from what the ground station received and persisted / reconciled."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import db as D
from ..schemas import (Envelope, Health, LinkMetric, Mission, Obstacle, Position, PresetRequest, TelemetryRecord)
from .common import envelope_ok, local_only, page

router = APIRouter()


def _svc(request: Request):
    return request.app.state.ingest


def _db(request: Request) -> D.Database:
    return request.app.state.db


def _mission_view(doc: dict, idle_after_s: float) -> dict:
    now = datetime.now(timezone.utc)
    age = (now - doc["last_rx_utc"]).total_seconds()
    out = {k: v for k, v in doc.items() if k != "_id"}
    out.update({"mission_id": doc["_id"], "last_rx_age_s": round(age, 3), "status": "active" if age <= idle_after_s else "idle"})
    return out


async def _mission(request: Request, mission_id: str) -> dict:
    doc = await _db(request)[D.MISSIONS].find_one({"_id": mission_id})
    if doc is None:
        raise HTTPException(404, detail={"code": "mission_not_found", "message": f"no mission {mission_id!r}"})
    return _mission_view(doc, request.app.state.settings.mission_idle_after_s)


async def health_payload(request: Request) -> dict:
    svc, db = _svc(request), _db(request)
    up = await db.ping()
    rs = svc.receiver.stats if svc.receiver else {}
    return {"status": "ok" if up and (svc.receiver is not None or not request.app.state.settings.receiver_enabled) else "degraded",
            "api_version": "v1", "mongodb": "up" if up else "down",
            "receiver": {"enabled": svc.receiver is not None, "valid": rs.get("valid", 0), "invalid": rs.get("invalid", 0),
                         "acks_sent": rs.get("acks_sent", 0), "last_rx_utc": None if not rs.get("last_rx_wall_s") else
                         datetime.fromtimestamp(rs["last_rx_wall_s"], tz=timezone.utc).isoformat()},
            "ingestion": svc.stats_flat(), "websocket_clients": request.app.state.hub.clients()}


@router.get("/health", response_model=Envelope[Health], summary="Service, MongoDB, receiver and ingestion health")
async def health(request: Request):
    return envelope_ok(request, await health_payload(request))


@router.get("/missions", response_model=Envelope[list[Mission]], summary="Missions (one per rover telemetry session)")
async def missions(request: Request, limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0),
                   status: Literal["active", "idle"] | None = None):
    col = _db(request)[D.MISSIONS]
    docs = [_mission_view(d, request.app.state.settings.mission_idle_after_s)
            async for d in col.find({}).sort("last_rx_utc", -1).skip(0).limit(1000)]
    if status:
        docs = [d for d in docs if d["status"] == status]
    return envelope_ok(request, docs[offset:offset + limit], page(limit, offset, len(docs)))


@router.get("/missions/{mission_id}", response_model=Envelope[Mission], summary="One mission")
async def mission(request: Request, mission_id: str):
    return envelope_ok(request, await _mission(request, mission_id))


@router.get("/missions/{mission_id}/overview", summary="Dashboard overview: mission, latest pose, obstacle picture, link, health")
async def overview(request: Request, mission_id: str):
    m = await _mission(request, mission_id)
    db = _db(request)
    pose = await db[D.POSITIONS].find_one({"mission_id": mission_id}, sort=[("t_obs_s", -1)], projection={"_id": 0})
    by_type, by_source = {}, {}
    async for o in db[D.OBSTACLES].find({"mission_id": mission_id, "present": True}, projection={"type": 1, "source": 1}):
        by_type[o["type"]] = by_type.get(o["type"], 0) + 1
        by_source[o["source"]] = by_source.get(o["source"], 0) + 1
    link = await db[D.LINK].find_one({"mission_id": mission_id}, sort=[("rx_utc", -1)], projection={"_id": 0})
    return envelope_ok(request, {"mission": m, "latest_position": pose, "health": m.get("health"),
                                 "obstacles": {"present": sum(by_type.values()), "by_type": by_type, "by_source": by_source},
                                 "link": link, "freshness": {"last_rx_age_s": m["last_rx_age_s"], "clock": "UTC wall clock"}})


@router.get("/missions/{mission_id}/positions", response_model=Envelope[list[Position]], summary="Received rover poses")
async def positions(request: Request, mission_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                    since_t_obs_s: float | None = Query(None, description="observation time lower bound (simulation s)"),
                    order: Literal["asc", "desc"] = "desc"):
    await _mission(request, mission_id)
    q = {"mission_id": mission_id, **({"t_obs_s": {"$gte": since_t_obs_s}} if since_t_obs_s is not None else {})}
    col = _db(request)[D.POSITIONS]
    total = await col.count_documents(q)
    docs = [d async for d in col.find(q, projection={"_id": 0}).sort("t_obs_s", -1 if order == "desc" else 1).skip(offset).limit(limit)]
    return envelope_ok(request, docs, page(limit, offset, total))


@router.get("/missions/{mission_id}/obstacles", response_model=Envelope[list[Obstacle]],
            summary="Reconciled obstacle picture (GroundMap): latest state per obstacle id")
async def obstacles(request: Request, mission_id: str, limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0),
                    present: bool | None = Query(True, description="true: current picture; false: removed; omit for all"),
                    source: Literal["yolo+depth", "depth"] | None = None,
                    type: Literal["rock", "obstacle", "negative_hazard"] | None = None):
    await _mission(request, mission_id)
    q = {"mission_id": mission_id}
    for k, v in (("present", present), ("source", source), ("type", type)):
        if v is not None:
            q[k] = v
    col = _db(request)[D.OBSTACLES]
    total = await col.count_documents(q)
    docs = [d async for d in col.find(q, projection={"_id": 0}).sort("obstacle_id", 1).skip(offset).limit(limit)]
    return envelope_ok(request, docs, page(limit, offset, total))


@router.get("/missions/{mission_id}/telemetry", response_model=Envelope[list[TelemetryRecord]],
            summary="Received packet log (validated packets only; duplicates stored once)")
async def telemetry(request: Request, mission_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                    packet_type: Literal["pose", "obstacles", "event"] | None = None):
    await _mission(request, mission_id)
    q = {"mission_id": mission_id, **({"packet_type": packet_type} if packet_type else {})}
    col = _db(request)[D.EVENTS]
    total = await col.count_documents(q)
    docs = [d async for d in col.find(q, projection={"_id": 0}).sort("seq", -1).skip(offset).limit(limit)]
    return envelope_ok(request, docs, page(limit, offset, total))


@router.get("/missions/{mission_id}/events", response_model=Envelope[list[TelemetryRecord]],
            summary="Mission events: health transitions and map resets")
async def events(request: Request, mission_id: str, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                 kind: Literal["health", "map_reset", "inference"] | None = None):
    await _mission(request, mission_id)
    q = {"mission_id": mission_id, "packet_type": "event", **({"kind": kind} if kind else {})}
    col = _db(request)[D.EVENTS]
    total = await col.count_documents(q)
    docs = [d async for d in col.find(q, projection={"_id": 0}).sort("t_obs_s", -1).skip(offset).limit(limit)]
    return envelope_ok(request, docs, page(limit, offset, total))


@router.get("/missions/{mission_id}/link", response_model=Envelope[list[LinkMetric]], summary="Link metrics history")
async def link(request: Request, mission_id: str, limit: int = Query(50, ge=1, le=1000), offset: int = Query(0, ge=0)):
    await _mission(request, mission_id)
    col = _db(request)[D.LINK]
    q = {"mission_id": mission_id}
    total = await col.count_documents(q)
    docs = [d async for d in col.find(q, projection={"_id": 0}).sort("rx_utc", -1).skip(offset).limit(limit)]
    return envelope_ok(request, docs, page(limit, offset, total))


@router.get("/missions/{mission_id}/stats", summary="Counts and delivery statistics for one mission")
async def stats(request: Request, mission_id: str):
    m = await _mission(request, mission_id)
    db = _db(request)
    counts = {c: await db[c].count_documents({"mission_id": mission_id}) for c in (D.POSITIONS, D.OBSTACLES, D.EVENTS, D.LINK)}
    present = await db[D.OBSTACLES].count_documents({"mission_id": mission_id, "present": True})
    link = await db[D.LINK].find_one({"mission_id": mission_id}, sort=[("rx_utc", -1)], projection={"_id": 0})
    return envelope_ok(request, {"mission_id": mission_id, "counters": m.get("counters", {}), "documents": counts,
                                 "obstacles_present": present, "receiver": (link or {}).get("receiver"),
                                 "emulator": (link or {}).get("emulator"), "ingestion": _svc(request).stats_flat(),
                                 "observation_span_s": None if m.get("last_t_obs_s") is None else
                                 round(m["last_t_obs_s"] - (m.get("first_t_obs_s") or 0.0), 3)})


# ---- network control: the REAL R5.3 emulator over its UDP control port; local clients only -----------------------------------
@router.get("/network/status", summary="Live status of the lunar link emulator (local only)", dependencies=[Depends(local_only)])
async def network_status(request: Request):
    from .. import network
    s = request.app.state.settings
    try:
        st = await network.status(s.emulator_addr)
    except network.EmulatorUnavailable as e:
        raise HTTPException(503, detail={"code": "emulator_unavailable", "message": str(e)})
    return envelope_ok(request, st)


@router.post("/network/preset", summary="Apply a link preset on the emulator (local only)", dependencies=[Depends(local_only)])
async def network_preset(request: Request, body: PresetRequest):
    from .. import network
    s = request.app.state.settings
    try:
        st = await network.apply_preset(s.emulator_addr, body.preset)
    except network.EmulatorUnavailable as e:
        raise HTTPException(503, detail={"code": "emulator_unavailable", "message": str(e)})
    if "error" in st:
        raise HTTPException(400, detail={"code": "emulator_rejected", "message": st["error"]})
    if st.get("preset") != body.preset:
        raise HTTPException(502, detail={"code": "preset_not_applied", "message": f"emulator reports {st.get('preset')!r}"})
    return envelope_ok(request, st)
