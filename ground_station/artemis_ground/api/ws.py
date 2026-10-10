"""WebSocket v1: ``/ws/v1/missions/{mission_id}`` (``*`` = every mission).

On connect the client first receives one ``mission.snapshot`` (persisted / reconciled state: latest pose, present
obstacles, health, counters) so a reconnecting dashboard needs no separate REST call, then live events in the standard
envelope (see ``hub.py``). The server never waits on a slow client: its bounded queue drops the oldest events.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder

from .. import db as D
from .. import logging as L
from ..hub import envelope

router = APIRouter()
log = L.get("artemis.ground.ws")


async def _snapshot(ws: WebSocket, mission_id: str) -> dict:
    db: D.Database = ws.app.state.db
    data = {"mission": None, "latest_position": None, "obstacles": [], "source": "mongodb"}
    if mission_id == "*":
        return data
    try:
        m = await db[D.MISSIONS].find_one({"_id": mission_id})
        if m is not None:
            m["mission_id"] = m.pop("_id")
        data["mission"] = m
        data["latest_position"] = await db[D.POSITIONS].find_one({"mission_id": mission_id}, sort=[("t_obs_s", -1)], projection={"_id": 0})
        data["obstacles"] = [o async for o in db[D.OBSTACLES].find({"mission_id": mission_id, "present": True}, projection={"_id": 0})]
    except Exception as e:                                  # database down: say so, keep the live stream
        data.update({"source": "unavailable", "error": type(e).__name__})
    return data


@router.websocket("/missions/{mission_id}")
async def mission_stream(ws: WebSocket, mission_id: str):
    hub = ws.app.state.hub
    await ws.accept()
    sub = hub.subscribe(mission_id)
    log.info("ws_connected", mission_id=mission_id, clients=hub.clients())
    try:
        await ws.send_json(jsonable_encoder(envelope("mission.snapshot", mission_id, await _snapshot(ws, mission_id))))

        async def pump():
            while True:
                await ws.send_json(jsonable_encoder(await sub.queue.get()))

        async def listen():                                 # detects disconnects; client messages are ignored
            while True:
                await ws.receive_text()

        done, pending = await asyncio.wait([asyncio.create_task(pump()), asyncio.create_task(listen())],
                                           return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            t.exception()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(sub)
        log.info("ws_disconnected", mission_id=mission_id, dropped=sub.dropped, clients=hub.clients())
