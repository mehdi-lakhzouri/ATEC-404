"""FastAPI application: lifespan (MongoDB, ingestion, receiver), versioned routes, CORS, structured logging.

    uv run --project ground_station uvicorn artemis_ground.main:app --host 127.0.0.1 --port 8000 --app-dir ground_station
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import API_VERSION, __version__
from . import logging as L
from .api import common, v1, ws
from .config import Settings
from .db import Database
from .hub import Hub
from .ingest import IngestionService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    L.configure(settings.log_level, settings.log_json)
    log = L.get("artemis.ground")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.db = Database(settings.mongo_uri, settings.mongo_db, settings.mongo_timeout_ms)
        app.state.hub = Hub(settings.ws_queue_max)
        app.state.ingest = IngestionService(settings, app.state.db, app.state.hub)
        await app.state.ingest.start()
        log.info("ground_station_started", version=__version__, api=API_VERSION, db=settings.mongo_db,
                 mongodb="up" if app.state.ingest.db_ok else "down", receiver=settings.receiver_enabled)
        try:
            yield
        finally:
            await app.state.ingest.stop()
            await app.state.db.close()
            log.info("ground_station_stopped")

    app = FastAPI(title="Artemis-X Ground Station API", version=__version__, lifespan=lifespan,
                  description="Mission data received through the lunar link: rover poses, reconciled obstacles, events and "
                              "link metrics. Every response is `{data, meta}`; live updates on `/ws/v1/missions/{id}`. "
                              "Observation times are simulation seconds (`t_obs_s`), reception times are UTC (`*_utc`).",
                  openapi_tags=[{"name": "v1", "description": "Mission data API, version 1"}])
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], expose_headers=["X-Request-ID"])
    common.install(app)
    app.include_router(v1.router, prefix=f"/api/{API_VERSION}", tags=[API_VERSION])
    app.include_router(ws.router, prefix=f"/ws/{API_VERSION}")

    @app.get("/health", tags=["service"], summary="Liveness and dependencies (unversioned, for probes)")
    async def health(request: Request):
        return common.envelope_ok(request, await v1.health_payload(request))

    return app


app = create_app()
