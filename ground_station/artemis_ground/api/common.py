"""Response envelope helpers, request-id middleware and the local-only guard."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import API_VERSION
from .. import logging as L

log = L.get("artemis.ground.api")
LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}


def meta(request: Request, pagination: dict | None = None) -> dict:
    return {"api_version": API_VERSION, "request_id": getattr(request.state, "request_id", None),
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "pagination": pagination}


def page(limit: int, offset: int, total: int) -> dict:
    return {"limit": limit, "offset": offset, "total": total, "next_offset": offset + limit if offset + limit < total else None}


def envelope_ok(request: Request, data, pagination: dict | None = None) -> dict:
    return {"data": data, "meta": meta(request, pagination)}


def error_response(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(status_code=status, content=jsonable_encoder(
        {"error": {"code": code, "message": message, "details": details}, "meta": meta(request)}))


def local_only(request: Request) -> None:
    host = request.client.host if request.client else None
    if not request.app.state.settings.network_control_enabled:
        raise HTTPException(403, detail={"code": "network_control_disabled", "message": "network control is disabled"})
    if host not in LOCAL:
        raise HTTPException(403, detail={"code": "local_only", "message": "network control is restricted to local clients"})


def install(app) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        structlog.contextvars.bind_contextvars(request_id=rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = rid
        log.info("http_request", request_id=rid, method=request.method, path=request.url.path, status=response.status_code,
                 duration_ms=round((time.perf_counter() - t0) * 1000, 2))
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) else {"code": f"http_{exc.status_code}", "message": str(exc.detail)}
        return error_response(request, exc.status_code, d.get("code", f"http_{exc.status_code}"), d.get("message", ""), d.get("details"))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(request, 422, "validation_error", "invalid request parameters", jsonable_encoder(exc.errors()))

    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def starlette_error(request: Request, exc: StarletteHTTPException):
        return error_response(request, exc.status_code, f"http_{exc.status_code}", str(exc.detail))
