"""Settings from environment variables (prefix ARTEMIS_GROUND_) or ground_station/.env. No credentials in code."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _addr(v: str | None) -> tuple[str, int] | None:
    if not v:
        return None
    host, port = v.rsplit(":", 1)
    return host, int(port)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARTEMIS_GROUND_", env_file=Path(__file__).resolve().parents[1] / ".env",
                                      extra="ignore")

    # MongoDB (docker/compose.yml); the URI carries the credentials and must come from the environment
    mongo_uri: str = Field("mongodb://127.0.0.1:27017", description="mongodb://user:password@127.0.0.1:27017/?authSource=admin")
    mongo_db: str = "artemis_mission"
    mongo_timeout_ms: int = 2000
    # telemetry reception (R5.2 / R5.4): the receiver listens where the emulator delivers, and ACKs to its reverse port
    receiver_enabled: bool = True
    udp_host: str = "127.0.0.1"
    udp_port: int = 5790
    ack_to: str | None = "127.0.0.1:5794"            # emulator reverse port; empty to disable ACKs
    jsonl_out: str | None = None                       # optional raw receiver log (same format as R5.2)
    # R5.3 emulator control (local only)
    emulator_control: str = "127.0.0.1:5792"
    network_control_enabled: bool = True
    # ingestion / fan-out bounds
    ingest_queue_max: int = 20000
    ws_queue_max: int = 512
    link_metrics_period_s: float = 2.0
    mission_idle_after_s: float = 30.0
    # API
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def ack_addr(self):
        return _addr(self.ack_to)

    @property
    def emulator_addr(self):
        return _addr(self.emulator_control)
