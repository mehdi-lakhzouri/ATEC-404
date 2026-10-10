"""Control of the REAL R5.3 lunar link emulator through its UDP control interface (default 127.0.0.1:5792).
Nothing here simulates a network state: a preset is applied by the emulator, and its reply (live stats) is returned."""

from __future__ import annotations

import asyncio

from artemis_x.telemetry.emulator import PRESETS, send_control

PRESET_NAMES = tuple(PRESETS)


class EmulatorUnavailable(RuntimeError):
    pass


async def command(addr: tuple[str, int], cmd: str, timeout_s: float = 1.5) -> dict:
    """Run the blocking UDP request in a worker thread so the event loop never waits on the network."""
    try:
        return await asyncio.to_thread(send_control, cmd, addr, timeout_s)
    except OSError as e:                                   # timeout / connection refused: emulator not running
        raise EmulatorUnavailable(f"emulator control {addr[0]}:{addr[1]} did not answer ({type(e).__name__})") from e


async def status(addr):
    return await command(addr, "STATUS")


async def apply_preset(addr, preset: str):
    name = preset.upper()
    if name not in PRESET_NAMES:
        raise ValueError(f"unknown preset {preset!r}; expected one of {', '.join(PRESET_NAMES)}")
    return await command(addr, f"PRESET {name}")
