"""CUDA / AMP compatibility check for the RTX 5060 Laptop (8 GB).

Default mode only reads device metadata (no kernels launched). `--compute` additionally runs a tiny fp16/bf16
autocast matmul + backward and an AMP GradScaler step - run it at the start of an AUTHORISED training session only.

Usage (from backend/):  uv run --no-sync python -m training.env_check [--compute] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


def check(compute: bool = False) -> dict[str, Any]:
    import torch
    import ultralytics
    r: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                         "torch_cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available()}
    if not r["cuda_available"]:
        r["verdict"] = "FAIL: CUDA not available"
        return r
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    r.update({"device": p.name, "capability": arch, "total_vram_gb": round(p.total_memory / 2**30, 2),
              "multiprocessors": p.multi_processor_count, "arch_in_torch_build": arch in torch.cuda.get_arch_list(),
              "torch_arch_list": torch.cuda.get_arch_list(), "bf16_supported": torch.cuda.is_bf16_supported(),
              "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32})
    if compute:
        torch.manual_seed(0)
        a = torch.randn(256, 256, device="cuda", requires_grad=True)
        res = {}
        for dt in (torch.float16, torch.bfloat16):
            with torch.autocast("cuda", dtype=dt):
                y = (a @ a.T).relu().mean()
            y.backward()
            res[str(dt)] = bool(torch.isfinite(a.grad).all().item())
            a.grad = None
        scaler = torch.amp.GradScaler("cuda")
        opt = torch.optim.SGD([a], lr=1e-3)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = (a @ a.T).float().pow(2).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        res["gradscaler_step_finite"] = bool(torch.isfinite(a).all().item())
        r["amp_compute_check"] = res
        r["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    ok = r["arch_in_torch_build"] and r["total_vram_gb"] >= 7.5 and (not compute or all(r["amp_compute_check"].values()))
    r["verdict"] = "PASS" if ok else "CHECK"
    r["mode"] = "metadata+compute" if compute else "metadata only (no kernels launched)"
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = check(a.compute)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
