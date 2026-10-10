"""Background GPU/CPU/RAM/disk sampler for training runs (pynvml + psutil + torch.cuda), written to CSV.

Sampling is read-only and cheap (no GPU synchronisation); it never changes training behaviour.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path


class ResourceMonitor:
    FIELDS = ("t_s", "gpu_util_pct", "gpu_mem_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w",
              "cuda_alloc_mb", "cuda_reserved_mb", "cpu_pct", "cpu_per_core_pct", "proc_tree_cpu_pct", "dataloader_workers",
              "ram_used_gb", "ram_available_gb", "proc_tree_rss_gb", "disk_read_mb_s")

    def __init__(self, out_csv: Path, period_s: float = 5.0, gpu_index: int = 0):
        self.out, self.period, self.gpu_index = Path(out_csv), period_s, gpu_index
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "ResourceMonitor":
        self.start(); return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set(); self._thread.join(timeout=2 * self.period)

    def _loop(self) -> None:
        import psutil
        try:
            import pynvml
            pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:  # monitoring must never break training
            pynvml = None; h = None
        try:
            import torch
            cuda = torch.cuda.is_available()
        except Exception:
            torch = None; cuda = False
        me = psutil.Process()
        procs: dict[int, psutil.Process] = {}
        psutil.cpu_percent(percpu=True)
        disk_prev = psutil.disk_io_counters(); t_prev = time.time()
        t0 = time.time()
        with self.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(self.FIELDS)
            while not self._stop.wait(self.period):
                now = time.time()
                row = dict.fromkeys(self.FIELDS, "")
                row["t_s"] = round(now - t0, 2)
                per = psutil.cpu_percent(percpu=True)
                row["cpu_pct"] = round(sum(per) / len(per), 1)
                row["cpu_per_core_pct"] = json.dumps(per)
                try:
                    tree = [me] + me.children(recursive=True)
                    cpu = 0.0; rss = 0
                    for p in tree:
                        if p.pid not in procs:
                            procs[p.pid] = p; p.cpu_percent(None)    # first call primes the counter
                        else:
                            cpu += procs[p.pid].cpu_percent(None)
                        rss += p.memory_info().rss
                    row["proc_tree_cpu_pct"] = round(cpu, 1)             # % of one core, summed over the process tree
                    row["dataloader_workers"] = sum(1 for p in tree[1:] if "python" in p.name().lower())
                    row["proc_tree_rss_gb"] = round(rss / 2**30, 2)
                except psutil.Error:
                    pass
                vm = psutil.virtual_memory()
                row["ram_used_gb"] = round(vm.used / 2**30, 2)
                row["ram_available_gb"] = round(vm.available / 2**30, 2)
                d = psutil.disk_io_counters()
                row["disk_read_mb_s"] = round((d.read_bytes - disk_prev.read_bytes) / 2**20 / max(now - t_prev, 1e-6), 2)
                disk_prev, t_prev = d, now
                if h is not None:
                    try:
                        u = pynvml.nvmlDeviceGetUtilizationRates(h); m = pynvml.nvmlDeviceGetMemoryInfo(h)
                        row.update(gpu_util_pct=u.gpu, gpu_mem_util_pct=u.memory, gpu_mem_used_mb=m.used // 2**20,
                                   gpu_mem_total_mb=m.total // 2**20, gpu_temp_c=pynvml.nvmlDeviceGetTemperature(h, 0),
                                   gpu_power_w=round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1))
                    except Exception:
                        pass
                if cuda:
                    row["cuda_alloc_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
                    row["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
                w.writerow([row[k] for k in self.FIELDS]); fh.flush()
