"""Bounded CPU/MPS use for a shared, always-on Apple-silicon host."""
from __future__ import annotations

import json
import os
import time
import sys
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


def memory_pressure() -> int | None:
    """Darwin's exported pressure bitmask: normal=1, warning=2, critical=4."""
    if sys.platform != 'darwin':
        return None
    value, length = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    libc = ctypes.CDLL(None)
    if libc.sysctlbyname(b'kern.memorystatus_vm_pressure_level', ctypes.byref(value),
                        ctypes.byref(length), None, 0) != 0:
        return None
    return value.value


class ResourceDeferred(Exception):
    """A private update must yield to foreground work; no samples consumed."""


class BackgroundGuard:
    """Recheck during long training, with at most one host sample per five seconds."""
    def __init__(self, policy, deadline=None, report=None):
        self.policy, self.deadline, self.report = policy, deadline, report
        self.next_sample = 0.0

    def __call__(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('background cycle deadline reached')
        if time.monotonic() < self.next_sample:
            return
        status = self.policy.sample()
        self.next_sample = time.monotonic() + 5
        if self.report:
            self.report(status)
        if hasattr(self.policy, '_storage_root'):
            from ml.data_budget import storage_status
            if not storage_status(self.policy._storage_root, self.policy.max_disk_gb)['background_allowed']:
                raise ResourceDeferred('live storage reserve protected; experience retained')
        if not status['training_allowed']:
            raise ResourceDeferred('host headroom changed; private update discarded, experience retained')


@dataclass
class ResourcePolicy:
    reserve_cores: int = 3
    max_workers: int = 4
    training_threads: int = 2
    min_available_gb: float = 2.0
    max_system_cpu_percent: float = 75.0
    worker_memory_mb: int = 400
    gpu_fraction: float = 0.12
    gpu_duty_fraction: float = 0.5
    backend: str = "auto"
    max_disk_gb: float = 8.0
    max_swapout_mb_per_second: float = 32.0

    def __post_init__(self):
        if self.backend not in ("auto", "cpu", "mps"):
            raise ValueError("backend must be auto, cpu or mps")
        if not 1 <= self.max_workers <= 8 or not 1 <= self.training_threads <= 4:
            raise ValueError("workers must be 1..8 and training threads 1..4")
        if not 0.01 <= self.gpu_fraction <= 0.25 or not 0.05 <= self.gpu_duty_fraction <= 1:
            raise ValueError("GPU memory fraction must be .01..25 and duty fraction .05..1")
        if self.reserve_cores < 1 or self.min_available_gb < 1 or not 10 <= self.max_system_cpu_percent <= 90:
            raise ValueError("reserve at least one core, 1 GB and 10% CPU for the host")
        if self.worker_memory_mb < 100 or self.max_disk_gb < 1 or self.max_swapout_mb_per_second <= 0:
            raise ValueError('worker memory >=100 MB, disk budget >=1 GB and positive swap-out limit required')
        self._swap_sample = None

    def sample(self, resident_workers: int = 0):
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        cores = os.cpu_count() or 1
        free = memory.available / 2**30
        pressure = memory_pressure()
        swap = psutil.swap_memory()
        clock = time.monotonic()
        swapout_rate = 0.0
        if self._swap_sample:
            previous_time, previous_bytes = self._swap_sample
            swapout_rate = max(0, swap.sout - previous_bytes) / max(.001, clock - previous_time) / 2**20
        self._swap_sample = (clock, swap.sout)
        memory_workers = max(0, int((free - self.min_available_gb) * 1024 / self.worker_memory_mb) + resident_workers)
        cpu_workers = max(0, int((self.max_system_cpu_percent - cpu) * cores / 100))
        workers = max(0, min(self.max_workers, cores - self.reserve_cores, memory_workers, cpu_workers))
        healthy = (free >= self.min_available_gb and cpu < self.max_system_cpu_percent
                   and (pressure is None or not pressure & 4)
                   and swapout_rate <= self.max_swapout_mb_per_second)
        if not healthy:
            workers = 0
        disk_free = psutil.disk_usage(str(Path.cwd())).free / 2**30
        if disk_free < 2:
            workers = 0
            healthy = False
        return {"cpu_cores": cores, "system_cpu_percent": cpu, "available_gb": round(free, 2),
                "disk_free_gb": round(disk_free, 2), "allowed_workers": workers,
                "memory_pressure": pressure, 'swap_used_gb': round(swap.used / 2**30, 2),
                'swapout_mb_per_second': round(swapout_rate, 2), 'training_allowed': healthy,
                "deferred": workers == 0, "policy": asdict(self)}

    def apply_background(self):
        # Changes affect this worker process and its children, never other services.
        if os.name == "posix":
            try:
                os.nice(10)
            except OSError:
                pass
        os.environ.setdefault("OMP_NUM_THREADS", str(self.training_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.training_threads))
        os.environ["PS_TORCH_THREADS"] = str(self.training_threads)
        import torch
        torch.set_num_threads(self.training_threads)
        if torch.backends.mps.is_available():
            torch.mps.set_per_process_memory_fraction(self.gpu_fraction)


def benchmark(policy: ResourcePolicy, repeats: int = 8) -> dict:
    """Measure actual masked actor training, including GPU synchronization."""
    import torch
    from ml.model import ActorCritic
    from ml.features import STATE_DIM, ACTION_DIM
    torch.set_num_threads(policy.training_threads)
    timings = {}
    for backend in ("cpu", "mps"):
        if backend == "mps" and not torch.backends.mps.is_available():
            continue
        try:
            if backend == "mps":
                torch.mps.set_per_process_memory_fraction(policy.gpu_fraction)
            net = ActorCritic().to(backend)
            optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
            states = torch.zeros((32, STATE_DIM), device=backend)
            actions = torch.randn((32, 80, ACTION_DIM), device=backend)
            def step():
                logits, values = net(states, actions)
                loss = logits.log_softmax(-1)[:, 0].neg().mean() + values.square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for _ in range(2):
                step()
            if backend == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            for _ in range(repeats):
                step()
            if backend == "mps":
                torch.mps.synchronize()
            timings[backend] = {"training_ms": round((time.perf_counter() - started) * 1000 / repeats, 3)}
            del optimizer, net, states, actions
            if backend == "mps":
                torch.mps.empty_cache()
        except (RuntimeError, NotImplementedError) as error:
            timings[backend] = {"error": str(error)[:200]}
    working = {k: v for k, v in timings.items() if "training_ms" in v}
    best = min(working, key=lambda k: working[k]["training_ms"]) if working else "cpu"
    # Avoid spending shared GPU resources for a marginal speedup.
    if best == "mps" and timings["mps"]["training_ms"] > timings["cpu"]["training_ms"] * 0.85:
        best = "cpu"
    return {"selected_backend": best, "measurements": timings, "gpu_memory_fraction": policy.gpu_fraction}


def backend_for(root: Path, policy: ResourcePolicy) -> str:
    if policy.backend != "auto":
        return policy.backend
    path = Path(root) / "compute-benchmark.json"
    if path.exists():
        return json.loads(path.read_text())["selected_backend"]
    result = benchmark(policy)
    path.write_text(json.dumps(result, indent=2))
    return result["selected_backend"]
