"""Resource monitoring for experiment execution."""

import os
import time
from pathlib import Path

import pandas as pd
import psutil

from aintelope.analytics.recording import write_csv

COLUMNS = [
    "label",
    "timestamp",
    "elapsed_s",
    "rss_mb",
    "rss_percent",
    "cpu_percent",
    "gpu_mem_mb",
    "gpu_mem_percent",
    "gpu_util_percent",
]


def _make_gpu_info():
    """Resolve GPU availability once. Returns (total_mb, sampler_fn)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, lambda: (0.0, 0.0)
        total = torch.cuda.get_device_properties(0).total_mem / (1024**2)
        has_util = hasattr(torch.cuda, "utilization")
        if has_util:
            return total, lambda: (
                torch.cuda.memory_allocated() / (1024**2),
                torch.cuda.utilization(),
            )
        return total, lambda: (torch.cuda.memory_allocated() / (1024**2), 0.0)
    except ImportError:
        return 0.0, lambda: (0.0, 0.0)


def _pct(part, whole):
    """Safe percentage."""
    return (part / whole * 100) if whole > 0 else 0.0


def _detect_trends(df):
    """Detect memory drift from repeated labels. Returns list of trend strings."""
    trends = []
    for col, unit in [("rss_mb", "MB"), ("gpu_mem_mb", "MB")]:
        for label, group in df.groupby("label", sort=False):
            if len(group) < 10:
                continue
            first = group[col].iloc[:5].mean()
            last = group[col].iloc[-5:].mean()
            delta = last - first
            tag = "stable" if abs(delta) < 1.0 else f"{delta:+.1f} {unit}"
            trends.append(f"  {col} [{label}]: {first:.1f} → {last:.1f} ({tag})")
    return trends


def _print_report(df, system, context):
    """Print a compact console report."""
    ram, cpus, gpu = system["ram_mb"], system["cpus"], system["gpu_mb"]
    wall = df["elapsed_s"].iloc[-1]

    print("\n── Performance Report ──────────────────────────────")

    ctx_parts = [f"{k}: {v}" for k, v in context.items()]
    ctx_parts.append(f"wall: {wall:.1f}s")
    print(" | ".join(ctx_parts))

    gpu_str = f" | GPU: {gpu:.0f} MB" if gpu > 0 else ""
    print(f"System: {ram:.0f} MB RAM | {cpus} CPUs{gpu_str}")

    summary = df.groupby("label", sort=False).agg(
        count=("elapsed_s", "size"),
        mean_rss=("rss_mb", "mean"),
        max_rss=("rss_mb", "max"),
        rss_pct=("rss_percent", "mean"),
        cpu_pct=("cpu_percent", "mean"),
        gpu_mem=("gpu_mem_mb", "mean"),
        gpu_pct=("gpu_mem_percent", "mean"),
    )
    print(f"\n{summary.round(1).to_string()}")

    trends = _detect_trends(df)
    if trends:
        print("\nTrends:")
        print("\n".join(trends))

    peak_rss = df["rss_mb"].max()
    peak_gpu = df["gpu_mem_mb"].max()
    print(
        f"\nPeak RSS: {peak_rss:.1f} MB ({_pct(peak_rss, ram):.1f}%)"
        f" | Peak GPU: {peak_gpu:.1f} MB ({_pct(peak_gpu, gpu):.1f}%)"
    )
    print("────────────────────────────────────────────────────\n")


class ResourceMonitor:
    """Collects labeled resource snapshots. Fully self-contained."""

    def __init__(self, context={}):
        self._t0 = time.monotonic()
        self._process = psutil.Process()
        self._ram_mb = psutil.virtual_memory().total / (1024**2)
        self._cpus = os.cpu_count()
        self._gpu_mb, self._gpu_sampler = _make_gpu_info()
        self._context = context
        self._rows = []

    def sample(self, label):
        """Record a labeled resource snapshot."""
        rss = self._process.memory_info().rss / (1024**2)
        gpu_mem, gpu_util = self._gpu_sampler()
        self._rows.append(
            [
                label,
                time.time(),
                time.monotonic() - self._t0,
                rss,
                _pct(rss, self._ram_mb),
                self._process.cpu_percent(),
                gpu_mem,
                _pct(gpu_mem, self._gpu_mb),
                gpu_util,
            ]
        )

    def report(self):
        """Print console report."""
        df = pd.DataFrame(self._rows, columns=COLUMNS)
        system = {
            "ram_mb": self._ram_mb,
            "cpus": self._cpus,
            "gpu_mb": self._gpu_mb,
        }
        _print_report(df, system, self._context)

    def save(self, folder):
        """Write performance_report.csv."""
        df = pd.DataFrame(self._rows, columns=COLUMNS)
        write_csv(Path(folder) / "performance_report.csv", df)
