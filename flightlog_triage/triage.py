"""Run every detector on one flight and rank the findings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import detectors
from .config import load_config
from .detectors import Finding
from .flight import FlightData


@dataclass
class TriageResult:
    source: str
    duration_s: float
    airborne_s: float
    findings: list[Finding]
    metrics: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)
    load_s: float = 0.0
    triage_s: float = 0.0
    size_bytes: int = 0
    info: dict = field(default_factory=dict)

    def kinds(self) -> set[str]:
        return {f.kind for f in self.findings}

    def to_dict(self) -> dict:
        return {
            "source": Path(self.source).name if self.source else "",
            "duration_s": round(self.duration_s, 2),
            "airborne_s": round(self.airborne_s, 2),
            "size_bytes": self.size_bytes,
            "load_s": round(self.load_s, 4),
            "triage_s": round(self.triage_s, 4),
            "info": self.info,
            "metrics": self.metrics,
            "findings": [f.to_dict() for f in self.findings],
        }


def triage(flight: FlightData, cfg: dict | None = None) -> TriageResult:
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    air = flight.airborne_intervals()
    findings: list[Finding] = []
    metrics: dict = {}
    series: dict = {}
    for det in detectors.ALL:
        out = det(flight, cfg, air)
        findings += out.findings
        metrics.update(out.metrics)
        series.update(out.series)
    findings.sort(key=lambda f: (-f.score, f.t_start))
    series["_airborne"] = air
    return TriageResult(
        source=flight.source,
        duration_s=flight.duration_s(),
        airborne_s=float(sum(b - a for a, b in air)),
        findings=findings,
        metrics=metrics,
        series=series,
        triage_s=time.perf_counter() - t0,
        size_bytes=flight.size_bytes,
        info={k: v for k, v in flight.info.items() if k in ("ver_sw_release", "sys_name")},
    )


def triage_file(path: str | Path, cfg: dict | None = None) -> TriageResult:
    from .ulog_io import load_ulog

    t0 = time.perf_counter()
    flight = load_ulog(path)
    load_s = time.perf_counter() - t0
    res = triage(flight, cfg)
    res.load_s = load_s
    return res
