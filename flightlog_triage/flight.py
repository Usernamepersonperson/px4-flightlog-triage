"""Neutral in-memory flight representation used by every detector.

A topic is a dict of 1-D numpy arrays keyed by the flat pyulog field name
(``accelerometer_m_s2[0]``), plus ``t``: seconds since log start. Keeping detectors on
this structure (not on pyulog objects) lets tests and fault injection edit signals
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Topic = dict[str, np.ndarray]


@dataclass
class FlightData:
    topics: dict[str, Topic] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)
    source: str = ""
    size_bytes: int = 0

    def has(self, topic: str, *fields: str) -> bool:
        t = self.topics.get(topic)
        if t is None or len(t.get("t", ())) == 0:
            return False
        return all(f in t for f in fields)

    def get(self, topic: str, name: str) -> np.ndarray:
        return self.topics[topic][name]

    def copy(self) -> FlightData:
        return FlightData(
            topics={k: {f: v.copy() for f, v in t.items()} for k, t in self.topics.items()},
            params=dict(self.params),
            info=dict(self.info),
            source=self.source,
            size_bytes=self.size_bytes,
        )

    def duration_s(self) -> float:
        ends = [float(t["t"][-1]) for t in self.topics.values() if len(t.get("t", ())) > 0]
        return max(ends) if ends else 0.0

    def airborne_intervals(self) -> list[tuple[float, float]]:
        """Airborne time ranges from vehicle_land_detected, else armed ranges."""
        if self.has("vehicle_land_detected", "landed"):
            tp = self.topics["vehicle_land_detected"]
            return intervals_where(tp["t"], tp["landed"] < 0.5, self.duration_s(), hold=True)
        if self.has("vehicle_status", "arming_state"):
            tp = self.topics["vehicle_status"]
            return intervals_where(tp["t"], tp["arming_state"] == 2, self.duration_s(), hold=True)
        d = self.duration_s()
        return [(0.0, d)] if d > 0 else []

    def airborne_s(self) -> float:
        return float(sum(b - a for a, b in self.airborne_intervals()))

    def vehicle_type(self) -> int:
        """1 = rotary wing, 2 = fixed wing, 0 = unknown (vehicle_status.vehicle_type)."""
        if self.has("vehicle_status", "vehicle_type"):
            vals = self.topics["vehicle_status"]["vehicle_type"]
            if len(vals):
                return int(np.bincount(vals.astype(int).clip(0, 10)).argmax())
        return 0


def intervals_where(
    t: np.ndarray, mask: np.ndarray, end_time: float | None = None, hold: bool = False
) -> list[tuple[float, float]]:
    """Contiguous [start, end] ranges where ``mask`` is True.

    With ``hold=True`` each sample's state holds until the next sample (on-change topics),
    and the last state holds until ``end_time``. Otherwise a run ends at its last sample.
    """
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if len(t) == 0:
        return []
    out: list[tuple[float, float]] = []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask) - 1)
    for s, e in zip(starts, ends, strict=True):
        if hold:
            stop = t[e + 1] if e + 1 < len(t) else (end_time if end_time is not None else t[e])
        else:
            stop = t[e]
        out.append((float(t[s]), float(stop)))
    return out


def in_intervals(t: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for a, b in intervals:
        m |= (t >= a) & (t <= b)
    return m


def step_values(t_src: np.ndarray, v_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Zero-order-hold lookup: value of the last source sample at or before each query."""
    idx = np.searchsorted(t_src, t_query, side="right") - 1
    out = v_src[np.clip(idx, 0, None)].astype(float)
    out[idx < 0] = np.nan
    return out

