"""Fault detectors. Each takes FlightData + config and returns findings, plot series and
metrics. All detectors only judge airborne time unless noted, so ground handling and
bench tests do not raise flags."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import KIND_WEIGHTS
from .flight import FlightData, in_intervals, intervals_where

Interval = tuple[float, float]


@dataclass
class Finding:
    kind: str
    t_start: float
    t_end: float
    peak: float
    threshold: float
    unit: str
    detail: str
    score: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = round(self.duration_s, 3)
        for k in ("t_start", "t_end", "peak", "threshold", "score"):
            d[k] = round(float(d[k]), 4)
        return d


@dataclass
class DetectorOutput:
    findings: list[Finding] = field(default_factory=list)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers


def merge_intervals(iv: list[Interval], gap: float) -> list[Interval]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def intersect(iv: list[Interval], air: list[Interval]) -> list[Interval]:
    out = []
    for a, b in iv:
        for c, d in air:
            lo, hi = max(a, c), min(b, d)
            if hi > lo or (hi == lo and a == b):
                out.append((lo, hi))
    return out


def shrink(air: list[Interval], margin: float) -> list[Interval]:
    return [(a + margin, b - margin) for a, b in air if b - a > 2 * margin]


def sample_events(
    t: np.ndarray, mask: np.ndarray, min_s: float, merge_gap: float, dt: float | None = None
) -> list[Interval]:
    """Runs of True in a regularly sampled mask; each sample covers ``dt`` seconds."""
    if len(t) == 0:
        return []
    if dt is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    runs = [(a, b + dt) for a, b in intervals_where(t, mask)]
    runs = merge_intervals(runs, merge_gap)
    return [r for r in runs if r[1] - r[0] >= min_s - 1e-9]


def hold_events(
    t: np.ndarray, mask: np.ndarray, end: float, air: list[Interval], min_s: float, merge_gap: float
) -> list[Interval]:
    """Like sample_events, for on-change topics where a state holds until the next sample."""
    runs = intersect(intervals_where(t, mask, end_time=end, hold=True), air)
    runs = merge_intervals(runs, merge_gap)
    return [r for r in runs if r[1] - r[0] >= min_s - 1e-9]


def peak_in(t: np.ndarray, y: np.ndarray, a: float, b: float, mode: str = "max") -> float:
    m = (t >= a) & (t <= b) & np.isfinite(y)
    if not m.any():
        return float("nan")
    return float(np.max(y[m]) if mode == "max" else np.min(y[m]))


def moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.astype(float)
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x.astype(float), (pad, n - 1 - pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n) | 1)  # odd
    if n == 1 or len(x) == 0:
        return x.astype(float)
    pad = n // 2
    xp = np.pad(x.astype(float), pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, n)
    out = np.empty(len(x))
    step = 4096  # chunked to bound memory on long flights
    for i in range(0, len(x), step):
        out[i : i + step] = np.nanmedian(win[i : i + step], axis=1)
    return out


def score(kind: str, ratio: float, duration_s: float) -> float:
    w = KIND_WEIGHTS.get(kind, 5.0)
    r = 1.0 + min(2.0, math.log2(max(1.0, ratio))) if np.isfinite(ratio) else 1.0
    d = 1.0 + min(1.0, max(duration_s, 0.0) / 10.0)
    return round(min(100.0, w * r * d * 100.0 / 54.0), 1)


def log_gaps(f: FlightData, min_gap: float) -> list[Interval]:
    """Spans where the highest-rate topic is silent: the logger dropped data, so silence
    in other topics there is not evidence of a sensor or link fault."""
    if not f.has("sensor_combined"):
        return []
    t = f.topics["sensor_combined"]["t"]
    d = np.diff(t)
    return [(float(t[i]), float(t[i + 1])) for i in np.flatnonzero(d > min_gap)]


def drop_overlapping(events: list[Interval], gaps: list[Interval], pad: float = 0.5) -> list[Interval]:
    return [(a, b) for a, b in events if not any(a < gb + pad and b > ga - pad for ga, gb in gaps)]


def _finding(kind, a, b, peak, thr, unit, detail, ratio=None) -> Finding:
    if ratio is None:
        ratio = peak / thr if thr and np.isfinite(peak) else 1.0
    return Finding(kind, a, b, peak, thr, unit, detail, score(kind, ratio, b - a))


# ----------------------------------------------------------------------------- vibration


def vibration(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    if not f.has("sensor_combined", "accelerometer_m_s2[0]"):
        return out
    sc = f.topics["sensor_combined"]
    t = sc["t"]
    if len(t) < 20:
        return out
    fs = 1.0 / float(np.median(np.diff(t)))
    n_hp = max(2, int(round(cfg["vib_highpass_s"] * fs)))
    win = cfg["vib_window_s"]
    bins = np.floor(t / win).astype(np.int64)
    bins -= bins.min()
    sumsq = np.zeros(bins.max() + 1)
    clipcount = np.zeros(bins.max() + 1)
    clipped = np.zeros(len(t), dtype=bool)
    for ax in range(3):
        a = sc[f"accelerometer_m_s2[{ax}]"].astype(float)
        a = np.where(np.isfinite(a), a, 0.0)
        hp = a - moving_mean(a, n_hp)
        sumsq += np.bincount(bins, weights=hp * hp, minlength=len(sumsq))
        clipped |= np.abs(a) >= cfg["clip_level_m_s2"]
    if "accelerometer_clipping" in sc:
        clipped |= sc["accelerometer_clipping"].astype(np.int64) != 0
    clipcount += np.bincount(bins, weights=clipped.astype(float), minlength=len(sumsq))
    count = np.bincount(bins, minlength=len(sumsq)).astype(float)
    ok = count >= 5
    rms = np.full(len(sumsq), np.nan)
    rms[ok] = np.sqrt(sumsq[ok] / count[ok])
    tw = (np.arange(len(sumsq)) + 0.5) * win + np.floor(t[0] / win) * win
    in_air = in_intervals(tw, air)
    out.series["vibration"] = (tw, rms)

    thr = cfg["vibration_warn_m_s2"]
    for a, b in sample_events(tw - win / 2, in_air & (rms > thr), cfg["vibration_min_s"],
                              cfg["merge_gap_s"], dt=win):
        pk = peak_in(tw, rms, a, b)
        out.findings.append(_finding("vibration", a, b, pk, thr, "m/s2",
                                     f"high-passed accel RMS peaks at {pk:.1f} m/s2"))
    cthr = cfg["clip_min_count"]
    for a, b in sample_events(tw - win / 2, in_air & (clipcount >= cthr), 0.0,
                              cfg["merge_gap_s"], dt=win):
        n = float(clipcount[(tw >= a) & (tw <= b)].sum())
        out.findings.append(_finding("accel_clipping", a, b, n, cthr, "samples",
                                     f"{int(n)} clipped accelerometer samples"))
    air_rms = rms[in_air & ok]
    out.metrics["vibration_p95_m_s2"] = (
        round(float(np.percentile(air_rms, 95)), 3) if len(air_rms) else None
    )
    out.metrics["clipped_samples"] = int(clipcount[in_air].sum())
    out.metrics["imu_log_rate_hz"] = round(fs, 1)
    return out


# ----------------------------------------------------------------------------- battery


def cell_count(f: FlightData, v: np.ndarray) -> int:
    bs = f.topics.get("battery_status", {})
    if "cell_count" in bs and len(bs["cell_count"]):
        n = int(np.median(bs["cell_count"]))
        if 1 <= n <= 16:
            return n
    for p in ("BAT1_N_CELLS", "BAT_N_CELLS"):
        n = int(f.params.get(p, 0))
        if 1 <= n <= 16:
            return n
    vmax = float(np.nanmax(v)) if len(v) else 0.0
    return max(1, math.ceil(vmax / 4.35))


def battery(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    if not f.has("battery_status", "voltage_v"):
        return out
    bs = f.topics["battery_status"]
    t, v = bs["t"], bs["voltage_v"].astype(float)
    valid = np.isfinite(v) & (v > 1.0)
    if valid.sum() < 20:
        return out
    t, v = t[valid], v[valid]
    cur = bs["current_a"].astype(float)[valid] if "current_a" in bs else np.full(len(t), np.nan)
    cells = cell_count(f, v)
    hz = cfg["batt_grid_hz"]
    g = np.arange(t[0], t[-1], 1.0 / hz)
    if len(g) < 10:
        return out
    V = np.interp(g, t, v)
    has_i = np.isfinite(cur).sum() > 20 and np.nanmax(cur) > 1.0
    curr = np.interp(g, t[np.isfinite(cur)], cur[np.isfinite(cur)]) if has_i else np.zeros(len(g))
    n_base = int(cfg["batt_baseline_s"] * hz)
    air_g = in_intervals(g, air)
    out.series["battery_v"] = (g, V)
    out.metrics["cells"] = cells
    if air_g.sum() > 10:
        high = float(np.mean(V[air_g] / cells > cfg["batt_max_cell_v"]))
        if high > 0.05:
            # configured cell count does not match the pack: re-derive it from the voltage
            guess = max(1, math.ceil(float(np.max(V)) / 4.35))
            if guess != cells:
                out.findings.append(_finding(
                    "battery_cell_count_mismatch", air[0][0], air[-1][1], guess, cells, "cells",
                    f"voltage fits {guess} cells but {cells} are configured/reported; using {guess}",
                    ratio=1.0))
                cells = guess
                out.metrics["cells"] = cells
        jump = float(np.median(np.abs(np.diff(V[air_g])))) / cells
        if jump > cfg["batt_jump_v_per_cell"]:
            out.findings.append(_finding(
                "battery_sensor_implausible", air[0][0], air[-1][1], jump, cfg["batt_jump_v_per_cell"],
                "V/cell", f"battery voltage reading jumps {jump:.2f} V/cell between samples (median); "
                "battery checks skipped", ratio=1.0))
            out.metrics["battery_note"] = "implausible voltage reading; checks skipped"
            return out

    r_pack = None
    if has_i and air_g.sum() > n_base:
        vhp = (V - rolling_median(V, n_base))[air_g]
        ihp = (curr - rolling_median(curr, n_base))[air_g]
        if np.std(ihp) >= cfg["batt_min_current_std_a"]:
            r_pack = float(np.clip(-np.cov(vhp, ihp)[0, 1] / np.var(ihp, ddof=1), 0.0, 1.0))
    comp = V + (r_pack or 0.0) * curr
    resid = comp - rolling_median(comp, n_base)
    out.series["battery_resid"] = (g, resid / cells)
    out.metrics["r_pack_mohm"] = None if r_pack is None else round(r_pack * 1000, 2)
    out.metrics["r_cell_mohm"] = None if r_pack is None else round(r_pack * 1000 / cells, 2)
    if r_pack is not None and air_g.any():
        out.metrics["sag_at_p95_current_v_per_cell"] = round(
            r_pack * float(np.percentile(curr[air_g], 95)) / cells, 3)
    out.metrics["min_cell_v_airborne"] = round(float(np.min(V[air_g])) / cells, 3) if air_g.any() else None

    thr = cfg["batt_sag_v_per_cell"]
    # without a current measurement a load step and a fault look the same: skip the sag check
    sag_mask = air_g & (resid / cells < -thr) if r_pack is not None else np.zeros(len(g), bool)
    if r_pack is None:
        out.metrics["battery_note"] = "no usable current signal; sag check skipped"
    for a, b in sample_events(g, sag_mask, cfg["batt_sag_min_s"], cfg["merge_gap_s"]):
        pk = -peak_in(g, resid / cells, a, b, "min")
        out.findings.append(_finding("battery_sag", a, b, pk, thr, "V/cell",
                                     f"voltage dips {pk:.2f} V/cell below baseline, not explained by current"))
    lthr = cfg["batt_min_cell_v"]
    for a, b in sample_events(g, air_g & (V / cells < lthr), cfg["batt_low_min_s"], cfg["merge_gap_s"]):
        pk = peak_in(g, V / cells, a, b, "min")
        out.findings.append(_finding("battery_low_voltage", a, b, pk, lthr, "V/cell",
                                     f"cell voltage under load down to {pk:.2f} V",
                                     ratio=1.0 + 10 * (lthr - pk)))
    if r_pack is not None and air and r_pack / cells > cfg["batt_r_cell_warn_ohm"]:
        rc = r_pack / cells
        out.findings.append(_finding("battery_high_resistance", air[0][0], air[-1][1], rc * 1000,
                                     cfg["batt_r_cell_warn_ohm"] * 1000, "mOhm/cell",
                                     f"estimated internal resistance {rc * 1000:.0f} mOhm per cell"))
    return out


# ----------------------------------------------------------------------------- GNSS


def gnss(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    if not f.has("vehicle_gps_position", "fix_type") or not air:
        return out
    gp = f.topics["vehicle_gps_position"]
    t = gp["t"]
    fix = gp["fix_type"].astype(float)
    good = fix >= cfg["gnss_min_fix"]
    in_air = in_intervals(t, air)
    if not (good & in_air).any():
        out.metrics["gnss_note"] = "no 3D fix while airborne; GNSS checks skipped"
        return out
    first_fix = float(t[good & in_air][0])
    watch = [(max(a, first_fix), b) for a, b in air if b > first_fix]
    out.series["gnss_fix"] = (t, fix)
    if "satellites_used" in gp:
        out.series["gnss_sats"] = (t, gp["satellites_used"].astype(float))

    lost = sample_events(t, in_intervals(t, watch) & ~good, cfg["gnss_loss_min_s"], cfg["merge_gap_s"])
    # message gaps (receiver silent), including silence until the end of the flight
    tw = t[in_intervals(t, watch)]
    gaps = []
    if len(tw):
        d = np.diff(tw)
        for i in np.flatnonzero(d > cfg["gnss_gap_s"]):
            gaps.append((float(tw[i]), float(tw[i + 1])))
        if watch[-1][1] - tw[-1] > cfg["gnss_gap_s"]:
            gaps.append((float(tw[-1]), watch[-1][1]))
    gaps = drop_overlapping(gaps, log_gaps(f, cfg["log_gap_s"]))
    for a, b in merge_intervals(lost + gaps, cfg["merge_gap_s"]):
        is_gap = any(ga >= a - 1e-6 and gb <= b + 1e-6 for ga, gb in gaps)
        detail = "no GNSS messages" if is_gap else "no 3D fix"
        out.findings.append(_finding("gnss_loss", a, b, b - a, cfg["gnss_loss_min_s"], "s",
                                     f"{detail} for {b - a:.1f} s", ratio=1.0))
    poor = np.zeros(len(t), dtype=bool)
    why = []
    if "eph" in gp:
        poor |= gp["eph"].astype(float) > cfg["gnss_eph_max_m"]
        why.append(f"eph>{cfg['gnss_eph_max_m']:g} m")
    if "satellites_used" in gp:
        poor |= gp["satellites_used"].astype(float) < cfg["gnss_min_sats"]
        why.append(f"sats<{cfg['gnss_min_sats']:g}")
    for a, b in sample_events(t, poor & good & in_intervals(t, watch), cfg["gnss_poor_min_s"],
                              cfg["merge_gap_s"]):
        pk = peak_in(t, gp["eph"].astype(float), a, b) if "eph" in gp else float("nan")
        out.findings.append(_finding("gnss_poor", a, b, pk, cfg["gnss_eph_max_m"], "m",
                                     f"degraded fix ({' or '.join(why)})"))
    if "satellites_used" in gp and in_air.any():
        out.metrics["gnss_min_sats_airborne"] = int(np.min(gp["satellites_used"][in_air]))
    return out


# ----------------------------------------------------------------------------- estimator

TEST_RATIOS = ("vel", "pos", "hgt", "mag", "hagl", "tas", "beta")


def estimator(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    if not f.has("estimator_status"):
        return out
    es = f.topics["estimator_status"]
    t = es["t"]
    in_air = in_intervals(t, air)
    thr = cfg["est_test_ratio_fail"]
    worst = np.zeros(len(t))
    for name in TEST_RATIOS:
        key = f"{name}_test_ratio"
        if key not in es:
            continue
        y = es[key].astype(float)
        y = np.where(np.isfinite(y), y, np.nan)  # inf/NaN = that aiding source is not in use
        worst = np.fmax(worst, np.nan_to_num(y))
        for a, b in sample_events(t, in_air & (np.nan_to_num(y) > thr), cfg["est_min_s"], cfg["merge_gap_s"]):
            pk = peak_in(t, y, a, b)
            out.findings.append(_finding("estimator_rejection", a, b, pk, thr, "ratio",
                                         f"{name} innovation test ratio up to {pk:.2f}"))
    out.series["est_worst_ratio"] = (t, worst)
    return out


# ----------------------------------------------------------------------------- RC / data link

# vehicle-level flags (with the firmware's own hysteresis) are preferred; the receiver's
# per-frame input_rc.rc_lost and input_rc message gaps are used only when neither exists
RC_VEHICLE = (("vehicle_status", "rc_signal_lost"), ("failsafe_flags", "manual_control_signal_lost"))
RC_RECEIVER = (("input_rc", "rc_lost"),)
LINK_SOURCES = (("vehicle_status", "data_link_lost"), ("failsafe_flags", "gcs_connection_lost"))


def _loss_events(f, sources, cfg, air, end) -> tuple[list[Interval], list[str]]:
    events, used = [], []
    for topic, fld in sources:
        if not f.has(topic, fld):
            continue
        tp = f.topics[topic]
        lost = tp[fld].astype(float) > 0.5
        if lost.all():
            continue  # link never present in this log: nothing was lost
        first_ok = float(tp["t"][~lost][0])
        watch = [(max(a, first_ok), b) for a, b in air if b > first_ok]
        events += hold_events(tp["t"], lost, end, watch, cfg["link_loss_min_s"], cfg["merge_gap_s"])
        used.append(f"{topic}.{fld}")
    return events, used


def links(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    end = f.duration_s()
    vehicle_level = any(f.has(t, k) for t, k in RC_VEHICLE)
    rc, used = _loss_events(f, RC_VEHICLE if vehicle_level else RC_RECEIVER, cfg, air, end)
    if not vehicle_level and f.has("input_rc", "rc_lost"):
        rt = f.topics["input_rc"]
        ok = rt["t"][rt["rc_lost"].astype(float) < 0.5]
        if len(ok):
            tw = rt["t"][in_intervals(rt["t"], air) & (rt["t"] >= ok[0])]
            d = np.diff(tw)
            gaps = [(float(tw[i]), float(tw[i + 1])) for i in np.flatnonzero(d > cfg["rc_gap_s"])]
            rc += drop_overlapping(gaps, log_gaps(f, cfg["log_gap_s"]))
    for a, b in merge_intervals(rc, cfg["merge_gap_s"]):
        out.findings.append(_finding("rc_loss", a, b, b - a, cfg["link_loss_min_s"], "s",
                                     f"RC signal lost for {b - a:.1f} s", ratio=1.0))
    dl, used2 = _loss_events(f, LINK_SOURCES, cfg, air, end)
    for a, b in merge_intervals(dl, cfg["merge_gap_s"]):
        out.findings.append(_finding("datalink_loss", a, b, b - a, cfg["link_loss_min_s"], "s",
                                     f"ground-station link lost for {b - a:.1f} s", ratio=1.0))
    out.metrics["link_sources"] = ",".join(used + used2) or None
    return out


# ----------------------------------------------------------------------------- motors


def _motor_matrix(f: FlightData, air: list[Interval]) -> tuple[np.ndarray, np.ndarray] | None:
    if f.has("actuator_motors", "control[0]"):
        am = f.topics["actuator_motors"]
        t = am["t"]
        cols = []
        in_air = in_intervals(t, air)
        for i in range(16):
            k = f"control[{i}]"
            if k in am:
                y = am[k].astype(float)
                ya = y[in_air]
                if len(ya) and np.isfinite(ya).mean() > 0.9 and np.nanmax(ya) > 0.01:
                    cols.append(y)
        return (t, np.stack(cols)) if cols else None
    if f.has("actuator_outputs", "output[0]"):
        ao = f.topics["actuator_outputs"]
        t = ao["t"]
        lo = f.params.get("PWM_MAIN_MIN", f.params.get("PWM_MIN", 1000.0))
        hi = f.params.get("PWM_MAIN_MAX", f.params.get("PWM_MAX", 2000.0))
        if hi - lo < 100:
            return None
        n = int(np.median(ao["noutputs"])) if "noutputs" in ao else 8
        in_air = in_intervals(t, air)
        cols = []
        for i in range(min(n, 16)):
            y = ao.get(f"output[{i}]")
            if y is None:
                continue
            ya = y[in_air].astype(float)
            if len(ya) and lo + 50 < np.median(ya) < hi and np.std(ya) > 1.0:
                cols.append((y.astype(float) - lo) / (hi - lo))
        return (t, np.stack(cols)) if len(cols) >= 3 else None
    return None


def motors(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    if f.vehicle_type() == 2:  # fixed wing: full throttle is normal
        return out
    mm = _motor_matrix(f, air)
    if mm is None:
        return out
    t, m = mm
    core = shrink(air, 2.0)  # ignore spool-up / spool-down
    in_air = in_intervals(t, core)
    hi_v = np.nanmax(m, axis=0)
    lo_v = np.nanmin(m, axis=0)
    out.series["motor_max"] = (t, hi_v)
    out.series["motor_min"] = (t, lo_v)
    out.metrics["motors"] = int(m.shape[0])
    sat_hi = hi_v >= cfg["motor_sat_high"]
    sat_lo = lo_v <= cfg["motor_sat_low"]
    events = [(ev, "max") for ev in sample_events(t, in_air & sat_hi, cfg["motor_sat_min_s"], cfg["merge_gap_s"])]
    events += [(ev, "min") for ev in sample_events(t, in_air & sat_lo, cfg["motor_sat_low_min_s"],
                                                     cfg["merge_gap_s"])]
    for (a, b), side in sorted(events):
        w = (t >= a) & (t <= b)
        hit = m[:, w] >= cfg["motor_sat_high"] if side == "max" else m[:, w] <= cfg["motor_sat_low"]
        idx = [int(i) + 1 for i in np.flatnonzero(hit.any(axis=1))]
        if side == "min" and len(idx) == m.shape[0]:
            continue  # every motor at idle = commanded descent, not an imbalance
        pk = peak_in(t, hi_v, a, b) if side == "max" else peak_in(t, lo_v, a, b, "min")
        detail = f"motor {','.join(map(str, idx))} pinned at {side} for {b - a:.1f} s"
        if side == "max" and len(idx) == m.shape[0]:
            detail += " (all motors: no thrust headroom)"
        thr = cfg["motor_sat_high"] if side == "max" else cfg["motor_sat_low"]
        out.findings.append(_finding("motor_saturation", a, b, pk, thr, "cmd", detail, ratio=1.0))
    if in_air.any():
        out.metrics["motor_sat_fraction"] = round(float(((sat_hi | sat_lo) & in_air).sum() / in_air.sum()), 4)
    return out


# ----------------------------------------------------------------------------- CPU


def cpu(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    if not f.has("cpuload", "load"):
        return out
    cl = f.topics["cpuload"]
    t, y = cl["t"], cl["load"].astype(float)
    out.series["cpu"] = (t, y)
    thr = cfg["cpu_load_warn"]
    for a, b in sample_events(t, in_intervals(t, air) & (y > thr), cfg["cpu_min_s"], cfg["merge_gap_s"]):
        pk = peak_in(t, y, a, b)
        out.findings.append(_finding("cpu_load", a, b, pk, thr, "load",
                                     f"CPU load up to {pk * 100:.0f} %"))
    ya = y[in_intervals(t, air)]
    out.metrics["cpu_max_airborne"] = round(float(np.nanmax(ya)), 3) if len(ya) else None
    return out


# ----------------------------------------------------------------------------- logging


def logging(f: FlightData, cfg: dict, air: list[Interval]) -> DetectorOutput:
    out = DetectorOutput()
    gaps = merge_intervals(intersect(log_gaps(f, cfg["log_gap_s"]), air), cfg["merge_gap_s"])
    for a, b in gaps:
        out.findings.append(_finding("log_dropout", a, b, b - a, cfg["log_gap_s"], "s",
                                     f"logger lost {b - a:.1f} s of data", ratio=1.0))
    out.metrics["log_dropout_s"] = round(sum(b - a for a, b in gaps), 2)
    return out


ALL = (vibration, battery, gnss, estimator, links, motors, cpu, logging)
