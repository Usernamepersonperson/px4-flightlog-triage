"""Measure the triage on real public logs.

(a) Fault injection: known faults are added to real logs' signals at random airborne
    times and sizes; we record whether the matching flag fires and its timing error.
(b) Untouched logs: how often each flag fires, per stratum, per flight hour.
(c) Speed: parse + triage wall time per MB and per flight hour (single thread).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import detectors, inject
from .config import load_config
from .triage import triage
from .ulog_io import load_npz, load_ulog

# fault -> (injector, detector, magnitudes, magnitude unit, duration s)
FAULTS = {
    "vibration": (inject.vibration, detectors.vibration, [1.5, 3.0, 6.0, 12.0], "m/s2 RMS per axis", 4.0),
    "voltage_sag": (inject.voltage_sag, detectors.battery, [0.06, 0.12, 0.24, 0.48], "V per cell", 3.0),
    "gnss_dropout": (inject.gnss_dropout, detectors.gnss, [0.5, 1.5, 3.0, 6.0], "s without 3D fix", None),
}
POSITIONS = 3  # injections per log, fault and magnitude
# The first DEV_LOGS manifest entries were looked at while choosing default thresholds;
# false-flag rates are reported separately for the remaining (held-out) logs.
DEV_LOGS = 25
MATCH_TOL_S = 2.0
# kinds that describe a vehicle fault; the rest are logging / configuration / link notes
FAULT_KINDS = {"vibration", "accel_clipping", "battery_sag", "battery_low_voltage", "gnss_loss", "gnss_poor",
               "estimator_rejection", "rc_loss", "motor_saturation", "cpu_load"}
EDGE_S = 20.0


def _usable(flight, res, fault) -> bool:
    m = res.metrics
    if fault == "vibration":
        return flight.has("sensor_combined", "accelerometer_m_s2[0]")
    if fault == "voltage_sag":  # the sag check needs a current signal and a plausible voltage
        return m.get("r_cell_mohm") is not None and "battery_note" not in m
    if fault == "gnss_dropout":
        return "gnss_fix" in res.series and "gnss_note" not in m
    return False


def _pick_times(air, busy, n, dur, rng) -> list[float]:
    spans = [(a + EDGE_S, b - EDGE_S - dur) for a, b in air if b - a > 2 * EDGE_S + dur]
    if not spans:
        return []
    lengths = np.array([b - a for a, b in spans])
    out = []
    for _ in range(200):
        if len(out) == n:
            break
        k = rng.choice(len(spans), p=lengths / lengths.sum())
        t0 = float(rng.uniform(*spans[k]))
        clash = any(t0 - 10 < b and t0 + dur + 10 > a for a, b in busy)
        near = any(abs(t0 - o) < dur + 10 for o in out)
        if not clash and not near:
            out.append(t0)
    return out


def run(logs_dir: Path, out_dir: Path, cfg: dict | None = None, seed: int = 11, limit: int | None = None) -> dict:
    cfg = cfg or load_config()
    logs_dir, out_dir = Path(logs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((logs_dir / "manifest.json").read_text(encoding="utf-8"))
    if limit:
        manifest = manifest[:limit]
    rng = np.random.default_rng(seed)
    per_log, trials, failures = [], [], []
    for i, m in enumerate(manifest):
        npz = logs_dir / f"{m['log_id']}.npz"
        raw = logs_dir / f"{m['log_id']}.ulg"
        if m.get("parse_error"):
            failures.append({"log_id": m["log_id"], "error": m["parse_error"]})
            continue
        try:
            t0 = time.perf_counter()
            flight = load_npz(npz) if npz.exists() else load_ulog(raw)
            # speed is reported for raw-ULog parsing (timed at fetch) + triage
            load_s = m.get("parse_s", time.perf_counter() - t0)
            res = triage(flight, cfg)
        except Exception as e:  # noqa: BLE001 - count unreadable logs, keep going
            failures.append({"log_id": m["log_id"], "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        res.load_s = load_s
        air = flight.airborne_intervals()
        kinds = {}
        for f in res.findings:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        per_log.append({
            "log_id": m["log_id"], "stratum": m["stratum"], "log_date": m["log_date"],
            "set": "dev" if i < DEV_LOGS else "holdout",
            "ver": m["ver_sw_release"], "size_mb": m["size_mb"], "airborne_s": round(res.airborne_s, 1),
            "load_s": round(load_s, 3), "triage_s": round(res.triage_s, 3), "kinds": kinds,
            "findings": [f.to_dict() for f in res.findings], "metrics": res.metrics,
        })
        for fault, (inj, det, mags, _unit, dur) in FAULTS.items():
            if not _usable(flight, res, fault):
                continue
            exp_kind = inject.KIND[fault]
            busy = [(f.t_start, f.t_end) for f in res.findings if f.kind == exp_kind]
            for mag in mags:
                d = mag if fault == "gnss_dropout" else dur
                for t_inj in _pick_times(air, busy, POSITIONS, d, rng):
                    if fault == "vibration":
                        g, truth = inj(flight, t_inj, dur, mag, seed=int(rng.integers(1 << 30)))
                    elif fault == "voltage_sag":
                        g, truth = inj(flight, t_inj, dur, mag, cells=res.metrics.get("cells") or 1)
                    else:
                        g, truth = inj(flight, t_inj, mag)
                    out = det(g, cfg, air)
                    hits = [f for f in out.findings if f.kind == exp_kind
                            and f.t_start <= truth.t1 + MATCH_TOL_S and f.t_end >= truth.t0 - MATCH_TOL_S]
                    err = min((h.t_start - truth.t0 for h in hits), key=abs) if hits else None
                    trials.append({"log_id": m["log_id"], "fault": fault, "magnitude": mag,
                                   "t0": round(truth.t0, 2), "detected": bool(hits),
                                   "timing_error_s": None if err is None else round(err, 3)})
        print(f"[{i + 1}/{len(manifest)}] {m['stratum']:13s} air {res.airborne_s / 60:5.1f} min "
              f"{len(res.findings):3d} flags  {m['log_id']}", flush=True)
    summary = summarize(per_log, trials, failures, cfg)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "per_log.json").write_text(json.dumps(per_log, indent=1), encoding="utf-8")
    with (out_dir / "injection_trials.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trials[0]) if trials else ["log_id"])
        w.writeheader()
        w.writerows(trials)
    return summary


def summarize(per_log: list[dict], trials: list[dict], failures: list[dict], cfg: dict) -> dict:
    out: dict = {"logs_parsed": len(per_log), "logs_failed": len(failures), "failures": failures}
    if per_log:
        dates = sorted(p["log_date"] for p in per_log)
        out["date_range"] = [dates[0], dates[-1]]
    # (a) injection
    inj = {}
    for fault, (_i, _d, mags, unit, _dur) in FAULTS.items():
        rows = [t for t in trials if t["fault"] == fault]
        by_mag = {}
        for mag in mags:
            r = [t for t in rows if t["magnitude"] == mag]
            by_mag[str(mag)] = {"trials": len(r), "detected": sum(t["detected"] for t in r),
                                "rate": round(sum(t["detected"] for t in r) / len(r), 4) if r else None}
        errs = np.array([abs(t["timing_error_s"]) for t in rows if t["timing_error_s"] is not None])
        inj[fault] = {
            "unit": unit, "logs": len({t["log_id"] for t in rows}), "by_magnitude": by_mag,
            "timing_abs_median_s": round(float(np.median(errs)), 3) if len(errs) else None,
            "timing_abs_p95_s": round(float(np.percentile(errs, 95)), 3) if len(errs) else None,
        }
    out["injection"] = inj
    # (b) untouched logs
    out["real_logs"] = _flag_stats(per_log)
    out["real_logs_holdout"] = _flag_stats([p for p in per_log if p.get("set") == "holdout"])
    # (c) speed
    mb = sum(p["size_mb"] for p in per_log)
    secs = sum(p["load_s"] + p["triage_s"] for p in per_log)
    hours = sum(p["airborne_s"] for p in per_log) / 3600
    out["speed"] = {
        "total_mb": round(mb, 1), "total_s": round(secs, 2), "mb_per_s": round(mb / secs, 2) if secs else None,
        "s_per_flight_hour": round(secs / hours, 1) if hours else None,
        "parse_share": round(sum(p["load_s"] for p in per_log) / secs, 3) if secs else None,
        "airborne_h": round(hours, 3),
    }
    out["config"] = cfg
    return out


def _flag_stats(per_log: list[dict]) -> dict:
    strata = {}
    for s in sorted({p["stratum"] for p in per_log}):
        rows = [p for p in per_log if p["stratum"] == s]
        hours = sum(p["airborne_s"] for p in rows) / 3600
        kinds: dict[str, int] = {}
        logs_with: dict[str, int] = {}
        for p in rows:
            for k, n in p["kinds"].items():
                kinds[k] = kinds.get(k, 0) + n
                logs_with[k] = logs_with.get(k, 0) + 1
        total = sum(kinds.values())
        faults = sum(n for k, n in kinds.items() if k in FAULT_KINDS)
        strata[s] = {
            "logs": len(rows), "airborne_h": round(hours, 3), "flags": total,
            "flags_per_h": round(total / hours, 2) if hours else None,
            "fault_flags": faults,
            "fault_flags_per_h": round(faults / hours, 2) if hours else None,
            "logs_without_fault_flags": sum(1 for p in rows if not set(p["kinds"]) & FAULT_KINDS),
            "logs_without_flags": sum(1 for p in rows if not p["kinds"]),
            "per_kind": {k: {"flags": n, "per_h": round(n / hours, 2) if hours else None,
                             "logs": logs_with[k]} for k, n in sorted(kinds.items())},
        }
    return strata


def resummarize(out_dir: Path, cfg: dict | None = None) -> dict:
    """Rebuild eval.json from per_log.json + injection_trials.csv (no re-processing)."""
    out_dir = Path(out_dir)
    per_log = json.loads((out_dir / "per_log.json").read_text(encoding="utf-8"))
    with (out_dir / "injection_trials.csv").open(encoding="utf-8") as fh:
        trials = []
        for r in csv.DictReader(fh):
            r["magnitude"] = float(r["magnitude"])
            r["detected"] = r["detected"] == "True"
            r["timing_error_s"] = float(r["timing_error_s"]) if r["timing_error_s"] else None
            trials.append(r)
    old = json.loads((out_dir / "eval.json").read_text(encoding="utf-8"))
    summary = summarize(per_log, trials, old.get("failures", []), cfg or old["config"])
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
