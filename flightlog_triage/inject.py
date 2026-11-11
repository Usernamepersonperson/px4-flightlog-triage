"""Inject known faults into a flight's signals (ground truth for evaluation).

Each injector returns a new FlightData and never edits its input. ``KIND`` maps each fault
to the finding kind the triage should report for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flight import FlightData

KIND = {
    "vibration": "vibration",
    "voltage_sag": "battery_sag",
    "gnss_dropout": "gnss_loss",
    "gnss_gap": "gnss_loss",
    "rc_loss": "rc_loss",
    "motor_saturation": "motor_saturation",
    "cpu_spike": "cpu_load",
    "estimator_fault": "estimator_rejection",
    "clipping": "accel_clipping",
}


@dataclass
class Injected:
    fault: str
    t0: float
    t1: float
    magnitude: float

    @property
    def kind(self) -> str:
        return KIND[self.fault]


def _window(t: np.ndarray, t0: float, dur: float) -> np.ndarray:
    return (t >= t0) & (t < t0 + dur)


def vibration(f: FlightData, t0: float, dur: float = 4.0, rms_m_s2: float = 6.0, seed: int = 0):
    """Add broadband noise of ``rms_m_s2`` per axis (white noise at the logged rate)."""
    g = f.copy()
    sc = g.topics["sensor_combined"]
    m = _window(sc["t"], t0, dur)
    rng = np.random.default_rng(seed)
    for ax in range(3):
        key = f"accelerometer_m_s2[{ax}]"
        sc[key] = sc[key].astype(float)
        sc[key][m] += rng.normal(0.0, rms_m_s2, int(m.sum()))
    return g, Injected("vibration", t0, t0 + dur, rms_m_s2)


def clipping(f: FlightData, t0: float, dur: float = 0.5, level: float = 160.0):
    g = f.copy()
    sc = g.topics["sensor_combined"]
    m = _window(sc["t"], t0, dur)
    sc["accelerometer_m_s2[2]"] = sc["accelerometer_m_s2[2]"].astype(float)
    sc["accelerometer_m_s2[2]"][m] = -level
    return g, Injected("clipping", t0, t0 + dur, level)


def voltage_sag(
    f: FlightData, t0: float, dur: float = 3.0, v_per_cell: float = 0.15, cells: int = 4,
    ramp_s: float = 0.3,
):
    """Subtract a trapezoid voltage dip that the measured current does not explain."""
    g = f.copy()
    bs = g.topics["battery_status"]
    t = bs["t"]
    depth = v_per_cell * cells
    up = np.clip((t - t0) / ramp_s, 0, 1)
    down = np.clip((t0 + dur - t) / ramp_s, 0, 1)
    shape = np.minimum(up, down)
    bs["voltage_v"] = bs["voltage_v"].astype(float) - depth * shape
    return g, Injected("voltage_sag", t0, t0 + dur, v_per_cell)


def gnss_dropout(f: FlightData, t0: float, dur: float = 5.0):
    """Receiver keeps reporting but loses the fix (fix_type 0, no satellites)."""
    g = f.copy()
    gp = g.topics["vehicle_gps_position"]
    m = _window(gp["t"], t0, dur)
    gp["fix_type"] = gp["fix_type"].copy()
    gp["fix_type"][m] = 0
    if "satellites_used" in gp:
        gp["satellites_used"] = gp["satellites_used"].copy()
        gp["satellites_used"][m] = 0
    if "eph" in gp:
        gp["eph"] = gp["eph"].astype(float)
        gp["eph"][m] = 99.0
    return g, Injected("gnss_dropout", t0, t0 + dur, dur)


def gnss_gap(f: FlightData, t0: float, dur: float = 5.0):
    """Receiver stops sending messages entirely (cable / power fault)."""
    g = f.copy()
    gp = g.topics["vehicle_gps_position"]
    keep = ~_window(gp["t"], t0, dur)
    g.topics["vehicle_gps_position"] = {k: v[keep] for k, v in gp.items()}
    return g, Injected("gnss_gap", t0, t0 + dur, dur)


def rc_loss(f: FlightData, t0: float, dur: float = 3.0):
    g = f.copy()
    rc = g.topics["input_rc"]
    rc["rc_lost"] = rc["rc_lost"].copy()
    rc["rc_lost"][_window(rc["t"], t0, dur)] = 1
    for topic, fld in (("failsafe_flags", "manual_control_signal_lost"), ("vehicle_status", "rc_signal_lost")):
        if g.has(topic, fld):
            tp = g.topics[topic]
            tp[fld] = tp[fld].copy()
            tp[fld][_window(tp["t"], t0, dur)] = 1
    return g, Injected("rc_loss", t0, t0 + dur, dur)


def motor_saturation(f: FlightData, t0: float, dur: float = 2.0, motor: int = 0, value: float = 1.0):
    g = f.copy()
    am = g.topics["actuator_motors"]
    key = f"control[{motor}]"
    am[key] = am[key].astype(float)
    am[key][_window(am["t"], t0, dur)] = value
    return g, Injected("motor_saturation", t0, t0 + dur, value)


def cpu_spike(f: FlightData, t0: float, dur: float = 3.0, load: float = 0.98):
    g = f.copy()
    cl = g.topics["cpuload"]
    cl["load"] = cl["load"].astype(float)
    cl["load"][_window(cl["t"], t0, dur)] = load
    return g, Injected("cpu_spike", t0, t0 + dur, load)


def estimator_fault(f: FlightData, t0: float, dur: float = 2.0, field: str = "vel_test_ratio",
                    value: float = 1.6):
    g = f.copy()
    es = g.topics["estimator_status"]
    es[field] = es[field].astype(float)
    es[field][_window(es["t"], t0, dur)] = value
    return g, Injected("estimator_fault", t0, t0 + dur, value)
