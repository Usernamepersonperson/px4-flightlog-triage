"""Simulated multicopter flight (clean), used by tests and the offline demo.

Signals are simple physical stand-ins, not a flight simulator: hover current with slow
manoeuvre swings, an open-circuit voltage that falls linearly with time, a resistive
voltage drop, broadband accelerometer noise, a steady GNSS fix and low estimator test
ratios. Faults are added afterwards with ``inject.py``.
"""

from __future__ import annotations

import numpy as np

from .flight import FlightData

SCHEMAS = {
    "sensor_combined": [("float", "accelerometer_m_s2", 3), ("uint8_t", "accelerometer_clipping", 0)],
    "vehicle_land_detected": [("bool", "landed", 0)],
    "vehicle_status": [
        ("uint8_t", "arming_state", 0),
        ("uint8_t", "vehicle_type", 0),
        ("uint8_t", "nav_state", 0),
    ],
    "battery_status": [
        ("float", "voltage_v", 0),
        ("float", "current_a", 0),
        ("float", "remaining", 0),
        ("uint8_t", "cell_count", 0),
    ],
    "vehicle_gps_position": [
        ("float", "eph", 0),
        ("float", "epv", 0),
        ("float", "hdop", 0),
        ("uint8_t", "fix_type", 0),
        ("uint8_t", "satellites_used", 0),
    ],
    "estimator_status": [
        ("float", "vel_test_ratio", 0),
        ("float", "pos_test_ratio", 0),
        ("float", "hgt_test_ratio", 0),
        ("float", "mag_test_ratio", 0),
    ],
    "input_rc": [("int32_t", "rssi", 0), ("bool", "rc_lost", 0)],
    "failsafe_flags": [("bool", "manual_control_signal_lost", 0), ("bool", "gcs_connection_lost", 0)],
    "actuator_motors": [("float", "control", 8)],
    "cpuload": [("float", "load", 0)],
}

PARAMS = {"BAT1_N_CELLS": 4, "MPC_THR_HOVER": 0.5}


def make_flight(
    duration_s: float = 300.0,
    takeoff_s: float = 15.0,
    land_s: float | None = None,
    cells: int = 4,
    r_cell_ohm: float = 0.008,
    accel_noise: float = 1.2,
    seed: int = 0,
    imu_hz: float = 100.0,
) -> FlightData:
    rng = np.random.default_rng(seed)
    land_s = duration_s - 15.0 if land_s is None else land_s

    def grid(hz: float) -> np.ndarray:
        return np.arange(0.0, duration_s, 1.0 / hz)

    def airborne(t: np.ndarray) -> np.ndarray:
        return (t >= takeoff_s) & (t < land_s)

    topics: dict[str, dict[str, np.ndarray]] = {}

    t = grid(imu_hz)
    air = airborne(t)
    sd = np.where(air, accel_noise, 0.05)
    acc = np.stack([rng.normal(0, 1, len(t)) * sd for _ in range(3)])
    acc[2] -= 9.81
    topics["sensor_combined"] = {
        "t": t,
        "accelerometer_m_s2[0]": acc[0],
        "accelerometer_m_s2[1]": acc[1],
        "accelerometer_m_s2[2]": acc[2],
        "accelerometer_clipping": np.zeros(len(t), dtype=np.uint8),
    }

    topics["vehicle_land_detected"] = {
        "t": np.array([0.0, takeoff_s, land_s]),
        "landed": np.array([1, 0, 1], dtype=np.uint8),
    }

    t = grid(1.0)
    armed = (t >= takeoff_s - 5) & (t < land_s + 3)
    topics["vehicle_status"] = {
        "t": t,
        "arming_state": np.where(armed, 2, 1).astype(np.uint8),
        "vehicle_type": np.ones(len(t), dtype=np.uint8),
        "nav_state": np.full(len(t), 2, dtype=np.uint8),
    }

    t = grid(5.0)
    air = airborne(t)
    swing = 4.0 * np.sin(2 * np.pi * t / 20.0) + 2.0 * np.sin(2 * np.pi * t / 7.0)
    current = np.where(air, 14.0 + swing + rng.normal(0, 0.4, len(t)), 0.6)
    frac = np.clip((t - takeoff_s) / max(land_s - takeoff_s, 1.0), 0, 1)
    voc = cells * (4.15 - 0.40 * frac)
    voltage = voc - r_cell_ohm * cells * current + rng.normal(0, 0.008, len(t))
    topics["battery_status"] = {
        "t": t,
        "voltage_v": voltage,
        "current_a": current,
        "remaining": 1.0 - 0.7 * frac,
        "cell_count": np.full(len(t), cells, dtype=np.uint8),
    }

    t = grid(5.0)
    topics["vehicle_gps_position"] = {
        "t": t,
        "eph": 0.7 + 0.1 * rng.random(len(t)),
        "epv": 1.1 + 0.1 * rng.random(len(t)),
        "hdop": np.full(len(t), 0.8),
        "fix_type": np.full(len(t), 3, dtype=np.uint8),
        "satellites_used": np.full(len(t), 14, dtype=np.uint8),
    }

    t = grid(10.0)
    topics["estimator_status"] = {
        "t": t,
        **{
            f"{k}_test_ratio": np.abs(rng.normal(0.15, 0.08, len(t)))
            for k in ("vel", "pos", "hgt", "mag")
        },
    }

    t = grid(10.0)
    topics["input_rc"] = {
        "t": t,
        "rssi": np.full(len(t), 100, dtype=np.int32),
        "rc_lost": np.zeros(len(t), dtype=np.uint8),
    }
    t = grid(1.0)
    topics["failsafe_flags"] = {
        "t": t,
        "manual_control_signal_lost": np.zeros(len(t), dtype=np.uint8),
        "gcs_connection_lost": np.zeros(len(t), dtype=np.uint8),
    }

    t = grid(50.0)
    air = airborne(t)
    motors = {"t": t}
    for i in range(8):
        if i < 4:
            base = 0.5 + 0.05 * np.sin(2 * np.pi * t / 9.0 + i) + rng.normal(0, 0.02, len(t))
            motors[f"control[{i}]"] = np.where(air, base, 0.0)
        else:
            motors[f"control[{i}]"] = np.full(len(t), np.nan)
    topics["actuator_motors"] = motors

    t = grid(2.0)
    topics["cpuload"] = {"t": t, "load": 0.45 + 0.03 * rng.random(len(t))}

    return FlightData(
        topics=topics,
        params=dict(PARAMS),
        info={"sys_name": "PX4", "ver_sw_release": "simulated"},
        source=f"simulated(seed={seed})",
    )
