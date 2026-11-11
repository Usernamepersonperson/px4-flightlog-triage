"""Detection thresholds.

Every number here is a default, not a tuned value for any airframe. Override any key
with a JSON file: ``flightlog-triage triage log.ulg --config my_thresholds.json``.
Keys are flat: ``{"vibration_warn_m_s2": 5.0}``. Unknown keys raise an error so that
typos do not silently fall back to defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULTS: dict[str, float] = {
    # --- vibration / clipping (sensor_combined accelerometer) ---------------------
    "vib_window_s": 1.0,  # metric is computed per window of this length
    "vib_highpass_s": 0.25,  # moving-mean length removed before taking the RMS
    "vibration_warn_m_s2": 10.0,  # 3-axis RMS of high-passed accel above this = flag
    "vibration_min_s": 2.0,  # flag only if above threshold this long
    "clip_level_m_s2": 152.0,  # |axis| at/above this counts as a clipped sample (~15.5 g)
    "clip_min_count": 1,  # clipped samples per window that make a flag
    # --- battery ------------------------------------------------------------------
    "batt_grid_hz": 5.0,  # battery series resampled to this rate
    "batt_baseline_s": 30.0,  # rolling-median baseline length
    "batt_sag_v_per_cell": 0.12,  # current-compensated dip below baseline (V per cell)
    "batt_sag_min_s": 1.0,
    "batt_min_cell_v": 3.3,  # per-cell voltage under load below this = flag
    "batt_low_min_s": 2.0,
    "batt_r_cell_warn_ohm": 0.02,  # estimated internal resistance per cell above this = flag
    "batt_min_current_std_a": 0.5,  # need this much current variation to estimate R
    "batt_jump_v_per_cell": 0.15,  # median step between 5 Hz samples above this = implausible sensor
    "batt_max_cell_v": 4.5,  # per-cell readings above this (airborne) = cell count setting wrong
    # --- GNSS ---------------------------------------------------------------------
    "gnss_min_fix": 3,  # fix_type below this = no 3D fix
    "gnss_loss_min_s": 1.0,
    "gnss_gap_s": 1.5,  # no GNSS message for this long while airborne = loss
    "gnss_eph_max_m": 3.0,
    "gnss_min_sats": 6,
    "gnss_poor_min_s": 3.0,
    # --- estimator ----------------------------------------------------------------
    "est_test_ratio_fail": 1.0,  # innovation test ratio above 1 = measurement rejected
    "est_min_s": 0.5,
    # --- links --------------------------------------------------------------------
    "link_loss_min_s": 0.5,
    "rc_gap_s": 1.0,  # no input_rc message for this long after RC was seen = loss
    # --- motors -------------------------------------------------------------------
    "motor_sat_high": 0.95,  # normalised motor command at/above this = saturated
    "motor_sat_min_s": 0.5,
    "motor_sat_low": 0.05,  # at/below this while airborne = pinned at minimum
    "motor_sat_low_min_s": 10.0,  # long, because idle phases are normal in descents and yaw moves
    # --- CPU ----------------------------------------------------------------------
    "cpu_load_warn": 0.90,
    "cpu_min_s": 1.0,
    # --- general ------------------------------------------------------------------
    "merge_gap_s": 2.0,  # events of the same kind closer than this are merged
    "log_gap_s": 0.5,  # sensor_combined silent this long = logging dropout (not a link/GNSS fault)
}

# Base weight per finding kind for the ranking score (0-10).
KIND_WEIGHTS: dict[str, float] = {
    "gnss_loss": 9.0,
    "rc_loss": 9.0,
    "battery_low_voltage": 8.0,
    "motor_saturation": 8.0,
    "accel_clipping": 7.0,
    "battery_sag": 7.0,
    "estimator_rejection": 7.0,
    "vibration": 6.0,
    "datalink_loss": 6.0,
    "cpu_load": 5.0,
    "gnss_poor": 4.0,
    "battery_high_resistance": 3.0,
    "battery_sensor_implausible": 5.0,
    "battery_cell_count_mismatch": 3.0,
    "log_dropout": 3.0,
}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Return DEFAULTS merged with a JSON file and/or a dict of overrides."""
    cfg = dict(DEFAULTS)
    extra: dict = {}
    if path is not None:
        extra.update(json.loads(Path(path).read_text(encoding="utf-8")))
    if overrides:
        extra.update(overrides)
    unknown = sorted(set(extra) - set(DEFAULTS))
    if unknown:
        raise KeyError(f"unknown threshold key(s): {', '.join(unknown)}")
    for key, value in extra.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"threshold {key} must be a number, got {value!r}")
        cfg[key] = float(value)
    return cfg
