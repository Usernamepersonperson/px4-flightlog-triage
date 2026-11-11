"""Detection on simulated flights with faults injected at known times."""

import numpy as np
import pytest

from flightlog_triage import detectors as D
from flightlog_triage import inject, synth
from flightlog_triage.triage import triage

CASES = [
    (inject.vibration, {"dur": 4, "rms_m_s2": 8}),
    (inject.clipping, {"dur": 0.5}),
    (inject.voltage_sag, {"dur": 3, "v_per_cell": 0.15}),
    (inject.gnss_dropout, {"dur": 5}),
    (inject.gnss_gap, {"dur": 5}),
    (inject.rc_loss, {"dur": 3}),
    (inject.motor_saturation, {"dur": 2}),
    (inject.cpu_spike, {"dur": 3}),
    (inject.estimator_fault, {"dur": 2}),
]


def test_clean_flight_has_no_findings(clean_flight, cfg):
    assert triage(clean_flight, cfg).findings == []


@pytest.mark.parametrize("fn,kw", CASES, ids=[c[0].__name__ for c in CASES])
def test_injected_fault_is_found_on_time(clean_flight, cfg, fn, kw):
    g, truth = fn(clean_flight, 100.0, **kw)
    res = triage(g, cfg)
    hits = [f for f in res.findings if f.kind == truth.kind and f.t_start <= truth.t1 + 2 and f.t_end >= truth.t0 - 2]
    assert hits, f"{truth.fault} not found; got {[f.kind for f in res.findings]}"
    assert abs(hits[0].t_start - truth.t0) <= 1.0
    assert res.findings[0].score >= hits[0].score  # ranked list is sorted


def test_injection_does_not_modify_input(clean_flight):
    before = clean_flight.topics["battery_status"]["voltage_v"].copy()
    inject.voltage_sag(clean_flight, 100.0)
    assert np.array_equal(before, clean_flight.topics["battery_status"]["voltage_v"])


def test_small_vibration_below_threshold_not_flagged(clean_flight, cfg):
    g, _ = inject.vibration(clean_flight, 100.0, 4.0, rms_m_s2=1.0)
    assert "vibration" not in triage(g, cfg).kinds()


def test_short_gnss_dropout_below_min_duration_ignored(clean_flight, cfg):
    g, _ = inject.gnss_dropout(clean_flight, 100.0, dur=0.4)
    assert "gnss_loss" not in triage(g, cfg).kinds()


def test_gnss_gap_detail_says_no_messages(clean_flight, cfg):
    g, _ = inject.gnss_gap(clean_flight, 100.0, dur=5.0)
    loss = [f for f in triage(g, cfg).findings if f.kind == "gnss_loss"]
    assert loss and "no GNSS messages" in loss[0].detail


def test_fault_on_ground_is_ignored(clean_flight, cfg):
    g, _ = inject.vibration(clean_flight, 2.0, 4.0, rms_m_s2=10.0)  # before takeoff at 15 s
    g, _ = inject.cpu_spike(g, 230.0, 3.0)  # after landing at 225 s
    assert triage(g, cfg).findings == []


@pytest.mark.parametrize("r_cell", [0.005, 0.010, 0.020])
def test_internal_resistance_estimate(cfg, r_cell):
    f = synth.make_flight(duration_s=300, seed=2, r_cell_ohm=r_cell)
    m = triage(f, cfg).metrics
    assert m["cells"] == 4
    assert m["r_cell_mohm"] == pytest.approx(r_cell * 1000, rel=0.2)


def test_high_resistance_and_low_voltage_flagged(cfg):
    f = synth.make_flight(duration_s=300, seed=2, r_cell_ohm=0.04)
    kinds = triage(f, cfg).kinds()
    assert "battery_high_resistance" in kinds
    assert "battery_low_voltage" in kinds  # 0.04 ohm x 18 A sags cells below 3.3 V late in flight


def test_threshold_override_changes_result(clean_flight):
    from flightlog_triage.config import load_config

    g, _ = inject.vibration(clean_flight, 100.0, 4.0, rms_m_s2=3.0)
    assert "vibration" not in triage(g, load_config()).kinds()
    assert "vibration" in triage(g, load_config(overrides={"vibration_warn_m_s2": 4.0})).kinds()


def test_fixed_wing_motor_saturation_skipped(clean_flight, cfg):
    g, _ = inject.motor_saturation(clean_flight, 100.0, 5.0)
    g.topics["vehicle_status"]["vehicle_type"][:] = 2
    assert "motor_saturation" not in triage(g, cfg).kinds()


def test_motor_saturation_from_pwm_outputs(clean_flight, cfg):
    g = clean_flight.copy()
    am = g.topics.pop("actuator_motors")
    ao = {"t": am["t"], "noutputs": np.full(len(am["t"]), 4)}
    for i in range(4):
        ao[f"output[{i}]"] = 1000.0 + 1000.0 * am[f"control[{i}]"]
    ao["output[2]"][(am["t"] >= 100) & (am["t"] < 102)] = 2000.0
    g.topics["actuator_outputs"] = ao
    g.params.update(PWM_MAIN_MIN=1000, PWM_MAIN_MAX=2000)
    sat = [f for f in triage(g, cfg).findings if f.kind == "motor_saturation"]
    assert len(sat) == 1 and "motor 3" in sat[0].detail


def test_rc_never_present_is_not_a_loss(clean_flight, cfg):
    g = clean_flight.copy()
    g.topics["input_rc"]["rc_lost"][:] = 1
    g.topics["failsafe_flags"]["manual_control_signal_lost"][:] = 1
    assert "rc_loss" not in triage(g, cfg).kinds()


def test_flaky_receiver_frames_ignored_when_vehicle_flag_is_fine(clean_flight, cfg):
    g = clean_flight.copy()
    rc = g.topics["input_rc"]
    rc["rc_lost"][(rc["t"] > 100) & (rc["t"] < 110) & (np.arange(len(rc["t"])) % 3 == 0)] = 1
    assert "rc_loss" not in triage(g, cfg).kinds()
    del g.topics["failsafe_flags"]["manual_control_signal_lost"]  # receiver flag is all we have
    assert "rc_loss" in triage(g, cfg).kinds()


def test_datalink_loss_from_failsafe_flags(clean_flight, cfg):
    g = clean_flight.copy()
    ff = g.topics["failsafe_flags"]
    ff["gcs_connection_lost"][(ff["t"] >= 100) & (ff["t"] < 106)] = 1
    dl = [f for f in triage(g, cfg).findings if f.kind == "datalink_loss"]
    assert len(dl) == 1 and dl[0].duration_s == pytest.approx(6.0)


def test_estimator_nan_ratios_ignored(clean_flight, cfg):
    g = clean_flight.copy()
    g.topics["estimator_status"]["mag_test_ratio"][:] = np.nan
    assert "estimator_rejection" not in triage(g, cfg).kinds()


def test_gnss_without_fix_in_flight_is_skipped(clean_flight, cfg):
    g = clean_flight.copy()
    g.topics["vehicle_gps_position"]["fix_type"][:] = 0
    out = D.gnss(g, cfg, g.airborne_intervals())
    assert out.findings == [] and "skipped" in out.metrics["gnss_note"]


def test_flaky_threshold_crossings_merge(clean_flight, cfg):
    g = clean_flight.copy()
    cl = g.topics["cpuload"]
    w = (cl["t"] >= 100) & (cl["t"] < 110)
    cl["load"][w] = np.where(np.arange(w.sum()) % 4 == 3, 0.5, 0.97)  # dips every 2 s
    cpu = [f for f in triage(g, cfg).findings if f.kind == "cpu_load"]
    assert len(cpu) == 1 and cpu[0].duration_s >= 9.0
