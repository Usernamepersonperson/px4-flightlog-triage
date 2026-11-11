import json

import numpy as np
import pytest

from flightlog_triage import detectors as D
from flightlog_triage.config import DEFAULTS, load_config
from flightlog_triage.flight import FlightData, in_intervals, intervals_where, step_values

# ----------------------------------------------------------------------------- config


def test_defaults_load():
    cfg = load_config()
    assert cfg == {k: v for k, v in DEFAULTS.items()}


def test_override_dict_and_file(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"vibration_warn_m_s2": 9}))
    cfg = load_config(p, {"cpu_load_warn": 0.8})
    assert cfg["vibration_warn_m_s2"] == 9.0 and cfg["cpu_load_warn"] == 0.8


def test_unknown_key_rejected():
    with pytest.raises(KeyError, match="vibe_typo"):
        load_config(overrides={"vibe_typo": 1})


@pytest.mark.parametrize("bad", ["high", True, None, [1]])
def test_non_numeric_rejected(bad):
    with pytest.raises(TypeError):
        load_config(overrides={"cpu_load_warn": bad})


# ----------------------------------------------------------------------------- intervals


def test_intervals_middle_run():
    t = np.arange(10.0)
    m = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], bool)
    assert intervals_where(t, m) == [(2.0, 4.0), (7.0, 7.0)]


def test_intervals_edges_and_empty():
    t = np.arange(4.0)
    assert intervals_where(t, np.ones(4, bool)) == [(0.0, 3.0)]
    assert intervals_where(t, np.zeros(4, bool)) == []
    assert intervals_where(np.array([]), np.array([], bool)) == []


def test_intervals_hold_extends_to_next_sample_and_end():
    t = np.array([0.0, 10.0, 50.0])
    m = np.array([False, True, False])
    assert intervals_where(t, m, end_time=60, hold=True) == [(10.0, 50.0)]
    m2 = np.array([False, False, True])
    assert intervals_where(t, m2, end_time=60, hold=True) == [(50.0, 60.0)]


def test_in_intervals_and_step_values():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert in_intervals(t, [(0.5, 2.0)]).tolist() == [False, True, True, False]
    v = step_values(np.array([1.0, 3.0]), np.array([5, 7]), np.array([0.0, 1.0, 2.5, 9.0]))
    assert np.isnan(v[0]) and v[1:].tolist() == [5, 5, 7]


def test_airborne_from_land_detector(clean_flight):
    assert clean_flight.airborne_intervals() == [(15.0, 225.0)]
    assert clean_flight.airborne_s() == pytest.approx(210.0)


def test_airborne_falls_back_to_armed():
    f = FlightData(topics={"vehicle_status": {"t": np.arange(10.0),
                                              "arming_state": np.array([1, 1, 2, 2, 2, 2, 1, 1, 1, 1])}})
    assert f.airborne_intervals() == [(2.0, 6.0)]


def test_empty_flight_does_not_crash(cfg):
    from flightlog_triage.triage import triage

    res = triage(FlightData(), cfg)
    assert res.findings == [] and res.airborne_s == 0.0


# ----------------------------------------------------------------------------- helpers


def test_merge_intervals():
    assert D.merge_intervals([(5, 6), (0, 1), (1.5, 2)], gap=1.0) == [(0, 2), (5, 6)]


def test_rolling_median_rejects_spike():
    x = np.ones(101)
    x[50] = 100.0
    assert D.rolling_median(x, 5)[50] == 1.0


def test_moving_mean_constant_and_length():
    y = D.moving_mean(np.full(37, 4.0), 6)
    assert len(y) == 37 and np.allclose(y, 4.0)


def test_score_grows_with_severity_and_duration():
    assert D.score("vibration", 2.0, 1.0) > D.score("vibration", 1.0, 1.0)
    assert D.score("vibration", 1.0, 10.0) > D.score("vibration", 1.0, 1.0)
    assert D.score("gnss_loss", 1.0, 1.0) > D.score("gnss_poor", 1.0, 1.0)
    assert D.score("rc_loss", 1e9, 1e9) <= 100.0


def test_cell_count_sources():
    f = FlightData(topics={"battery_status": {"t": np.arange(3.0)}}, params={"BAT1_N_CELLS": 6})
    assert D.cell_count(f, np.array([25.0])) == 6
    f2 = FlightData(topics={"battery_status": {"t": np.arange(3.0)}})
    assert D.cell_count(f2, np.array([16.7])) == 4  # 4S from max voltage
