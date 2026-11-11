"""ULog writer/reader round trip, report output, CLI and index streaming."""

import gzip
import io
import json

import numpy as np
import pytest
from pyulog.exceptions import ULogException

from flightlog_triage import fetch, inject, synth
from flightlog_triage.cli import main
from flightlog_triage.report import render_html, render_png
from flightlog_triage.triage import triage, triage_file
from flightlog_triage.ulog_io import HEADER_MAGIC, load_npz, load_ulog, save_npz, write_ulog


@pytest.fixture(scope="module")
def ulog_path(tmp_path_factory, clean_flight):
    g, _ = inject.gnss_dropout(clean_flight, 120.0, 5.0)
    return write_ulog(tmp_path_factory.mktemp("u") / "flight.ulg", g, synth.SCHEMAS, synth.PARAMS)


def test_ulog_header(ulog_path):
    raw = ulog_path.read_bytes()
    assert raw[:7] == HEADER_MAGIC and raw[7] == 1


def test_ulog_round_trip_values(ulog_path, clean_flight):
    f = load_ulog(ulog_path)
    src = clean_flight.topics["battery_status"]
    got = f.topics["battery_status"]
    assert np.allclose(got["t"], src["t"], atol=1e-6)
    assert np.allclose(got["voltage_v"], src["voltage_v"].astype(np.float32))
    assert got["cell_count"].tolist() == src["cell_count"].tolist()
    assert np.isnan(f.topics["actuator_motors"]["control[5]"]).all()
    assert f.params["BAT1_N_CELLS"] == 4
    assert f.info["ver_sw_release"] == "simulated"
    assert f.size_bytes == ulog_path.stat().st_size


def test_triage_from_file_matches_memory(ulog_path):
    res = triage_file(ulog_path)
    assert [f.kind for f in res.findings] == ["gnss_loss"]
    assert res.findings[0].t_start == pytest.approx(120.0, abs=0.01)
    assert res.load_s > 0


def test_invalid_file_raises(tmp_path):
    p = tmp_path / "bad.ulg"
    p.write_bytes(b"not a ulog file at all")
    with pytest.raises(ULogException):
        load_ulog(p)


def test_truncated_file_still_parses_prefix(tmp_path, ulog_path):
    raw = ulog_path.read_bytes()
    p = tmp_path / "cut.ulg"
    p.write_bytes(raw[: len(raw) // 2])
    f = load_ulog(p)
    assert 0 < f.duration_s() < 240


def test_npz_round_trip(tmp_path, clean_flight):
    p = save_npz(clean_flight, tmp_path / "f.npz")
    g = load_npz(p)
    assert g.params == clean_flight.params
    assert np.array_equal(g.topics["cpuload"]["load"], clean_flight.topics["cpuload"]["load"])
    assert [f.kind for f in triage(g).findings] == []


def test_report_png_and_html(clean_flight, cfg):
    g, _ = inject.cpu_spike(clean_flight, 100.0)
    res = triage(g, cfg)
    png = render_png(res, cfg=cfg)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    page = render_html(res, png, cfg)
    assert "cpu_load" in page and "data:image/png;base64," in page


def test_cli_single_file(tmp_path, ulog_path, capsys):
    assert main(["triage", str(ulog_path), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "gnss_loss" in out
    assert (tmp_path / "o" / "flight.html").exists() and (tmp_path / "o" / "flight.png").exists()


def test_cli_folder_json_and_index(tmp_path, ulog_path, capsys):
    d = tmp_path / "logs"
    d.mkdir()
    (d / "a.ulg").write_bytes(ulog_path.read_bytes())
    (d / "broken.ulg").write_bytes(b"xx")
    assert main(["triage", str(d), "--out", str(tmp_path / "o"), "--json", "--no-png"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    assert json.loads(lines[0])["findings"][0]["kind"] == "gnss_loss"
    assert (tmp_path / "o" / "summary.csv").read_text().count("\n") == 2


def test_cli_empty_folder(tmp_path):
    assert main(["triage", str(tmp_path)]) == 2


def test_cli_demo_finds_all_faults(tmp_path):
    assert main(["demo", "--out", str(tmp_path)]) == 0


# ----------------------------------------------------------------------------- index streaming

RECS = [{"log_id": f"id{i}", "description": "café ✈" * (i % 3)} for i in range(50)]


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("chunk", [7, 64, 1 << 16])
def test_iter_index_streams_json_array(compress, chunk):
    raw = json.dumps(RECS).encode()
    if compress:
        raw = gzip.compress(raw)
    got = list(fetch.iter_index(io.BytesIO(raw), chunk=chunk))
    assert got == RECS


def test_candidate_filter():
    base = {"mav_type": "Quadrotor", "estimator": "EKF2", "duration_s": 600, "ver_sw_release": "v1.14.0 0",
            "log_date": "2025-01-01", "sys_hw": "FMU_V6", "type": "flightreport"}
    assert fetch.is_candidate(base)
    assert not fetch.is_candidate({**base, "mav_type": "Fixed Wing"})
    assert not fetch.is_candidate({**base, "sys_hw": "PX4_SITL"})
    assert not fetch.is_candidate({**base, "duration_s": 30})
    assert not fetch.is_candidate({**base, "ver_sw_release": "v1.5.0"})


def test_pick_sample_strata():
    cands = (
        [{"log_id": f"g{i}", "rating": "good", "error_labels": [], "num_logged_errors": 0,
          "log_date": "2018-01-01", "ver_sw_release": "v1.8.0"} for i in range(10)]
        + [{"log_id": f"r{i}", "rating": "", "error_labels": [], "num_logged_errors": 0,
            "log_date": "2025-01-01", "ver_sw_release": "v1.15.0"} for i in range(10)]
        + [{"log_id": f"c{i}", "rating": "crash_pilot", "error_labels": [], "num_logged_errors": 3,
            "log_date": "2019-01-01", "ver_sw_release": "v1.9.0"} for i in range(10)]
    )
    s = fetch.pick_sample(cands, 2, 2, 1, seed=1)
    strata = {c["stratum"] for c in s}
    assert strata == {"rated_good", "recent", "rated_problem"}
    assert all(c["log_id"].startswith("g") for c in s if c["stratum"] == "rated_good")
