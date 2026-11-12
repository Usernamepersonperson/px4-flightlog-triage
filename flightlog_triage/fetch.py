"""Download a sample of public logs from the PX4 Flight Review database into cache/.

Logs are not committed to this repo. The database index (~385 MB of JSON) is streamed and
reduced to a small candidate list on the fly, so it never sits on disk in full.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
import zlib
from pathlib import Path

INDEX_URL = "https://cdn.logs.px4.io/dbinfo.json"
MULTIROTOR = {"Quadrotor", "Hexarotor", "Octorotor"}
GOOD = {"good", "great", "BRAVO!"}
PROBLEM = {"unsatisfactory", "crash_sw_hw", "crash_pilot"}
KEEP = ("log_id", "log_date", "rating", "duration_s", "mav_type", "ver_sw_release", "error_labels",
        "num_logged_errors", "download_url")
UA = {"User-Agent": "px4-flightlog-triage/0.1 (research sample; github.com/VB-Builds)"}


def iter_index(stream, chunk: int = 1 << 20):
    """Yield the objects of a (gzip or plain) JSON array without loading it whole."""
    dec = json.JSONDecoder()
    gz = None
    buf = ""
    first = True
    tail = b""
    while True:
        raw = stream.read(chunk)
        if first and raw[:2] == b"\x1f\x8b":
            gz = zlib.decompressobj(16 + zlib.MAX_WBITS)
        first = False
        if not raw:
            break
        data = gz.decompress(raw) if gz else raw
        data = tail + data
        try:
            text = data.decode("utf-8")
            tail = b""
        except UnicodeDecodeError as e:  # multi-byte char split across chunks
            text = data[: e.start].decode("utf-8")
            tail = data[e.start :]
        buf += text
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in "[, \n\r\t":
                pos += 1
            if pos >= len(buf) or buf[pos] == "]":
                break
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                break  # incomplete object, wait for more data
            yield obj
            pos = end
        buf = buf[pos:]


def _minor(rec: dict) -> int:
    rel = str(rec.get("ver_sw_release", ""))
    try:
        return int(rel.split(".")[1]) if rel.startswith("v1.") else 0
    except (IndexError, ValueError):
        return 0


def is_candidate(rec: dict, min_s: float = 240, max_s: float = 1200, since: str = "2017-01-01") -> bool:
    return (
        rec.get("mav_type") in MULTIROTOR
        and rec.get("estimator") == "EKF2"
        and min_s <= float(rec.get("duration_s") or 0) <= max_s
        and _minor(rec) >= 6
        and str(rec.get("log_date", "")) >= since
        and "SITL" not in str(rec.get("sys_hw", "")).upper()
        and str(rec.get("type", "")) == "flightreport"
    )


def build_candidates(out: Path, source: str | Path = INDEX_URL) -> list[dict]:
    cands = []
    if str(source).startswith("http"):
        req = urllib.request.Request(str(source), headers={**UA, "Accept-Encoding": "gzip"})
        stream = urllib.request.urlopen(req, timeout=120)
    else:
        stream = open(source, "rb")
    with stream:
        for rec in iter_index(stream):
            if is_candidate(rec):
                cands.append({k: rec.get(k) for k in KEEP})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cands), encoding="utf-8")
    return cands


def pick_sample(cands: list[dict], n_good: int, n_other: int, n_problem: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # "rated_good": uploader rated the flight good/great and the log has no logged errors
    # or error labels. "recent": unrated logs from 2024+ on firmware v1.13+ (current topic
    # layout). "rated_problem": uploader reported a crash or an unsatisfactory flight.
    good = [c for c in cands if c["rating"] in GOOD and not c["error_labels"]
            and not c["num_logged_errors"]]
    prob = [c for c in cands if c["rating"] in PROBLEM]
    other = [c for c in cands if not c["rating"] and c["log_date"] >= "2024-01-01"
             and _minor(c) >= 13]
    out = []
    for pool, n, stratum in ((good, n_good, "rated_good"), (other, n_other, "recent"),
                             (prob, n_problem, "rated_problem")):
        rng.shuffle(pool)
        for c in pool[: n * 4]:  # extra so oversize files can be skipped
            out.append({**c, "stratum": stratum, "_want": n})
    return out


def _compact(raw: Path, npz: Path) -> float:
    """Parse a raw log once (timed) and store the needed topics as .npz."""
    from .ulog_io import load_ulog, save_npz

    t0 = time.perf_counter()
    flight = load_ulog(raw)
    parse_s = time.perf_counter() - t0
    save_npz(flight, npz)
    return parse_s


def download_sample(cache: Path, n_good=35, n_other=25, n_problem=10, max_mb=25.0, keep_raw_mb=150.0,
                    seed=7, source: str | Path = INDEX_URL) -> list[dict]:
    """Download, parse and compact a stratified sample.

    Raw .ulg files are kept until ``keep_raw_mb`` is used; later ones are deleted right
    after parsing, so the cache stays small. The parse time of every raw file is recorded
    in the manifest (it is the dominant cost of a triage run).
    """
    cache = Path(cache)
    cand_path = cache / "index_candidates.json"
    if cand_path.exists():
        cands = json.loads(cand_path.read_text(encoding="utf-8"))
    else:
        cands = build_candidates(cand_path, source)
    logs_dir = cache / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = logs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    raw_mb = 0.0
    for m in manifest:  # compact anything downloaded by an older run
        raw = logs_dir / f"{m['log_id']}.ulg"
        npz = logs_dir / f"{m['log_id']}.npz"
        if raw.exists() and ("parse_s" not in m or not npz.exists()):
            try:
                m["parse_s"] = round(_compact(raw, npz), 4)
            except Exception as e:  # noqa: BLE001
                m["parse_error"] = f"{type(e).__name__}: {e}"[:200]
        if raw.exists():
            if raw_mb + m["size_mb"] > keep_raw_mb:
                raw.unlink()
            else:
                raw_mb += m["size_mb"]
        m["raw_kept"] = raw.exists()
    save()

    have = {m["log_id"] for m in manifest}
    tried = set()
    count: dict[str, int] = {}
    for m in manifest:
        count[m["stratum"]] = count.get(m["stratum"], 0) + 1
    for c in pick_sample(cands, n_good, n_other, n_problem, seed):
        s = c["stratum"]
        if count.get(s, 0) >= c["_want"] or c["log_id"] in have or c["log_id"] in tried:
            continue
        tried.add(c["log_id"])
        url = c["download_url"]
        try:
            head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60)
            size_mb = int(head.headers.get("Content-Length", 0)) / 1e6
        except OSError:
            continue
        if size_mb <= 0 or size_mb > max_mb:
            continue
        raw = logs_dir / f"{c['log_id']}.ulg"
        npz = logs_dir / f"{c['log_id']}.npz"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                raw.write_bytes(r.read())
        except OSError:
            continue
        rec = {k: c[k] for k in KEEP}
        rec.update(stratum=s, size_mb=round(size_mb, 3))
        try:
            rec["parse_s"] = round(_compact(raw, npz), 4)
        except Exception as e:  # noqa: BLE001 - keep the record, evaluation counts failures
            rec["parse_error"] = f"{type(e).__name__}: {e}"[:200]
        if raw_mb + size_mb <= keep_raw_mb:
            raw_mb += size_mb
        else:
            raw.unlink()
        rec["raw_kept"] = raw.exists()
        count[s] = count.get(s, 0) + 1
        manifest.append(rec)
        have.add(c["log_id"])
        save()
        print(f"[{len(manifest)}] {s:14s} {size_mb:6.2f} MB  {c['log_date']}  {c['log_id']}", flush=True)
        time.sleep(0.5)
    return manifest
