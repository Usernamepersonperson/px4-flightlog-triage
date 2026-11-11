"""Command line: triage / demo / fetch / evaluate / figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# keep numeric libraries to one thread (the tool is meant to run beside other work)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from .config import load_config  # noqa: E402
from .report import mmss, write_index, write_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _print(res) -> None:
    print(f"{Path(res.source).name}: airborne {res.airborne_s / 60:.1f} min, "
          f"{len(res.findings)} finding(s), {res.load_s + res.triage_s:.2f} s")
    for i, f in enumerate(res.findings, 1):
        print(f"  {i:2d}. [{f.score:5.1f}] {f.kind:24s} at {mmss(f.t_start):>7s} "
              f"for {f.duration_s:6.1f} s  {f.detail}")


def cmd_triage(a) -> int:
    from .triage import triage_file

    cfg = load_config(a.config)
    p = Path(a.path)
    files = sorted(p.glob("*.ulg")) if p.is_dir() else [p]
    if not files:
        print(f"no .ulg files in {p}", file=sys.stderr)
        return 2
    out = Path(a.out)
    results = []
    for f in files:
        try:
            res = triage_file(f, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"{f.name}: could not parse ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        results.append(res)
        if a.json:
            print(json.dumps(res.to_dict()))
        else:
            _print(res)
        if not a.no_report:
            write_report(res, out, cfg, png=not a.no_png)
    if len(files) > 1 and results and not a.no_report:
        idx = write_index(results, out)
        print(f"index: {idx['html']}")
    elif results and not a.no_report:
        print(f"report: {out / (Path(results[0].source).stem + '.html')}")
    return 0 if results else 1


def cmd_demo(a) -> int:
    from . import inject, synth
    from .triage import triage_file
    from .ulog_io import write_ulog

    cfg = load_config(a.config)
    f = synth.make_flight(duration_s=360, seed=3)
    truth = []
    for fn, t0, kw in (
        (inject.vibration, 60, {"dur": 4, "rms_m_s2": 8}),
        (inject.voltage_sag, 100, {"dur": 3, "v_per_cell": 0.15}),
        (inject.gnss_dropout, 140, {"dur": 6}),
        (inject.rc_loss, 180, {"dur": 3}),
        (inject.motor_saturation, 210, {"dur": 2}),
        (inject.estimator_fault, 240, {"dur": 2}),
        (inject.cpu_spike, 270, {"dur": 3}),
        (inject.clipping, 300, {"dur": 0.5}),
    ):
        f, tr = fn(f, t0, **kw)
        truth.append(tr)
    out = Path(a.out)
    path = write_ulog(out / "demo_flight.ulg", f, synth.SCHEMAS, synth.PARAMS)
    res = triage_file(path, cfg)
    _print(res)
    found = 0
    print("\nground truth (simulated faults):")
    for tr in truth:
        hit = [x for x in res.findings if x.kind == tr.kind and x.t_start <= tr.t1 + 2 and x.t_end >= tr.t0 - 2]
        found += bool(hit)
        err = f"{hit[0].t_start - tr.t0:+.2f} s" if hit else "-"
        print(f"  {tr.fault:18s} at {tr.t0:6.1f} s -> {'found' if hit else 'MISSED'} ({err})")
    print(f"{found}/{len(truth)} injected faults found")
    paths = write_report(res, out, cfg)
    print(f"report: {paths['html']}")
    return 0 if found == len(truth) else 1


def cmd_fetch(a) -> int:
    from .fetch import download_sample

    m = download_sample(Path(a.cache), n_good=a.good, n_other=a.recent, n_problem=a.problem,
                        max_mb=a.max_mb, keep_raw_mb=a.keep_raw_mb, seed=a.seed)
    print(f"{len(m)} logs, {sum(x['size_mb'] for x in m):.1f} MB in {Path(a.cache) / 'logs'}")
    return 0


def cmd_evaluate(a) -> int:
    from .evaluate import run

    s = run(Path(a.cache) / "logs", Path(a.out), load_config(a.config), limit=a.limit)
    print(json.dumps({k: s[k] for k in ("logs_parsed", "logs_failed", "speed")}, indent=1))
    return 0


def cmd_figures(a) -> int:
    from .figures import make_all

    make_all(Path(a.results), Path(a.docs))
    return 0


def cmd_reproduce(a) -> int:
    """fetch + evaluate + figures with default settings (downloads ~1 GB, keeps <300 MB)."""
    root = Path(a.root)
    base = ["--cache", str(root / "cache")]
    main(["fetch", *base])
    main(["evaluate", *base, "--out", str(root / "results")])
    return main(["figures", "--results", str(root / "results"), "--docs", str(root / "docs")])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flightlog-triage", description="Rank problems in PX4 ULog flight logs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="triage one .ulg file or every .ulg in a folder")
    t.add_argument("path")
    t.add_argument("--out", default="triage_out")
    t.add_argument("--config", help="JSON file with threshold overrides")
    t.add_argument("--json", action="store_true", help="print one JSON object per log")
    t.add_argument("--no-png", action="store_true")
    t.add_argument("--no-report", action="store_true")
    t.set_defaults(fn=cmd_triage)

    d = sub.add_parser("demo", help="offline demo on a simulated flight with 8 injected faults")
    d.add_argument("--out", default=str(ROOT / "cache" / "demo"))
    d.add_argument("--config")
    d.set_defaults(fn=cmd_demo)

    f = sub.add_parser("fetch", help="download a sample of public logs into cache/")
    f.add_argument("--cache", default=str(ROOT / "cache"))
    f.add_argument("--good", type=int, default=35)
    f.add_argument("--recent", type=int, default=25)
    f.add_argument("--problem", type=int, default=10)
    f.add_argument("--max-mb", type=float, default=25.0, help="skip logs larger than this")
    f.add_argument("--keep-raw-mb", type=float, default=150.0,
                   help="raw .ulg kept up to this total; later logs are kept only as parsed .npz")
    f.add_argument("--seed", type=int, default=7)
    f.set_defaults(fn=cmd_fetch)

    e = sub.add_parser("evaluate", help="fault injection + flag statistics on the fetched logs")
    e.add_argument("--cache", default=str(ROOT / "cache"))
    e.add_argument("--out", default=str(ROOT / "results"))
    e.add_argument("--config")
    e.add_argument("--limit", type=int)
    e.set_defaults(fn=cmd_evaluate)

    g = sub.add_parser("figures", help="draw docs/figure.png and docs/post-figure.png from results/")
    g.add_argument("--results", default=str(ROOT / "results"))
    g.add_argument("--docs", default=str(ROOT / "docs"))
    g.set_defaults(fn=cmd_figures)

    r = sub.add_parser("reproduce", help="fetch + evaluate + figures (the README numbers)")
    r.add_argument("--root", default=str(ROOT))
    r.set_defaults(fn=cmd_reproduce)

    a = p.parse_args(argv)
    return a.fn(a)
