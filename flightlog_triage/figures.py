"""README figure (1600x900) and LinkedIn figure (960x700, dark) from results/eval.json."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

LABELS = {"vibration": "Vibration burst", "voltage_sag": "Battery voltage sag", "gnss_dropout": "GNSS dropout"}
DARK = {"bg": "#16213A", "grid": "#2A3750", "text": "#F8FAFC", "muted": "#A9B6CB", "accent": "#38BDF8",
        "green": "#34D399", "amber": "#FBBF24", "red": "#F87171", "grey": "#64748B", "violet": "#A78BFA"}
FAULT_COLORS = {"vibration": DARK["amber"], "voltage_sag": DARK["red"], "gnss_dropout": DARK["accent"]}
LIGHT_COLORS = {"vibration": "#d97706", "voltage_sag": "#dc2626", "gnss_dropout": "#2563eb"}


def relative_size(fault: str, mag: float, cfg: dict) -> float:
    """Injected fault size divided by the detector threshold it has to cross."""
    if fault == "vibration":  # added noise is per axis; the metric is the 3-axis RMS
        return math.sqrt(3) * mag / cfg["vibration_warn_m_s2"]
    if fault == "voltage_sag":
        return mag / cfg["batt_sag_v_per_cell"]
    return mag / cfg["gnss_loss_min_s"]


def _curves(ev: dict):
    cfg = ev["config"]
    for fault, d in ev["injection"].items():
        xs, ys, ns = [], [], []
        for mag, r in d["by_magnitude"].items():
            if r["trials"]:
                xs.append(relative_size(fault, float(mag), cfg))
                ys.append(100 * r["rate"])
                ns.append(r["trials"])
        yield fault, np.array(xs), np.array(ys), ns, d


def _font(plt):
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"


def figure_main(ev: dict, trials: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _font(plt)
    plt.rcParams.update({"font.size": 13})
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3, left=0.07, right=0.98, top=0.86, bottom=0.08)
    d0, d1 = ev["date_range"]
    fig.suptitle("PX4 flight-log triage: injected faults found on real logs, and how often flags fire on "
                 "untouched logs", x=0.05, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.05, 0.925, f"Open data: {ev['logs_parsed']} public logs from the PX4 Flight Review database "
             f"({d0} to {d1}); faults injected by this repo's code into the logged signals.",
             fontsize=13, color="#475569")

    for k, (fault, xs, ys, ns, d) in enumerate(_curves(ev)):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(xs, ys, "o-", color=LIGHT_COLORS[fault], lw=2.5, ms=8)
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                        fontsize=12, fontweight="bold")
        ax.axvline(1.0, color="#94a3b8", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.18, 9)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["1/4x", "1/2x", "1x", "2x", "4x", "8x"])
        ax.set_ylim(-5, 120)
        ax.set_xlabel("fault size / detection threshold")
        ax.set_ylabel("detected (%)")
        ax.set_title(f"{LABELS[fault]}: {d['logs']} logs, {min(ns)}-{max(ns)} trials/point",
                     fontsize=14, loc="left")
        ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    data, names = [], []
    for fault in LABELS:
        e = [abs(float(t["timing_error_s"])) for t in trials if t["fault"] == fault and t["timing_error_s"] != ""]
        if e:
            data.append(e)
            names.append({"vibration": "vibration", "voltage_sag": "voltage\nsag", "gnss_dropout": "GNSS"}[fault])
    ax.boxplot(data, vert=False, tick_labels=names, showfliers=True, whis=(5, 95))
    ax.set_xlabel("|flag start - injected start| (s)")
    ax.set_title("Timing error of detected faults", fontsize=14, loc="left")
    ax.grid(alpha=0.3, axis="x")

    ax = fig.add_subplot(gs[1, 1])
    hold = ev["real_logs_holdout"]
    good = hold.get("rated_good", {})
    names = {"rated_good": "rated good\nby uploader", "recent": "recent,\nunrated",
             "rated_problem": "rated crash /\nunsatisfactory"}
    keys = [k for k in ("rated_good", "recent", "rated_problem") if k in hold]
    vals = [hold[k]["fault_flags_per_h"] for k in keys]
    bars = ax.bar([names[k] for k in keys], vals, color=["#059669", "#64748b", "#dc2626"][: len(keys)])
    for bar, k in zip(bars, keys, strict=True):
        h = hold[k]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{h['fault_flags_per_h']:.0f}/h\n{h['logs_without_fault_flags']}/{h['logs']} with 0",
                ha="center", va="bottom", fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.35 if vals else 1)
    ax.set_ylabel("fault flags per flight hour")
    ax.set_title("Untouched held-out logs (x/y logs with 0 fault flags)", fontsize=13, loc="left")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    sp = ev["speed"]
    lines = [
        ("Speed (1 thread)", f"{sp['mb_per_s']:.1f} MB/s of ULog"),
        ("", f"{sp['s_per_flight_hour']:.0f} s per flight hour"),
        ("Parsing share", f"{sp['parse_share'] * 100:.0f} % of run time"),
        ("Logs parsed", f"{ev['logs_parsed']} of {ev['logs_parsed'] + ev['logs_failed']}"),
        ("Rated-good logs", f"{good.get('fault_flags_per_h', 0):.1f} fault flags/h"),
    ]
    y = 0.92
    ax.set_title("Measured", fontsize=14, loc="left")
    for a, b in lines:
        ax.text(0.0, y, a, fontsize=14, color="#475569", transform=ax.transAxes)
        ax.text(0.45, y, b, fontsize=15, fontweight="bold", transform=ax.transAxes)
        y -= 0.17
    fig.savefig(out, dpi=100)
    plt.close(fig)


def figure_post(ev: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _font(plt)
    c = DARK
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=c["bg"])
    ax = fig.add_axes([0.11, 0.13, 0.85, 0.66], facecolor=c["bg"])
    d0, d1 = ev["date_range"]
    fig.text(0.03, 0.94, "Known faults injected into real drone logs: share found", fontsize=19,
             fontweight="bold", color=c["text"])
    fig.text(0.03, 0.895, f"Open data: {ev['logs_parsed']} public PX4 logs ({d0[:4]}-{d1[:4]}), "
             "faults injected by this repo", fontsize=15, color=c["muted"])
    for fault, xs, ys, _ns, _d in _curves(ev):
        ax.plot(xs, ys, "o-", color=FAULT_COLORS[fault], lw=3.5, ms=10, label=LABELS[fault])
    ax.axvline(1.0, color=c["grey"], ls="--", lw=1.5)
    ax.text(1.04, 6, "threshold", color=c["muted"], fontsize=15)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×", "8×"])
    ax.set_ylim(-4, 108)
    ax.set_xlim(0.2, 9)
    ax.set_xlabel("fault size relative to detection threshold", color=c["text"], fontsize=16)
    ax.set_ylabel("faults detected (%)", color=c["text"], fontsize=16)
    ax.grid(color=c["grid"], lw=1)
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["text"], labelsize=15)
    leg = ax.legend(loc="lower right", fontsize=15, frameon=True, facecolor=c["bg"], edgecolor=c["grid"])
    for t in leg.get_texts():
        t.set_color(c["text"])
    fig.savefig(out, dpi=100, facecolor=c["bg"])
    plt.close(fig)


def make_all(results: Path, docs: Path) -> None:
    ev = json.loads((results / "eval.json").read_text(encoding="utf-8"))
    with (results / "injection_trials.csv").open(encoding="utf-8") as fh:
        trials = list(csv.DictReader(fh))
    docs.mkdir(parents=True, exist_ok=True)
    figure_main(ev, trials, docs / "figure.png")
    figure_post(ev, docs / "post-figure.png")
