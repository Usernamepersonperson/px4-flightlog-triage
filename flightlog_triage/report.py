"""One-page PNG + HTML summary for a triaged flight, and a folder index."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

from .triage import TriageResult

KIND_COLORS = {
    "vibration": "#d97706",
    "accel_clipping": "#b45309",
    "battery_sag": "#dc2626",
    "battery_low_voltage": "#991b1b",
    "battery_high_resistance": "#f87171",
    "gnss_loss": "#2563eb",
    "gnss_poor": "#93c5fd",
    "estimator_rejection": "#7c3aed",
    "rc_loss": "#db2777",
    "datalink_loss": "#f472b6",
    "motor_saturation": "#059669",
    "cpu_load": "#475569",
}

PANELS = [
    ("vibration", "Vibration\nRMS m/s²", ("vibration", "accel_clipping")),
    ("battery_v", "Battery\nV/cell", ("battery_sag", "battery_low_voltage")),
    ("gnss_sats", "GNSS\nsatellites", ("gnss_loss", "gnss_poor")),
    ("est_worst_ratio", "Estimator\ntest ratio", ("estimator_rejection",)),
    ("motor_max", "Motor cmd\nmin/max", ("motor_saturation",)),
    ("cpu", "CPU load", ("cpu_load", "rc_loss", "datalink_loss")),
]


def mmss(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):d}:{t % 60:04.1f}"


def render_png(res: TriageResult, path: Path | None = None, cfg: dict | None = None) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 10), sharex=True, dpi=100)
    air = res.series.get("_airborne", [])
    cells = res.metrics.get("cells") or 1
    for ax, (key, label, kinds) in zip(axes, PANELS, strict=True):
        for a, b in air:
            ax.axvspan(a, b, color="#e2e8f0", lw=0, zorder=0)
        if key in res.series:
            t, y = res.series[key]
            if key == "battery_v":
                y = y / cells
            ax.plot(t, y, lw=0.8, color="#0f172a")
            if key == "motor_max" and "motor_min" in res.series:
                t2, y2 = res.series["motor_min"]
                ax.plot(t2, y2, lw=0.8, color="#64748b")
        else:
            ax.text(0.5, 0.5, "not logged", transform=ax.transAxes, ha="center", va="center",
                    color="#94a3b8")
        if cfg:
            thr = {"vibration": cfg["vibration_warn_m_s2"], "est_worst_ratio": cfg["est_test_ratio_fail"],
                   "cpu": cfg["cpu_load_warn"], "battery_v": cfg["batt_min_cell_v"]}.get(key)
            if thr is not None:
                ax.axhline(thr, color="#ef4444", lw=0.8, ls="--")
        for fnd in res.findings:
            if fnd.kind in kinds:
                ax.axvspan(fnd.t_start, max(fnd.t_end, fnd.t_start + 0.5),
                           color=KIND_COLORS.get(fnd.kind, "#f59e0b"), alpha=0.35, lw=0)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time since log start (s); grey band = airborne")
    n = len(res.findings)
    fig.suptitle(
        f"{Path(res.source).name or 'flight'} - airborne {res.airborne_s / 60:.1f} min - {n} finding(s)",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def render_html(res: TriageResult, png: bytes | None, cfg: dict) -> str:
    rows = "".join(
        f"<tr><td>{i}</td><td><span class='dot' style='background:{KIND_COLORS.get(f.kind, '#999')}'></span>"
        f"{html.escape(f.kind)}</td><td>{mmss(f.t_start)}</td><td>{f.duration_s:.1f}</td>"
        f"<td>{f.peak:.3g} {html.escape(f.unit)}</td><td>{f.threshold:.3g}</td><td>{f.score:.0f}</td>"
        f"<td>{html.escape(f.detail)}</td></tr>"
        for i, f in enumerate(res.findings, 1)
    ) or "<tr><td colspan='8'>No findings above the configured thresholds.</td></tr>"
    metrics = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in res.metrics.items()
    )
    img = f"<img alt='flight timeline' src='data:image/png;base64,{base64.b64encode(png).decode()}'>" if png else ""
    thr = html.escape(json.dumps(cfg, indent=1))
    name = html.escape(Path(res.source).name or "flight")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Flight triage - {name}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:Segoe UI,DejaVu Sans,sans-serif;margin:16px;color:#0f172a;max-width:1200px}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #e2e8f0;padding:4px 8px;text-align:left}}
img{{max-width:100%}}.dot{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px}}
details{{margin-top:12px;font-size:12px}}</style></head><body>
<h2>Flight triage: {name}</h2>
<p>Log length {res.duration_s / 60:.1f} min, airborne {res.airborne_s / 60:.1f} min, {len(res.findings)} finding(s),
processed in {res.load_s + res.triage_s:.2f} s.</p>
<h3>Ranked findings</h3>
<table><tr><th>#</th><th>kind</th><th>start (m:s)</th><th>dur (s)</th><th>peak</th><th>threshold</th>
<th>score</th><th>detail</th></tr>{rows}</table>
{img}
<h3>Metrics</h3><table>{metrics}</table>
<details><summary>Thresholds used</summary><pre>{thr}</pre></details>
</body></html>"""


def write_report(res: TriageResult, out_dir: Path, cfg: dict, png: bool = True) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.source).stem or "flight"
    paths = {}
    data = None
    if png:
        paths["png"] = out_dir / f"{stem}.png"
        data = render_png(res, paths["png"], cfg)
    paths["html"] = out_dir / f"{stem}.html"
    paths["html"].write_text(render_html(res, data, cfg), encoding="utf-8")
    paths["json"] = out_dir / f"{stem}.json"
    paths["json"].write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
    return paths


def write_index(results: list[TriageResult], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(results, key=lambda r: -max([f.score for f in r.findings], default=0.0))
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["log", "airborne_min", "findings", "top_score", "top_kind", "kinds"])
        for r in order:
            top = r.findings[0] if r.findings else None
            w.writerow([Path(r.source).name, round(r.airborne_s / 60, 2), len(r.findings),
                        top.score if top else 0, top.kind if top else "",
                        ";".join(sorted(r.kinds()))])
    rows = "".join(
        f"<tr><td><a href='{html.escape(Path(r.source).stem)}.html'>{html.escape(Path(r.source).name)}</a></td>"
        f"<td>{r.airborne_s / 60:.1f}</td><td>{len(r.findings)}</td>"
        f"<td>{max([f.score for f in r.findings], default=0):.0f}</td>"
        f"<td>{html.escape(', '.join(sorted(r.kinds())))}</td></tr>"
        for r in order
    )
    idx = out_dir / "index.html"
    idx.write_text(
        "<!doctype html><meta charset='utf-8'><title>Flight triage index</title>"
        "<style>body{font-family:Segoe UI,sans-serif;margin:16px}td,th{padding:4px 8px;"
        "border-bottom:1px solid #ddd;text-align:left}</style><h2>Flight triage index</h2>"
        "<table><tr><th>log</th><th>airborne min</th><th>findings</th><th>top score</th><th>kinds</th></tr>"
        f"{rows}</table>",
        encoding="utf-8",
    )
    return {"csv": csv_path, "html": idx}

