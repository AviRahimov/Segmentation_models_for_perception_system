#!/usr/bin/env python3
"""Stage 6b: Generate Markdown and HTML comparison reports from benchmark CSV.

Reads reports/segmentation/optimization/benchmark_results.csv (produced by benchmark_jetson.py)
and writes:
  - reports/segmentation/optimization/RESULTS.md   — Markdown table
  - reports/segmentation/optimization/RESULTS.html — styled HTML with colour-coding

Colour coding in HTML:
  Green  — best FPS in the table
  Yellow — mIoU drop between 1% and 2% vs the highest-mIoU variant
  Red    — mIoU drop > 2% vs the highest-mIoU variant

Usage
-----
    python scripts/segmentation/optimization/generate_report.py \\
        --csv reports/segmentation/optimization/benchmark_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Data loading                                                                  #
# --------------------------------------------------------------------------- #

def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fmt(val, decimals: int = 3) -> str:
    if val is None or val == "":
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


# --------------------------------------------------------------------------- #
# Markdown                                                                      #
# --------------------------------------------------------------------------- #

def _write_markdown(rows: list[dict], out: Path) -> None:
    header = (
        "| Variant | Backbone | Precision | Sparsity | Res | "
        "mIoU (PyTorch) | mIoU (Engine) | Latency p50 (ms) | Latency p99 (ms) | "
        "FPS | FPS sustained | Notes |\n"
        "|---------|----------|-----------|----------|-----|"
        "----------------|---------------|------------------|------------------|"
        "-----|---------------|-------|\n"
    )
    lines = [
        "# Optimization Benchmark Results\n\n",
        "> **Hardware**: Jetson AGX Orin 64GB — MAXN power mode, clocks locked.\n",
        "> **Dataset**: ORFD validation split.  mIoU: higher = better.\n\n",
        header,
    ]
    for r in rows:
        line = (
            f"| {r.get('variant_name', '')} "
            f"| {r.get('backbone', '')} "
            f"| {r.get('precision', '')} "
            f"| {r.get('sparsity', '')} "
            f"| {r.get('resolution', '')} "
            f"| {_fmt(r.get('miou_pytorch'), 4)} "
            f"| {_fmt(r.get('miou_engine'),  4)} "
            f"| {_fmt(r.get('latency_ms_p50'), 2)} "
            f"| {_fmt(r.get('latency_ms_p99'), 2)} "
            f"| {_fmt(r.get('fps'), 1)} "
            f"| {_fmt(r.get('sustained_fps_30min'), 1)} "
            f"| {r.get('notes', '')} |\n"
        )
        lines.append(line)

    out.write_text("".join(lines))
    print(f"Markdown saved: {out}")


# --------------------------------------------------------------------------- #
# HTML                                                                          #
# --------------------------------------------------------------------------- #

_HTML_STYLE = """
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }
h1 { color: #7ec8e3; }
p.meta { color: #888; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
th { background: #222; color: #ccc; padding: 8px 12px; text-align: left;
     border-bottom: 2px solid #444; }
td { padding: 7px 12px; border-bottom: 1px solid #2a2a2a; }
tr:hover td { background: #1e1e1e; }
.best-fps  { background: #1a3d1a !important; color: #6fdf6f !important; font-weight: bold; }
.warn-miou { background: #3d3200 !important; color: #ffd966 !important; }
.bad-miou  { background: #3d0000 !important; color: #ff6666 !important; }
.note-warn { color: #ff9900; font-size: 0.8em; }
</style>
"""

def _row_class(val: str, best_fps: float, best_miou: float) -> dict:
    """Return dict of CSS classes per column."""
    classes: dict[str, str] = {}
    try:
        fps = float(val.get("fps", "") or 0)
        if best_fps > 0 and abs(fps - best_fps) < 0.05 * best_fps:
            classes["fps"] = "best-fps"
    except (ValueError, TypeError):
        pass
    try:
        eng = float(val.get("miou_engine", "") or 0)
        drop = best_miou - eng
        if 0.01 < drop <= 0.02:
            classes["miou_engine"] = "warn-miou"
        elif drop > 0.02:
            classes["miou_engine"] = "bad-miou"
    except (ValueError, TypeError):
        pass
    return classes


def _write_html(rows: list[dict], out: Path) -> None:
    valid_fps = [float(r.get("fps") or 0) for r in rows if r.get("fps")]
    best_fps = max(valid_fps) if valid_fps else 0.0

    valid_miou = [float(r.get("miou_engine") or 0) for r in rows if r.get("miou_engine")]
    best_miou = max(valid_miou) if valid_miou else 0.0

    cols = [
        ("variant_name",       "Variant"),
        ("backbone",           "Backbone"),
        ("precision",          "Precision"),
        ("sparsity",           "Sparsity"),
        ("resolution",         "Resolution"),
        ("miou_pytorch",       "mIoU (PyTorch)"),
        ("miou_engine",        "mIoU (Engine)"),
        ("latency_ms_p50",     "Latency p50 (ms)"),
        ("latency_ms_p99",     "Latency p99 (ms)"),
        ("fps",                "FPS"),
        ("sustained_fps_30min","FPS sustained"),
        ("notes",              "Notes"),
    ]

    th_cells = "".join(f"<th>{name}</th>" for _, name in cols)
    tbody_lines = []
    for r in rows:
        css = _row_class(r, best_fps, best_miou)
        td_cells = []
        for key, _ in cols:
            val = r.get(key, "") or ""
            cell_cls = css.get(key, "")
            note_cls = "note-warn" if key == "notes" and val else ""
            cls_attr = f' class="{cell_cls or note_cls}"' if (cell_cls or note_cls) else ""
            td_cells.append(f"<td{cls_attr}>{val}</td>")
        tbody_lines.append("<tr>" + "".join(td_cells) + "</tr>")

    tbody = "\n".join(tbody_lines)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Optimization Benchmark Results</title>
{_HTML_STYLE}
</head>
<body>
<h1>SegFormer-B2 Optimization Benchmark Results</h1>
<p class="meta">
  Hardware: Jetson AGX Orin 64GB — MAXN power mode, clocks locked.<br>
  Dataset: ORFD validation split.&nbsp;
  <span style="color:#6fdf6f">■</span> Best FPS &nbsp;
  <span style="color:#ffd966">■</span> mIoU drop 1–2% &nbsp;
  <span style="color:#ff6666">■</span> mIoU drop &gt;2%
</p>
<div style="overflow-x:auto">
<table>
<thead><tr>{th_cells}</tr></thead>
<tbody>{tbody}</tbody>
</table>
</div>
</body>
</html>
"""
    out.write_text(html)
    print(f"HTML saved: {out}")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="Stage 6b: generate Markdown/HTML report from CSV")
    p.add_argument("--csv", default="reports/segmentation/optimization/benchmark_results.csv")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: same as CSV directory)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _ROOT / csv_path

    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run benchmark_jetson.py first to generate the results file.", file=sys.stderr)
        return 1

    rows = _load_csv(csv_path)
    if not rows:
        print("CSV is empty.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_markdown(rows, out_dir / "RESULTS.md")
    _write_html(rows,     out_dir / "RESULTS.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
