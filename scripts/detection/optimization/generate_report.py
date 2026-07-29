#!/usr/bin/env python3
"""Stage 3: Generate Markdown and HTML reports from benchmark_jetson.py's CSV.

Reads reports/detection/optimization/benchmark_results.csv and writes:
  - reports/detection/optimization/RESULTS.md
  - reports/detection/optimization/RESULTS.html

Colour coding in HTML:
  Green  — best FPS in the table
  Red    — recall collapsed (<0.3) at that precision, i.e. a broken engine

Usage
-----
    python scripts/detection/optimization/generate_report.py \\
        --csv reports/detection/optimization/benchmark_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

_COLS = [
    ("model_name",       "Model"),
    ("precision",        "Precision"),
    ("harness",          "Harness"),
    ("engine_build_ok",  "Built OK"),
    ("fps_video_mean",   "FPS (gaza videos)"),
    ("latency_ms_p50",   "Latency p50 (ms)"),
    ("latency_ms_p99",   "Latency p99 (ms)"),
    ("precision_at_0.4", "Precision @0.4"),
    ("recall_at_0.4",    "Recall @0.4"),
    ("fp_per_image",     "FP/image"),
    ("n_gt_boxes",       "GT boxes"),
    ("n_images",         "Val images"),
    ("notes",            "Notes"),
]


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


def _write_markdown(rows: list[dict], out: Path) -> None:
    header = "| " + " | ".join(name for _, name in _COLS) + " |\n"
    header += "|" + "|".join("---" for _ in _COLS) + "|\n"
    lines = [
        "# RF-DETR-m TensorRT Optimization — Benchmark Results\n\n",
        "> **Hardware**: Jetson AGX Orin 64GB — MAXN power mode, clocks locked, TensorRT 8.6.2.3.\n",
        "> **FPS source**: real decode+infer over 11 raw gaza-road clips "
        "(`~/Music/gaza_road_videos/`), not synthetic dummy tensors.\n",
        "> **Accuracy source**: fixed conf=0.4 operating point against "
        "`datasets/detection/Detection_Dataset/valid` (34 images, 115 GT boxes) — a coarse "
        "regression guard, not a publishable mAP number (see benchmark_jetson.py's docstring).\n\n",
        header,
    ]
    _string_cols = {"model_name", "precision", "harness", "engine_build_ok", "notes", "n_gt_boxes", "n_images"}
    for r in rows:
        cells = [_fmt(r.get(key)) if key not in _string_cols
                 else str(r.get(key, "") or "—") for key, _ in _COLS]
        lines.append("| " + " | ".join(cells) + " |\n")
    out.write_text("".join(lines))
    print(f"Markdown saved: {out}")


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
.best-fps    { background: #1a3d1a !important; color: #6fdf6f !important; font-weight: bold; }
.bad-recall  { background: #3d0000 !important; color: #ff6666 !important; font-weight: bold; }
.note-warn   { color: #ff9900; font-size: 0.8em; }
</style>
"""


def _write_html(rows: list[dict], out: Path) -> None:
    fps_vals = [float(r.get("fps_video_mean") or 0) for r in rows if r.get("fps_video_mean")]
    best_fps = max(fps_vals) if fps_vals else 0.0

    th_cells = "".join(f"<th>{name}</th>" for _, name in _COLS)
    tbody_lines = []
    for r in rows:
        recall = r.get("recall_at_0.4", "")
        recall_broken = False
        try:
            recall_broken = recall != "" and float(recall) < 0.3
        except (ValueError, TypeError):
            pass
        td_cells = []
        for key, _ in _COLS:
            val = r.get(key, "") or "—"
            cls = ""
            if key == "fps_video_mean" and best_fps > 0:
                try:
                    if abs(float(r.get(key) or 0) - best_fps) < 0.05 * best_fps:
                        cls = "best-fps"
                except (ValueError, TypeError):
                    pass
            elif key == "recall_at_0.4" and recall_broken:
                cls = "bad-recall"
            elif key == "notes" and val != "—":
                cls = "note-warn"
            cls_attr = f' class="{cls}"' if cls else ""
            td_cells.append(f"<td{cls_attr}>{val}</td>")
        tbody_lines.append("<tr>" + "".join(td_cells) + "</tr>")
    tbody = "\n".join(tbody_lines)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RF-DETR-m Jetson Optimization Results</title>
{_HTML_STYLE}
</head>
<body>
<h1>RF-DETR-m TensorRT Optimization — Benchmark Results</h1>
<p class="meta">
  Hardware: Jetson AGX Orin 64GB — MAXN power mode, clocks locked, TensorRT 8.6.2.3.<br>
  FPS: real decode+infer over 11 raw gaza-road clips.
  Accuracy: fixed conf=0.4 operating point, Detection_Dataset/valid (34 images, 115 GT boxes).&nbsp;
  <span style="color:#6fdf6f">■</span> Best FPS &nbsp;
  <span style="color:#ff6666">■</span> Recall collapsed (&lt;0.3) — broken precision, do not deploy
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


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3: generate Markdown/HTML report from CSV")
    p.add_argument("--csv", default="reports/detection/optimization/benchmark_results.csv")
    p.add_argument("--out-dir", default=None, help="Output directory (default: same as CSV directory)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = _ROOT / csv_path
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run benchmark_jetson.py on the Jetson first, then scp the CSV back.", file=sys.stderr)
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
    _write_html(rows, out_dir / "RESULTS.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
