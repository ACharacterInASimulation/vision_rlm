from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from vision_rlm.paths import build_project_paths
from vision_rlm.preprocess.common import ensure_dir, write_json


def _extract_field(payload: dict, field: str) -> object:
    current: object = payload
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(field)
    return current


def _iter_metric_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(sorted(root.rglob("metrics.json")))
    return sorted(set(files))


def _is_better(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    return candidate["x"] <= incumbent["x"] and candidate["y"] >= incumbent["y"] and (
        candidate["x"] < incumbent["x"] or candidate["y"] > incumbent["y"]
    )


def _pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    numeric_rows = [row for row in rows if isinstance(row["x"], (int, float)) and isinstance(row["y"], (int, float))]
    for row in numeric_rows:
        if any(_is_better(other, row) for other in numeric_rows if other is not row):
            continue
        frontier.append(row)
    return sorted(frontier, key=lambda item: (float(item["x"]), -float(item["y"])))


def build_frontier(
    metric_roots: list[Path],
    output_dir: Path,
    x_field: str,
    y_field: str,
    label_field: str,
    title: str,
) -> dict[str, object]:
    metric_files = _iter_metric_files(metric_roots)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for metric_file in metric_files:
        payload = json.loads(metric_file.read_text())
        try:
            x_value = _extract_field(payload, x_field)
            y_value = _extract_field(payload, y_field)
        except KeyError as exc:
            skipped.append({"path": metric_file.as_posix(), "reason": f"missing field: {exc.args[0]}"})
            continue

        try:
            label_value = _extract_field(payload, label_field)
        except KeyError:
            label_value = payload.get("run_name", metric_file.parent.name)

        rows.append(
            {
                "metrics_path": metric_file.as_posix(),
                "label": str(label_value),
                "x": float(x_value),
                "y": float(y_value),
                "raw": payload,
            }
        )

    frontier = _pareto_frontier(rows)
    ensure_dir(output_dir)

    csv_path = output_dir / "frontier_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "x", "y", "metrics_path", "on_frontier"])
        writer.writeheader()
        frontier_paths = {row["metrics_path"] for row in frontier}
        for row in sorted(rows, key=lambda item: (item["x"], -item["y"], item["label"])):
            writer.writerow(
                {
                    "label": row["label"],
                    "x": row["x"],
                    "y": row["y"],
                    "metrics_path": row["metrics_path"],
                    "on_frontier": row["metrics_path"] in frontier_paths,
                }
            )

    summary = {
        "metric_file_count": len(metric_files),
        "row_count": len(rows),
        "skipped": skipped,
        "x_field": x_field,
        "y_field": y_field,
        "label_field": label_field,
        "frontier_labels": [row["label"] for row in frontier],
        "csv_path": csv_path.as_posix(),
    }
    write_json(output_dir / "frontier_summary.json", summary)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(output_dir / "plot_status.json", {"plotted": False, "reason": str(exc)})
        return summary

    figure_path = output_dir / "frontier.png"
    plt.figure(figsize=(8, 5))
    for row in rows:
        color = "#d62728" if row in frontier else "#1f77b4"
        plt.scatter(row["x"], row["y"], color=color)
        plt.annotate(row["label"], (row["x"], row["y"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    if frontier:
        xs = [row["x"] for row in frontier]
        ys = [row["y"] for row in frontier]
        plt.plot(xs, ys, color="#d62728", linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    write_json(output_dir / "plot_status.json", {"plotted": True, "figure_path": figure_path.as_posix()})
    return summary


def parse_args() -> argparse.Namespace:
    paths = build_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        nargs="+",
        default=[paths.artifacts_root],
        help="One or more roots to scan recursively for metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.artifacts_root / "frontiers" / "default",
        help="Output directory for CSV, summary, and plot.",
    )
    parser.add_argument("--x-field", type=str, required=True, help="Dot-path metric for x-axis; smaller is better.")
    parser.add_argument("--y-field", type=str, required=True, help="Dot-path metric for y-axis; larger is better.")
    parser.add_argument(
        "--label-field",
        type=str,
        default="run_name",
        help="Dot-path metric used for labels. Falls back to run_name or folder name.",
    )
    parser.add_argument("--title", type=str, default="Accuracy-Cost Frontier", help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_frontier(
        metric_roots=args.metrics_root,
        output_dir=args.output_dir,
        x_field=args.x_field,
        y_field=args.y_field,
        label_field=args.label_field,
        title=args.title,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
