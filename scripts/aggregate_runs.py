"""Aggregate multi-seed results as mean +/- std."""

from __future__ import annotations

import argparse
import glob as globlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OVERALL_METRICS = (
    "mae",
    "rmse",
    "bias",
    "relative_count_error",
    "mean_image_relative_error",
    "gt_total",
    "pred_total",
)
TIER_METRICS = ("mae", "rmse", "bias", "relative_count_error")

# usage:
#   python scripts/aggregate_runs.py --run name=path/metrics.json --run ...
#   python scripts/aggregate_runs.py --glob "outputs/eval/*/metrics.json"


def parse_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--run expects NAME=PATH, got {text!r}")
    name, path = text.split("=", 1)
    return name.strip(), Path(path.strip())


def discover_runs(pattern: str) -> list[tuple[str, Path]]:
    paths = sorted(globlib.glob(pattern))
    if not paths:
        raise SystemExit(f"glob matched no metrics.json: {pattern}")
    return [(Path(p).parent.name, Path(p)) for p in paths]


def load_report(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for name, path in runs:
        report = load_report(path)
        overall = report.get("overall", {})
        for metric in OVERALL_METRICS:
            if metric in overall and overall[metric] is not None:
                rows.append({"run": name, "scope": "overall", "tier": "ALL", "metric": metric, "value": float(overall[metric])})
        for tier in report.get("tiers", []):
            for metric in TIER_METRICS:
                if metric in tier and tier[metric] is not None:
                    rows.append({"run": name, "scope": "tier", "tier": tier["tier"], "metric": metric, "value": float(tier[metric])})
    if not rows:
        raise SystemExit("no metrics found in the provided metrics.json files")
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["scope", "tier", "metric"], sort=False)["value"]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary = summary.rename(columns={"count": "n"})
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def to_wide(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["label"] = np.where(frame["tier"] == "ALL", frame["metric"], frame["tier"] + "_" + frame["metric"])
    frame["value"] = [f"{mean:.4f} +/- {std:.4f}" for mean, std in zip(frame["mean"], frame["std"])]
    return frame.pivot(index="scope", columns="label", values="value").reset_index()


def print_table(summary: pd.DataFrame) -> None:
    print(f"\n{'scope':<8}{'tier':<6}{'metric':<28}{'mean':>12}{'std':>12}{'n':>4}")
    for _, row in summary.iterrows():
        print(f"{row['scope']:<8}{row['tier']:<6}{row['metric']:<28}{row['mean']:>12.4f}{row['std']:>12.4f}{int(row['n']):>4}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results as mean +/- std")
    parser.add_argument("--run", action="append", type=parse_run, default=[], help="NAME=PATH/metrics.json")
    parser.add_argument("--glob", default=None, help="glob pattern for metrics.json")
    parser.add_argument("--out-dir", default="outputs/eval/aggregated", help="CSV output directory")
    args = parser.parse_args(argv)
    if not args.run and not args.glob:
        parser.error("provide at least one --run or --glob")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = list(args.run)
    if args.glob:
        runs.extend(discover_runs(args.glob))

    frame = collect_rows(runs)
    summary = summarize(frame)
    wide = to_wide(summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(out_dir / "aggregate_wide.csv", index=False, encoding="utf-8-sig")

    names = sorted({name for name, _ in runs})
    print(f"runs ({len(names)}): {', '.join(names)}")
    print_table(summary)
    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
