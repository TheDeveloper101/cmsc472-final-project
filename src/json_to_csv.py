import argparse
import json

from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import RESULTS_PATH


def main():
    parser = argparse.ArgumentParser(description="Convert one or more experiment JSON files to a CSV.")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=None,
        help="Input JSON file(s). Default: all JSON files in RESULTS_PATH/experiments.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path. Default: a unique CSV in RESULTS_PATH/tables.",
    )
    args = parser.parse_args()

    results_path = Path(RESULTS_PATH)

    if args.input is None:
        exp_dir = results_path / "experiments"
        args.input = [str(p) for p in sorted(exp_dir.glob("*.json"))]
        if not args.input:
            raise SystemExit(f"No input JSON files found in: {exp_dir}")
    else:
        expanded: list[str] = []
        for p in args.input:
            pp = Path(p)
            if pp.is_dir():
                expanded.extend(str(x) for x in sorted(pp.glob("*.json")))
            else:
                expanded.append(p)
        args.input = expanded
        if not args.input:
            raise SystemExit("No input JSON files provided/found.")

    if args.output is None:
        tables_dir = results_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output = str(tables_dir / f"experiments_{stamp}.csv")

    dfs: list[pd.DataFrame] = []
    for path in args.input:
        with open(path, "r") as f:
            data = json.load(f)
        dfs.append(pd.json_normalize(data))

    df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    df.to_csv(
        args.output,
        index=False,
        columns=[
            "model",
            "quantize",
            "profile.text.execution_summary.all_inference_times_med",
            "profile.text.execution_summary.estimated_inference_peak_memory",
            "profile.image.execution_summary.all_inference_times_med",
            "profile.image.execution_summary.estimated_inference_peak_memory",
            "recall",
        ],
        header=[
            "model",
            "quantize",
            "Text Latency (μs)",
            "Text Peak Memory (Bytes)",
            "Image Latency (μs)",
            "Image Peak Memory (Bytes)",
            "Recall @ Top10",
        ],
    )


if __name__ == "__main__":
    main()
