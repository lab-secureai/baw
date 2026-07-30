#!/usr/bin/env python3
"""Export per-seed Track A results from results_real.json to CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results_real/results_real.json")
    parser.add_argument("--output", default="results_real/track_a_per_seed.csv")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for scheme, results in data["track_a_per_seed"].items():
        for index, result in enumerate(results):
            row = {"scheme": scheme, "run_index": index}
            row.update(result)
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(rows).to_csv(output, index=False)
    print(f"Wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
