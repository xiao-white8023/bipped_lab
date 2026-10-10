"""Analyze an existing V3 collection on CPU; no Isaac Sim imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from legged_lab.selector.analysis import write_collection_analysis


def main():
    parser = argparse.ArgumentParser(description="Audit LP/VP selector V3 datasets.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    path = write_collection_analysis(args.dataset, args.output)
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"rows={report['overall']['rows']} complete={report['collection_complete']}")
    for head, count, rate in zip(report["label_order"], report["overall"]["positive"], report["overall"]["positive_rate"]):
        print(f"{head}: {count} failures; rate={rate:.6%}" if rate is not None else f"{head}: no rows")
    print(f"analysis: {path}")


if __name__ == "__main__":
    main()
