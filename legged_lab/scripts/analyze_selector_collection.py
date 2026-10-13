"""Analyze saved selector tensors in a CPU process without launching Isaac."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rsl_rl") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))


def print_report_summary(report):
    """Expose failure yield and filtering without interpreting rejected labels as truth."""
    overall = report["overall"]
    print(f"[selector analysis] collection_complete={report['collection_complete']} "
          f"rows={overall['rows']}", flush=True)
    print("[selector analysis] pair_order="+str(report["pair_order"]), flush=True)
    for horizon, pairs in zip(report["horizon_order"], overall["pair_counts"]):
        print(f"[selector analysis] {horizon}={pairs}", flush=True)
    for name, positive, rate in zip(report["label_order"], overall["positive"], overall["positive_rate"]):
        percentage = f"{100 * rate:.4f}%" if rate is not None else "unavailable"
        print(f"[selector analysis] {name}: failures={positive} rate={percentage}", flush=True)

    for key in ("sampling_progress", "safety_label_filter", "per_candidate_replay_filter"):
        value = report.get(key)
        if value is not None:
            print(f"[selector analysis filter] {key}="+json.dumps(value, sort_keys=True), flush=True)
    audit = report.get("collection_audit") or {}
    funnel = audit.get("funnel_by_active_scenario_severity")
    if funnel is not None:
        totals = {name: sum(sum(severities) for severities in stage)
                  for name, stage in zip(audit["funnel_order"], funnel)}
        print("[selector analysis filter] funnel="+json.dumps(totals), flush=True)
    stages = audit.get("pairs_by_stage_horizon_active_scenario_severity")
    if stages is not None:
        print("[selector analysis filter] before_replay_filter uses second-replay diagnostic labels; "
              "rejected labels are not reliable ground truth", flush=True)
        for name, stage in zip(audit["pair_stage_order"], stages):
            pairs = [[sum(severities[pair] for scenario in horizon for severities in scenario)
                      for pair in range(len(report["pair_order"]))] for horizon in stage]
            print(f"[selector analysis filter] {name}="+json.dumps(
                dict(zip(audit["pair_horizon_order"], pairs))), flush=True)

    coverage = report.get("supplement_coverage") or {}
    if coverage:
        print("[selector analysis filter] eligible_1s="+str(coverage["eligible_1s"]), flush=True)
        print("[selector analysis filter] admission_rejections="+json.dumps(coverage["rejections"]), flush=True)
        print("[selector analysis filter] pair_quotas="+str(coverage["config"]["pair_quotas"]), flush=True)
    temporal = coverage.get("temporal_coverage")
    if temporal:
        print(f"[selector analysis] real_span_s={temporal['collection_elapsed_s']:.2f} "
              f"warmup_s={temporal['warmup_s']:.2f}", flush=True)
        for i, name in enumerate(temporal["lifecycle_order"]):
            seen, eligible, written = (temporal[k][i] for k in ("seen", "eligible", "written"))
            if seen or eligible or written:
                print(f"[selector analysis] {name}: seen={seen} eligible={eligible} "
                      f"written={written} eligible_1s={temporal['eligible_1s_by_lifecycle'][i]} "
                      f"written_1s={temporal['written_1s_by_lifecycle'][i]}", flush=True)
        print("[selector analysis] missing_observed_stages="+str(temporal["missing_observed_stages"]), flush=True)

    legends = report["group_key_legend"]
    for dimension in ("active_lp_fault_kind", "active_vp_fault_kind", "active_fault_severity",
                      "terrain_level", "behavior_mode"):
        for key, counts in (report.get("by_group", {}).get(dimension) or {}).items():
            if dimension.endswith("fault_kind"):
                value = legends["fault_kind"][int(key)]
            elif dimension == "active_fault_severity":
                value = legends["severity"][int(key)]
            elif dimension == "behavior_mode":
                value = legends["behavior_mode"][key]
            else:
                value = key
            print(f"[selector analysis group] {dimension}={value} rows={counts['rows']} "
                  f"1s={counts['pair_counts'][0]}", flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", type=Path)
    mode.add_argument("--check-dependencies", action="store_true",
                      help="Check the CPU analysis imports without reading a dataset or launching Isaac.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    from legged_lab.selector.analysis import write_collection_analysis
    if args.check_dependencies:
        print("[selector analysis] CPU analysis dependencies ready", flush=True)
        return
    print(f"[selector analysis] reading {args.dataset}", flush=True)
    path = write_collection_analysis(args.dataset)
    report = json.loads(path.read_text())
    print(f"[selector analysis] saved {path}", flush=True)
    print_report_summary(report)
    if args.require_complete and not report["collection_complete"]:
        raise SystemExit("collection is incomplete; inspect missing quotas and filters")
if __name__ == "__main__":
    main()