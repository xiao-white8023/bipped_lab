"""Print compact first-step diagnostics from a saved report; no Isaac imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def probe_summary(report):
    summary = {key: report.get(key) for key in (
        "passed", "failed_checks", "branch", "label_agreement", "label_mismatch_count",
        "initial_restore_max_abs_error", "first_divergent_step", "physics_settings")}
    summary["restore_version"] = report.get("snapshot_restore_contract", {}).get("version")
    summary["actuator_models"] = report.get("actuator_models", {})
    summary["first_control_step_max_error"] = {
        key: value.get("max_abs_error") if isinstance(value, dict) else value
        for key, value in report.get("first_control_step_errors", {}).items()}
    trace = report.get("physics_substep_comparison", {})
    summary["physics_trace_available"] = trace.get("available", False)
    steps = trace.get("substeps", [])
    if steps:
        first = steps[0]
        summary["first_physics_substep"] = {"step": first.get("physics_substep")}
        fields = ("physx.get_root_transforms", "physx.get_root_velocities",
                  "physx.get_dof_positions", "physx.get_dof_velocities",
                  "_joint_pos_target_sim", "_joint_vel_target_sim", "_joint_effort_target_sim")
        for phase in ("before_sim_step", "after_scene_update"):
            summary["first_physics_substep"][phase] = {
                key: first.get(phase, {}).get(key) for key in fields if key in first.get(phase, {})}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path, help="manifest.json, replay_diagnostics.json, or a dataset folder")
    source.add_argument("--latest", action="store_true", help="Most recent saved LP pilot/probe under this repo's outputs")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    path = args.report
    if args.latest:
        outputs = Path(__file__).resolve().parents[2] / "outputs"
        paths = list(outputs.glob("g1_renet_selector_temporal_pilot_*/pilot_lp_map/dataset/manifest.json"))
        paths.extend(outputs.glob("g1_renet_selector_replay_probe_*/replay_diagnostics.json"))
        if not paths:
            parser.error("No saved LP replay report found; use --report with its exact path")
        path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    document = json.loads(path.read_text())
    if "probes" in document:
        probes = document["probes"]
        passed = document.get("passed") is True and len(probes) == 4
    elif "metadata" in document:
        replay = document["metadata"].get("snapshot_replay_validation")
        if not replay:
            parser.error(f"No snapshot_replay_validation in {path}")
        probes = {"startup": replay}
        passed = replay.get("passed") is True
    else:
        probes = {"replay": document}
        passed = document.get("passed") is True
    print(f"[selector replay report] {path}", flush=True)
    for name, report in probes.items():
        print(f"[selector replay summary] {name} " + json.dumps(probe_summary(report), ensure_ascii=False), flush=True)
    if args.require_pass and (not passed or not all(r.get("passed") is True for r in probes.values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
