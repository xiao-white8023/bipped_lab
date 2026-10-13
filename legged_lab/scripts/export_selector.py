"""Export risk and optional joint-outcome methods after CPU reload/parity checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

from legged_lab.selector.config import SelectorConfig
from legged_lab.selector.frozen_policy import policy_identity, sha256
from legged_lab.selector.labels import validate_safety_label_contract
from legged_lab.selector.model import selector_from_artifact
from legged_lab.selector.selection import SelectionRule
from legged_lab.selector.metrics import json_safe_metrics
from legged_lab.selector.feature_contract import validate_feature_contract


@torch.inference_mode()
def compare(eager, scripted, inputs, outcomes):
    r = eager(inputs)
    s = scripted(inputs)
    if r.shape != (len(inputs),2) or not torch.isfinite(s).all() or not torch.allclose(r,s,rtol=1e-5,atol=1e-6):
        raise RuntimeError(f"risk TorchScript parity failed for batch={len(inputs)}")
    errors = {"risks_max_abs_error":float((r-s).abs().max())}
    if outcomes:
        er,eq = eager.decision_outputs(inputs)
        sr,sq = scripted.decision_outputs(inputs)
        if (eq.shape != (len(inputs),4) or not torch.isfinite(sq).all()
                or not torch.allclose(er,sr,rtol=1e-5,atol=1e-6)
                or not torch.allclose(eq,sq,rtol=1e-5,atol=1e-6)
                or not torch.allclose(er,r,rtol=1e-5,atol=1e-6)):
            raise RuntimeError(f"outcome TorchScript parity failed for batch={len(inputs)}")
        errors["outcomes_max_abs_error"] = float((eq-sq).abs().max())
    return errors


def main():
    parser = argparse.ArgumentParser(description="Export the one-second LP/VP selector.")
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args = parser.parse_args()
    destination = args.output_dir/"lp_vp_selector.pt"
    contract_path = args.output_dir/"selector_contract.json"
    if destination.exists() or contract_path.exists():
        parser.error("use a new export directory; existing artifacts are never overwritten")
    artifact = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    outcomes = artifact.get("version") == 5
    if outcomes:
        from legged_lab.selector.outcome_decision import outcome_guard, OutcomeGuardConfig, validate_outcome_guard_record
        fit = artifact.get("selection_rule_fit",{})
        validate_outcome_guard_record(fit.get("guard"))
        if (fit.get("selected") != artifact.get("selection_rule")
                or fit["guard"]["requirements"] != artifact.get("outcome_guard_config")):
            raise ValueError("outcome parameters differ from the validation-fitted decision record")
        verified = outcome_guard(fit["baseline_metrics"],fit["selected_metrics"],OutcomeGuardConfig(**artifact["outcome_guard_config"]))
        if verified != fit["guard"]:
            raise ValueError("outcome validation guard is inconsistent with recorded validation metrics")
    elif artifact.get("selection_rule",{}).get("version") == 2:
        fit = artifact.get("selection_rule_fit",{})
        if fit.get("guard",{}).get("passed") is not True:
            raise ValueError("directional candidate did not pass its validation guard; export is blocked")
        from legged_lab.selector.decision_tuning import decision_guard
        if not decision_guard(fit["baseline_metrics"],fit["selected_metrics"],SelectionRule.from_dict(artifact["selection_rule"]))["passed"]:
            raise ValueError("directional candidate has no effective validation holds; rerun decision fitting")
    frozen = policy_identity(artifact.get("frozen_policy",{}))
    labels = validate_safety_label_contract(artifact.get("safety_label_contract"))
    features = validate_feature_contract(artifact.get("selector_feature_contract"))
    cfg = SelectorConfig(**artifact["selector_config"])
    model = selector_from_artifact(artifact).cpu().eval()
    risk_model = model.risk_selector if outcomes else model
    dummy = torch.randn(8,cfg.history_frames,cfg.frame_dim,generator=torch.Generator().manual_seed(927))
    with torch.inference_mode():
        if outcomes:
            traced = torch.jit.trace_module(model,{"forward":dummy,"decision_outputs":dummy},check_trace=True)
        else:
            traced = torch.jit.trace(model,dummy,check_trace=True)
    probes = [torch.randn(size,cfg.history_frames,cfg.frame_dim,generator=torch.Generator().manual_seed(size+927)) for size in (1,7,8)]
    for probe in probes:
        compare(model,traced,probe,outcomes)
    metadata = {
        "version":5 if outcomes else 4,"feature_shape":[cfg.history_frames,cfg.frame_dim],
        "output_order":["lp_1.0s","vp_1.0s"],"mask_contract":{"vp":0,"lp":1},
        "source_checkpoint":str(args.checkpoint.resolve()),"source_checkpoint_sha256":sha256(args.checkpoint),
        "frozen_policy":frozen,"safety_label_contract":labels,"selector_feature_contract":features,
        "calibration":{"method":"affine_logit","temperatures":risk_model.temperatures.tolist(),
                       "biases":risk_model.calibration_biases.tolist(),"fit":artifact.get("calibration",{"method":"temperature_only"})},
        "selection_rule":artifact.get("selection_rule") or SelectionRule().to_dict(),
        "validation_decision_guard":artifact.get("selection_rule_fit",{}).get("guard"),
        "threshold_source":"validation" if "selection_rule_fit" in artifact else "default_not_fitted",
        "metrics":json_safe_metrics(artifact.get("metrics",{})),
    }
    if outcomes:
        metadata.update({"outcome_order":artifact["outcome_order"],"decision_method":"decision_outputs",
                         "decision_outputs":["risks_batch_2","outcomes_batch_4"],
                         "outcome_calibration":artifact["outcome_calibration"],"joint_warning_is_switch_veto":False,
                         "decision_runtime":{"selector_period_s":.1,"min_dwell_s":1.,
                                             "switch_confirmations":3,"normal_blend_steps":5}})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".selector_export_",dir=args.output_dir) as temporary:
        staged = Path(temporary)/"lp_vp_selector.pt"
        torch.jit.save(traced,str(staged),_extra_files={"selector_contract.json":json.dumps(metadata,sort_keys=True,allow_nan=False)})
        extra = {"selector_contract.json":""}
        loaded = torch.jit.load(str(staged),map_location="cpu",_extra_files=extra).eval()
        if json.loads(extra["selector_contract.json"]) != metadata:
            raise RuntimeError("reloaded selector identity contract differs")
        validation = [compare(model,loaded,probe,outcomes) for probe in probes]
        staged_contract = Path(temporary)/"selector_contract.json"
        staged_contract.write_text(json.dumps(metadata,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
        os.link(staged,destination)
        try:
            os.link(staged_contract,contract_path)
        except BaseException:
            destination.unlink()
            raise
    print(f"[selector] exported {destination}; CPU parity batches=1,7,8; errors={validation}")


if __name__ == "__main__":
    main()
