"""Run the reviewed outcome trainer using this experiment's newly fitted risks.

This is stage two of full selector retraining. Its risk checkpoint must carry
the stage-one random-initialization proof. During stage two it stays fixed.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.scripts import train_selector_clean_outcomes as outcome_stage


def main():
    args = outcome_stage.parse_args()
    if args.checkpoint.resolve() != args.baseline_checkpoint.resolve():
        raise ValueError("stage two must use its own newly trained risk checkpoint as the risk-only reference")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = source.get("risk_training", {})
    if (source.get("version") != 3 or provenance.get("initialization") != "random"
            or provenance.get("initial_checkpoint") is not None
            or provenance.get("risk_network_retrained") is not True
            or provenance.get("weights_fit_split") != "train"
            or provenance.get("normalizer_fit_split") != "train"
            or provenance.get("calibration_fit_split") != "validation"
            or provenance.get("test_used_for_selection") is not False
            or not all(provenance.get("updated_components", {}).get(key, 0) > 0
                       for key in ("frame_encoder", "temporal", "head"))):
        raise ValueError("expected a risk checkpoint created by train_selector_clean_risks.py in stage one")
    source_audit = source["data_sources"]
    original_auditor = outcome_stage.audit_filtered_training_sources

    def audited_sources(*arguments, **keywords):
        audit = original_auditor(*arguments, **keywords)
        for key in ("base_manifest_sha256", "verified_original_manifest_sha256", "training_samples"):
            if audit.get(key) != source_audit.get(key):
                raise ValueError(f"risk/outcome training sources differ: {key}")
        identities = lambda value: sorted((item["root"], item["manifest_sha256"])
                                         for item in value["additional_sources"])
        if identities(audit) != identities(source_audit):
            raise ValueError("risk/outcome supplement sources differ")
        audit["risk_parent_provenance"] = "stage one risk network retrained from random initialization on these same training rows"
        audit["risk_pretraining"] = provenance
        return audit

    outcome_stage.audit_filtered_training_sources = audited_sources
    print("[selector full] stage two: using this arm's NEWLY TRAINED risks; risk-only reference is the same new model", flush=True)
    outcome_stage.main()
    report_path = args.output / "outcome_training_report.json"
    import json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["network_training"] = "full two-stage retraining: stage-one risk network from random initialization; stage-two new outcome encoder/head with stage-one risks fixed"
    report["risk_network_retrained_this_experiment"] = True
    report["risk_pretraining"] = provenance
    report["risk_pretraining_checkpoint"] = str(args.checkpoint.resolve())
    report["risk_pretraining_checkpoint_sha256"] = outcome_stage.sha256(args.checkpoint)
    outcome_stage.save_json(report_path, report)


if __name__ == "__main__":
    main()