"""Train decision outcomes on train; fit/calibrate on validation; report test once.

The existing calibrated two-risk network and its feature normalizer are frozen.
All seed/snapshot selection and acceptance finish before test predictions.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legged_lab.selector.dataset import SelectorShardDataset, ShardBatchSampler
from legged_lab.selector.evaluation import Predictions, collect_predictions, report_predictions
from legged_lab.selector.feature_contract import dataset_feature_contract
from legged_lab.selector.frozen_policy import assert_same_policy, dataset_policy_identity, sha256
from legged_lab.selector.labels import dataset_safety_label_contract
from legged_lab.selector.metrics import json_safe_metrics
from legged_lab.selector.model import selector_from_artifact
from legged_lab.selector.outcomes import (
    OUTCOME_ORDER, OutcomeNetworkConfig, OutcomeNetwork, OutcomeSelector,
    fit_outcome_calibration, outcome_classes, outcome_loss, weights_from_outcome_counts,
)
from legged_lab.selector.outcome_decision import OutcomeGuardConfig, fit_outcome_rule
from legged_lab.selector.selection import SelectionRule, current_mask_from_features
from legged_lab.selector.supplement_dataset import SelectorTrainingDataset, audit_training_sources


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_safe_metrics(data), ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k,v in module.state_dict().items()}


def unchanged_source(artifact, model):
    risk = model.risk_selector
    for key, actual in (("model_state_dict", risk.model.state_dict()),
                        ("normalizer_state_dict", risk.normalizer.state_dict())):
        if artifact[key].keys() != actual.keys() or any(not torch.equal(v, actual[k].cpu()) for k,v in artifact[key].items()):
            raise RuntimeError(f"outcome training changed frozen {key}")
    for key in ("temperatures", "calibration_biases"):
        if not torch.equal(artifact[key], getattr(risk,key).detach().cpu()):
            raise RuntimeError(f"outcome training changed frozen {key}")


def loader_for(dataset, args, device, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_sampler=ShardBatchSampler(dataset, args.batch_size,
        shuffle=shuffle, generator=generator), num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)


@torch.no_grad()
def validation_outputs(model, loader, device, class_weights, contrast_weights, coefficient):
    model.eval()
    logits, risks, labels, masks = [], [], [], []
    for batch in loader:
        r,z = model.raw_decision_outputs(batch["features"].to(device,non_blocking=True))
        logits.append(z.float().cpu())
        risks.append(r.float().cpu())
        labels.append(batch["labels"].float().cpu())
        masks.append(current_mask_from_features(batch["features"]).cpu())
    z,y = torch.cat(logits),torch.cat(labels)
    loss,joint,contrast = outcome_loss(z,y,class_weights.cpu(),contrast_weights.cpu(),coefficient)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite outcome validation loss")
    return float(loss),float(joint),float(contrast),z,torch.cat(risks),y,torch.cat(masks)


def train_seed(seed, source, risk_template, train_dataset, validation_loader, counts, args, device, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    network_cfg = OutcomeNetworkConfig()
    head = OutcomeNetwork(risk_template.model.cfg.frame_dim,risk_template.model.cfg.gru_hidden_dim,network_cfg)
    model = OutcomeSelector(copy.deepcopy(risk_template),head).to(device)
    class_weights,contrast_weights = weights_from_outcome_counts(counts)
    class_weights,contrast_weights = class_weights.to(device),contrast_weights.to(device)
    train_loader = loader_for(train_dataset,args,device,True,seed)
    optimizer = torch.optim.AdamW(model.outcome_model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda",enabled=device.type == "cuda")
    best_loss,stale = math.inf,0
    snapshots,history = [],[]
    snapshot_dir = output/"training_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    for epoch in range(1,args.max_epochs+1):
        model.train()
        total,samples = 0.,0
        for batch in train_loader:
            features = batch["features"].to(device,non_blocking=True)
            labels = batch["labels"].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type == "cuda"):
                _,logits = model.raw_decision_outputs(features)
                loss,_,_ = outcome_loss(logits,labels,class_weights,contrast_weights,args.contrast_coefficient)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss, seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.outcome_model.parameters(),1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*len(labels)
            samples += len(labels)
        v,j,c,*_ = validation_outputs(model,validation_loader,device,class_weights,contrast_weights,args.contrast_coefficient)
        row = {"epoch":epoch,"train_loss":total/samples,"validation_loss":v,
               "validation_joint_loss":j,"validation_contrast_loss":c}
        history.append(row)
        print(f"[selector outcomes seed={seed}] epoch={epoch:03d} train={row['train_loss']:.6f} "
              f"validation={v:.6f} joint={j:.6f} contrast={c:.6f}",flush=True)
        if len(snapshots)<args.top_k or v<snapshots[-1]["loss"]:
            state = cpu_state(model.outcome_model)
            snapshots.append({"loss":v,"epoch":epoch,"state":state})
            snapshots.sort(key=lambda item:item["loss"])
            snapshots = snapshots[:args.top_k]
        if v<best_loss-1e-5:
            best_loss,stale = v,0
            save_torch(snapshot_dir/f"seed_{seed}_best.pt",{
                "artifact":"outcome_training_snapshot_not_deployable","seed":seed,"epoch":epoch,
                "outcome_network_config":asdict(network_cfg),"outcome_state_dict":snapshots[0]["state"],
                "source_checkpoint_sha256":sha256(args.checkpoint),"history":history,
            })
        else:
            stale += 1
        save_json(output/f"outcome_seed_{seed}_history.json",history)
        if stale>=args.patience:
            print(f"[selector outcomes seed={seed}] early stop; best_epoch={snapshots[0]['epoch']}",flush=True)
            break
    unchanged_source(source,model)
    del train_loader
    return network_cfg,snapshots,history,class_weights.cpu(),contrast_weights.cpu()


def parse_args():
    p = argparse.ArgumentParser(description="Train direct branch choices and independent joint-failure warnings.")
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--supplement-datasets",nargs="+",default=[],type=Path)
    p.add_argument("--checkpoint",required=True,type=Path,help="Existing affine-calibrated two-risk checkpoint.")
    p.add_argument("--baseline-checkpoint",required=True,type=Path,help="Original two-risk model before decision tuning.")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--batch-size",type=int,default=4096)
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--max-epochs",type=int,default=100)
    p.add_argument("--patience",type=int,default=15)
    p.add_argument("--top-k",type=int,default=2,help="Validation-loss snapshots per seed to compare before test.")
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--contrast-coefficient",type=float,default=1.)
    p.add_argument("--max-dangerous-rate",type=float,default=.03)
    p.add_argument("--max-both-safe-switch-rate",type=float,default=.05)
    p.add_argument("--max-both-failure-switch-rate",type=float,default=.10)
    p.add_argument("--min-both-failure-recall",type=float,default=.80)
    p.add_argument("--min-both-failure-precision",type=float,default=.50)
    p.add_argument("--min-class-samples",type=int,default=20)
    p.add_argument("--min-contrast-samples",type=int,default=50)
    args = p.parse_args()
    if (min(args.batch_size,args.max_epochs,args.patience,args.top_k)<1 or args.num_workers<0
            or len(set(args.seeds)) != len(args.seeds)):
        p.error("batch/epoch/patience/top-k must be positive; workers nonnegative; seeds distinct")
    if not all(math.isfinite(x) and x>0 for x in (args.learning_rate,args.contrast_coefficient)):
        p.error("learning rate and contrast coefficient must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay<0:
        p.error("weight decay must be finite and nonnegative")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        p.error("use a new or empty output directory")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(min(torch.get_num_threads(),8))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; activate env_tienkung or choose --device cpu")
    if hasattr(torch.backends,"cuda"): torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    guard_cfg = OutcomeGuardConfig(args.max_dangerous_rate,args.max_both_safe_switch_rate,
        args.max_both_failure_switch_rate,args.min_both_failure_recall,args.min_both_failure_precision,
        args.min_class_samples,args.min_contrast_samples)
    guard_cfg.validate()
    source = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    baseline = torch.load(args.baseline_checkpoint,map_location="cpu",weights_only=False)
    if source.get("version") not in (3,4) or "calibration_biases" not in source:
        raise ValueError("source must be the existing affine-calibrated two-risk model, not a new outcome checkpoint")
    if baseline.get("version") != 3 or "selection_rule" in baseline:
        raise ValueError("baseline must be the original version 3 two-risk training checkpoint")
    assert_same_policy(source["frozen_policy"],baseline["frozen_policy"])
    for key in ("model_state_dict","normalizer_state_dict"):
        if source[key].keys()!=baseline[key].keys() or any(not torch.equal(v,baseline[key][k]) for k,v in source[key].items()):
            raise ValueError("source and original baseline must have identical risk network and feature statistics")
    print("[selector outcomes] checking split identities and contracts",flush=True)
    audit = audit_training_sources(args.dataset,args.supplement_datasets)
    for parent in (source,baseline):
        expected = (parent.get("data_sources") or {}).get("base_manifest_sha256")
        if expected is not None and expected != audit["base_manifest_sha256"]:
            raise ValueError("checkpoint was trained on a different base dataset manifest")
    train_dataset = SelectorTrainingDataset([args.dataset,*args.supplement_datasets])
    validation_dataset = SelectorTrainingDataset([args.dataset],"validation")
    if min(len(train_dataset),len(validation_dataset))==0:
        raise ValueError("train and validation splits must be nonempty")
    manifest = train_dataset.manifest
    assert_same_policy(source["frozen_policy"],dataset_policy_identity(manifest))
    if (dataset_feature_contract(manifest)!=source["selector_feature_contract"]
            or dataset_safety_label_contract(manifest)!=source["safety_label_contract"]):
        raise ValueError("dataset/source feature or safety label contract differs")
    counts = torch.zeros(4,dtype=torch.long)
    for path in train_dataset.shards:
        data = torch.load(path,map_location="cpu",weights_only=False)
        counts += torch.bincount(outcome_classes(data["labels"]),minlength=4)
    class_weights,contrast_weights = weights_from_outcome_counts(counts)
    print(f"[selector outcomes] train class order={list(OUTCOME_ORDER)} counts={counts.tolist()}",flush=True)
    print(f"[selector outcomes] device={device}; frozen risk/normalizer/calibration; requirements={asdict(guard_cfg)}",flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    save_json(args.output/"training_sources.json",audit)
    risk_template = selector_from_artifact(source).eval()
    reference = selector_from_artifact(baseline).to(device).eval()
    validation_loader = loader_for(validation_dataset,args,device)
    baseline_prediction = collect_predictions(reference,validation_loader,device)
    candidates = []
    training_cfg = {k:v for k,v in vars(args).items() if k not in ("dataset","checkpoint","baseline_checkpoint","output","supplement_datasets")}
    for seed in args.seeds:
        cfg,snapshots,history,cw,rw = train_seed(seed,source,risk_template,train_dataset,
            validation_loader,counts,args,device,args.output)
        for rank,snapshot in enumerate(snapshots,1):
            head = OutcomeNetwork(risk_template.model.cfg.frame_dim,risk_template.model.cfg.gru_hidden_dim,cfg)
            head.load_state_dict(snapshot["state"],strict=True)
            model = OutcomeSelector(copy.deepcopy(risk_template),head).to(device).eval()
            v,_,_,z,risks,labels,masks = validation_outputs(model,validation_loader,device,cw,rw,args.contrast_coefficient)
            temperature,biases,calibration = fit_outcome_calibration(z,labels,cw)
            # Match inference's calibration arithmetic on the training device;
            # fitting thresholds from a separate CPU softmax can move boundary
            # rows when the model is subsequently evaluated on CUDA.
            q = torch.softmax(z.to(device)/temperature.to(device)+biases.to(device),dim=1).cpu()
            print(f"[selector outcomes seed={seed} snapshot={rank}] fit decision on validation; epoch={snapshot['epoch']}",flush=True)
            rule,fit = fit_outcome_rule(risks,q,labels,masks,baseline_prediction.probabilities,guard_cfg)
            unchanged_source(source,model)
            artifact = copy.deepcopy(source)
            artifact.update({"version":5,"seed":seed,"outcome_network_config":asdict(cfg),
                "outcome_state_dict":snapshot["state"],"outcome_order":list(OUTCOME_ORDER),
                "outcome_temperature":temperature,"outcome_biases":biases,"outcome_calibration":calibration,
                "outcome_training_config":training_cfg,"outcome_train_class_counts":counts,
                "outcome_class_weights":cw,"outcome_contrast_weights":rw,"outcome_history":history,
                "outcome_best_epoch":snapshot["epoch"],"outcome_guard_config":asdict(guard_cfg),
                "selection_rule":rule.to_dict(),"selection_rule_fit":fit,"data_sources":audit,
                "source_metrics_before_outcomes":copy.deepcopy(source.get("metrics",{})),
                "metrics":{f"validation/{k}":v for k,v in fit["selected_metrics"].items()},
                "outcome_source":{"checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":sha256(args.checkpoint),
                    "baseline_checkpoint":str(args.baseline_checkpoint.resolve()),"baseline_sha256":sha256(args.baseline_checkpoint),
                    "risk_network_unchanged":True,"normalizer_unchanged":True,"risk_calibration_unchanged":True,
                    "weights_fit_split":"train","calibration_and_decision_fit_split":"validation",
                    "test_used_for_acceptance":False}})
            artifact["metrics"]["validation/decision_guard_passed"] = float(fit["accepted"])
            artifact["metrics"]["validation/outcome_training_loss"] = v
            artifact["metrics"]["outcome_training/epochs"] = len(history)
            path = args.output/f"outcome_seed_{seed}_snapshot_{rank}.pt"
            save_torch(path,artifact)
            candidates.append((path,artifact))
            print(f"[selector outcomes seed={seed} snapshot={rank}] accepted={fit['accepted']} "
                  f"dangerous={fit['selected_metrics']['selection/dangerous_branch_rate']:.4%} "
                  f"joint_recall={fit['selected_metrics']['selection/both_failure_flag_recall']:.4%}",flush=True)
    def key(item):
        artifact = item[1]
        fit,m = artifact["selection_rule_fit"],artifact["selection_rule_fit"]["selected_metrics"]
        failed = sum(not v for v in fit["guard"]["checks"].values())
        return (not fit["accepted"],failed,m["selection/dangerous_branch_count"],
                max(m["selection/lp_only_failure_wrong_rate"],m["selection/vp_only_failure_wrong_rate"]),
                m["switch/missed_rescue_count"],m["switch/unnecessary_switch_count"]+m["switch/both_failure_switch_count"])
    candidates.sort(key=key)
    path,artifact = candidates[0]
    accepted = artifact["selection_rule_fit"]["accepted"]
    destination = args.output/("lp_vp_selector_checkpoint.pt" if accepted else "candidate_selector_checkpoint.pt")
    shutil.copyfile(path,destination)
    digest_before_test = sha256(destination)
    selection_freeze = {"selected":path.name,"checkpoint":str(destination.resolve()),
        "checkpoint_sha256_before_test":digest_before_test,"accepted_on_validation":accepted,
        "decision_guard":artifact["selection_rule_fit"]["guard"],"test_used_for_selection":False,
        "selection_priority":["guard_passed","fewest_failed_validation_guards","fewest_dangerous_choices",
                              "lowest_worst_direction_error_rate","fewest_missed_rescues","fewest_redundant_requests"]}
    save_json(args.output/"selection_freeze.json",selection_freeze)
    print(f"[selector outcomes] frozen selection={path.name}; validation accepted={accepted}",flush=True)
    selected_model = selector_from_artifact(artifact).to(device).eval()
    rule = SelectionRule.from_dict(artifact["selection_rule"])
    previous_rule = SelectionRule.from_dict(source["selection_rule"]) if "selection_rule" in source else None
    validation_predictions = collect_predictions(selected_model,validation_loader,device)
    def comparison(predictions,original):
        frozen = Predictions(predictions.probabilities,predictions.labels,predictions.current_mask,predictions.audit)
        return {"baseline":report_predictions(original,None),
                "previous_rule":report_predictions(frozen,previous_rule),
                "candidate":report_predictions(predictions,rule,min_class_samples=guard_cfg.min_class_samples,
                    min_contrast_samples=guard_cfg.min_contrast_samples,outcome_guard_config=guard_cfg)}
    validation_report = comparison(validation_predictions,baseline_prediction)
    print("[selector outcomes] reading test for reporting only; no further fitting or reselection",flush=True)
    test_dataset = SelectorShardDataset(args.dataset,"test")
    test_loader = loader_for(test_dataset,args,device)
    test_predictions = collect_predictions(selected_model,test_loader,device)
    test_baseline = collect_predictions(reference,test_loader,device)
    test_report = comparison(test_predictions,test_baseline)
    unchanged_source(source,selected_model)
    if sha256(destination)!=digest_before_test:
        raise RuntimeError("frozen checkpoint changed after test reporting")
    summary = {"report_version":6,**selection_freeze,"recommended_checkpoint":str(destination.resolve()) if accepted else None,
        "network_training":"new outcome encoder/head only; risk predictor remains frozen",
        "train_class_order":list(OUTCOME_ORDER),"train_class_counts":counts.tolist(),
        "guard_config":asdict(guard_cfg),"calibration":artifact["outcome_calibration"],
        "selection_rule_fit":artifact["selection_rule_fit"],"validation":validation_report,"test":test_report,
        "all_candidates":{p.name:{"accepted_on_validation":a["selection_rule_fit"]["accepted"],
            "best_epoch":a["outcome_best_epoch"],"validation_metrics":a["selection_rule_fit"]["selected_metrics"],
            "guard":a["selection_rule_fit"]["guard"]} for p,a in candidates},
        "notes":["Only the training split updates outcome network parameters.",
                 "Validation alone fits outcome calibration, warning/action thresholds and seed/snapshot selection.",
                 "Only the selected frozen model reads test labels; the checkpoint is not rewritten afterwards.",
                 "No candidate passing all guards means diagnostic output only, not a relaxed target.",
                 "Offline request rates are not closed-loop rollout switching or physical fall rates."]}
    report = args.output/"outcome_training_report.json"
    save_json(report,summary)
    print(json.dumps(json_safe_metrics({"accepted_on_validation":accepted,"checkpoint":str(destination),
        "report":str(report),"guard":summary["decision_guard"],"test_candidate":test_report["candidate"]["metrics"],
        "test_checks":test_report["candidate"]["checks"]}),indent=2,sort_keys=True,allow_nan=False),flush=True)
    if not accepted:
        print("[selector outcomes] no candidate passed; diagnostic checkpoint only; source retained",flush=True)


if __name__ == "__main__":
    main()
