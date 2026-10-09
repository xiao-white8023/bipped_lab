from __future__ import annotations

import torch


def validate_renet_vp_checkpoint_contract(saved, current, legacy_history_frames):
    """Permit explicit temporal-window changes while rejecting other VP mismatches."""
    if saved is None:
        if current.get("version", 1) != 1:
            raise ValueError("Legacy checkpoint has no multi-head VP contract; train the new VP from scratch.")
        saved = {**current, "proprio_history_frames": legacy_history_frames}
    expected_keys = set(current)
    if set(saved) != expected_keys or any(saved[key] != current[key]
                                        for key in expected_keys - {"proprio_history_frames"}):
        raise ValueError("Checkpoint VP encoder contract is incompatible.")
    previous = saved["proprio_history_frames"]
    if isinstance(previous, bool) or not isinstance(previous, int) or not 1 <= previous <= legacy_history_frames:
        raise ValueError("Checkpoint VP proprio history length is invalid.")
    if previous != current["proprio_history_frames"]:
        return [f"VP proprio history changed from {previous} to {current['proprio_history_frames']} frames; "
                "preserved network weights/Adam state for continued training. VP actions may change."]
    return []


def validate_renet_critic_checkpoint_contract(saved, current, legacy_history_frames=None):
    """Permit the explicit ten-to-five-frame layout migration and VP masking."""
    if current is None:
        if saved is not None:
            raise ValueError("Checkpoint critic observation contract is incompatible.")
        return []
    if saved is None:
        previous_frames = current["history_frames"] if legacy_history_frames is None else legacy_history_frames
        saved = {**current, "history_frames": previous_frames, "vp_history_frames": previous_frames}
    if (set(saved) != set(current)
            or any(saved[key] != current[key] for key in set(current) - {"history_frames", "vp_history_frames"})):
        raise ValueError("Checkpoint critic observation contract is incompatible.")
    previous_frames = saved["history_frames"]
    if (type(previous_frames) is not int or previous_frames < 1
            or (legacy_history_frames is not None and previous_frames != legacy_history_frames)):
        raise ValueError("Checkpoint critic history does not match its input weights.")
    if (previous_frames != current["history_frames"]
            and (previous_frames, current["history_frames"]) != (10, 5)):
        raise ValueError("Only a ten-to-five-frame critic layout migration is supported.")
    previous = saved["vp_history_frames"]
    if type(previous) is not int or not 1 <= previous <= previous_frames:
        raise ValueError("Checkpoint VP critic history length is invalid.")
    if previous_frames != current["history_frames"]:
        return ["Critic history changed from 10 to 5 frames for LP/VP; "
                "retain the latest-frame/terrain weights and matching Adam columns. LP/VP values may change."]
    if previous != current["vp_history_frames"]:
        return [f"VP critic history changed from {previous} to {current['vp_history_frames']} frames; "
                "preserved shared critic weights/Adam state. VP values may change."]
    return []


def upgrade_renet_auxiliary_checkpoint(
    checkpoint, policy, optimizer, contract, *, load_optimizer=True,
    discriminator=None, amp_normalizer=None, amp_contract=None,
):
    """Upgrade new AUX parameters, target scales, and AMP preprocessing.

    Call on fresh modules. Preserve actor and unchanged Adam moments by name;
    crop critic input columns/moments when reducing ten frames to five.
    Reset AMP jointly when its input statistics were incompatible.
    Unknown keys, partially missing heads and incompatible shapes still fail.
    """
    upgraded = dict(checkpoint)
    saved_model = dict(checkpoint["model_state_dict"])
    current_model = policy.state_dict()
    lp_keys = {name for name in current_model if name.startswith("lp_feet_height_decoder.")}
    missing_lp = lp_keys - saved_model.keys()
    if missing_lp and missing_lp != lp_keys:
        raise ValueError("Checkpoint contains an incomplete LP feet-height decoder.")
    other_missing = current_model.keys() - saved_model.keys() - missing_lp
    unexpected = saved_model.keys() - current_model.keys()
    if other_missing or unexpected:
        raise ValueError(f"Incompatible RENet checkpoint: missing={sorted(other_missing)}, unexpected={sorted(unexpected)}")
    critic_contract = policy.get_critic_observation_contract() if hasattr(policy, "get_critic_observation_contract") else None
    crop_columns = {}
    if critic_contract is not None:
        old_width = saved_model["critic.0.weight"].shape[1]
        history_dim = old_width - critic_contract["terrain_dim"]
        if history_dim <= 0 or history_dim % critic_contract["single_frame_dim"]:
            raise ValueError("Checkpoint critic input has an incompatible observation layout.")
        old_frames = history_dim // critic_contract["single_frame_dim"]
        validate_renet_critic_checkpoint_contract(
            checkpoint.get("critic_observation_contract"), critic_contract, old_frames
        )
    for name in saved_model:
        if saved_model[name].shape != current_model[name].shape:
            if (name == "critic.0.weight" and critic_contract is not None
                    and old_frames == 10 and critic_contract["history_frames"] == 5
                    and saved_model[name].shape[0] == current_model[name].shape[0]):
                first_column = 5 * critic_contract["single_frame_dim"]
                cropped = saved_model[name][:, first_column:]
                if cropped.shape != current_model[name].shape:
                    raise ValueError("Checkpoint critic latest-frame columns have an incompatible shape.")
                crop_columns[name] = first_column
                saved_model[name] = cropped.clone()
            else:
                raise ValueError(f"Incompatible RENet checkpoint tensor shape: {name}")

    old_contract = checkpoint.get("auxiliary_training_contract", {})
    if contract.get("version", 1) == 2 and old_contract.get("vp_vae") != contract.get("vp_vae"):
        raise ValueError("Checkpoint VP beta-VAE training contract is incompatible.")
    target_changed = (
        old_contract.get("vp_terrain_target_normalization", "batch")
        != contract["vp_terrain_target_normalization"]
        or old_contract.get("terrain_target_clip", 1.0) != contract["terrain_target_clip"]
    )
    reset_names = {name for name in current_model if target_changed and name.startswith(("terrain_decoder.", "vp_terrain_decoder."))}
    for name in current_model:
        if name in missing_lp or name in reset_names:
            saved_model[name] = current_model[name].detach().clone()
    upgraded["model_state_dict"] = {name: saved_model[name] for name in current_model}
    upgraded["auxiliary_training_contract"] = dict(contract)
    if critic_contract is not None:
        upgraded["critic_observation_contract"] = dict(critic_contract)
    messages = []
    if crop_columns:
        messages.append("Reduced critic input from 1037 to 612; kept the latest five privileged frames and terrain weights.")
        if "privileged_obs_norm_state_dict" in checkpoint:
            normalization = dict(checkpoint["privileged_obs_norm_state_dict"])
            if set(normalization) != {"_mean", "_var", "_std", "count"}:
                raise ValueError("Checkpoint critic normalizer layout is incompatible.")
            for key in ("_mean", "_var", "_std"):
                if normalization[key].shape != (1, old_width):
                    raise ValueError("Checkpoint critic normalizer shape is incompatible.")
                normalization[key] = normalization[key][:, crop_columns["critic.0.weight"]:].clone()
            upgraded["privileged_obs_norm_state_dict"] = normalization
    if missing_lp:
        messages.append("Initialized the new LP feet-height decoder; existing policy weights are preserved.")
    if reset_names:
        messages.append("VP terrain target contract changed; initialized its AUX decoder for the new target scale.")

    reset_amp = False
    if discriminator is not None:
        saved_discriminator = checkpoint["discriminator_state_dict"]
        current_discriminator = discriminator.state_dict()
        if saved_discriminator.keys() != current_discriminator.keys():
            raise ValueError("Checkpoint AMP discriminator module layout does not match RENet.")
        shape_changes = {name for name in current_discriminator
                         if saved_discriminator[name].shape != current_discriminator[name].shape}
        if shape_changes:
            reset_amp = (
                shape_changes == {"trunk.0.weight"}
                and (saved_discriminator["trunk.0.weight"].shape[1],
                     current_discriminator["trunk.0.weight"].shape[1]) in ((100, 116), (116, 100))
                and amp_contract is not None
                and amp_contract["dimension"] * 2 == current_discriminator["trunk.0.weight"].shape[1]
            )
            if not reset_amp:
                raise ValueError("Incompatible AMP discriminator shapes; only 50/58-D layout switches are supported.")
            new_dim = current_discriminator["trunk.0.weight"].shape[1] // 2
            change = ("AMP upgraded from 50 to 58 features including ankles" if new_dim == 58
                      else "AMP switched from 58 to 50 non-ankle features")
            messages.append(f"AMP reset: {change}; initialized the {2 * new_dim}-input discriminator and AMP normalizer.")
        saved_amp_contract = checkpoint.get("amp_observation_contract")
        if not reset_amp:
            preprocessing_keys = {"version", "normalizer_statistics", "gradient_penalty_space"}
            if saved_amp_contract is not None:
                if amp_contract is None:
                    raise ValueError("Checkpoint AMP feature contract requires a current AMP contract.")
                saved_features = {k: v for k, v in saved_amp_contract.items() if k not in preprocessing_keys}
                current_features = {k: v for k, v in amp_contract.items() if k not in preprocessing_keys}
                if saved_features != current_features:
                    raise ValueError("Checkpoint AMP feature contract does not match the current joint order/layout.")
            if amp_contract is not None and "normalizer_statistics" in amp_contract:
                legacy_preprocessing = saved_amp_contract is None or (
                    saved_amp_contract.get("version", 1) == 1
                    and "normalizer_statistics" not in saved_amp_contract
                    and "gradient_penalty_space" not in saved_amp_contract
                )
                if legacy_preprocessing:
                    reset_amp = True
                    messages.append("AMP reset: legacy mixed-scale statistics/gradient penalty; initialized the discriminator and raw-data normalizer, preserving the policy.")
                elif saved_amp_contract != amp_contract:
                    raise ValueError("Checkpoint AMP preprocessing contract is incompatible.")
        if reset_amp:
            if amp_normalizer is None:
                raise ValueError("The AMP upgrade requires a fresh normalizer.")
            upgraded["discriminator_state_dict"] = {name: value.detach().clone() for name, value in current_discriminator.items()}
            upgraded["amp_normalizer"] = amp_normalizer
        upgraded["amp_observation_contract"] = amp_contract

    if load_optimizer and (missing_lp or reset_names or reset_amp or crop_columns):
        saved_optimizer = checkpoint["optimizer_state_dict"]
        current_optimizer = optimizer.state_dict()
        saved_groups = saved_optimizer["param_groups"]
        current_groups = current_optimizer["param_groups"]
        if len(saved_groups) != len(current_groups):
            raise ValueError("Checkpoint optimizer group count does not match RENet.")
        parameters = dict(policy.named_parameters())
        current_names = list(parameters)
        old_names = checkpoint.get("policy_parameter_names")
        if old_names is None:
            old_names = [name for name in checkpoint["model_state_dict"] if name in parameters]
        expected_old_names = [name for name in current_names if name not in missing_lp]
        if old_names != expected_old_names:
            raise ValueError("Checkpoint policy parameter registration order is incompatible.")
        migrated_state, migrated_groups = {}, []
        for saved_group, current_group, live_group in zip(saved_groups, current_groups, optimizer.param_groups, strict=True):
            if saved_group.get("name") != current_group.get("name"):
                raise ValueError("Checkpoint optimizer group names/order do not match RENet.")
            if current_group.get("name") == "policy":
                if len(saved_group["params"]) != len(old_names):
                    raise ValueError("Checkpoint policy optimizer parameter count does not match model keys.")
                current_by_name = dict(zip(current_names, current_group["params"], strict=True))
                pairs = [(old_id, current_by_name[name], parameters[name], name)
                         for name, old_id in zip(old_names, saved_group["params"], strict=True)]
            else:
                if len(saved_group["params"]) != len(current_group["params"]):
                    raise ValueError("Checkpoint discriminator optimizer parameter count does not match.")
                pairs = [(old_id, new_id, parameter, None) for old_id, new_id, parameter in
                         zip(saved_group["params"], current_group["params"], live_group["params"], strict=True)]
            for old_id, new_id, parameter, name in pairs:
                if reset_amp and current_group.get("name") in ("amp_trunk", "amp_head"):
                    continue
                state = saved_optimizer["state"].get(old_id)
                if state is None or name in reset_names:
                    continue
                state = dict(state)
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = state.get(key)
                    if name in crop_columns and isinstance(value, torch.Tensor):
                        if value.shape != checkpoint["model_state_dict"][name].shape:
                            raise ValueError(f"Checkpoint Adam state shape is incompatible: {name}/{key}")
                        value = value[:, crop_columns[name]:].clone()
                        state[key] = value
                    if isinstance(value, torch.Tensor) and value.shape != parameter.shape:
                        raise ValueError(f"Checkpoint Adam state shape is incompatible: {name or new_id}/{key}")
                migrated_state[new_id] = state
            migrated_groups.append({**saved_group, "params": current_group["params"]})
        upgraded["optimizer_state_dict"] = {"state": migrated_state, "param_groups": migrated_groups}
        messages.append("Preserved Adam state for unchanged parameters; new/reset modules start without moments.")
    return upgraded, messages
