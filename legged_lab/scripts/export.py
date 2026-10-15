import os
import copy
import json
import warnings
import torch
import torch.nn as nn

class FullPolicyExporter(nn.Module):
    """
    包装器：它不只拷贝 Actor MLP，而是拷贝整个 Policy 架构。
    """
    def __init__(self, policy, normalizer=None):
        super().__init__()
        # 深拷贝完整的 policy（包含你的 ActorCritic 和内部的 CNNModule）
        self.policy = copy.deepcopy(policy)
        # 拷贝归一化器
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        # 🔥 关键：调用 act_inference 才会触发 _process_obs 进行视觉切分和提取
        return self.policy.act_inference(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass


class RENetPolicyExporter(nn.Module):
    """Export-safe RENet policy with dynamic LP/VP routing.

    This wrapper evaluates both estimators and selects their latents with tensor
    operations, so the estimator mask remains a runtime input after tracing.
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.policy = copy.deepcopy(policy)
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        observations = self.normalizer(x)
        actor_input = self.policy._process_actor_obs(observations)
        return self.policy.actor(actor_input)

    @torch.jit.export
    def reset(self):
        pass


def _is_renet_policy(policy):
    return all(
        hasattr(policy, name)
        for name in (
            "estimator_mask_dim",
            "vp_gru",
            "_split_actor_obs",
            "_process_actor_obs",
        )
    )


def _make_policy_exporter(policy, normalizer):
    if _is_renet_policy(policy):
        return RENetPolicyExporter(policy, normalizer)
    return FullPolicyExporter(policy, normalizer)


def _validate_renet_torchscript(policy, normalizer, traced_model, obs_dim):
    """Check eager parity and LP/VP runtime routing of an exported policy."""

    expected_obs_dim = (
        policy.proprio_actor_dim
        + policy.lidar_flat_dim
        + policy.depth_flat_dim
        + policy.estimator_mask_dim
    )
    if obs_dim != expected_obs_dim:
        raise RuntimeError(
            f"RENet observation layout mismatch: exporter got {obs_dim}, "
            f"policy fields describe {expected_obs_dim}."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    sample = 0.05 * torch.randn(2, obs_dim, generator=generator)
    mask_index = policy.proprio_actor_dim + policy.lidar_flat_dim + policy.depth_flat_dim
    sample[:, mask_index] = torch.tensor([0.0, 1.0])

    reference = FullPolicyExporter(policy, normalizer).to("cpu").eval()
    with torch.inference_mode():
        expected = reference(sample)
        actual = traced_model(sample)
    if not torch.allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5):
        max_error = (actual - expected).abs().max().item()
        raise RuntimeError(
            "RENet TorchScript validation failed for LP/VP; "
            f"maximum absolute error: {max_error:.6g}."
        )

    # The inactive sensor branch must not leak into the selected action path.
    base = 0.05 * torch.randn(1, obs_dim, generator=generator)
    lidar_slice = slice(policy.proprio_actor_dim, policy.proprio_actor_dim + policy.lidar_flat_dim)
    depth_slice = slice(
        policy.proprio_actor_dim + policy.lidar_flat_dim,
        mask_index,
    )
    lp_base = base.clone()
    lp_base[:, mask_index] = 1.0
    lp_depth_changed = lp_base.clone()
    lp_depth_changed[:, depth_slice] += 0.5
    lp_lidar_changed = lp_base.clone()
    lp_lidar_changed[:, lidar_slice] += 0.5
    lidar_map = lp_base[:, lidar_slice].reshape(1, policy.lidar_h, policy.lidar_w)
    lp_lidar_center = lp_base.clone()
    center_map = lidar_map.clone()
    center_y, center_x = policy.lidar_h // 2, policy.lidar_w // 2
    center_map[
        :,
        max(0, center_y - 1) : min(policy.lidar_h, center_y + 2),
        max(0, center_x - 1) : min(policy.lidar_w, center_x + 2),
    ] += 0.75
    lp_lidar_center[:, lidar_slice] = center_map.reshape(1, -1)
    lp_lidar_half = lp_base.clone()
    half_map = lidar_map.clone()
    half_map[:, :, policy.lidar_w // 2 :] += 0.5
    lp_lidar_half[:, lidar_slice] = half_map.reshape(1, -1)
    lp_lidar_shuffled = lp_base.clone()
    lp_lidar_shuffled[:, lidar_slice] = torch.flip(lidar_map, dims=(1, 2)).reshape(1, -1)
    vp_base = base.clone()
    vp_base[:, mask_index] = 0.0
    vp_lidar_changed = vp_base.clone()
    vp_lidar_changed[:, lidar_slice] += 0.5
    vp_depth_changed = vp_base.clone()
    vp_depth_changed[:, depth_slice] += 0.5

    with torch.inference_mode():
        lp_action = traced_model(lp_base)
        lp_depth_action = traced_model(lp_depth_changed)
        lp_lidar_action = traced_model(lp_lidar_changed)
        lp_lidar_center_action = traced_model(lp_lidar_center)
        lp_lidar_half_action = traced_model(lp_lidar_half)
        lp_lidar_shuffled_action = traced_model(lp_lidar_shuffled)
        vp_action = traced_model(vp_base)
        vp_lidar_action = traced_model(vp_lidar_changed)
        vp_depth_action = traced_model(vp_depth_changed)

    if lp_action.ndim != 2 or lp_action.shape[1] != int(policy.num_actions):
        raise RuntimeError(
            f"RENet TorchScript action shape must be (N, {policy.num_actions}), "
            f"got {tuple(lp_action.shape)}."
        )
    if not torch.allclose(lp_action, lp_depth_action, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError("LP action unexpectedly depends on the inactive depth input.")
    if not torch.allclose(vp_action, vp_lidar_action, rtol=1.0e-5, atol=1.0e-6):
        raise RuntimeError("VP action unexpectedly depends on the inactive LiDAR input.")
    lidar_probe_actions = (
        lp_lidar_action,
        lp_lidar_center_action,
        lp_lidar_half_action,
        lp_lidar_shuffled_action,
    )
    if all(
        torch.allclose(lp_action, action, rtol=1.0e-6, atol=1.0e-7)
        for action in lidar_probe_actions
    ):
        raise RuntimeError(
            "LP action is insensitive to uniform, local, half-map, and spatial LiDAR "
            "perturbations; refusing to export."
        )
    if torch.allclose(vp_action, vp_depth_action, rtol=1.0e-6, atol=1.0e-7):
        raise RuntimeError("VP action is insensitive to the depth input; refusing to export.")
    print("[SUCCESS] TorchScript LP/VP routing validation passed.")


def _get_policy_obs_dim(policy):
    if hasattr(policy, "num_actor_obs"):
        return int(policy.num_actor_obs)
    return int(
        getattr(policy, "proprio_actor_dim", 0)
        + getattr(policy, "lidar_flat_dim", 0)
        + getattr(policy, "depth_flat_dim", 0)
        + getattr(policy, "estimator_mask_dim", 0)
    )


class G1StudentDeploymentExporter(nn.Module):
    """
    g1_student 部署包装器。

    导出的策略必须包含完整控制逻辑：
        action = blind_action + gate * clipped_residual
    而不是只导出 student residual actor。
    """

    def __init__(
        self,
        student_policy,
        blind_policy,
        normalizer=None,
        residual_action_clip=0.1,
        blind_obs_dim=100,
        blind_obs_history_length=5,
    ):
        super().__init__()
        self.student_policy = copy.deepcopy(student_policy)
        self.blind_policy = copy.deepcopy(blind_policy)
        if normalizer is not None:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

        self.residual_action_clip = residual_action_clip
        self.blind_obs_dim = blind_obs_dim
        self.blind_obs_history_length = blind_obs_history_length
        self.blind_obs_hist_dim = blind_obs_dim * blind_obs_history_length
        self.proprio_dim = student_policy.proprio_actor_dim
        self.has_cnn = student_policy.has_cnn
        self.single_proprio_dim = student_policy.single_proprio_dim
        self.cnn_channels = 0
        self.cnn_h = 0
        self.cnn_w = 0
        if self.has_cnn:
            self.cnn_channels = student_policy.cnn_channels
            self.cnn_h = student_policy.cnn_h
            self.cnn_w = student_policy.cnn_w

    def forward(self, x):
        student_obs = self.normalizer(x)

        if self.has_cnn:
            proprio_history = student_obs[:, :self.proprio_dim]
            depth_flat = student_obs[:, self.proprio_dim:]
            current_proprio = proprio_history[:, -self.single_proprio_dim:]
            his_latent = self.student_policy.history_encoder(proprio_history)
            depth_img = depth_flat.view(-1, self.cnn_channels, self.cnn_h, self.cnn_w)
            depth_latent = self.student_policy.cnn(depth_img)
            actor_input = torch.cat([current_proprio, his_latent, depth_latent], dim=-1)
        else:
            actor_input = student_obs

        residual_raw = self.student_policy.residual_actor(actor_input)
        reliability_gate = torch.sigmoid(self.student_policy.reliability_gate_head(actor_input))
        residual = self.residual_action_clip * torch.tanh(residual_raw)

        blind_obs_hist = x[:, self.proprio_dim - self.blind_obs_hist_dim : self.proprio_dim]
        blind_obs = blind_obs_hist[:, -self.blind_obs_dim:]
        blind_actions = self.blind_policy.act_inference_deterministic(blind_obs, blind_obs_hist)

        return blind_actions + reliability_gate * residual

    @torch.jit.export
    def reset(self):
        pass


def export_policy_as_jit(policy, normalizer, path, filename="policy.pt", metadata=None, metadata_filename="renet_contract.json"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    exporter = _make_policy_exporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()

    obs_dim = _get_policy_obs_dim(policy)
    dummy_input = torch.zeros(1, obs_dim)

    traced_model = torch.jit.trace(
        exporter,
        dummy_input,
        strict=False,
        # MultiheadAttention may select a different eager fast path during
        # PyTorch's second trace pass.  The graph is validated immediately
        # below against eager LP/VP outputs and structured sensor probes.
        check_trace=False,
    )

    if _is_renet_policy(policy):
        _validate_renet_torchscript(policy, normalizer, traced_model, obs_dim)

    save_path = os.path.join(path, filename)
    extra_files = {}
    if metadata is not None:
        extra_files[metadata_filename] = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True
        )
    torch.jit.save(traced_model, save_path, _extra_files=extra_files)

    print(f"\n[SUCCESS] TorchScript policy exported to: {save_path}")
    return save_path


def export_g1_student_policy_as_jit(runner, path, filename="policy.pt"):
    """
    导出 g1_student 的完整部署策略。

    runner.alg.policy 只会输出 student residual；这个导出器会同时封装 blind policy、
    reliability gate 和 residual clip，供 sim2sim 直接调用。
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    exporter = G1StudentDeploymentExporter(
        student_policy=runner.alg.policy,
        blind_policy=runner.blind_policy,
        normalizer=runner.obs_normalizer,
        residual_action_clip=runner.residual_action_clip,
        blind_obs_dim=runner.blind_obs_dim,
        blind_obs_history_length=runner.blind_obs_history_length,
    )
    exporter.to("cpu")
    exporter.eval()

    scripted_model = torch.jit.script(exporter)
    save_path = os.path.join(path, filename)
    scripted_model.save(save_path)
    print(f"\n[SUCCESS] g1_student 完整部署策略已导出至: {save_path}")


def export_policy_as_onnx(policy, normalizer, path, filename="policy.onnx"):
    """Export a deployment policy as an opset-17 ONNX model."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    exporter = _make_policy_exporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()

    # Compute the raw actor-observation width. RENet includes one estimator mask.
    obs_dim = _get_policy_obs_dim(policy)
    dummy_input = torch.zeros(1, obs_dim)

    # Deployment is one robot per policy call. Keeping batch=1 also avoids the
    # unsupported variable-batch/variable-length GRU combination in ONNX.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Exporting a model to ONNX with a batch_size other than 1.*",
        )
        torch.onnx.export(
            exporter,
            dummy_input,
            os.path.join(path, filename),
            opset_version=17,
            input_names=['obs'],
            output_names=['actions'],
        )
    save_path = os.path.join(path, filename)
    import onnx

    onnx.checker.check_model(onnx.load(save_path))
    print("[SUCCESS] ONNX checker validation passed.")
    print(f"[SUCCESS] ONNX 模型已导出至: {save_path}")
    return save_path
