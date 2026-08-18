import os
import copy
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
    """Export-safe RENet policy with dynamic VP/OP/Recovery routing.

    ``RENetActorCritic`` skips inactive branches at runtime.  A trace made with
    a single example would therefore permanently remove the other modes from
    TorchScript/ONNX.  This deployment wrapper evaluates all three encoders and
    selects their latent with tensor operations, so ``actor_mode`` remains a
    real runtime input in the exported graph.
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
        proprio_history, depth_flat, actor_mode, beta_obs, current_proprio = (
            self.policy._split_actor_obs(observations)
        )

        proprio_embed = self.policy._embed_proprio_history(proprio_history)
        depth_embed = self.policy._embed_depth_history(depth_flat)
        depth_context = self.policy._align_depth_to_proprio_history(depth_embed)

        op_features = self.policy._fuse_op_features(proprio_embed)
        vp_features = self.policy._fuse_vp_features(
            proprio_embed,
            depth_embed,
            depth_context,
        )
        _, op_hidden = self.policy.op_gru(op_features)
        _, vp_hidden = self.policy.vp_gru(vp_features)
        op_latent = op_hidden[-1]
        vp_latent = vp_hidden[-1]

        recovery_embed = self.policy._embed_recovery_proprio_history(proprio_history)
        recovery_features = self.policy._fuse_recovery_features(recovery_embed)
        _, recovery_hidden = self.policy.recovery_gru(recovery_features)
        recovery_latent = recovery_hidden[-1]

        is_op = actor_mode == 1.0
        is_recovery = actor_mode == 2.0
        is_vp = actor_mode == 0.0
        zeros = torch.zeros_like(op_latent)
        first_latent_slot = torch.where(
            is_recovery,
            recovery_latent,
            torch.where(is_op, op_latent, zeros),
        )
        second_latent_slot = torch.where(is_vp, vp_latent, zeros)
        actor_input = torch.cat(
            [
                current_proprio,
                first_latent_slot,
                second_latent_slot,
                is_op.to(current_proprio.dtype),
                is_recovery.to(current_proprio.dtype),
                beta_obs,
            ],
            dim=-1,
        )
        return self.policy.actor(actor_input)

    @torch.jit.export
    def reset(self):
        pass


def _is_renet_policy(policy):
    return all(
        hasattr(policy, name)
        for name in (
            "actor_control_dim",
            "op_gru",
            "vp_gru",
            "recovery_gru",
            "recovery_encoder",
        )
    )


def _make_policy_exporter(policy, normalizer):
    if _is_renet_policy(policy):
        return RENetPolicyExporter(policy, normalizer)
    return FullPolicyExporter(policy, normalizer)


def _validate_renet_torchscript(policy, normalizer, traced_model, obs_dim):
    """Check that the exported graph matches eager RENet in all three modes."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    sample = 0.05 * torch.randn(3, obs_dim, generator=generator)
    control_start = policy.proprio_actor_dim + policy.depth_flat_dim
    sample[:, control_start] = torch.tensor([0.0, 1.0, 2.0])
    if policy.actor_control_dim >= 2:
        sample[:, control_start + 1] = 0.25

    reference = FullPolicyExporter(policy, normalizer).to("cpu").eval()
    with torch.inference_mode():
        expected = reference(sample)
        actual = traced_model(sample)
    if not torch.allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5):
        max_error = (actual - expected).abs().max().item()
        raise RuntimeError(
            "RENet TorchScript validation failed for VP/OP/Recovery; "
            f"maximum absolute error: {max_error:.6g}."
        )
    print("[SUCCESS] TorchScript VP/OP/Recovery routing validation passed.")


def _get_policy_obs_dim(policy):
    if hasattr(policy, "num_actor_obs"):
        return int(policy.num_actor_obs)
    return int(
        getattr(policy, "proprio_actor_dim", 0)
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


def export_policy_as_jit(policy, normalizer, path, filename="policy.pt"):
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
    )

    if _is_renet_policy(policy):
        _validate_renet_torchscript(policy, normalizer, traced_model, obs_dim)

    save_path = os.path.join(path, filename)
    traced_model.save(save_path)

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

    # Compute the raw actor-observation width. RENet includes actor_mode and
    # recovery beta in its control suffix.
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
