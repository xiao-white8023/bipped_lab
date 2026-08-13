import os
import copy
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

    exporter = FullPolicyExporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()

    obs_dim = _get_policy_obs_dim(policy)
    dummy_input = torch.zeros(1, obs_dim)

    traced_model = torch.jit.trace(
        exporter,
        dummy_input,
        strict=False,
    )

    save_path = os.path.join(path, filename)
    traced_model.save(save_path)

    print(f"\n[SUCCESS] traced 视觉策略模型已导出至: {save_path}")


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
    """
    同理更新 ONNX 导出逻辑
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    exporter = FullPolicyExporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()

    # 动态计算输入总维度，RENet 还包含最后 1 维 estimator_mask。
    obs_dim = _get_policy_obs_dim(policy)
    dummy_input = torch.zeros(1, obs_dim)

    torch.onnx.export(
        exporter,
        dummy_input,
        os.path.join(path, filename),
        opset_version=17,
        input_names=['obs'],
        output_names=['actions']
    )
    print(f"[SUCCESS] ONNX 模型已导出至: {os.path.join(path, filename)}")
