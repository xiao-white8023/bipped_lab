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

def export_policy_as_jit(policy, normalizer, path, filename="policy.pt"):
    """
    替代官方函数，由 play.py 直接调用
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # 使用自定义包装器
    exporter = FullPolicyExporter(policy, normalizer)
    exporter.to("cpu") # 强制转 CPU 以兼容所有部署环境
    exporter.eval()
    
    # 导出为 TorchScript
    scripted_model = torch.jit.script(exporter)
    save_path = os.path.join(path, filename)
    scripted_model.save(save_path)
    print(f"\n[SUCCESS] 带完整视觉管线的模型已导出至: {save_path}")

def export_policy_as_onnx(policy, normalizer, path, filename="policy.onnx"):
    """
    同理更新 ONNX 导出逻辑
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    exporter = FullPolicyExporter(policy, normalizer)
    exporter.to("cpu")
    exporter.eval()

    # 动态计算输入总维度: 1020 (本体) + 4608 (视觉)
    # 注意：这里使用的属性名必须和你 ActorCritic 类中定义的一致
    obs_dim = policy.proprio_actor_dim + policy.depth_flat_dim
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