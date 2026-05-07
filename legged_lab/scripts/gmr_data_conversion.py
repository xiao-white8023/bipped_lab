import pickle
import numpy as np
import torch
import argparse
from scipy.spatial.transform import Rotation 

# =====================================================================
# 🚀 纯 PyTorch 实现的四元数运算 (彻底摆脱 isaaclab 和 pxr 的依赖)
# =====================================================================
def quat_conjugate(q):
    """计算四元数的共轭"""
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)

def quat_mul(q1, q2):
    """计算两个四元数相乘"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def axis_angle_from_quat(q):
    """将四元数转换为轴角表示"""
    w = q[..., 0]
    vec = q[..., 1:]
    norm = torch.norm(vec, dim=-1, keepdim=True)
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0)).unsqueeze(-1)
    axis = vec / (norm + 1e-6)
    return axis * angle
# =====================================================================

def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    # 注意：标准 AMP 通常需要四元数，如果训练报错维度不对，把这里的 euler 换回四元数
    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)