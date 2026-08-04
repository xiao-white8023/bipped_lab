from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _to_numpy(value: Any, name: str) -> np.ndarray:
    """Convert NumPy arrays, Torch tensors, or array-like objects to NumPy."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)

    if array.dtype == object:
        raise TypeError(f"{name} 的 dtype=object，无法作为数值数组处理。")

    return array.astype(np.float64, copy=False)


def _read_scalar(value: Any, name: str) -> float:
    """Read a scalar stored as Python number, NumPy scalar, or one-element array."""
    array = np.asarray(value)

    if array.size != 1:
        raise ValueError(f"{name} 应当是标量，实际 shape={array.shape}")

    return float(array.reshape(-1)[0])


def _resolve_motion_dict(
    raw_data: Any,
    motion_key: str | None,
) -> tuple[str, dict[str, Any], str]:
    """
    Support two PKL layouts.

    1. Legacy flat layout:
       {
           "root_pos": ...,
           "root_rot": ...,
           "dof_pos": ...,
           "fps": ...
       }

    2. Hugging Face nested layout:
       {
           "motion_name.npz": {
               "root_trans_offset": ...,
               "root_rot": ...,
               "dof": ...,
               "fps": ...
           }
       }
    """
    if not isinstance(raw_data, dict):
        raise TypeError(
            "PKL 顶层必须是 dict，"
            f"实际类型为 {type(raw_data).__name__}"
        )

    legacy_required = {"root_pos", "root_rot", "dof_pos"}
    hf_required = {"root_trans_offset", "root_rot", "dof"}

    if legacy_required.issubset(raw_data.keys()):
        return "legacy_flat_motion", raw_data, "legacy"

    candidates: list[tuple[str, dict[str, Any]]] = []

    for key, value in raw_data.items():
        if isinstance(value, dict) and hf_required.issubset(value.keys()):
            candidates.append((str(key), value))

    if motion_key is not None:
        if motion_key not in raw_data:
            available = list(raw_data.keys())
            raise KeyError(
                f"找不到 --motion_key={motion_key!r}。\n"
                f"可用顶层 keys: {available}"
            )

        selected = raw_data[motion_key]

        if not isinstance(selected, dict):
            raise TypeError(
                f"motion_key={motion_key!r} 对应的值不是 dict。"
            )

        if not hf_required.issubset(selected.keys()):
            raise KeyError(
                f"motion_key={motion_key!r} 缺少字段。\n"
                f"当前字段: {list(selected.keys())}\n"
                f"需要字段: {sorted(hf_required)}"
            )

        return motion_key, selected, "huggingface_nested"

    if len(candidates) == 0:
        raise KeyError(
            "没有找到支持的动作字段。\n"
            f"顶层 keys: {list(raw_data.keys())}\n"
            "支持的旧格式字段: root_pos/root_rot/dof_pos\n"
            "支持的嵌套格式字段: root_trans_offset/root_rot/dof"
        )

    if len(candidates) > 1:
        names = [name for name, _ in candidates]
        raise ValueError(
            "PKL 中包含多段动作，请通过 --motion_key 指定一段。\n"
            f"可选动作: {names}"
        )

    return candidates[0][0], candidates[0][1], "huggingface_nested"


def _detect_quaternion_order(root_rot_raw: np.ndarray) -> str:
    """
    Heuristic detection for typical walking motions.

    Upright humanoid root orientations usually have a relatively large scalar
    quaternion component. Compare the first and last components and choose the
    more likely scalar position.

    Use --quat_order explicitly if the printed result is wrong.
    """
    first_score = float(np.median(np.abs(root_rot_raw[:, 0])))
    last_score = float(np.median(np.abs(root_rot_raw[:, 3])))

    if last_score >= first_score:
        return "xyzw"

    return "wxyz"


def _convert_quaternion_to_xyzw(
    root_rot_raw: np.ndarray,
    quat_order: str,
) -> tuple[np.ndarray, str]:
    if root_rot_raw.ndim != 2 or root_rot_raw.shape[1] != 4:
        raise ValueError(
            "root_rot 应为 (T, 4)，"
            f"实际 shape={root_rot_raw.shape}"
        )

    resolved_order = quat_order

    if quat_order == "auto":
        resolved_order = _detect_quaternion_order(root_rot_raw)

    if resolved_order == "xyzw":
        root_rot_xyzw = root_rot_raw.copy()
    elif resolved_order == "wxyz":
        root_rot_xyzw = root_rot_raw[:, [1, 2, 3, 0]].copy()
    else:
        raise ValueError(f"不支持的四元数顺序: {resolved_order}")

    norms = np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True)

    if np.any(norms < 1.0e-8):
        bad_ids = np.nonzero(norms[:, 0] < 1.0e-8)[0]
        raise ValueError(
            "root_rot 中存在零长度四元数，"
            f"异常帧索引示例: {bad_ids[:10].tolist()}"
        )

    root_rot_xyzw /= norms

    # q and -q represent the same rotation, but sign jumps harm interpolation.
    for frame_id in range(1, root_rot_xyzw.shape[0]):
        if np.dot(
            root_rot_xyzw[frame_id - 1],
            root_rot_xyzw[frame_id],
        ) < 0.0:
            root_rot_xyzw[frame_id] *= -1.0

    return root_rot_xyzw, resolved_order


def _validate_motion(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    expected_dofs: int,
) -> None:
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            "root position 应为 (T, 3)，"
            f"实际 shape={root_pos.shape}"
        )

    if dof_pos.ndim != 2:
        raise ValueError(
            "dof position 应为二维数组，"
            f"实际 shape={dof_pos.shape}"
        )

    if expected_dofs > 0 and dof_pos.shape[1] != expected_dofs:
        raise ValueError(
            f"期望 {expected_dofs} DoF，"
            f"但 PKL 中为 {dof_pos.shape[1]} DoF。"
        )

    frame_counts = (
        root_pos.shape[0],
        root_rot_xyzw.shape[0],
        dof_pos.shape[0],
    )

    if len(set(frame_counts)) != 1:
        raise ValueError(
            "root_pos、root_rot、dof_pos 帧数不一致："
            f"{frame_counts}"
        )

    if root_pos.shape[0] < 2:
        raise ValueError(
            f"动作至少需要两帧，实际只有 {root_pos.shape[0]} 帧。"
        )

    for name, array in (
        ("root_pos", root_pos),
        ("root_rot", root_rot_xyzw),
        ("dof_pos", dof_pos),
    ):
        if not np.all(np.isfinite(array)):
            bad_count = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
            raise ValueError(
                f"{name} 中存在 {bad_count} 个 NaN 或 Inf。"
            )


def _resample_motion(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_fps <= 0.0:
        raise ValueError(f"非法 source_fps={source_fps}")

    if target_fps <= 0.0:
        raise ValueError(f"非法 target_fps={target_fps}")

    if np.isclose(source_fps, target_fps, rtol=0.0, atol=1.0e-8):
        return root_pos, root_rot_xyzw, dof_pos

    num_source_frames = root_pos.shape[0]
    duration = (num_source_frames - 1) / source_fps

    source_times = np.arange(num_source_frames, dtype=np.float64) / source_fps
    num_target_frames = int(np.floor(duration * target_fps + 1.0e-9)) + 1
    target_times = np.arange(num_target_frames, dtype=np.float64) / target_fps

    # Numerical guard: never query past the final source frame.
    target_times = np.minimum(target_times, source_times[-1])

    root_pos_resampled = np.column_stack(
        [
            np.interp(target_times, source_times, root_pos[:, axis])
            for axis in range(3)
        ]
    )

    dof_pos_resampled = np.column_stack(
        [
            np.interp(target_times, source_times, dof_pos[:, joint_id])
            for joint_id in range(dof_pos.shape[1])
        ]
    )

    source_rotations = Rotation.from_quat(root_rot_xyzw)
    slerp = Slerp(source_times, source_rotations)
    root_rot_resampled = slerp(target_times).as_quat()

    return (
        root_pos_resampled,
        root_rot_resampled,
        dof_pos_resampled,
    )


def _compute_motion_features(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    Output one transition row per pair of adjacent frames:

    root_pos(3)
    root_euler_XYZ(3)
    dof_pos(D)
    root_lin_vel(3)
    root_ang_vel(3)
    dof_vel(D)

    For D=23, the output dimension is 58.
    """
    dt = 1.0 / fps

    root_lin_vel = np.diff(root_pos, axis=0) / dt
    dof_vel = np.diff(dof_pos, axis=0) / dt

    rotations = Rotation.from_quat(root_rot_xyzw)

    relative_rotations = rotations[:-1].inv() * rotations[1:]
    root_ang_vel = relative_rotations.as_rotvec() / dt

    euler_angles = rotations[:-1].as_euler(
        "XYZ",
        degrees=False,
    )
    euler_angles = np.unwrap(euler_angles, axis=0)

    output = np.concatenate(
        (
            root_pos[:-1],
            euler_angles,
            dof_pos[:-1],
            root_lin_vel,
            root_ang_vel,
            dof_vel,
        ),
        axis=1,
    )

    return output.astype(np.float32)


def _write_motion_file(
    output_txt: Path,
    frames: np.ndarray,
    fps: float,
    loop_mode: str,
    motion_weight: float,
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    wrap = loop_mode == "Wrap"

    with output_txt.open("w", encoding="utf-8") as file:
        file.write("{\n")
        file.write(f'"LoopMode": "{loop_mode}",\n')
        file.write(f'"FrameDuration": {1.0 / fps:.9f},\n')
        file.write(
            '"EnableCycleOffsetPosition": '
            f'{"true" if wrap else "false"},\n'
        )
        file.write(
            '"EnableCycleOffsetRotation": '
            f'{"true" if wrap else "false"},\n'
        )
        file.write(f'"MotionWeight": {motion_weight:.9g},\n\n')
        file.write('"Frames":\n')
        file.write("[\n")

        for frame_id, frame in enumerate(frames):
            values = ", ".join(f"{float(value):.8f}" for value in frame)
            suffix = "," if frame_id < len(frames) - 1 else ""
            file.write(f"  [{values}]{suffix}\n")

        file.write("]\n")
        file.write("}\n")


def convert_pkl_to_custom(
    input_pkl: str | Path,
    output_txt: str | Path,
    target_fps: float | None,
    expected_dofs: int,
    quat_order: str,
    motion_key: str | None,
    loop_mode: str,
    motion_weight: float,
) -> None:
    input_pkl = Path(input_pkl).expanduser().resolve()
    output_txt = Path(output_txt).expanduser().resolve()

    if not input_pkl.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_pkl}")

    with input_pkl.open("rb") as file:
        raw_data = pickle.load(file)

    motion_name, motion_data, data_format = _resolve_motion_dict(
        raw_data,
        motion_key,
    )

    if data_format == "legacy":
        root_pos = _to_numpy(
            motion_data["root_pos"],
            "root_pos",
        )
        root_rot_raw = _to_numpy(
            motion_data["root_rot"],
            "root_rot",
        )
        dof_pos = _to_numpy(
            motion_data["dof_pos"],
            "dof_pos",
        )
    else:
        root_pos = _to_numpy(
            motion_data["root_trans_offset"],
            "root_trans_offset",
        )
        root_rot_raw = _to_numpy(
            motion_data["root_rot"],
            "root_rot",
        )
        dof_pos = _to_numpy(
            motion_data["dof"],
            "dof",
        )

    root_rot_xyzw, resolved_quat_order = _convert_quaternion_to_xyzw(
        root_rot_raw,
        quat_order,
    )

    _validate_motion(
        root_pos,
        root_rot_xyzw,
        dof_pos,
        expected_dofs,
    )

    source_fps_value = motion_data.get("fps")

    if source_fps_value is None:
        source_fps_value = motion_data.get("mocap_framerate")

    if source_fps_value is None:
        source_fps_value = motion_data.get("mocap_frame_rate")

    if source_fps_value is None:
        if target_fps is None:
            raise ValueError(
                "PKL 中没有 fps，请通过 --fps 指定输出帧率。"
            )

        source_fps = float(target_fps)
        print(
            "警告: PKL 中没有源 fps，"
            f"将假定源帧率也是 {source_fps:.6g} FPS，"
            "不会执行时间重采样。"
        )
    else:
        source_fps = _read_scalar(
            source_fps_value,
            "fps",
        )

    if target_fps is None:
        target_fps = source_fps

    root_pos, root_rot_xyzw, dof_pos = _resample_motion(
        root_pos,
        root_rot_xyzw,
        dof_pos,
        source_fps,
        float(target_fps),
    )

    frames = _compute_motion_features(
        root_pos,
        root_rot_xyzw,
        dof_pos,
        float(target_fps),
    )

    expected_output_dim = 12 + 2 * dof_pos.shape[1]

    if frames.shape[1] != expected_output_dim:
        raise RuntimeError(
            "输出维度计算异常："
            f"实际 {frames.shape[1]}，"
            f"期望 {expected_output_dim}"
        )

    _write_motion_file(
        output_txt,
        frames,
        float(target_fps),
        loop_mode,
        motion_weight,
    )

    print("=" * 72)
    print("转换完成")
    print(f"输入文件       : {input_pkl}")
    print(f"动作名称       : {motion_name}")
    print(f"PKL 格式       : {data_format}")
    print(f"PKL 字段       : {list(motion_data.keys())}")
    print(f"四元数输入顺序 : {resolved_quat_order}")
    print(f"源 FPS         : {source_fps:.6g}")
    print(f"输出 FPS       : {float(target_fps):.6g}")
    print(f"原始/重采样帧数: {root_pos.shape[0]}")
    print(f"DoF 数量       : {dof_pos.shape[1]}")
    print(f"输出帧数       : {frames.shape[0]}")
    print(f"每帧维度       : {frames.shape[1]}")
    print(f"LoopMode       : {loop_mode}")
    print(f"输出文件       : {output_txt}")
    print("=" * 72)

    if quat_order == "auto":
        print(
            "提示: 四元数顺序由脚本自动推断。"
            "如果回放时机器人根姿态明显错误，"
            "请分别尝试 --quat_order xyzw 或 --quat_order wxyz。"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将 GMR/Hugging Face 的 G1 动作 PKL "
            "转换为 TienKung-Lab 自定义 motion txt。"
        )
    )

    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="输入 PKL 文件。",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        required=True,
        help="输出 motion txt 文件。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "目标输出 FPS。默认使用 PKL 中的 fps；"
            "与源 FPS 不同时会执行重采样。"
        ),
    )
    parser.add_argument(
        "--expected_dofs",
        type=int,
        default=23,
        help=(
            "期望的机器人 DoF 数量。默认 23；"
            "设置为 0 可关闭检查。"
        ),
    )
    parser.add_argument(
        "--quat_order",
        choices=("auto", "xyzw", "wxyz"),
        default="auto",
        help=(
            "PKL 中 root_rot 的四元数顺序。"
            "默认 auto；姿态异常时请显式指定。"
        ),
    )
    parser.add_argument(
        "--motion_key",
        type=str,
        default=None,
        help=(
            "当一个 PKL 包含多段嵌套动作时，"
            "指定要转换的顶层 key。"
        ),
    )
    parser.add_argument(
        "--loop_mode",
        choices=("Clamp", "Wrap"),
        default="Clamp",
        help=(
            "动作边界模式。站立到行走、侧步、转身等有限片段"
            "建议使用 Clamp；完整无缝周期才使用 Wrap。"
        ),
    )
    parser.add_argument(
        "--motion_weight",
        type=float,
        default=0.5,
        help="写入动作文件的 MotionWeight，默认 0.5。",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0.0:
        parser.error("--fps 必须大于 0。")

    if args.expected_dofs < 0:
        parser.error("--expected_dofs 不能小于 0。")

    if args.motion_weight <= 0.0:
        parser.error("--motion_weight 必须大于 0。")

    convert_pkl_to_custom(
        input_pkl=args.input_pkl,
        output_txt=args.output_txt,
        target_fps=args.fps,
        expected_dofs=args.expected_dofs,
        quat_order=args.quat_order,
        motion_key=args.motion_key,
        loop_mode=args.loop_mode,
        motion_weight=args.motion_weight,
    )


if __name__ == "__main__":
    main()
