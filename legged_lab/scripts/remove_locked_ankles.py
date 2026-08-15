import argparse
import json
from pathlib import Path

import numpy as np


ANKLE_JOINT_INDICES = {4, 5, 10, 11}
BASE_FULL_FRAME_SIZE = 58
BASE_REDUCED_FRAME_SIZE = 50
RECOVERY_FULL_FRAME_SIZE = 61
RECOVERY_REDUCED_FRAME_SIZE = 53

KEEP_JOINT_INDICES = [
    index
    for index in range(23)
    if index not in ANKLE_JOINT_INDICES
]

KEEP_FRAME_INDICES = (
    KEEP_JOINT_INDICES
    + [23 + index for index in KEEP_JOINT_INDICES]
    + list(range(46, 58))
)

assert len(KEEP_JOINT_INDICES) == 19
assert len(KEEP_FRAME_INDICES) == 50


def convert_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as file:
        motion = json.load(file)

    frames = np.asarray(
        motion["Frames"],
        dtype=np.float32,
    )

    if frames.ndim != 2 or frames.shape[1] not in (
        BASE_FULL_FRAME_SIZE,
        RECOVERY_FULL_FRAME_SIZE,
    ):
        raise ValueError(
            f"{input_path} 应为 (T, 58) 或 (T, 61)，"
            f"实际为 {frames.shape}"
        )

    if not np.isfinite(frames).all():
        raise ValueError(
            f"{input_path} 含有 NaN 或 Inf"
        )

    reduced_base = frames[:, KEEP_FRAME_INDICES]
    extras = frames[:, BASE_FULL_FRAME_SIZE:]
    reduced_frames = np.concatenate((reduced_base, extras), axis=1)

    expected_output_dim = (
        BASE_REDUCED_FRAME_SIZE
        if frames.shape[1] == BASE_FULL_FRAME_SIZE
        else RECOVERY_REDUCED_FRAME_SIZE
    )
    if reduced_frames.shape[1] != expected_output_dim:
        raise RuntimeError(
            f"输出维度错误：{reduced_frames.shape}"
        )

    if frames.shape[1] == RECOVERY_FULL_FRAME_SIZE:
        projected_gravity = frames[:, BASE_FULL_FRAME_SIZE:RECOVERY_FULL_FRAME_SIZE]
        gravity_norms = np.linalg.norm(projected_gravity, axis=1)
        if not np.all(np.abs(gravity_norms - 1.0) < 1e-3):
            bad_indices = np.flatnonzero(np.abs(gravity_norms - 1.0) >= 1e-3)[:10]
            raise ValueError(
                f"{input_path} 的 projected_gravity_b 必须为单位向量；"
                f"异常帧索引: {bad_indices.tolist()}"
            )
        if not np.array_equal(
            reduced_frames[:, BASE_REDUCED_FRAME_SIZE:RECOVERY_REDUCED_FRAME_SIZE],
            projected_gravity,
        ):
            raise RuntimeError("Recovery projected_gravity_b 在降维过程中发生了变化。")

    motion["Frames"] = reduced_frames.tolist()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            motion,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    print(
        f"{input_path.name}: "
        f"{frames.shape} -> {reduced_frames.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    motion_files = sorted(input_dir.glob("*.txt"))

    if not motion_files:
        raise ValueError(
            f"没有在 {input_dir} 找到 txt 文件"
        )

    for input_path in motion_files:
        convert_file(
            input_path,
            output_dir / input_path.name,
        )


if __name__ == "__main__":
    main()
