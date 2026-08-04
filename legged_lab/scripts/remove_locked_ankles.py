import argparse
import json
from pathlib import Path

import numpy as np


ANKLE_JOINT_INDICES = {4, 5, 10, 11}

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

    if frames.ndim != 2 or frames.shape[1] != 58:
        raise ValueError(
            f"{input_path} 应为 (T, 58)，"
            f"实际为 {frames.shape}"
        )

    if not np.isfinite(frames).all():
        raise ValueError(
            f"{input_path} 含有 NaN 或 Inf"
        )

    reduced_frames = frames[:, KEEP_FRAME_INDICES]

    if reduced_frames.shape[1] != 50:
        raise RuntimeError(
            f"输出维度错误：{reduced_frames.shape}"
        )

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