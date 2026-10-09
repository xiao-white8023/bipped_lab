# 中文导读
# 文件用途：无需启动 Isaac Sim，将 model_60000 单独导出为冻结 RENet 双分支 TorchScript。
# 使用位置：用户单独准备双分支模型时运行；采集脚本与此脚本共用 artifacts.py 的导出流程。
# 说明：生成六个输出：LP/VP 特征、LP/VP 速度估计和 LP/VP 动作；在线运行与 selector 模型配合使用。
# 说明：会校验原部署 policy.pt，并拒绝覆盖输入模型或已有输出文件。
#
# 类、函数及方法说明（包含私有方法、内部函数和属性访问器）：
# - parse_args：解析 checkpoint、参考 policy.pt 和输出模型路径，不导入或启动 Isaac Sim。
# - main：严格加载冻结策略，导出并重新加载双分支模型，校验动作一致性和嵌入元数据。
# 中文导读结束

"""Export the frozen G1 RENet checkpoint as one dual-branch TorchScript model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
LOCAL_RSL_RL = REPO_ROOT / "rsl_rl"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_RSL_RL) not in sys.path:
    sys.path.insert(0, str(LOCAL_RSL_RL))

import torch  # noqa: E402
from legged_lab.selector.config import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH, DEFAULT_REFERENCE_POLICY_PATH, DEFAULT_RUN_DIR,
)
from legged_lab.selector.artifacts import export_dual_branch_artifact  # noqa: E402
from legged_lab.selector.frozen_policy import (  # noqa: E402
    load_frozen_adapter, sha256, validate_actions, validation_inputs,
)


def parse_args() -> argparse.Namespace:
    """Parse artifact paths without importing or starting Isaac Sim."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--reference-policy",
        type=Path,
        default=DEFAULT_REFERENCE_POLICY_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN_DIR / "exported/renet_dual_branch.pt",
    )
    return parser.parse_args()


def main() -> None:
    """Load, export, reload, and validate the frozen dual-branch artifact."""

    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    reference_path = args.reference_policy.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    for path in (checkpoint_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path in (checkpoint_path, reference_path):
        raise ValueError("dual-branch output must not overwrite a source artifact")
    if output_path.exists():
        raise FileExistsError(output_path)

    adapter, reference, frozen_metadata = load_frozen_adapter(checkpoint_path, reference_path)
    export_dual_branch_artifact(
        adapter, output_path, frozen_metadata, reference=reference
    )
    probes = validation_inputs()

    extra_files = {"renet_dual_branch_contract.json": ""}
    reloaded = torch.jit.load(
        str(output_path), map_location="cpu", _extra_files=extra_files
    ).eval()
    final_lp_error, final_vp_error = validate_actions(reloaded, reference, probes)
    print(f"[SUCCESS] exported: {output_path}")
    print(f"[SUCCESS] output SHA-256: {sha256(output_path)}")
    print(
        "[SUCCESS] reference parity: "
        f"LP max_abs={final_lp_error:.9g}, VP max_abs={final_vp_error:.9g}"
    )
    print(
        "[SUCCESS] embedded metadata bytes: "
        f"{len(extra_files['renet_dual_branch_contract.json'])}"
    )


if __name__ == "__main__":
    main()
