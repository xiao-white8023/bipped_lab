from __future__ import annotations

import math

import torch
from torch import nn

from rsl_rl.utils import get_param, resolve_nn_activation


class FiLM(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        num_channels: int,
        hidden_dims: list[int] | int | None = None,
        activation: str = "elu",
    ):
        super().__init__()
        self.num_channels = num_channels

        if hidden_dims is None:
            hidden_dims = [condition_dim, condition_dim]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        act_fn = resolve_nn_activation(activation)
        layers = []
        last_dim = condition_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_fn)
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 2 * num_channels)) # 2 * num_channels 不是随便写的，而是为了给每个 CNN 通道分别生成一对 FiLM 参数：缩放系数 gamma 和偏移量 beta
        self.net = nn.Sequential(*layers)

        # Start from identity modulation: gamma = 1, beta = 0.
        final_layer = self.net[-1]  # 取 self.net 这个 Sequential 网络里的最后一层。
        '''
        就是把这个最后的 Linear 层的权重和偏置都初始化成 0。这样 FiLM 网络一开始输出的是 0。
        '''
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        gamma = 1.0 + gamma
        return gamma[:, :, None, None], beta[:, :, None, None]


class FiLMCnnMlp(nn.Module):
    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: list[int],
        kernel_size: list[int] | int,
        film_condition_dim: int,
        stride: int | tuple[int, ...] | list[int] = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str] | list[str] = "none",
        activation: str = "relu",
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
        mlp_hidden_dim: list[int] | None = None,
        mlp_output_dim: int = 128,
        mlp_activation: str = "elu",
        film_hidden_dims: list[int] | int | None = [128],
        film_activation: str = "elu",
        **kwargs,
    ):
        super().__init__()

        if not flatten:
            raise ValueError("FiLMCnnMlp requires flatten=True before the MLP head.")

        nn_activation = resolve_nn_activation(activation)
        self.conv_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        last_channels = input_channels
        last_dim = input_dim

        for idx in range(len(output_channels)):
            k = get_param(kernel_size, idx)
            s = get_param(stride, idx)
            d = get_param(dilation, idx)
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ["zeros", "reflect", "replicate", "circular"]
                else (0, 0)
            )

            block_layers = [
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
                )
            ]

            block_output_dim = _compute_output_dim(last_dim, k, s, d, p)

            n = get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                block_layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == "layer":
                block_layers.append(nn.LayerNorm([output_channels[idx], block_output_dim[0], block_output_dim[1]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )

            self.conv_blocks.append(
                nn.ModuleDict(
                    {
                        "pre_film": nn.Sequential(*block_layers),
                        "activation": nn_activation,
                        "pool": nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                        if get_param(max_pool, idx)
                        else nn.Identity(),
                    }
                )
            )
            self.film_layers.append(
                FiLM(
                    film_condition_dim,
                    output_channels[idx],
                    hidden_dims=film_hidden_dims,
                    activation=film_activation,
                )
            )

            last_channels = output_channels[idx]
            last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))

        post_conv_layers = []
        if global_pool == "none":
            pass
        elif global_pool == "max":
            post_conv_layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            last_dim = (1, 1)
        elif global_pool == "avg":
            post_conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            last_dim = (1, 1)
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )

        self.post_conv = nn.Sequential(*post_conv_layers)

        mlp_layers = [nn.Flatten(start_dim=1)]
        cnn_output_dim = last_channels * last_dim[0] * last_dim[1]

        if mlp_hidden_dim is None:
            mlp_hidden_dim = []
        elif isinstance(mlp_hidden_dim, int):
            mlp_hidden_dim = [mlp_hidden_dim]
        elif len(mlp_hidden_dim) == 0:
            raise Exception("mlp_hidden_dim是空的，没有mlp的隐藏层的维度")

        mlp_nn_activation = resolve_nn_activation(mlp_activation)
        if len(mlp_hidden_dim) > 0:
            mlp_layers.append(nn.Linear(cnn_output_dim, mlp_hidden_dim[0]))
            mlp_layers.append(mlp_nn_activation)
            for idx in range(len(mlp_hidden_dim)):
                if idx == len(mlp_hidden_dim) - 1:
                    mlp_layers.append(nn.Linear(mlp_hidden_dim[idx], mlp_output_dim))
                else:
                    mlp_layers.append(nn.Linear(mlp_hidden_dim[idx], mlp_hidden_dim[idx + 1]))
                    mlp_layers.append(mlp_nn_activation)
        else:
            mlp_layers.append(nn.Linear(cnn_output_dim, mlp_output_dim))

        self.mlp = nn.Sequential(*mlp_layers)
        self._output_channels = None
        self._output_dim = mlp_output_dim

    @property
    def output_channels(self) -> int | None:
        return self._output_channels

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for block, film in zip(self.conv_blocks, self.film_layers):
            x = block["pre_film"](x)
            gamma, beta = film(condition)
            x = gamma * x + beta
            x = block["activation"](x)
            x = block["pool"](x)
        x = self.post_conv(x)
        return self.mlp(x)


def _compute_padding(input_hw: tuple[int, int], kernel: int, stride: int, dilation: int) -> tuple[int, int]:
    h = math.ceil((stride * math.floor(input_hw[0] / stride) - input_hw[0] - stride + dilation * (kernel - 1) + 1) / 2)
    w = math.ceil((stride * math.floor(input_hw[1] / stride) - input_hw[1] - stride + dilation * (kernel - 1) + 1) / 2)
    return (h, w)


def _compute_output_dim(
    input_hw: tuple[int, int],
    kernel: int,
    stride: int,
    dilation: int,
    padding: tuple[int, int],
    is_max_pool: bool = False,
) -> tuple[int, int]:
    h = math.floor((input_hw[0] + 2 * padding[0] - dilation * (kernel - 1) - 1) / stride + 1)
    w = math.floor((input_hw[1] + 2 * padding[1] - dilation * (kernel - 1) - 1) / stride + 1)

    if is_max_pool:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)
    return (h, w)
