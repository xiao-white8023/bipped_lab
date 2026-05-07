from __future__ import annotations

import torch
from torch import nn as nn
import math
from rsl_rl.utils import get_param, resolve_nn_activation

class CnnMlp(nn.Sequential):
    def __init__(
                    self,
                    input_dim:tuple[int,int],  # 图像的高和宽
                    input_channels:int, # 图像的通道数
                    output_channels:list[int],# 每一层卷积输出的通道数列表
                    kernel_size:list[int] | int,# 卷积核大小，可以是一个固定的整数（比如 3，代表所有层都是 3x3），也可以是列表指定每一层。
                    stride: int | tuple[int, ...] | list[int] = 1, # 步长，决定了每次卷积图像缩小的比例
                    dilation: int | tuple[int, ...] | list[int] = 1, # 膨胀系数（用于扩大感受野，一般填 1 即可） 
                    padding: str = "none", # 边缘填充方式（防止图像越卷越小）
                    norm: str | tuple[str] | list[str] = "none", # 归一化方式（可以选不加 'none'，或者 'batch'、'layer'）
                    activation: str = "elu", # 激活函数的名字（比如 'elu'、'relu'）
                    max_pool: bool | tuple[bool] | list[bool] = False, # 是否在卷积后加最大池化层（进一步降维）
                    global_pool: str = "none",  # 设定在所有卷积层结束后，要不要加全局池化，把整个特征图直接压缩成 1×1 的大小。
                    flatten: bool = True, # 设为 True 时，网络会在最后把 2D 的图像特征图（Height x Width x Channels）强行压扁成一个一维的特征向量。这样才能和机器人的 1D 本体数据（关节角度等）进行拼接。
                    mlp_hidden_dim: list[int] | None =None,
                    mlp_output_dim:int=128,
                    mlp_activation:str="relu",
                    **kwargs
                ):
        super().__init__()
        
        #激活函数
        nn_activation=resolve_nn_activation(activation)
        
        layers = [] # 拿出一个空箱子，准备一会把造好的网络层一层层放进去。

        # 最开始的通道数是就是图片的通道数  输入的维度就是图片的维度
        last_channels = input_channels 
        last_dim = input_dim

        for idx in range(len(output_channels)):
            '''
            get_param是一个工具函数
            如果 kernel_size 是单个整数（比如 3）：直接返回这个数，所有层都用 3×3；
            如果 kernel_size 是列表（比如 [3, 5, 3]）：返回列表里第 idx 个值，第 1 层用 3，第 2 层用 5，第 3 层用 3。
            '''
            k = get_param(kernel_size, idx)
            '''
            大白话解释：挑出当前层卷积核的滑动步长。代码逻辑作用：和 k 完全一样，智能选择步长：
            单个值（比如 1）：所有层步长都是 1；
            列表（比如 [2, 1, 1]）：第 1 层步长 2（快速缩小），后面两层步长 1。
            '''
            s = get_param(stride, idx)
            '''
            挑出当前层的膨胀系数。
            '''
            d = get_param(dilation, idx)

            '''
            计算当前层的 Padding（给图片边缘补边）
            如果参数是"zeros", "reflect", "replicate", "circular" 就进入到函数中进行计算补多少的像素

            否则 不补充像素
            '''
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ["zeros", "reflect", "replicate", "circular"]
                else (0, 0)
            )

            layers.append(nn.Conv2d(
                  in_channels=last_channels,
                  out_channels=output_channels[idx],
                  kernel_size=k,
                  stride=s,
                  padding=p,
                  dilation=d,
                  padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
            ))
            # Append normalization layer if specified
            n = get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == "layer":
                norm_input_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0], norm_input_dim[1]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )
            layers.append(nn_activation)

            # Apply max pooling if specified
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            last_channels = output_channels[idx]
            last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))
        
        # Apply global pooling if specified
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            last_dim = (1, 1)
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            last_dim = (1, 1)
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )
        
        # Apply flattening if specified
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        # Store final output dimension
        self._output_channels = last_channels if not flatten else None  # 如果展平了 就没有通道数了
        self._output_dim = last_dim if not flatten else last_channels * last_dim[0] * last_dim[1] # 通道数*宽度*高度
        
        # 构建mlp
        if not flatten and mlp_hidden_dim is not None:
            raise Exception("卷积神经网络的输出不是1D的向量，接下来的mlp无法接收输入。必须开启 flatten=True 才能在后面拼接 MLP 层")
        
        if mlp_hidden_dim is None:
            mlp_hidden_dim=[]
        elif len(mlp_hidden_dim)==0:
            raise Exception("mlp_hidden_dim是空的，没有mlp的隐藏层的维度")
        elif isinstance(mlp_hidden_dim, int):
            mlp_hidden_dim=[mlp_hidden_dim]
        
        mlp_nn_activation=resolve_nn_activation(mlp_activation)
        layers.append(nn.Linear(self._output_dim,mlp_hidden_dim[0]))
        layers.append(mlp_nn_activation)
        for idx in range(len(mlp_hidden_dim)):
            if idx==len(mlp_hidden_dim)-1:
                layers.append(nn.Linear(mlp_hidden_dim[idx],mlp_output_dim))
            else:
                layers.append(nn.Linear(mlp_hidden_dim[idx],mlp_hidden_dim[idx+1]))
                layers.append(mlp_nn_activation)

        self._output_dim = mlp_output_dim  # 最终的维度

        # Register the layers   # 把上面装进 layers 列表里的所有网络层，正式注册到当前的 nn.Sequential 模块中
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels or None if output is flattened."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """得到mlp最终输出的维度"""
        # 提供一个对外接口，获取网络最终输出的特征维度
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize the weights of the CNN with Kaiming initialization."""
        # # 初始化网络权重，能让网络训练得更快、更稳
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN."""
        for layer in self:
            x = layer(x)
        return x

def _compute_padding(input_hw: tuple[int, int], kernel: int, stride: int, dilation: int) -> tuple[int, int]:
    """Compute the optimal padding for the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
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
    """
    Compute the output height and width of the current layer.
    """
    h = math.floor((input_hw[0] + 2 * padding[0] - dilation * (kernel - 1) - 1) / stride + 1)
    w = math.floor((input_hw[1] + 2 * padding[1] - dilation * (kernel - 1) - 1) / stride + 1)

    if is_max_pool:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)
    return (h, w)