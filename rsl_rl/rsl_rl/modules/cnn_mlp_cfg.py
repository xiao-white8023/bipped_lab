from isaaclab.utils import configclass
from dataclasses import MISSING

# @configclass 强制要求：每一个类属性，要么必须给一个默认值，要么必须显式地赋值为 MISSING。
@configclass
class CnnMlpCfg:
    
    input_dim:tuple[int,int] = MISSING                   # 图像的高和宽
    input_channels:int = MISSING                         # 图像的通道数
    output_channels:list[int] = MISSING                  # 每一层卷积输出的通道数列表
    kernel_size:list[int] | int= MISSING                 # 卷积核大小，可以是一个固定的整数（比如 3，代表所有层都是 3x3），也可以是列表指定每一层。
    stride: int | tuple[int, ...] | list[int] = 1        # 步长，决定了每次卷积图像缩小的比例
    dilation: int | tuple[int, ...] | list[int] = 1      # 膨胀系数（用于扩大感受野，一般填 1 即可） 
    padding: str = "none"                                # 边缘填充方式（防止图像越卷越小）
    norm: str | tuple[str] | list[str] = "none"          # 归一化方式（可以选不加 'none'，或者 'batch'、'layer'）
    activation: str = "elu"                              # 激活函数的名字（比如 'elu'、'relu'）
    max_pool: bool | tuple[bool] | list[bool] = False    # 是否在卷积后加最大池化层（进一步降维）
    global_pool: str = "none"                            # 设定在所有卷积层结束后，要不要加全局池化，把整个特征图直接压缩成 1×1 的大小。
    flatten: bool = False                                 # 设为 True 时，网络会在最后把 2D 的图像特征图（Height x Width x Channels）强行压扁成一个一维的特征向量。这样才能和机器人的 1D 本体数据（关节角度等）进行拼接。
    mlp_hidden_dim: list[int] | None =None
    mlp_output_dim:int=128
    mlp_activation:str="relu"
    num_heads:int = 16
    embed_dim:int = 64