import torch
from collections.abc import Sequence
from typing import Union

from isaaclab.utils.buffers import CircularBuffer

# max_len：缓冲区最大时间步长度（最多存多少步历史数据）。
# batch_size 环境数目
class AsyncCircularBuffer(CircularBuffer):
    def __init__(self, max_len: int, batch_size: int, device: str):
        super().__init__(max_len, batch_size, device)
    
    #返回所有 batch 的、整理好顺序的完整缓冲区数据（“整理顺序” 指：最旧的数据在时间步开头，最新的数据在结尾）
    '''
    这几行就是因为是异步的 允许不同时更新，所以有的更新了有的没有更新，为了不读到没有更新时的全0数据，所以加了这三行，而isaaclab中是所有环境一起更新的，就不存在有的更新了，有的没有更新了、
    '''
    @property
    def buffer(self) -> torch.Tensor:
        if any(self._num_pushes == 0):
            raise RuntimeError("Attempting to access a buffer that is not fully initialized.")
        return self.get_by_batch_ids()

    # 把 “循环指针乱序” 的原始缓冲区，重新排列成 “最旧数据在开头、最新数据在结尾” 的整齐顺序。
    def get_by_batch_ids(self, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        # Index seems too large, potentially needing speed optimization. But we may wait and see.
        batch_ids = self._ALL_INDICES if batch_ids is None else torch.as_tensor(batch_ids, device=self._device)
        shifts = self.max_length - self._pointer - 1
        selected_shifts = shifts if batch_ids is None else shifts[batch_ids]
        selected_buf = self._buffer.clone() if batch_ids is None else self._buffer[:, batch_ids, ...].clone()
        selected_batch_size = self._batch_size if batch_ids is None else batch_ids.size(0)
        T = self.max_length
        arange = torch.arange(T, device=self._device)  # (T,)
        index = ((arange[:, None] - selected_shifts[None, :]) % T).long()  # (T, 1) - (1, selected_B) -> (T, selected_B)
        extra_shape = selected_buf.shape[2:]  # (*D)
        index = index.view(T, selected_batch_size, *([1] * len(extra_shape)))  # (T, selected_B, 1....)
        index = index.expand(T, selected_batch_size, *extra_shape)  # (T, selected_B, *D)
        buf = torch.gather(selected_buf, dim=0, index=index)
        return torch.transpose(buf, dim0=0, dim1=1)

    def append(self, data: torch.Tensor, batch_ids: Sequence[int] | None = None):
        if batch_ids is None:
            return super().append(data)
        else:
            # 数据的个数和batch的长度要相等
            if data.shape[0] != len(batch_ids):
                raise ValueError(f"Data shape {data.shape[0]} does not match batch_ids length {len(batch_ids)}.")

        data = data.to(self._device)

        if self._buffer is None:
            # 给每一个环境创建一个指针 初始是-1
            self._pointer = -torch.ones(self._batch_size, dtype=torch.int, device=self._device)
            # (self.max_length, self._batch_size)：前两维是 (时间步, 批量)。
            #   + data.shape[1:]：后面加上数据的额外维度
            self._buffer = torch.empty(
                (self.max_length, self._batch_size) + data.shape[1:], device=self._device, dtype=data.dtype
            )
       
        '''
        self._pointer[batch_ids] 取对应batch的指针 （意思是只更新传入的batch） 这就是异步更新的核心
        self._pointer[batch_ids] + 1  把对应位置的指针+1  取模值之后在送回到对应的self._pointer[batch_ids]
        '''
        self._pointer[batch_ids] = (self._pointer[batch_ids] + 1) % self.max_length
        # 把传进来的新数据放入到对应的地方 只更新对应的id
        self._buffer[self._pointer[batch_ids], batch_ids] = data
        is_first_push = self._num_pushes[batch_ids] == 0

        if torch.any(is_first_push):
            batch_ids = torch.as_tensor(batch_ids, device=self._device)
            first_push_batch_ids = batch_ids[is_first_push]
            '''
            self._buffer[:, first_push_batch_ids] = data[is_first_push]：
            ：第 0 维（时间步）全选。
            first_push_batch_ids：第 1 维（批量）选首次推送的 batch。
            作用：把当前数据，填充到该 batch 的所有时间步。
            '''
            self._buffer[:, first_push_batch_ids] = data[is_first_push]
        # 只给指定的 batch_ids 加 1
        self._num_pushes[batch_ids] += 1

    def __getitem__(self, key: torch.Tensor | None = None, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        if batch_ids is None:
            return super().__getitem__(key)
        elif key is None:
            return self.get_by_batch_ids(batch_ids)
        else:
            if len(batch_ids) != key.shape[0]:
                raise ValueError(f"Batch IDs length {len(batch_ids)} does not match key shape {key.shape[0]}.")

        if torch.any(self._num_pushes[batch_ids] == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")

        current_pointers = self._pointer[batch_ids]

        valid_keys = torch.minimum(key, self._num_pushes[batch_ids] - 1)
        index_in_buffer = torch.remainder(current_pointers - valid_keys, self.max_length)
        return self._buffer[index_in_buffer, batch_ids]
