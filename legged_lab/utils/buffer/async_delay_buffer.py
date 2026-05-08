import torch
from collections.abc import Sequence
from typing import Union

from isaaclab.utils.buffers import DelayBuffer

from .async_circular_buffer import AsyncCircularBuffer


class AsyncDelayBuffer(DelayBuffer):
    """Asynchronous delay buffer that allows retrieving stored data with delays asynchronously for each batch index."""

    def __init__(self, history_length: int, batch_size: int, device: str):
        """Initialize the asynchronous delay buffer.

        Args:
            history_length: The history of the buffer, i.e., the number of time steps in the past that the data
                will be buffered. It is recommended to set this value equal to the maximum time-step lag that
                is expected. The minimum acceptable value is zero, which means only the latest data is stored.
            batch_size: The batch dimension of the data.
            device: The device used for processing.
        """
        '''
        history_length 是 “最大延迟步数”（比如 4 步，即可以取到 4 步前的数据）。
        要取到 4 步前的数据，缓冲区必须存：当前步（0 延迟） + 4 步历史，总共 5 步。
        '''
        super().__init__(history_length, batch_size, device)
        self._circular_buffer = AsyncCircularBuffer(self._history_length + 1, batch_size, device)

    '''
    data：本次的新数据，形状 (selected_batch_size, *D)。
    batch_ids：本次要追加数据的 batch 索引，None表示所有 batch
    '''
    def compute(self, data: torch.Tensor, batch_ids: Sequence[int] | None = None) -> torch.Tensor:
        if batch_ids is None:
            return super().compute(data)
        else:
            if len(batch_ids) != data.shape[0]:
                raise ValueError(f"Batch IDs length {len(batch_ids)} does not match data shape {data.shape[0]}.")

        # 把新数据追加到内部异步循环缓冲区
        # add the new data to the last layer
        self._circular_buffer.append(data, batch_ids)
        # return the output
        #用固定延迟步数取历史数据
        delayed_data = self._circular_buffer.__getitem__(self._time_lags[batch_ids], batch_ids)
        return delayed_data.clone()
