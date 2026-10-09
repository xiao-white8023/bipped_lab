"""Minimal fixed-shape TensorRT runner for the G1 RENet policy."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


class TensorRTPolicyRunner:
    """Run the mask-selected ONNX policy twice to obtain LP and VP actions."""

    def __init__(self, engine_path: Path, input_dim: int, action_dim: int):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT backend requested, but the tensorrt Python module "
                "is unavailable"
            ) from exc

        self._trt = trt
        self._cuda = ctypes.CDLL("libcudart.so")
        self._configure_cuda_api()
        self._closed = False

        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        self._engine = self._runtime.deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        self._input_index = self._binding_index("obs")
        self._output_index = self._binding_index("actions")
        self._validate_binding(
            self._input_index,
            "obs",
            (1, input_dim),
            is_input=True,
        )
        self._validate_binding(
            self._output_index,
            "actions",
            (1, action_dim),
            is_input=False,
        )

        self._host_input = np.empty((1, input_dim), dtype=np.float32)
        self._host_output = np.empty((1, action_dim), dtype=np.float32)
        self._device_input = ctypes.c_void_p()
        self._device_output = ctypes.c_void_p()
        self._cuda_check(
            self._cuda.cudaMalloc(
                ctypes.byref(self._device_input), self._host_input.nbytes
            ),
            "cudaMalloc(obs)",
        )
        try:
            self._cuda_check(
                self._cuda.cudaMalloc(
                    ctypes.byref(self._device_output), self._host_output.nbytes
                ),
                "cudaMalloc(actions)",
            )
        except Exception:
            self._cuda.cudaFree(self._device_input)
            raise

        self._bindings = [0] * self._engine.num_bindings
        self._bindings[self._input_index] = self._device_input.value
        self._bindings[self._output_index] = self._device_output.value

    def _configure_cuda_api(self) -> None:
        self._cuda.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._cuda.cudaMalloc.restype = ctypes.c_int
        self._cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self._cuda.cudaFree.restype = ctypes.c_int
        self._cuda.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._cuda.cudaMemcpy.restype = ctypes.c_int

    @staticmethod
    def _cuda_check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def _binding_index(self, name: str) -> int:
        index = int(self._engine.get_binding_index(name))
        if index < 0:
            raise RuntimeError(f"TensorRT engine has no {name!r} binding")
        return index

    def _validate_binding(
        self,
        index: int,
        name: str,
        expected_shape: tuple[int, ...],
        *,
        is_input: bool,
    ) -> None:
        actual_shape = tuple(int(value) for value in self._engine.get_binding_shape(index))
        actual_input = bool(self._engine.binding_is_input(index))
        actual_dtype = self._engine.get_binding_dtype(index)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"TensorRT binding {name!r} has shape {actual_shape}, "
                f"expected {expected_shape}"
            )
        if actual_input != is_input:
            raise RuntimeError(
                f"TensorRT binding {name!r} has the wrong input/output direction"
            )
        if actual_dtype != self._trt.float32:
            raise RuntimeError(
                f"TensorRT binding {name!r} must expose float32, got {actual_dtype}"
            )

    def _infer_mask(self, observation: np.ndarray, mask: float) -> np.ndarray:
        np.copyto(
            self._host_input,
            np.asarray(observation, dtype=np.float32).reshape(self._host_input.shape),
        )
        self._host_input[0, -1] = mask
        self._cuda_check(
            self._cuda.cudaMemcpy(
                self._device_input,
                self._host_input.ctypes.data_as(ctypes.c_void_p),
                self._host_input.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "cudaMemcpy(obs H2D)",
        )
        if not self._context.execute_v2(self._bindings):
            raise RuntimeError("TensorRT execute_v2 failed")
        self._cuda_check(
            self._cuda.cudaMemcpy(
                self._host_output.ctypes.data_as(ctypes.c_void_p),
                self._device_output,
                self._host_output.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            ),
            "cudaMemcpy(actions D2H)",
        )
        return self._host_output.reshape(-1).copy()

    def infer_dual(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return LP then VP action, matching renet_dual_branch.pt ordering."""

        lp_action = self._infer_mask(observation, 1.0)
        vp_action = self._infer_mask(observation, 0.0)
        return lp_action, vp_action

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._device_output.value is not None:
            self._cuda_check(
                self._cuda.cudaFree(self._device_output),
                "cudaFree(actions)",
            )
        if self._device_input.value is not None:
            self._cuda_check(
                self._cuda.cudaFree(self._device_input),
                "cudaFree(obs)",
            )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
