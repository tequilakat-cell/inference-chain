"""
Abstract inference backend.
All GPU backends (CUDA, MLX, Vulkan, llama.cpp) implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class InferenceBackend(ABC):
    """Every backend must be able to load a model and generate text."""

    @abstractmethod
    def supports(self, model_id: str) -> bool:
        ...

    @abstractmethod
    async def load(self, model_id: str) -> None:
        ...

    @abstractmethod
    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short backend identifier e.g. 'cuda', 'mlx', 'vulkan', 'llama.cpp'."""
        ...

    @property
    def info(self) -> dict[str, Any]:
        """Optional metadata about the device / driver."""
        return {"name": self.name}

    def supports_tensor_parallel(self) -> bool:
        """Return True if this backend supports pipeline-parallel tensor sharding."""
        return False

    async def num_layers(self, model_id: str) -> int:
        """Return the number of transformer layers in the model."""
        raise NotImplementedError("tensor_parallel not supported by this backend")

    async def embed_and_forward(
        self, model_id: str, prompt: str, layer_end: int
    ) -> tuple:
        """Embed prompt and forward through layers[0:layer_end]. Returns (hidden_bytes, dtype, shape)."""
        raise NotImplementedError("tensor_parallel not supported by this backend")

    async def forward_layers(
        self, model_id: str, hidden_bytes: bytes, shape: list, dtype: str,
        layer_start: int, layer_end: int
    ) -> tuple:
        """Forward hidden state through layers[layer_start:layer_end]. Returns (hidden_bytes, dtype, shape)."""
        raise NotImplementedError("tensor_parallel not supported by this backend")

    async def forward_and_decode(
        self, model_id: str, hidden_bytes: bytes, shape: list, dtype: str,
        layer_start: int, max_tokens: int
    ) -> str:
        """Forward through final layers and decode to text."""
        raise NotImplementedError("tensor_parallel not supported by this backend")


class MockBackend(InferenceBackend):
    """Fallback when no real backend is available (development / testing)."""

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map = model_map

    @property
    def name(self) -> str:
        return "mock"

    def supports(self, model_id: str) -> bool:
        return model_id in self._map

    async def load(self, model_id: str) -> None:
        pass  # nothing to load

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return (
            f"[MOCK inference — install a real GPU backend]\n"
            f"model={model_id} prompt={prompt[:120]} max_tokens={max_tokens}"
        )
