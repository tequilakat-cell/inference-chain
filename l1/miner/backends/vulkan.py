"""
Vulkan Inference Backend
========================
Uses llama.cpp compiled with Vulkan support for cross-vendor GPU inference.
Works on AMD (AMDGPU), Intel (ANV), and NVIDIA (NVK via nouveau) GPUs.

Requirements:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan

    # Or build from source with Vulkan:
    # CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python

Environment variables:
    MINER_VULKAN_GPU_LAYERS=-1       # Layers to offload (-1 = all)
    MINER_VULKAN_N_CTX=4096          # Context window size
    MINER_VULKAN_BATCH_SIZE=512      # Batch size for prompt processing
    MINER_VULKAN_UBATCH_SIZE=512     # Batch size for generation
    MINER_VULKAN_USE_MLOCK=true      # Lock memory to avoid swapping
    MINER_VULKAN_DEVICE=0            # Vulkan device index
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional
from .base import InferenceBackend

log = logging.getLogger("miner.backend.vulkan")

try:
    from llama_cpp import Llama
    import llama_cpp

    HAS_VULKAN_LLAMACPP = True

    # Try to detect Vulkan at import time
    HAS_VULKAN = True  # optimistic; actual GPU detection happens on model load
except ImportError:
    HAS_VULKAN_LLAMACPP = False
    HAS_VULKAN = False
    Llama = None
    llama_cpp = None


class VulkanBackend(InferenceBackend):
    """
    Inference via llama.cpp with Vulkan GPU offloading.
    Works with AMD, Intel, and NVIDIA GPUs that support Vulkan.
    Uses GGUF model format.
    """

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map = model_map
        self._models: dict[str, Any] = {}
        self._batch_size = int(os.environ.get("MINER_VULKAN_BATCH_SIZE", "512"))
        self._ubatch_size = int(os.environ.get("MINER_VULKAN_UBATCH_SIZE", "512"))
        self._n_gpu_layers = int(os.environ.get("MINER_VULKAN_GPU_LAYERS", "-1"))
        self._n_ctx = int(os.environ.get("MINER_VULKAN_N_CTX", "4096"))
        self._use_mlock = os.environ.get("MINER_VULKAN_USE_MLOCK", "true").lower() == "true"

        if HAS_VULKAN_LLAMACPP:
            log.info(
                "Vulkan backend ready  layers=%s ctx=%d batch=%d ubatch=%d",
                self._n_gpu_layers, self._n_ctx,
                self._batch_size, self._ubatch_size,
            )
        else:
            log.info("Vulkan backend disabled — install llama-cpp-python with Vulkan support")

    @property
    def name(self) -> str:
        return "vulkan"

    @property
    def info(self) -> dict[str, Any]:
        """Report Vulkan device and backend info."""
        info: dict[str, Any] = {
            "name": "vulkan",
            "n_gpu_layers": self._n_gpu_layers,
            "n_ctx": self._n_ctx,
        }
        if HAS_VULKAN_LLAMACPP and llama_cpp:
            try:
                # llama_cpp may expose system_info or device count
                info["system_info"] = str(llama_cpp.llama_supports_gpu_offload())
            except Exception:
                pass
        return info

    def supports(self, model_id: str) -> bool:
        return model_id in self._map

    def _model_path(self, model_id: str) -> str:
        return self._map[model_id]

    async def load(self, model_id: str) -> None:
        if model_id in self._models:
            return
        if not HAS_VULKAN_LLAMACPP:
            raise RuntimeError(
                "llama-cpp-python with Vulkan support is not installed.\n"
                "  pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/vulkan"
            )

        model_path = self._model_path(model_id)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loop = asyncio.get_event_loop()
        log.info("loading_vulkan_model model=%s path=%s", model_id, model_path)

        def _load() -> Any:
            return Llama(
                model_path=model_path,
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
                n_batch=self._batch_size,
                n_ubatch=self._ubatch_size,
                use_mlock=self._use_mlock,
                verbose=False,
                # Vulkan-specific: select device if multiple GPUs
                main_gpu=int(os.environ.get("MINER_VULKAN_DEVICE", "0")),
            )

        try:
            model = await loop.run_in_executor(None, _load)
            self._models[model_id] = model
            log.info("vulkan_model_loaded model=%s", model_id)
        except Exception as exc:
            log.error("vulkan_load_failed model=%s err=%s", model_id, exc)
            raise

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if model_id not in self._models:
            await self.load(model_id)

        model = self._models.get(model_id)
        if model is None:
            raise RuntimeError(f"Model {model_id} not loaded")

        loop = asyncio.get_event_loop()

        def _generate() -> str:
            result = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[INST]", "[/INST]", "###"],
                echo=False,
            )
            return result["choices"][0]["text"].strip()

        try:
            return await loop.run_in_executor(None, _generate)
        except Exception as exc:
            log.error("vulkan_generate_failed model=%s err=%s", model_id, exc)
            raise
