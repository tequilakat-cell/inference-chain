"""
MLX Inference Backend
=====================
Uses Apple's MLX framework for Apple Silicon (M1/M2/M3/M4).
Supports HuggingFace models directly via mlx-lm.
Extremely fast on Apple Silicon — often beats CUDA (F32) on M-series.

Requirements:
    pip install mlx mlx-lm

Environment variables:
    MINER_MLX_MAX_MEMORY=8GB   # Memory limit (default: no limit)
    MINER_MLX_MODEL_CACHE=     # Path to cache directory (default: ~/.cache/huggingface)
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional
from .base import InferenceBackend

log = logging.getLogger("miner.backend.mlx")

try:
    import mlx.core as mx
    import mlx_lm

    HAS_MLX = hasattr(mx, "metal") and mx.metal.is_available()
except ImportError:
    HAS_MLX = False
    mx = None
    mlx_lm = None


class MLXBackend(InferenceBackend):
    """
    Inference via Apple MLX on Apple Silicon.
    Uses mlx_lm.generate() for text generation.
    """

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map = model_map
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}

        if HAS_MLX:
            import mlx.core as mx
            try:
                mem_info = mx.metal.get_memory_info()
                self._device_info = {
                    "metal":       True,
                    "peak_memory": mem_info.peak,
                    "free_memory": mem_info.free,
                    "active_memory": hasattr(mem_info, "active") and mem_info.active,
                }
            except Exception:
                self._device_info = {"metal": True}
            log.info("MLX backend ready  device=%s", self._device_info)
        else:
            self._device_info = {"metal": False}
            log.info("MLX backend disabled — no Metal available")

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def info(self) -> dict[str, Any]:
        return {"name": "mlx", "device": self._device_info}

    def supports(self, model_id: str) -> bool:
        return model_id in self._map

    async def load(self, model_id: str) -> None:
        if model_id in self._models:
            return
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed or Metal is unavailable")

        hf_id = self._map[model_id]
        log.info("loading_mlx_model model=%s hf_id=%s", model_id, hf_id)

        loop = asyncio.get_event_loop()

        def _load() -> tuple[Any, Any]:
            return mlx_lm.load(
                hf_id,
                tokenizer_config={"trust_remote_code": True},
            )

        try:
            model, tokenizer = await loop.run_in_executor(None, _load)
            self._models[model_id] = model
            self._tokenizers[model_id] = tokenizer
            log.info("mlx_model_loaded model=%s", model_id)
        except Exception as exc:
            log.error("mlx_load_failed model=%s err=%s", model_id, exc)
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
        tokenizer = self._tokenizers.get(model_id)
        if model is None or tokenizer is None:
            raise RuntimeError(f"Model {model_id} not loaded")

        loop = asyncio.get_event_loop()

        def _generate() -> str:
            from mlx_lm.sample_utils import make_sampler

            # Apply chat template when the tokenizer supports it
            if hasattr(tokenizer, "apply_chat_template") and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            return mlx_lm.generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature),
            )

        try:
            result = await loop.run_in_executor(None, _generate)
            return result.strip()
        except Exception as exc:
            log.error("mlx_generate_failed model=%s err=%s", model_id, exc)
            raise
