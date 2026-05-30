"""
CUDA Inference Backend
======================
Uses HuggingFace transformers + PyTorch with CUDA.
Best for NVIDIA GPUs with large VRAM (8 GB+).
Supports all HuggingFace model formats (not just GGUF).

Requirements:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers accelerate bitsandbytes

Environment variables:
    CUDA_VISIBLE_DEVICES=0          # GPU selection
    MINER_CUDA_DEVICE_MAP=auto      # "auto" | "sequential" (default auto)
    MINER_CUDA_LOAD_IN_8BIT=false   # true to use 8-bit quantization
    MINER_CUDA_LOAD_IN_4BIT=true    # true to use 4-bit quantization
    MINER_CUDA_MAX_MEMORY=8GiB      # Max memory per GPU
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional
from .base import InferenceBackend

log = logging.getLogger("miner.backend.cuda")

try:
    import torch
    import transformers
    import accelerate

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False
    torch = None
    transformers = None
    accelerate = None

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cuda_device_info() -> list[dict[str, Any]]:
    if not HAS_CUDA:
        return []
    infos = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        infos.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return infos


# ── Quantization configs ──────────────────────────────────────────────────────

def _quantization_config() -> Optional[Any]:
    """Return a BitsAndBytesConfig or None."""
    try:
        from transformers import BitsAndBytesConfig
        import torch

        load_4bit = os.environ.get("MINER_CUDA_LOAD_IN_4BIT", "true").lower() == "true"
        load_8bit = os.environ.get("MINER_CUDA_LOAD_IN_8BIT", "false").lower() == "true"

        if load_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if load_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        pass
    return None


# ── Backend ───────────────────────────────────────────────────────────────────

class CUDABackend(InferenceBackend):
    """
    Inference via HuggingFace transformers on CUDA GPUs.
    Supports 'device_map=auto' and quantization (4-bit / 8-bit).
    """

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map = model_map
        self._pipelines: dict[str, Any] = {}
        self._device_info = _cuda_device_info()
        log.info(
            "CUDA backend ready  devices=%d gpus=%s",
            len(self._device_info),
            [d["name"] for d in self._device_info],
        )

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def info(self) -> dict[str, Any]:
        return {"name": "cuda", "devices": self._device_info}

    def supports(self, model_id: str) -> bool:
        return model_id in self._map

    async def load(self, model_id: str) -> None:
        if model_id in self._pipelines:
            return
        hf_id = self._map[model_id]
        loop = asyncio.get_event_loop()
        log.info("loading_cuda_model model=%s hf_id=%s", model_id, hf_id)
        try:
            pipe = await loop.run_in_executor(
                None,
                lambda: transformers.pipeline(
                    "text-generation",
                    model=hf_id,
                    device_map=os.environ.get("MINER_CUDA_DEVICE_MAP", "auto"),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    quantization_config=_quantization_config(),
                    max_memory={i: os.environ.get("MINER_CUDA_MAX_MEMORY", "8GiB")
                                for i in range(torch.cuda.device_count())}
                    if torch.cuda.is_available() else None,
                ),
            )
            self._pipelines[model_id] = pipe
            log.info("cuda_model_loaded model=%s", model_id)
        except Exception as exc:
            log.error("cuda_load_failed model=%s err=%s", model_id, exc)
            raise

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if model_id not in self._pipelines:
            await self.load(model_id)

        pipe = self._pipelines.get(model_id)
        if pipe is None:
            raise RuntimeError(f"Model {model_id} not loaded")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=pipe.tokenizer.eos_token_id,
            ),
        )
        return result[0]["generated_text"].removeprefix(prompt).strip()
