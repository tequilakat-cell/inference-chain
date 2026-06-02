"""
Auto-detection of best GPU inference backend.
Scans available hardware and returns the optimal backend:

  1. CUDA        — NVIDIA GPU with PyTorch + transformers
  2. metal       — Apple Silicon via llama-cpp-python (Metal GPU, GGUF)
  3. jacobi      — Jacobi parallel decoding (standalone C++ jacobi-server)
  4. vulkan      — AMD / Intel / NVIDIA via llama.cpp + Vulkan (GGUF)
  5. llama_cpp   — CPU fallback via llama.cpp (GGUF)
  6. mock        — No hardware found (dev/test)

All non-CUDA backends use GGUF models via llama-cpp-python.
MLX is no longer used — Metal (llama.cpp) replaces it on Apple Silicon.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional
from .base import InferenceBackend, MockBackend

log = logging.getLogger("miner.backend.detect")

_IS_MACOS = sys.platform == "darwin"


def get_backend(
    model_map: dict[str, str],
    preferred: Optional[str] = None,
) -> InferenceBackend:
    """
    Detect and return the best available inference backend.

    Args:
        model_map:  Mapping of HF model IDs to local GGUF paths.
        preferred:  Force a specific backend: "cuda", "metal", "vulkan",
                    "llama_cpp", "mock".  If None, auto-detect.
    """
    if preferred:
        preferred = preferred.lower().strip()
        # Aliases for backward compat
        preferred = {"mlx": "metal", "llama": "vulkan"}.get(preferred, preferred)
        return _load_preferred(model_map, preferred)

    # Priority order — Jacobi before Metal/Vulkan when jacobi-server is available.
    # Jacobi parallel decoding gives 2-5x speedup over vanilla AR generation.
    candidates = [
        ("cuda",     _try_cuda),
        ("jacobi",   _try_jacobi),
        ("metal",    _try_metal),
        ("vulkan",   _try_vulkan),
        ("llama_cpp", _try_llama_cpp),
    ]

    for backend_name, try_fn in candidates:
        instance = try_fn(model_map)
        if instance is not None:
            log.info("backend_selected name=%s info=%s", backend_name, instance.info)
            return instance

    log.warning("no_gpu_backend_found — falling back to mock backend")
    return MockBackend(model_map)


def get_available_backends() -> list[str]:
    """Return list of available backend names on this machine."""
    available = []
    for name, try_fn in [
        ("cuda",     _try_cuda),
        ("jacobi",   _try_jacobi),
        ("metal",    _try_metal),
        ("vulkan",   _try_vulkan),
        ("llama_cpp", _try_llama_cpp),
    ]:
        if try_fn({}) is not None:
            available.append(name)
    return available or ["mock"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_preferred(model_map: dict[str, str], name: str) -> InferenceBackend:
    all_backends = {
        "cuda":     _try_cuda,
        "jacobi":   _try_jacobi,
        "metal":    _try_metal,
        "vulkan":   _try_vulkan,
        "llama_cpp": _try_llama_cpp,
        "mock":     lambda m: MockBackend(m),
    }
    if name not in all_backends:
        raise RuntimeError(f"Unknown backend '{name}'. Options: {list(all_backends.keys())}")

    instance = all_backends[name](model_map)
    if instance is None:
        log.warning("preferred_backend_unavailable preferred=%s — falling back", name)
        for fallback, try_fn in [("llama_cpp", _try_llama_cpp), ("mock", lambda m: MockBackend(m))]:
            if fallback == name:
                continue
            instance = try_fn(model_map)
            if instance is not None:
                log.info("backend_fallback from=%s to=%s", name, fallback)
                return instance
    return instance


def _try_cuda(model_map: dict[str, str]) -> Optional[InferenceBackend]:
    try:
        from .cuda import CUDABackend, HAS_CUDA
        if HAS_CUDA:
            backend = CUDABackend(model_map)
            if backend.info.get("devices"):
                return backend
    except Exception as exc:
        log.debug("cuda_unavailable err=%s", exc)
    return None


def _try_metal(model_map: dict[str, str]) -> Optional[InferenceBackend]:
    """Apple Metal via llama-cpp-python (macOS only)."""
    if not _IS_MACOS:
        return None
    try:
        from .llama_cpp_backend import LlamaCppBackend, HAS_LLAMA_CPP
        if HAS_LLAMA_CPP:
            return LlamaCppBackend(model_map)
    except Exception as exc:
        log.debug("metal_unavailable err=%s", exc)
    return None


def _try_vulkan(model_map: dict[str, str]) -> Optional[InferenceBackend]:
    try:
        from .vulkan import VulkanBackend, HAS_VULKAN_LLAMACPP
        if HAS_VULKAN_LLAMACPP:
            return VulkanBackend(model_map)
    except Exception as exc:
        log.debug("vulkan_unavailable err=%s", exc)
    return None


def _try_llama_cpp(model_map: dict[str, str]) -> Optional[InferenceBackend]:
    """llama.cpp CPU/GPU fallback — works anywhere llama-cpp-python is installed."""
    try:
        from llama_cpp import Llama  # noqa: F401
        from .llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend(model_map)
    except Exception as exc:
        log.debug("llama_cpp_unavailable err=%s", exc)
    return None


def _try_jacobi(model_map: dict[str, str]) -> Optional[InferenceBackend]:
    """Lookahead Decoding backend (replaces Jacobi; 'jacobi' name kept for config compat)."""
    try:
        import llama_cpp  # noqa: F401 — need llama_cpp for in-process inference
        from .jacobi_backend import LookaheadBackend
        return LookaheadBackend(model_map)
    except Exception as exc:
        log.debug("lookahead_unavailable err=%s", exc)
    return None
