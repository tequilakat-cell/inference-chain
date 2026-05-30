"""
Unified llama.cpp inference backend.

Works on all platforms via llama-cpp-python:
  - macOS Apple Silicon: Metal GPU  (n_gpu_layers=-1, Metal is default on macOS)
  - Linux + Vulkan GPU:  Vulkan offload (install llama-cpp-python with Vulkan support)
  - Fallback:            CPU

Also handles pipeline-parallel coordinator role via the llama.cpp RPC backend:
  - Coordinator: loads GGUF, passes rpc_servers to Llama() to distribute layers
  - Worker:      runs rpc-server subprocess (see l2_miner.py sidecar startup)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from .base import InferenceBackend

log = logging.getLogger("miner.backend.llama_cpp")

try:
    from llama_cpp import Llama
    import llama_cpp as _llama_cpp_mod
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    Llama = None
    _llama_cpp_mod = None

# Detect platform for logging
_IS_MACOS = sys.platform == "darwin"
_BACKEND_LABEL = "metal" if _IS_MACOS else "llama_cpp"


def _find_binary(name: str) -> Optional[str]:
    """
    Find a binary, preferring the source build (which has RPC support compiled in)
    over any system/brew install that may lack it.
    """
    import shutil
    # Prefer source build — it was compiled with -DGGML_RPC=on
    priority = [
        f"/tmp/llama_cpp_src/build/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/homebrew/bin/{name}",
    ]
    for p in priority:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Fall back to PATH
    return shutil.which(name)


class LlamaCppBackend(InferenceBackend):
    """
    llama.cpp inference via llama-cpp-python.

    On macOS Apple Silicon, Metal is used automatically when n_gpu_layers > 0.
    On Linux with Vulkan build, Vulkan is used.
    """

    def __init__(self, model_map: dict[str, str]) -> None:
        self._map = model_map
        self._models: dict[str, Any] = {}
        self._n_gpu_layers = int(os.environ.get("MINER_LLAMA_GPU_LAYERS", "-1"))
        self._n_ctx = int(os.environ.get("MINER_LLAMA_N_CTX", "4096"))
        self._n_batch = int(os.environ.get("MINER_LLAMA_BATCH", "512"))
        self._use_mlock = os.environ.get("MINER_LLAMA_MLOCK", "true").lower() == "true"

        if HAS_LLAMA_CPP:
            log.info(
                "llama_cpp_backend ready platform=%s gpu_layers=%s ctx=%d",
                "macos-metal" if _IS_MACOS else "linux",
                self._n_gpu_layers, self._n_ctx,
            )
        else:
            log.warning("llama-cpp-python not installed — backend disabled")

    @property
    def name(self) -> str:
        return "metal" if _IS_MACOS else "llama_cpp"

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_gpu_layers": self._n_gpu_layers,
            "n_ctx": self._n_ctx,
            "has_llama_cpp": HAS_LLAMA_CPP,
        }

    def supports(self, model_id: str) -> bool:
        if model_id not in self._map or not HAS_LLAMA_CPP:
            return False
        path = os.path.expanduser(self._map[model_id])
        return os.path.isfile(path)

    def _model_path(self, model_id: str) -> str:
        path = self._map[model_id]
        expanded = os.path.expanduser(path)
        return expanded

    def _make_llama(self, model_id: str, n_gpu_layers: Optional[int] = None, n_ctx: Optional[int] = None) -> "Llama":
        """Create a Llama instance with optional override for gpu layers and context size."""
        model_path = self._model_path(model_id)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers if n_gpu_layers is not None else self._n_gpu_layers,
            "n_ctx": n_ctx if n_ctx is not None else self._n_ctx,
            "n_batch": self._n_batch,
            "use_mlock": self._use_mlock,
            "verbose": False,
        }

        return Llama(**kwargs)

    def _model_cache_key(self, model_id: str, n_gpu_layers: int, n_ctx: int) -> str:
        return f"{model_id}::ngl{n_gpu_layers}::ctx{n_ctx}"

    async def _get_or_load(self, model_id: str, n_gpu_layers: int, n_ctx: int) -> "Llama":
        """Return a cached Llama instance, loading it if necessary."""
        key = self._model_cache_key(model_id, n_gpu_layers, n_ctx)
        if key not in self._models:
            loop = asyncio.get_event_loop()
            log.info("loading model=%s ngl=%d ctx=%d", model_id, n_gpu_layers, n_ctx)
            model = await loop.run_in_executor(
                None, self._make_llama, model_id, n_gpu_layers, n_ctx
            )
            self._models[key] = model
            log.info("model_loaded model=%s ngl=%d ctx=%d", model_id, n_gpu_layers, n_ctx)
        return self._models[key]

    async def load(self, model_id: str) -> None:
        if model_id in self._models:
            return
        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama-cpp-python not installed")

        await self._get_or_load(model_id, self._n_gpu_layers, self._n_ctx)
        # Also store under plain model_id for backwards compatibility
        key = self._model_cache_key(model_id, self._n_gpu_layers, self._n_ctx)
        self._models[model_id] = self._models[key]

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if not HAS_LLAMA_CPP:
            raise RuntimeError("llama-cpp-python not installed")
        if model_id not in self._models:
            await self.load(model_id)

        llm = self._models[model_id]
        loop = asyncio.get_event_loop()

        def _run() -> str:
            return _infer(llm, prompt, max_tokens, temperature)

        return await loop.run_in_executor(None, _run)

    async def generate_with_rpc(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        rpc_servers: str,
        prompt_cache_path: str = "",
        max_memory_gb: int = 0,
        worker_memory_gb: list[int] | None = None,
        tensor_split_fracs: list[float] | None = None,
    ) -> str:
        """
        Pipeline-parallel coordinator inference.

        Each node in a pipeline_parallel job loads its assigned model layers
        from its OWN local model file. The coordinator runs inference locally
        (no layer-weight transfer over the network). Workers participate via
        the rpc-server sidecar and receive compute requests for their layers
        once the model is loaded, but the coordinator always loads from its
        own local copy — network bandwidth is only used for inter-layer
        activations during generation, not for weight distribution.

        If a fast RPC path becomes available (rpc_server_bin with shared
        weights / RDMA), set MINER_LLAMA_RPC_WEIGHT_DISTRIBUTION=1 to enable
        the legacy llama-cli --rpc weight-push path.

        prompt_cache_path (Phase 4): if provided, passes --prompt-cache to
        llama-cli so KV activations for the context prefix are saved/reused
        between jobs that share the same context_hash.
        """
        # Use the weight-distribution RPC path only when explicitly enabled.
        # By default the coordinator loads locally to avoid sending the full
        # model over the network (which at LAN speeds can take minutes).
        use_rpc_weights = os.environ.get("MINER_LLAMA_RPC_WEIGHT_DISTRIBUTION", "").lower() in ("1", "true", "yes")

        if use_rpc_weights and rpc_servers:
            return await self._generate_with_rpc_weights(
                model_id, prompt, max_tokens, temperature, rpc_servers,
                prompt_cache_path, max_memory_gb, worker_memory_gb, tensor_split_fracs,
            )

        # Default path: coordinator runs inference locally via Python bindings.
        # The model is cached after first load so subsequent jobs are fast.
        # Workers hold their own layer slices and participate via rpc-server sidecar.
        #
        # Calculate n_gpu_layers from max_memory_gb budget so we don't OOM.
        # When the model file is larger than the memory budget, use pure CPU
        # (n_gpu_layers=0) so that Metal allocations don't eat into the OS
        # page cache — Apple Silicon CPU inference is memory-bandwidth-bound,
        # and maximizing the page cache hit rate matters more than GPU layers
        # when RAM < model size.
        if max_memory_gb > 0:
            try:
                model_size_gb = os.path.getsize(self._model_path(model_id)) / (1024 ** 3)
            except OSError:
                model_size_gb = 16.0
            if model_size_gb > max_memory_gb:
                # Memory-constrained: CPU-only maximizes page-cache headroom.
                n_gpu_layers = 0
            else:
                per_layer_mb = 280
                n_gpu_layers = max(0, int(max_memory_gb * 1024 * 0.6 / per_layer_mb))
            # Reduce context to free KV-cache memory on constrained machines.
            n_ctx = min(self._n_ctx, max(512, int(max_memory_gb * 128)))
        else:
            n_gpu_layers = self._n_gpu_layers
            n_ctx = self._n_ctx

        log.info(
            "pipeline_coordinator_local model=%s rpc_peers=%s ngl=%d ctx=%d cache=%s ts=%s",
            model_id, rpc_servers or "none", n_gpu_layers, n_ctx, prompt_cache_path or "none",
            [f"{f:.2f}" for f in (tensor_split_fracs or [])],
        )

        # Disable chain-of-thought for thinking models (Qwen3, DeepSeek-R1, etc.)
        # so the token budget is spent on the answer, not internal reasoning.
        effective_prompt = prompt
        if not prompt.lstrip().startswith("/no_think"):
            effective_prompt = "/no_think\n" + prompt

        llm = await self._get_or_load(model_id, n_gpu_layers, n_ctx)
        loop = asyncio.get_event_loop()

        def _run_local() -> str:
            return _infer(llm, effective_prompt, max_tokens, temperature)

        return await loop.run_in_executor(None, _run_local)

    async def _generate_with_rpc_weights(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        rpc_servers: str,
        prompt_cache_path: str = "",
        max_memory_gb: int = 0,
        worker_memory_gb: list[int] | None = None,
        tensor_split_fracs: list[float] | None = None,
    ) -> str:
        """
        Legacy RPC path: coordinator sends layer weights to workers over the
        network via llama-cli --rpc. Only use when all nodes are on a fast
        interconnect (1 Gbps+) since the full model weights are transferred.
        Enable with MINER_LLAMA_RPC_WEIGHT_DISTRIBUTION=1.
        """
        model_path = self._model_path(model_id)
        llama_cli  = _find_binary("llama-cli")

        if not llama_cli:
            log.warning("llama-cli not found — running local inference (no RPC)")
            return await self.generate(model_id, prompt, max_tokens, temperature)

        # Tensor-split priority:
        # 1. tensor_split_fracs from benchmark scores (most accurate).
        # 2. Memory-budget heuristic (fallback when no scores available).
        # 3. Equal split (last resort).
        ts_args: list[str] = []
        workers = worker_memory_gb or []
        n_rpc_workers = len([s for s in rpc_servers.split(",") if s.strip()])

        if tensor_split_fracs and len(tensor_split_fracs) >= 1:
            fracs = [f"{f:.4f}" for f in tensor_split_fracs]
            ts_args = ["-ts", ",".join(fracs)]
            log.info(
                "pipeline_coordinator_rpc rpc=%s model=%s binary=%s cache=%s ts=benchmark(%s)",
                rpc_servers, model_id, llama_cli, prompt_cache_path or "none",
                ",".join(fracs),
            )
        elif max_memory_gb > 0 and workers and any(w > 0 for w in workers):
            total = max_memory_gb + sum(w for w in workers if w > 0)
            fracs = [f"{max_memory_gb / total:.4f}"] + [
                f"{w / total:.4f}" for w in workers if w > 0
            ]
            ts_args = ["-ts", ",".join(fracs)]
            log.info(
                "pipeline_coordinator_rpc rpc=%s model=%s binary=%s cache=%s ts=memory(%s)",
                rpc_servers, model_id, llama_cli, prompt_cache_path or "none",
                ",".join(fracs),
            )
        elif n_rpc_workers > 0:
            n_backends = n_rpc_workers + 1
            frac = f"{1.0 / n_backends:.4f}"
            ts_args = ["-ts", ",".join([frac] * n_backends)]
            log.info(
                "pipeline_coordinator_rpc rpc=%s model=%s binary=%s cache=%s ts=equal(%d)",
                rpc_servers, model_id, llama_cli, prompt_cache_path or "none", n_backends,
            )
        else:
            log.info(
                "pipeline_coordinator_rpc rpc=%s model=%s binary=%s cache=%s",
                rpc_servers, model_id, llama_cli, prompt_cache_path or "none",
            )

        cmd = [
            llama_cli,
            "--model",    model_path,
            "--rpc",      rpc_servers,
            "-ngl",       "999",
            "-n",         str(max_tokens),
            "--temp",     str(temperature),
            "--ctx-size", str(self._n_ctx),
            "--prompt",   prompt,
            "--log-disable",
            "-st",
        ] + ts_args

        if prompt_cache_path:
            import os as _os
            _os.makedirs(_os.path.dirname(prompt_cache_path), exist_ok=True)
            cmd += ["--prompt-cache", prompt_cache_path, "--prompt-cache-all"]

        loop = asyncio.get_event_loop()

        def _run() -> str:
            import subprocess as _sp
            result = _sp.run(
                cmd,
                stdin=_sp.DEVNULL,
                stdout=_sp.PIPE,
                stderr=_sp.DEVNULL,
                timeout=600,
            )
            raw = result.stdout.decode("utf-8", errors="replace")
            lines = raw.splitlines()
            in_response = False
            response_lines: list[str] = []
            for line in lines:
                if not in_response:
                    if line.startswith("> "):
                        in_response = True
                else:
                    if line.startswith("[ Prompt:") or line == "Exiting...":
                        break
                    clean: list[str] = []
                    for ch in line:
                        if ch == "\x08" and clean:
                            clean.pop()
                        elif ch != "\r":
                            clean.append(ch)
                    response_lines.append("".join(clean))
            return "\n".join(response_lines).strip()

        return await loop.run_in_executor(None, _run)

    def supports_tensor_parallel(self) -> bool:
        return HAS_LLAMA_CPP


def _infer(llm: "Llama", prompt: str, max_tokens: int, temperature: float) -> str:
    """Run inference, applying chat template if the model supports it."""
    # Check for chat template in model metadata
    has_template = False
    try:
        meta = llm.metadata or {}
        has_template = bool(meta.get("tokenizer.chat_template"))
    except Exception:
        pass

    if has_template:
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"].strip()
    else:
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "[INST]", "[/INST]", "###", "<|im_end|>"],
            echo=False,
        )
        return result["choices"][0]["text"].strip()
