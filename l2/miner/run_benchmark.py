"""
Standalone miner benchmark — measures tokens/sec and publishes the score on-chain.

Usage:
    python -m miner.run_benchmark --config test_local/config_miner1.json
    python -m miner.run_benchmark --config test_local/config_khadas.json --n-tokens 32

The script:
  1. Loads the miner config (private key, model paths, chain RPC URL).
  2. Runs inference on the target model (default: Qwen/Qwen2.5-0.5B-Instruct).
  3. Signs the measured score with the miner's private key.
  4. Calls inft_submitBenchmarkScore on the chain node, which verifies the
     signature, writes a BENCHMARK_COMMIT tx, and stores the score in StateDB
     (and pg_inft if configured).

The chain node then uses this score to proportion layer allocations during
pipeline-parallel jobs — faster miners get more layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_PROMPT  = (
    "Explain the difference between supervised and unsupervised learning "
    "in machine learning, with two examples of each."
)
DEFAULT_TOKENS  = 64


# ── Inference runners ─────────────────────────────────────────────────────────

def _run_python_bindings(model_path: str, prompt: str, n_tokens: int, backend: str) -> tuple[int, int]:
    """Run via llama-cpp-python. Returns (actual_tokens, elapsed_ms)."""
    from llama_cpp import Llama  # type: ignore

    n_gpu_layers = -1 if backend == "metal" else 0
    print(f"  Loading model (n_gpu_layers={n_gpu_layers})…", flush=True)
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    print("  Running inference…", flush=True)
    t0 = time.monotonic()
    out = llm(prompt, max_tokens=n_tokens, temperature=0.0, echo=False)
    elapsed_ms = max(1, int((time.monotonic() - t0) * 1000))
    actual_tokens = out["usage"]["completion_tokens"]
    return actual_tokens, elapsed_ms


def _run_llama_cli(cli_bin: str, model_path: str, prompt: str, n_tokens: int) -> tuple[int, int]:
    """Run via llama-cli subprocess. Returns (actual_tokens, elapsed_ms)."""
    cmd = [
        cli_bin,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", "0",
        "--no-display-prompt",
        "--log-disable",
        "-e",
    ]
    print(f"  Running: {' '.join(cmd[:4])} …", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed_ms = max(1, int((time.monotonic() - t0) * 1000))

    # Parse actual token count from llama-cli timing footer if present
    actual_tokens = n_tokens  # fallback
    for line in (result.stderr + result.stdout).splitlines():
        # "llama_print_timings:        eval time =  3412.13 ms /    64 runs"
        if "eval time" in line and "runs" in line:
            try:
                part = line.split("/")[-1].strip()  # "64 runs"
                actual_tokens = int(part.split()[0])
            except (IndexError, ValueError):
                pass
            break

    return actual_tokens, elapsed_ms


def run_inference(cfg: dict, model_path: str, prompt: str, n_tokens: int) -> tuple[int, int]:
    """
    Run inference on model_path and return (actual_tokens, elapsed_ms).
    Tries llama-cpp-python first, falls back to llama-cli subprocess.
    """
    backend = cfg.get("backend", "llama_cpp")

    # Try Python bindings
    try:
        return _run_python_bindings(model_path, prompt, n_tokens, backend)
    except ImportError:
        pass

    # Fall back to llama-cli
    # Look next to rpc_server_bin, or search PATH
    cli_bin: str | None = None
    rpc_bin = cfg.get("rpc_server_bin", "")
    if rpc_bin:
        candidate = str(Path(rpc_bin).parent / "llama-cli")
        if os.path.isfile(candidate):
            cli_bin = candidate

    if cli_bin is None:
        import shutil
        cli_bin = shutil.which("llama-cli")

    if cli_bin is None:
        print(
            "ERROR: Neither llama-cpp-python nor llama-cli found.\n"
            "  • Install llama-cpp-python: pip install llama-cpp-python\n"
            "  • Or set 'rpc_server_bin' in config so llama-cli can be found next to it."
        )
        sys.exit(1)

    return _run_llama_cli(cli_bin, model_path, prompt, n_tokens)


# ── Signing + RPC ─────────────────────────────────────────────────────────────

def _address_from_key(privkey_hex: str) -> str:
    from eth_account import Account
    key = privkey_hex if privkey_hex.startswith("0x") else "0x" + privkey_hex
    return Account.from_key(key).address.lower()


def _sign(privkey_hex: str, payload_bytes: bytes) -> str:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    key = privkey_hex if privkey_hex.startswith("0x") else "0x" + privkey_hex
    signed = Account.sign_message(encode_defunct(payload_bytes), private_key=key)
    return "0x" + signed.signature.hex()


def _rpc(url: str, method: str, params: list) -> object:
    import urllib.request
    body = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    req  = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read())
    if "error" in d:
        raise RuntimeError(d["error"])
    return d["result"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark this miner's hardware and publish the score on-chain."
    )
    parser.add_argument("--config",   default="test_local/config_miner1.json",
                        help="Path to miner JSON config file")
    parser.add_argument("--model",    default=None,
                        help=f"Model ID to benchmark (default: {DEFAULT_MODEL})")
    parser.add_argument("--n-tokens", type=int, default=DEFAULT_TOKENS,
                        help=f"Number of tokens to generate (default: {DEFAULT_TOKENS})")
    parser.add_argument("--prompt",   default=DEFAULT_PROMPT,
                        help="Prompt for the benchmark run")
    parser.add_argument("--rpc",      default=None,
                        help="Override the chain RPC URL from config")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run inference and print score without submitting to chain")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    cfg     = json.loads(config_path.read_text())
    privkey = cfg["private_key"]
    models  = cfg.get("models", {})
    rpc_url = args.rpc or cfg.get("l2_rpc_url")

    if not rpc_url:
        print("ERROR: no l2_rpc_url in config and --rpc not specified")
        sys.exit(1)

    # ── Choose model ──────────────────────────────────────────────────────────
    model_id = args.model or DEFAULT_MODEL
    if model_id not in models:
        if models:
            model_id = next(iter(models))
            print(f"Model '{args.model}' not in config — using '{model_id}'")
        else:
            print("ERROR: no models defined in config")
            sys.exit(1)

    model_path = os.path.expanduser(models[model_id])
    if not os.path.isfile(model_path):
        print(f"ERROR: model file not found: {model_path}")
        sys.exit(1)

    address = _address_from_key(privkey)

    print("=" * 60)
    print(f"  Miner     : {address}")
    print(f"  Model     : {model_id}")
    print(f"  File      : {model_path}")
    print(f"  Tokens    : {args.n_tokens}")
    print(f"  Chain RPC : {rpc_url}")
    print("=" * 60)
    print()

    # ── Run inference ─────────────────────────────────────────────────────────
    print("Running benchmark…")
    actual_tokens, elapsed_ms = run_inference(cfg, model_path, args.prompt, args.n_tokens)
    tokens_per_sec = round(actual_tokens / (elapsed_ms / 1000.0), 4)

    print()
    print("=" * 60)
    print(f"  Tokens generated : {actual_tokens}")
    print(f"  Elapsed          : {elapsed_ms} ms")
    print(f"  Tokens / sec     : {tokens_per_sec:.2f} t/s")
    print("=" * 60)
    print()

    if args.dry_run:
        print("Dry run — score NOT submitted to chain.")
        return

    # ── Sign and submit ───────────────────────────────────────────────────────
    nonce = hashlib.sha256(
        f"{address}{model_id}{time.time()}".encode()
    ).hexdigest()[:24]

    payload = json.dumps({
        "miner":          address,
        "model_id":       model_id,
        "tokens_per_sec": tokens_per_sec,
        "n_tokens":       actual_tokens,
        "elapsed_ms":     elapsed_ms,
        "nonce":          nonce,
    }, sort_keys=True, separators=(",", ":"))

    signature = _sign(privkey, payload.encode())

    print("Submitting score to chain…")
    try:
        result = _rpc(rpc_url, "inft_submitBenchmarkScore", [{
            "miner":          address,
            "model_id":       model_id,
            "tokens_per_sec": tokens_per_sec,
            "n_tokens":       actual_tokens,
            "elapsed_ms":     elapsed_ms,
            "nonce":          nonce,
            "signature":      signature,
        }])
        print()
        print("Score committed:")
        print(f"  miner          : {result.get('miner', address)}")
        print(f"  model_id       : {result.get('model_id', model_id)}")
        print(f"  tokens_per_sec : {result.get('tokens_per_sec', tokens_per_sec):.2f} t/s")
        print(f"  block_number   : {result.get('block_number', '?')}")
        print(f"  expires_block  : {result.get('expires_at_block', '?')}")
        print()
        print("Done. Score is now visible in the Miners explorer.")
    except Exception as exc:
        print(f"ERROR: failed to submit score: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
