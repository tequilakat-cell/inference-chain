"""
HuggingFace Model Crawler
=========================
Fetches popular text-generation models from the HuggingFace Hub API and
registers them in the InferenceToken contract.

Usage:
    python scripts/crawl_hf_models.py \\
        --deployment deployment.json \\
        --private-key 0xYOUR_KEY \\
        --rpc https://sepolia.base.org \\
        --limit 20

Requirements:
    pip install requests web3 eth-account

The script pages through HF's model search ranked by downloads, filters for
models that are likely runnable with llama.cpp (text-generation, non-gated),
and calls registerModel() for each one not already registered.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hf-crawler")

HF_API = "https://huggingface.co/api/models"

# Model IDs known to work well with llama.cpp / Ollama.
# The crawler will also discover new ones, but these are always seeded first.
SEED_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "01-ai/Yi-1.5-34B-Chat",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "NousResearch/Meta-Llama-3-8B-Instruct",
]


def fetch_hf_models(limit: int, min_downloads: int = 10_000) -> list[str]:
    """Fetch popular text-generation models from the HF Hub."""
    log.info("Fetching HF models (limit=%d, min_downloads=%d)...", limit, min_downloads)
    results = []
    page = 0
    page_size = 50

    while len(results) < limit:
        params = {
            "pipeline_tag": "text-generation",
            "sort": "downloads",
            "direction": "-1",
            "limit": page_size,
            "skip": page * page_size,
            "full": "false",
        }
        try:
            r = requests.get(HF_API, params=params, timeout=15)
            r.raise_for_status()
        except requests.RequestException as exc:
            log.error("HF API error: %s", exc)
            break

        models = r.json()
        if not models:
            break

        for m in models:
            model_id = m.get("id", "")
            downloads = m.get("downloads", 0)
            gated     = m.get("gated", False)
            private   = m.get("private", False)

            # Skip gated/private models and low-download ones
            if gated or private or downloads < min_downloads:
                continue

            # Skip known non-llama.cpp-friendly architectures
            tags = m.get("tags", [])
            if any(t in tags for t in ["diffusers", "stable-diffusion", "vae"]):
                continue

            results.append(model_id)
            if len(results) >= limit:
                break

        page += 1
        time.sleep(0.5)  # rate limit

    log.info("Fetched %d models from HF", len(results))
    return results


def register_models(
    models: list[str],
    contract,
    account,
    w3: Web3,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Register each model in the contract. Returns (registered, skipped)."""
    registered = skipped = 0

    for model_id in models:
        try:
            info = contract.functions.getModelInfo(model_id).call()
            already = info[2]  # exists field
        except Exception as exc:
            log.warning("Could not check %s: %s", model_id, exc)
            skipped += 1
            continue

        if already:
            log.debug("Already registered: %s", model_id)
            skipped += 1
            continue

        if dry_run:
            log.info("[DRY RUN] Would register: %s", model_id)
            registered += 1
            continue

        try:
            tx = contract.functions.registerModel(model_id, 0).build_transaction({
                "from":     account.address,
                "nonce":    w3.eth.get_transaction_count(account.address),
                "gas":      200_000,
                "gasPrice": w3.eth.gas_price,
            })
            signed  = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if receipt.status == 1:
                log.info("Registered: %s  tx=%s", model_id, tx_hash.hex()[:10] + "...")
                registered += 1
            else:
                log.error("Reverted for %s", model_id)
                skipped += 1

        except Exception as exc:
            log.error("Error registering %s: %s", model_id, exc)
            skipped += 1

        time.sleep(0.2)  # avoid nonce racing

    return registered, skipped


def main():
    parser = argparse.ArgumentParser(description="Crawl HuggingFace and register models on-chain")
    parser.add_argument("--deployment",   default="deployment.json",
                        help="Path to deployment.json from `npm run deploy`")
    parser.add_argument("--rpc",          default="",
                        help="Override RPC URL")
    parser.add_argument("--private-key",  required=True,
                        help="Ethereum private key (0x...)")
    parser.add_argument("--limit",        type=int, default=50,
                        help="Max number of HF models to consider (default: 50)")
    parser.add_argument("--min-downloads",type=int, default=10_000,
                        help="Minimum HF downloads to include (default: 10000)")
    parser.add_argument("--seed-only",    action="store_true",
                        help="Only register the curated seed list, skip HF crawl")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print what would be registered without sending transactions")
    args = parser.parse_args()

    dep_path = Path(args.deployment)
    if not dep_path.exists():
        print(f"deployment.json not found at {dep_path}")
        sys.exit(1)

    with open(dep_path) as f:
        dep = json.load(f)

    rpc_url = args.rpc or dep.get("rpc_url", "https://sepolia.base.org")
    w3      = Web3(Web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    account = Account.from_key(args.private_key)
    contract = w3.eth.contract(
        address=Web3.to_checksum_address(dep["address"]),
        abi=dep["abi"],
    )

    log.info("Connected to chain %s  account=%s", w3.eth.chain_id, account.address)
    log.info("Contract: %s", dep["address"])

    # Build model list
    models = list(SEED_MODELS)
    if not args.seed_only:
        hf_models = fetch_hf_models(limit=args.limit, min_downloads=args.min_downloads)
        # Merge without duplicates, preserving seed priority
        seen = set(models)
        for m in hf_models:
            if m not in seen:
                models.append(m)
                seen.add(m)

    log.info("Total models to process: %d", len(models))

    registered, skipped = register_models(
        models, contract, account, w3, dry_run=args.dry_run
    )

    log.info("Done — registered=%d  skipped=%d", registered, skipped)
    total = contract.functions.getModelCount().call()
    log.info("Total registered in contract: %d", total)


if __name__ == "__main__":
    main()
