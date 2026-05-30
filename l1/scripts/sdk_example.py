"""
SDK usage example — run after deploying the contract locally.

1. Terminal 1: cd inference && npm run node
2. Terminal 2: npm run deploy:local
3. Terminal 3: cd miner && python miner.py --config config.json
4. Terminal 4: python scripts/sdk_example.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sdk"))

from inference_sdk import InferenceClient

# Hardhat local account #2 (for testing)
PRIVATE_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
RPC_URL     = "http://127.0.0.1:8545"

client = InferenceClient.from_deployment(
    deployment_json="deployment.json",
    private_key=PRIVATE_KEY,
    rpc_url=RPC_URL,
)

print("=== Token stats ===")
stats = client.token_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n=== Available models ===")
models = client.list_models()
for m in models:
    print(" ", m)

if not models:
    print("  (no models registered — deploy first)")
    sys.exit(0)

print("\n=== Posting inference job ===")
try:
    response = client.infer(
        model=models[0],
        prompt="Explain what a transformer neural network is in 3 sentences.",
        max_tokens=512,
        timeout=900,
    )
    print(f"\nJob #{response.job_id} complete in {response.elapsed_sec:.1f}s")
    print(f"Miner earned: {response.tokens_minted} INFT")
    print(f"\nOutput:\n{response.text}")
except TimeoutError as e:
    print(f"Timeout: {e}")
except Exception as e:
    print(f"Error: {e}")
