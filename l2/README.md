# InferenceChain

A purpose-built L2 blockchain for parallelised AI inference, settling on Ethereum Sepolia.

## What makes this different from the L1

| | L1 InferenceToken | InferenceChain L2 |
|---|---|---|
| Block time | ~12s (Ethereum) | **1 second** |
| Finality | 10-min challenge window | **~1 second** (L2), 7 days (L1 settlement) |
| Miners per job | 1 | **1–8 in parallel** |
| Speedup | baseline | **2–5× (parallel_sample), ~4× (speculative)** |
| Settlement | direct on-chain | optimistic rollup → Sepolia |

## Parallel shard modes

### `parallel_sample` — race mode
N miners all run the full prompt simultaneously. The first valid result wins. Eliminates worst-case latency caused by a slow or stalled miner. Best for short, latency-sensitive tasks.

### `context_split` — map mode
The prompt is divided into N chunks. Each miner handles one chunk and returns a partial result. The sequencer concatenates the outputs in order. Best for long-document summarisation, RAG over many sources.

### `speculative` — draft + verify mode
Miner 0 uses a small, fast model (draft miner) to generate tokens quickly. Miner 1 runs the full target model but only needs a single forward pass to accept/reject the draft — 3–4× faster than autoregressive generation alone. Best for high-quality generation where a large model is required.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        InferenceChain L2                         │
│                                                                   │
│  ┌──────────────┐  1s blocks   ┌──────────────────────────────┐ │
│  │  Sequencer   │─────────────▶│  Block (txs + state root)    │ │
│  │              │              └──────────────────────────────┘ │
│  │  • Mempool   │  VRF assign  ┌──────────────────────────────┐ │
│  │  • ShardProto│─────────────▶│  ShardOffer (P2P gossip)     │ │
│  │  • RPC :8545 │              └──────────────────────────────┘ │
│  └──────────────┘                          │                     │
│                                            ▼                     │
│  ┌──────────────┐  inference   ┌──────────────────────────────┐ │
│  │  L2 Miner 1  │─────────────▶│  ShardResult (P2P gossip)    │ │
│  │  L2 Miner 2  │─────────────▶│  (parallel, N miners)        │ │
│  │  L2 Miner N  │─────────────▶│                              │ │
│  └──────────────┘              └──────────────────────────────┘ │
│                                            │                     │
│  ┌──────────────────────────────────────── ▼ ─────────────────┐ │
│  │  Assembler  →  TX_SHARD_COMMIT  →  mempool  →  block       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Every 100 blocks: RollupPoster → L1 InferenceChainRollup.sol    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Ethereum Sepolia (L1)                          │
│                                                                   │
│  InferenceChainRollup.sol    — state root commitments            │
│  InferenceChainBridge.sol    — INFT lock/release                 │
│  InferenceToken.sol          — existing L1 INFT token            │
└─────────────────────────────────────────────────────────────────┘
```

## Quick start

### 1. Deploy L1 contracts (Sepolia)

```bash
cd inference_chain
cp /home/trinity/inference/.env .env   # reuse existing keys
npm install
npm run deploy:sepolia
# → writes l1_deployment.json
```

### 2. Configure genesis

Edit `genesis.json`:
```json
{
  "sequencer_address": "0xYOUR_ADDRESS",
  "rollup_l1_address":  "0x...",   // from l1_deployment.json
  "bridge_l1_address":  "0x...",   // from l1_deployment.json
  "initial_validators": [
    {"address": "0xYOUR_ADDRESS", "stake_inft": 1000}
  ],
  "initial_balances": [
    {"address": "0xYOUR_ADDRESS", "balance_inft": 10000}
  ]
}
```

### 3. Start the sequencer

```bash
pip install -e ".[miner]"
SEQUENCER_PRIVATE_KEY=0x... python -m chain.sequencer --genesis genesis.json
```

### 4. Start an L2 miner

```bash
# Extend the existing miner config
cp ../inference/miner/config.json config_l2.json
# Add L2 fields: l2_rpc_url, bootstrap_peers, p2p_port, min_stake_inft
python -m miner.l2_miner --config config_l2.json
```

### 5. Post a parallel inference job

```python
from sdk.client import InferenceChainClient

client = InferenceChainClient(
    rpc_url="http://127.0.0.1:8545",
    private_key="0x...",
)

# 3 miners in parallel — returns the fastest result
result = client.infer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    prompt="Explain neural attention in one sentence.",
    n_shards=3,
    shard_mode="parallel_sample",
)
print(result.output)
print(f"Completed in {result.elapsed_sec:.1f}s using {result.shard_count} miners")
```

## File structure

```
inference_chain/
├── chain/
│   ├── types.py           ← All data structures (Block, Transaction, ShardSpec…)
│   ├── crypto.py          ← keccak256, merkle tree, ECDSA helpers
│   ├── state.py           ← L2 account + job state machine
│   ├── genesis.py         ← Genesis block construction
│   ├── mempool.py         ← Priority transaction queue
│   ├── block_builder.py   ← Assembles blocks from mempool
│   ├── block_validator.py ← Structural block validation
│   ├── sequencer.py       ← Main block production loop
│   ├── rollup_poster.py   ← Commits state roots to L1 every 100 blocks
│   ├── shard/
│   │   ├── vrf.py         ← Deterministic stake-weighted miner selection
│   │   ├── protocol.py    ← THE CORE: parallel shard orchestration
│   │   ├── assembler.py   ← Mode-dispatch result assembly
│   │   ├── slash.py       ← Non-responsive miner slashing
│   │   └── modes/
│   │       ├── parallel_sample.py  ← Race mode
│   │       ├── context_split.py    ← Chunk mode
│   │       └── speculative.py      ← Draft + verify mode
│   ├── p2p/
│   │   ├── node.py        ← WebSocket gossip node
│   │   ├── messages.py    ← Signed message envelopes
│   │   └── discovery.py   ← Bootstrap peer management
│   └── rpc/
│       ├── server.py      ← JSON-RPC 2.0 server (eth_ + inft_ namespace)
│       └── handlers.py    ← RPC method implementations
├── miner/
│   ├── l2_miner.py        ← Extends L1 Miner with shard execution
│   ├── shard_worker.py    ← Single-shard inference + signing
│   ├── stake_manager.py   ← L2 INFT stake management
│   └── config.py          ← L2MinerConfig
├── bridge/
│   ├── watcher.py         ← L1 deposit → L2 mint
│   └── relayer.py         ← L2 withdrawal → L1 release
├── contracts/
│   ├── InferenceChainRollup.sol  ← State root commitments + fraud proofs
│   └── InferenceChainBridge.sol  ← INFT lock/release bridge
├── sdk/client.py          ← Python SDK for application developers
├── genesis.json           ← Chain configuration
└── docker-compose.yml     ← Full-stack deployment
```

## Key design decisions

**VRF uses parent block hash** — shard assignments are deterministic and verifiable but unpredictable before the block is produced, preventing sequencer manipulation.

**Single sequencer (v1)** — the sequencer can censor but cannot steal funds (fraud proofs on L1 protect this). Decentralized sequencer rotation is v2.

**Data availability** — full block data lives on L2 nodes. L1 receives only state roots. Any full node can reconstruct history.

**L1 miners continue working** — L2 miners extend L1 miners. Running `L2Miner` keeps both loops active: L1 `finalizeJob` calls for L1 INFT, and L2 shard execution for L2 INFT.
