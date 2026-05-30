"""
L2 state database.

Holds account balances, stakes, nonces, and a job registry.
Designed to be copy-on-write: apply_transaction() returns a new StateDB
rather than mutating the current one, making rollback trivial.

The state root is a merkle root over sorted (address → account_hash) pairs,
which lets the L1 rollup contract verify proofs against committed roots.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Optional

from .types import AccountState, JobState, TxType, JobStatus, ShardStatus, ShardResult
from .crypto import keccak256_hex, hash_dict, merkle_root

_CONTEXT_LOAD_MAX_PER_JOB = 32  # safety cap; one entry per miner shard

_HISTORY_MAX_PER_WALLET = 500  # sliding window cap


class StateError(Exception):
    pass


class StateDB:
    def __init__(self):
        # Core account ledger
        self._accounts: dict[str, AccountState] = {}  # address (lower) → AccountState

        # Job registry (in-flight and completed)
        self._jobs: dict[str, JobState] = {}           # job_id → JobState

        # Pending bridge deposits (deposit_nonce → dict)
        self._bridge_deposits: dict[int, dict] = {}

        # MINE token proof-of-work state
        from .mine import MineState
        self._mine = MineState()

        # Pending bridge withdrawals (l2_tx_hash → dict)
        self._bridge_withdrawals: dict[str, dict] = {}

        # Block number (updated by sequencer on each block application)
        self.block_number: int = 0

        # Jobs that transitioned WAITING → PENDING in the current block.
        # Drained by the sequencer after each block to trigger dispatch.
        self._newly_pending: list[str] = []

        # Model weight registry: address → model_id → (root_hex, leaf_count)
        # Populated by MODEL_REGISTER transactions.
        self._model_roots:       dict[str, dict[str, str]] = {}
        self._model_leaf_counts: dict[str, dict[str, int]] = {}

        # Per-wallet Q&A history — keyed by wallet.lower().
        # Each entry: {job_id, model_id, prompt_hash, output_hash, prompt, output, timestamp, block_number}
        # Capped at _HISTORY_MAX_PER_WALLET entries per wallet (sliding window).
        self._history: dict[str, list[dict]] = {}

        # Context load commits — keyed by job_id.
        # Each value is a list of {shard_index, miner, chunk_hash, cache_hit, latency_ms, block_number}.
        # Records which miners pre-loaded which context chunks before inference began.
        self._context_loads: dict[str, list[dict]] = {}

        # Miner benchmark scores — keyed by "{miner_addr}:{model_id}" (both lowercased).
        # Each value: {tokens_per_sec, n_tokens, elapsed_ms, block_number, expires_at_block, nonce}.
        # Written by BENCHMARK_COMMIT (sequencer-synthesized; miner cannot self-report).
        # Scores expire after benchmark_validity_blocks; expired entries return None from get_miner_score().
        self._miner_scores: dict[str, dict] = {}

    # ── Account helpers ───────────────────────────────────────────────────────

    def _key(self, address: str) -> str:
        return address.lower()

    def account(self, address: str) -> AccountState:
        return self._accounts.get(self._key(address), AccountState())

    def set_account(self, address: str, state: AccountState) -> None:
        self._accounts[self._key(address)] = state

    def balance(self, address: str) -> int:
        return self.account(address).balance_inft

    def stake(self, address: str) -> int:
        return self.account(address).stake_inft

    def nonce(self, address: str) -> int:
        return self.account(address).nonce

    def reputation(self, address: str) -> int:
        return self.account(address).reputation

    # ── Active validator set ──────────────────────────────────────────────────

    def active_validators(self) -> list[tuple[str, int]]:
        """Return (address, stake_inft) for all addresses with stake > 0."""
        return [
            (addr, acc.stake_inft)
            for addr, acc in self._accounts.items()
            if acc.stake_inft > 0
        ]

    # ── Benchmark score queries ───────────────────────────────────────────────

    def get_miner_score(self, addr: str, model_id: str) -> Optional[dict]:
        """Return the benchmark score for (addr, model_id), or None if absent/expired."""
        key = f"{addr.lower()}:{model_id}"
        entry = self._miner_scores.get(key)
        if entry is None:
            return None
        if entry.get("expires_at_block", 0) < self.block_number:
            return None  # expired; force re-benchmark
        return entry

    def all_miner_scores(self) -> list[dict]:
        """Return all non-expired benchmark scores with miner/model_id fields included."""
        result = []
        for key, entry in self._miner_scores.items():
            if entry.get("expires_at_block", 0) >= self.block_number:
                addr, model_id = key.split(":", 1)
                result.append({"miner": addr, "model_id": model_id, **entry})
        return result

    # ── Transaction application ───────────────────────────────────────────────

    def apply_transaction(self, tx) -> "StateDB":
        """
        Apply a single transaction and return a new StateDB.
        Raises StateError on invalid transactions.
        """
        new = self.copy()
        new._apply(tx)
        return new

    def apply_transactions(self, txs) -> "StateDB":
        """Apply a sequence of transactions and return the final state."""
        state = self
        for tx in txs:
            state = state.apply_transaction(tx)
        return state

    def _apply(self, tx) -> None:
        t = tx.tx_type
        p = tx.payload_dict()
        sender = tx.sender.lower() if tx.sender else ""
        acc = self.account(sender) if sender else AccountState()

        # Nonce check (skip for sequencer-synthesized txs that have no sender)
        if sender and tx.tx_type not in (TxType.BRIDGE_DEPOSIT, TxType.SLASH, TxType.SLASH_HARD, TxType.SHARD_COMMIT, TxType.HISTORY_COMMIT, TxType.BENCHMARK_COMMIT):
            if tx.nonce != acc.nonce:
                raise StateError(f"nonce mismatch: got {tx.nonce}, want {acc.nonce}")

        if t == TxType.TRANSFER:
            self._apply_transfer(sender, acc, p)

        elif t == TxType.STAKE:
            self._apply_stake(sender, acc, p)

        elif t == TxType.UNSTAKE:
            self._apply_unstake(sender, acc, p)

        elif t == TxType.JOB_POST:
            self._apply_job_post(sender, acc, p, tx)

        elif t == TxType.SHARD_COMMIT:
            self._apply_shard_commit(p)

        elif t == TxType.BRIDGE_DEPOSIT:
            self._apply_bridge_deposit(p)

        elif t == TxType.BRIDGE_WITHDRAW:
            self._apply_bridge_withdraw(sender, acc, p, tx)

        elif t in (TxType.SLASH, TxType.SLASH_HARD):
            self._apply_slash(p, hard=(t == TxType.SLASH_HARD))

        elif t == TxType.MINE_SUBMIT:
            self._apply_mine_submit(sender, p)

        elif t == TxType.MINE_BRIDGE:
            self._apply_mine_bridge(sender, acc, p, tx)

        elif t == TxType.MODEL_REGISTER:
            self._apply_model_register(sender, acc, p)

        elif t == TxType.HISTORY_COMMIT:
            self._apply_history_commit(p)

        elif t == TxType.CONTEXT_LOAD_COMMIT:
            self._apply_context_load_commit(p)

        elif t == TxType.BENCHMARK_COMMIT:
            self._apply_benchmark_commit(p)

        else:
            raise StateError(f"unknown tx_type: {t}")

    def _bump_nonce(self, sender: str, acc: AccountState) -> AccountState:
        return AccountState(
            balance_inft=acc.balance_inft,
            stake_inft=acc.stake_inft,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        )

    def _apply_transfer(self, sender: str, acc: AccountState, p: dict) -> None:
        to = p["to"].lower()
        amount = int(p["amount"])
        if acc.balance_inft < amount:
            raise StateError("insufficient balance")
        to_acc = self.account(to)
        self.set_account(sender, AccountState(
            balance_inft=acc.balance_inft - amount,
            stake_inft=acc.stake_inft,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        ))
        self.set_account(to, AccountState(
            balance_inft=to_acc.balance_inft + amount,
            stake_inft=to_acc.stake_inft,
            unlock_block=to_acc.unlock_block,
            nonce=to_acc.nonce,
            reputation=to_acc.reputation,
        ))

    def _apply_stake(self, sender: str, acc: AccountState, p: dict) -> None:
        amount = int(p["amount"])
        if acc.balance_inft < amount:
            raise StateError("insufficient balance to stake")
        self.set_account(sender, AccountState(
            balance_inft=acc.balance_inft - amount,
            stake_inft=acc.stake_inft + amount,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        ))

    def _apply_unstake(self, sender: str, acc: AccountState, p: dict) -> None:
        amount = int(p["amount"])
        if acc.stake_inft < amount:
            raise StateError("insufficient stake")
        unlock_block = self.block_number + 86_400  # 24 h at 1 s/block
        self.set_account(sender, AccountState(
            balance_inft=acc.balance_inft,
            stake_inft=acc.stake_inft - amount,
            unlock_block=unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        ))

    def _apply_job_post(self, sender: str, acc: AccountState, p: dict, tx) -> None:
        fee = int(p.get("fee_inft", 0))
        if acc.balance_inft < fee:
            raise StateError("insufficient balance for job fee")
        # Deduct fee immediately; refunded proportionally when shards complete
        self.set_account(sender, AccountState(
            balance_inft=acc.balance_inft - fee,
            stake_inft=acc.stake_inft,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        ))

        parent_job_id   = p.get("parent_job_id") or None
        prompt_template = p.get("prompt_template", "") or p.get("prompt", "")
        chain_step      = int(p.get("chain_step", 0))

        # Determine initial status: WAITING if a parent exists and hasn't completed yet.
        if parent_job_id:
            parent = self._jobs.get(parent_job_id)
            if parent is None or parent.status != JobStatus.COMPLETE:
                initial_status = JobStatus.WAITING
                resolved_prompt = ""
            else:
                # Parent already complete — resolve immediately and go straight to PENDING.
                initial_status  = JobStatus.PENDING
                resolved_prompt = prompt_template.replace("{prev_output}", parent.final_output or "")
                self._newly_pending.append(p["job_id"])
        else:
            initial_status  = JobStatus.PENDING
            resolved_prompt = prompt_template

        job = JobState(
            job_id=p["job_id"],
            requester=sender,
            model_id=p["model_id"],
            prompt=resolved_prompt,
            mode=p["shard_mode"],
            n_shards=int(p["n_shards"]),
            max_tokens=int(p["max_tokens"]),
            fee_inft=fee,
            block_number=self.block_number,
            deadline_ms=int(time.time() * 1000) + int(p.get("timeout_ms", 35_000)),
            status=initial_status,
            parent_job_id=parent_job_id,
            prompt_template=prompt_template,
            chain_step=chain_step,
            original_prompt=p.get("original_prompt", resolved_prompt),
            context_hash=p.get("context_hash") or None,
            context_entries=int(p.get("context_entries", 0)),
        )
        self._jobs[p["job_id"]] = job

    def _apply_shard_commit(self, p: dict) -> None:
        """Record a completed shard and pay the miner their fee share."""
        job_id = p["job_id"]
        shard_index = int(p["shard_index"])
        miner = p["miner"].lower()
        output_hash = p["output_hash"]

        job = self._jobs.get(job_id)
        if job is None:
            raise StateError(f"unknown job {job_id}")

        result = ShardResult(
            shard_index=shard_index,
            miner=miner,
            output=p.get("output_preview", ""),
            latency_ms=int(p.get("latency_ms", 0)),
            signature=p.get("miner_sig", ""),
        )
        job.results[shard_index] = result
        job.shard_status[shard_index] = ShardStatus.SUBMITTED

        # Pay miner their fee share
        fee_share = job.fee_inft // job.n_shards
        miner_acc = self.account(miner)
        self.set_account(miner, AccountState(
            balance_inft=miner_acc.balance_inft + fee_share,
            stake_inft=miner_acc.stake_inft,
            unlock_block=miner_acc.unlock_block,
            nonce=miner_acc.nonce,
            reputation=min(1000, miner_acc.reputation + 1),
        ))

        # Mark job complete if all shards are in
        if job.all_shards_in() or (job.mode == "parallel_sample" and len(job.results) >= 1):
            job.status = JobStatus.COMPLETE
            job.output_hash = output_hash
            # Cascade to any jobs waiting on this one
            self._cascade_dependents(job_id, output_hash)

    def _cascade_dependents(self, completed_job_id: str, assembled_output_hash: str) -> None:
        """Transition any WAITING jobs whose parent just completed to PENDING.

        Prompt resolution (replacing {prev_output} with actual text) happens at
        dispatch time in ShardProtocol.dispatch_chained_job, which has access to
        the full assembled output.  Here we only advance the status and record
        the parent's output hash so the sequencer knows which jobs to dispatch.
        """
        for dep_id, dep in self._jobs.items():
            if dep.parent_job_id == completed_job_id and dep.status == JobStatus.WAITING:
                dep.status = JobStatus.PENDING
                dep.parent_output_hash = assembled_output_hash
                self._newly_pending.append(dep_id)

    def pop_newly_pending(self) -> list[str]:
        """Return and clear the list of job IDs that became PENDING in this block."""
        result = self._newly_pending[:]
        self._newly_pending.clear()
        return result

    def _apply_bridge_deposit(self, p: dict) -> None:
        recipient = p["recipient"].lower()
        amount = int(p["amount"])
        nonce = int(p["l1_deposit_nonce"])
        acc = self.account(recipient)
        self.set_account(recipient, AccountState(
            balance_inft=acc.balance_inft + amount,
            stake_inft=acc.stake_inft,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce,
            reputation=acc.reputation,
        ))
        self._bridge_deposits[nonce] = p

    def _apply_bridge_withdraw(self, sender: str, acc: AccountState, p: dict, tx) -> None:
        amount = int(p["amount"])
        if acc.balance_inft < amount:
            raise StateError("insufficient balance for withdrawal")
        self.set_account(sender, AccountState(
            balance_inft=acc.balance_inft - amount,
            stake_inft=acc.stake_inft,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        ))
        self._bridge_withdrawals[tx.tx_hash] = {
            "sender": sender,
            "l1_recipient": p["recipient_l1"].lower(),
            "amount": amount,
            "block": self.block_number,
        }

    def _apply_slash(self, p: dict, hard: bool) -> None:
        miner = p["miner"].lower()
        acc = self.account(miner)
        pct = 0.30 if hard else 0.10
        slash_amount = int(acc.stake_inft * pct)
        slash_amount = min(slash_amount, acc.stake_inft)
        new_rep = max(0, acc.reputation - (30 if hard else 10))
        self.set_account(miner, AccountState(
            balance_inft=acc.balance_inft,
            stake_inft=acc.stake_inft - slash_amount,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce,
            reputation=new_rep,
        ))

    def _apply_model_register(self, sender: str, acc: AccountState, p: dict) -> None:
        model_id   = p.get("model_id", "")
        model_root = p.get("model_root", "")
        leaf_count = int(p.get("leaf_count", 0))

        if not model_id:
            raise StateError("model_id required")
        try:
            if not model_root.startswith("0x"):
                raise ValueError
            decoded = bytes.fromhex(model_root[2:])
            if len(decoded) != 32:
                raise ValueError
        except (ValueError, AttributeError):
            raise StateError("model_root must be a 0x-prefixed 32-byte hex string")
        if leaf_count < 1:
            raise StateError("leaf_count must be >= 1")

        addr = sender
        if addr not in self._model_roots:
            self._model_roots[addr]       = {}
            self._model_leaf_counts[addr] = {}
        self._model_roots[addr][model_id]       = model_root
        self._model_leaf_counts[addr][model_id] = leaf_count

        self.set_account(sender, AccountState(
            balance_inft=acc.balance_inft,
            stake_inft=acc.stake_inft,
            unlock_block=acc.unlock_block,
            nonce=acc.nonce + 1,
            reputation=acc.reputation,
        ))

    def assemble_context(
        self,
        wallet: str,
        model_id: str,
        char_budget: int = 2000,
    ) -> tuple[str, str, int]:
        """
        Build a context prefix from the wallet's history for model_id.

        Fills the budget prioritising the most-recent entries and presents them
        oldest-first so the model sees a natural conversation order.

        Returns (context_text, context_hash, n_entries).
        context_hash is ZERO_HASH when no history is available.
        """
        from .crypto import ZERO_HASH
        entries = self.get_history(wallet, model_id=model_id, limit=20)
        if not entries:
            return "", ZERO_HASH, 0

        # entries is newest-first; greedily pick from newest → oldest
        selected: list[tuple[str, str]] = []  # (job_id, formatted_block)
        budget = char_budget
        for entry in entries:
            block = f"User: {entry['prompt']}\nAssistant: {entry['output']}\n\n"
            if len(block) <= budget:
                selected.append((entry["job_id"], block))
                budget -= len(block)

        if not selected:
            return "", ZERO_HASH, 0

        # Reverse to chronological order (oldest first) for the model
        selected.reverse()
        context_text = "".join(b for _, b in selected)
        leaves = [keccak256_hex(jid.encode()) for jid, _ in selected]
        context_hash = merkle_root(leaves)
        return context_text, context_hash, len(selected)

    def _apply_history_commit(self, p: dict) -> None:
        wallet = p.get("wallet", "").lower()
        if not wallet:
            return
        entry = {
            "job_id":       p.get("job_id", ""),
            "model_id":     p.get("model_id", ""),
            "prompt_hash":  p.get("prompt_hash", ""),
            "output_hash":  p.get("output_hash", ""),
            "prompt":       p.get("prompt", ""),
            "output":       p.get("output", ""),
            "timestamp":    int(p.get("timestamp", 0)),
            "block_number": int(p.get("block_number", self.block_number)),
        }
        bucket = self._history.setdefault(wallet, [])
        if len(bucket) >= _HISTORY_MAX_PER_WALLET:
            del bucket[0]
        bucket.append(entry)

    def _apply_context_load_commit(self, p: dict) -> None:
        job_id = p.get("job_id", "")
        if not job_id:
            return
        entry = {
            "shard_index": int(p.get("shard_index", 0)),
            "miner":       p.get("miner", "").lower(),
            "chunk_hash":  p.get("chunk_hash", ""),
            "cache_hit":   bool(p.get("cache_hit", False)),
            "latency_ms":  int(p.get("latency_ms", 0)),
            "block_number":int(p.get("block_number", self.block_number)),
        }
        bucket = self._context_loads.setdefault(job_id, [])
        if len(bucket) < _CONTEXT_LOAD_MAX_PER_JOB:
            bucket.append(entry)

    def get_context_loads(self, job_id: str) -> list[dict]:
        """Return context load commit records for a job (one per miner shard)."""
        return list(self._context_loads.get(job_id, []))

    def get_history(self, wallet: str, model_id: Optional[str] = None, limit: int = 20) -> list[dict]:
        """Return history entries for wallet, newest first. Optional model_id filter."""
        bucket = self._history.get(wallet.lower(), [])
        if model_id:
            bucket = [e for e in bucket if e["model_id"] == model_id]
        return list(reversed(bucket[-limit * 2:]))[:limit]

    # ── Model registry queries ────────────────────────────────────────────────

    def get_model_root(self, address: str, model_id: str) -> Optional[str]:
        """Return the registered Merkle root for (address, model_id), or None."""
        return self._model_roots.get(address.lower(), {}).get(model_id)

    def get_model_leaf_count(self, address: str, model_id: str) -> int:
        return self._model_leaf_counts.get(address.lower(), {}).get(model_id, 0)

    def get_miner_models(self, address: str) -> list[str]:
        """Return all model IDs registered by address."""
        return list(self._model_roots.get(address.lower(), {}).keys())

    def miners_with_model(self, model_id: str) -> list[str]:
        """Return all addresses that have registered model_id."""
        return [
            addr for addr, models in self._model_roots.items()
            if model_id in models
        ]

    # ── State root ────────────────────────────────────────────────────────────

    def state_root(self) -> str:
        """Merkle root over sorted (address → account_hash) pairs."""
        if not self._accounts:
            from .crypto import ZERO_HASH
            return ZERO_HASH
        leaves = [
            keccak256_hex((addr + ":" + hash_dict(acc.to_dict())).encode())
            for addr, acc in sorted(self._accounts.items())
        ]
        return merkle_root(leaves)

    def shard_root(self) -> str:
        """Merkle root over all committed shard output hashes."""
        entries = []
        for job_id, job in sorted(self._jobs.items()):
            for idx, result in sorted(job.results.items()):
                key = f"{job_id}:{idx}:{result.miner}"
                entries.append(keccak256_hex(key.encode()))
        if not entries:
            from .crypto import ZERO_HASH
            return ZERO_HASH
        return merkle_root(entries)

    # ── Snapshot / restore ────────────────────────────────────────────────────

    def copy(self) -> "StateDB":
        new = StateDB()
        new._accounts            = copy.deepcopy(self._accounts)
        new._jobs                = copy.deepcopy(self._jobs)
        new._bridge_deposits     = copy.deepcopy(self._bridge_deposits)
        new._bridge_withdrawals  = copy.deepcopy(self._bridge_withdrawals)
        new._mine                = copy.deepcopy(self._mine)
        new._model_roots         = copy.deepcopy(self._model_roots)
        new._model_leaf_counts   = copy.deepcopy(self._model_leaf_counts)
        new._history             = copy.deepcopy(self._history)
        new._context_loads       = copy.deepcopy(self._context_loads)
        new._miner_scores        = copy.deepcopy(self._miner_scores)
        new.block_number         = self.block_number
        new._newly_pending       = self._newly_pending[:]
        return new

    def to_snapshot(self) -> dict:
        return {
            "block_number":      self.block_number,
            "accounts":          {addr: acc.to_dict() for addr, acc in self._accounts.items()},
            "bridge_deposits":   self._bridge_deposits,
            "bridge_withdrawals":self._bridge_withdrawals,
            "mine":              self._mine.to_dict(),
            "history":           self._history,
            "context_loads":     self._context_loads,
            "miner_scores":      self._miner_scores,
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "StateDB":
        from .mine import MineState
        db = cls()
        db.block_number = snap.get("block_number", 0)
        for addr, acc_dict in snap.get("accounts", {}).items():
            db._accounts[addr] = AccountState.from_dict(acc_dict)
        db._bridge_deposits    = snap.get("bridge_deposits", {})
        db._bridge_withdrawals = snap.get("bridge_withdrawals", {})
        if "mine" in snap:
            db._mine = MineState.from_dict(snap["mine"])
        if "history" in snap:
            db._history = snap["history"]
        if "context_loads" in snap:
            db._context_loads = snap["context_loads"]
        if "miner_scores" in snap:
            db._miner_scores = snap["miner_scores"]
        return db

    # ── MINE token ────────────────────────────────────────────────────────────

    def _apply_mine_submit(self, sender: str, p: dict) -> None:
        """Apply a PoW solution. Raises StateError on any validation failure."""
        nonce             = int(p["nonce"])
        challenge_digest  = p["challenge_digest"]
        block_number      = int(p.get("block_number", self.block_number))
        block_hash        = bytes.fromhex(p.get("block_hash", "00" * 32))
        try:
            reward = self._mine.apply_solution(
                sender, nonce, challenge_digest, block_number, block_hash
            )
        except ValueError as exc:
            raise StateError(str(exc))
        # Increment the miner's nonce (normal account bookkeeping)
        acc = self.account(sender)
        self.set_account(sender, AccountState(
            balance_inft  = acc.balance_inft,
            stake_inft    = acc.stake_inft,
            nonce         = acc.nonce + 1,
            reputation    = acc.reputation,
            unlock_block  = acc.unlock_block,
        ))

    def _apply_mine_bridge(self, sender: str, acc: AccountState, p: dict, tx) -> None:
        """Burn MINE on L2 to initiate a bridge to L1."""
        amount = int(p["amount"])
        l1_recipient = p.get("l1_recipient", sender)
        try:
            self._mine.bridge_burn(sender, amount)
        except ValueError as exc:
            raise StateError(str(exc))
        self.set_account(sender, AccountState(
            balance_inft = acc.balance_inft,
            stake_inft   = acc.stake_inft,
            nonce        = acc.nonce + 1,
            reputation   = acc.reputation,
            unlock_block = acc.unlock_block,
        ))
        # Record withdrawal for the bridge relayer
        withdrawal_id = tx.tx_hash if hasattr(tx, "tx_hash") else f"mine-{sender}-{amount}"
        self._bridge_withdrawals[withdrawal_id] = {
            "type":          "mine",
            "sender_l2":     sender,
            "recipient_l1":  l1_recipient,
            "amount":        amount,
            "tx_hash":       withdrawal_id,
            "block_number":  self.block_number,
        }

    # ── Benchmark score application ───────────────────────────────────────────

    def _apply_benchmark_commit(self, p: dict) -> None:
        """
        Record a benchmark score for a miner.
        Accepts both sequencer-measured (sender="") and self-reported (sender=miner) txs.
        """
        miner    = p["miner"].lower()
        model_id = p["model_id"]
        key      = f"{miner}:{model_id}"
        validity = int(p.get("validity_blocks", 5760))
        self._miner_scores[key] = {
            "miner":            miner,
            "model_id":         model_id,
            "tokens_per_sec":   float(p["tokens_per_sec"]),
            "n_tokens":         int(p["n_tokens"]),
            "elapsed_ms":       int(p["elapsed_ms"]),
            "block_number":     self.block_number,
            "expires_at_block": self.block_number + validity,
            "nonce":            p.get("nonce", ""),
            "self_reported":    bool(p.get("self_reported", False)),
        }

    # ── Convenience getters ───────────────────────────────────────────────────

    def job(self, job_id: str) -> Optional[JobState]:
        return self._jobs.get(job_id)

    def all_jobs(self) -> dict[str, JobState]:
        return self._jobs

    def pending_withdrawals(self) -> list[dict]:
        return list(self._bridge_withdrawals.values())
