"""
JSON-RPC method handlers.

Implements both eth_-compatible methods (for wallet compatibility)
and the inft_ namespace (InferenceChain-specific).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

log = logging.getLogger("chain.rpc.handlers")


class RPCHandlers:
    def __init__(self, sequencer, shard_protocol=None, benchmark=None, thought_store=None):
        self._seq           = sequencer
        self._shards        = shard_protocol
        self._benchmark     = benchmark
        self._thought_store = thought_store

    # ── eth_ compatibility ────────────────────────────────────────────────────

    async def eth_chainId(self, params) -> str:
        return hex(self._seq.chain_id)

    async def eth_blockNumber(self, params) -> str:
        return hex(self._seq.head().header.block_number)

    async def eth_getBalance(self, params) -> str:
        address = params[0] if params else "0x0"
        balance = self._seq.state().balance(address)
        return hex(balance)

    async def eth_getTransactionCount(self, params) -> str:
        address = params[0] if params else "0x0"
        nonce   = self._seq.state().nonce(address)
        return hex(nonce)

    async def eth_sendRawTransaction(self, params) -> str:
        raw = params[0] if params else ""
        try:
            tx_dict = json.loads(bytes.fromhex(raw.removeprefix("0x")).decode())
            from ..types import Transaction
            tx = Transaction.from_dict(tx_dict)
            ok, reason = await self._seq.submit_transaction(tx)
            if not ok:
                raise ValueError(reason)
            return tx.tx_hash
        except Exception as exc:
            raise ValueError(f"sendRawTransaction failed: {exc}") from exc

    async def eth_getBlockByNumber(self, params) -> Optional[dict]:
        num_hex = params[0] if params else "latest"
        head    = self._seq.head()
        if num_hex in ("latest", "pending"):
            block = head
        else:
            num = int(num_hex, 16)
            if num != head.header.block_number:
                return None
            block = head

        return {
            "number":     hex(block.header.block_number),
            "hash":       block.block_hash,
            "parentHash": block.header.parent_hash,
            "timestamp":  hex(block.header.timestamp),
            "miner":      block.header.sequencer,
            "stateRoot":  block.header.state_root,
            "gasUsed":    hex(block.header.gas_used),
            "transactions": [tx.tx_hash for tx in block.transactions],
        }

    # ── inft_ namespace ───────────────────────────────────────────────────────

    async def inft_postJobOpen(self, params) -> str:
        """
        Post an inference job without providing a private key.
        The sequencer signs the transaction using its own key.
        Intended for public demos and dashboards.

        params: [model_id, prompt, max_tokens, shard_mode=parallel_sample, n_shards=1]
        Returns: job_id
        """
        if len(params) < 2:
            raise ValueError("inft_postJobOpen requires [model_id, prompt, max_tokens?, shard_mode?, n_shards?]")

        model_id   = params[0]
        prompt     = params[1]
        max_tokens = int(params[2]) if len(params) > 2 else 128
        shard_mode = params[3] if len(params) > 3 else "parallel_sample"
        n_shards   = int(params[4]) if len(params) > 4 else 1

        # Sign with the sequencer's own private key (stored on the node)
        privkey = self._seq._privkey
        if not privkey:
            raise ValueError("Sequencer has no signing key configured")

        # Delegate to regular postJob using the sequencer key
        return await self.inft_postJob([model_id, prompt, max_tokens, shard_mode, n_shards, privkey])

    async def inft_postJobBatch(self, params) -> list:
        """
        Post multiple inference jobs in a single call with correct sequential nonces.
        All jobs land in the mempool immediately; no separate RPC call per job.

        params: [jobs_array] where each element is either:
          {model_id, prompt, max_tokens?, shard_mode?, n_shards?}   (dict)
          [model_id, prompt, max_tokens?, shard_mode?, n_shards?]   (list/tuple)

        Returns: [{job_id, ok}, ...] in input order.
        """
        if not params:
            raise ValueError("inft_postJobBatch requires [jobs_array]")

        jobs_input = params[0] if isinstance(params[0], list) else params
        privkey    = self._seq._privkey
        if not privkey:
            raise ValueError("Sequencer has no signing key configured")

        import uuid, json
        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key

        sender = address_from_key(privkey)

        # Start nonce from state + any already-queued pending txs
        base_nonce = self._seq.state().nonce(sender)
        pending    = await self._seq.mempool.pending_for(sender)
        nonce      = (max(pending) + 1) if pending else base_nonce

        results = []
        for job_spec in jobs_input:
            if isinstance(job_spec, dict):
                model_id   = job_spec.get("model_id", job_spec.get("model", ""))
                prompt     = job_spec.get("prompt", "")
                max_tokens = int(job_spec.get("max_tokens", 128))
                shard_mode = job_spec.get("shard_mode", "parallel_sample")
                n_shards   = int(job_spec.get("n_shards", 1))
            elif isinstance(job_spec, (list, tuple)):
                model_id   = job_spec[0] if len(job_spec) > 0 else ""
                prompt     = job_spec[1] if len(job_spec) > 1 else ""
                max_tokens = int(job_spec[2]) if len(job_spec) > 2 else 128
                shard_mode = job_spec[3] if len(job_spec) > 3 else "parallel_sample"
                n_shards   = int(job_spec[4]) if len(job_spec) > 4 else 1
            else:
                results.append({"job_id": None, "ok": False, "error": "invalid spec"})
                continue

            if not model_id or not prompt.strip():
                results.append({"job_id": None, "ok": False, "error": "model_id and prompt required"})
                continue

            try:
                job_id  = str(uuid.uuid4())
                fee     = n_shards * 10
                raw_prompt = prompt.strip()
                ctx_text, ctx_hash, ctx_entries = self._seq.state().assemble_context(sender, model_id)
                aug_prompt = ctx_text + raw_prompt if ctx_text else raw_prompt
                tx = build_transaction(
                    tx_type=TxType.JOB_POST,
                    sender=sender,
                    nonce=nonce,
                    payload={
                        "job_id":          job_id,
                        "model_id":        model_id,
                        "prompt":          aug_prompt,
                        "original_prompt": raw_prompt,
                        "max_tokens":      max_tokens,
                        "shard_mode":      shard_mode,
                        "n_shards":        n_shards,
                        "fee_inft":        fee,
                        "timeout_ms":      35_000,
                        "context_hash":    ctx_hash,
                        "context_entries": ctx_entries,
                        "context_text":    ctx_text,
                    },
                    chain_id=self._seq.chain_id,
                    private_key=privkey,
                    gas_price=1,
                )
                ok, reason = await self._seq.submit_transaction(tx)
                if ok:
                    nonce += 1   # only advance nonce on successful queue
                    results.append({"job_id": job_id, "ok": True})
                else:
                    results.append({"job_id": None, "ok": False, "error": reason})
            except Exception as exc:
                results.append({"job_id": None, "ok": False, "error": str(exc)})

        return results

    async def inft_postJob(self, params) -> str:
        """
        Post an inference job to the L2 mempool.
        params: [model_id, prompt, max_tokens, shard_mode, n_shards, private_key]
        Returns: job_id
        """
        if len(params) < 6:
            raise ValueError("inft_postJob requires [model_id, prompt, max_tokens, shard_mode, n_shards, private_key]")

        model_id, prompt, max_tokens, shard_mode, n_shards, privkey = params[:6]

        import uuid
        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key

        job_id  = str(uuid.uuid4())
        sender  = address_from_key(privkey)
        nonce   = self._seq.state().nonce(sender)

        # Assemble context prefix from wallet's history for this model
        ctx_text, ctx_hash, ctx_entries = self._seq.state().assemble_context(sender, model_id)
        augmented_prompt = ctx_text + prompt if ctx_text else prompt

        # Fee in raw INFT units (state stores whole INFT, not wei).
        # 10 INFT-units per shard; genesis initial_balances are in the same units.
        fee = int(n_shards) * 10

        tx = build_transaction(
            tx_type=TxType.JOB_POST,
            sender=sender,
            nonce=nonce,
            payload={
                "job_id":          job_id,
                "model_id":        model_id,
                "prompt":          augmented_prompt,
                "original_prompt": prompt,
                "max_tokens":      int(max_tokens),
                "shard_mode":      shard_mode,
                "n_shards":        int(n_shards),
                "fee_inft":        fee,
                "timeout_ms":      35_000,
                "context_hash":    ctx_hash,
                "context_entries": ctx_entries,
                "context_text":    ctx_text,   # passed through to shard protocol for context load phase
            },
            chain_id=self._seq.chain_id,
            private_key=privkey,
            gas_price=1,
        )

        ok, reason = await self._seq.submit_transaction(tx)
        if not ok:
            raise ValueError(f"postJob rejected: {reason}")

        return job_id

    async def inft_buildJobTx(self, params) -> dict:
        """
        Build an unsigned job transaction for wallet-based signing.
        params: [model_id, prompt, max_tokens, shard_mode, n_shards, sender]
        Returns: { job_id, preimage_hex, tx }
          preimage_hex — 0x-prefixed hex bytes to sign with personal_sign
          tx           — unsigned transaction dict (add signature before submitting)
        """
        if len(params) < 6:
            raise ValueError("inft_buildJobTx requires [model_id, prompt, max_tokens, shard_mode, n_shards, sender]")

        model_id, prompt, max_tokens, shard_mode, n_shards, sender = params[:6]

        import uuid
        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import encode_transaction

        job_id = str(uuid.uuid4())
        nonce  = self._seq.state().nonce(sender)
        fee    = int(n_shards) * 10

        ctx_text, ctx_hash, ctx_entries = self._seq.state().assemble_context(sender, model_id)
        augmented_prompt = ctx_text + prompt if ctx_text else prompt

        tx = build_transaction(
            tx_type=TxType.JOB_POST,
            sender=sender,
            nonce=nonce,
            payload={
                "job_id":          job_id,
                "model_id":        model_id,
                "prompt":          augmented_prompt,
                "original_prompt": prompt,
                "max_tokens":      int(max_tokens),
                "shard_mode":      shard_mode,
                "n_shards":        int(n_shards),
                "fee_inft":        fee,
                "timeout_ms":      35_000,
                "context_hash":    ctx_hash,
                "context_entries": ctx_entries,
                "context_text":    ctx_text,
            },
            chain_id=self._seq.chain_id,
            private_key=None,
            gas_price=1,
        )

        preimage = encode_transaction({
            **tx.to_dict(),
            "chain_id": self._seq.chain_id,
        })

        return {
            "job_id":       job_id,
            "preimage_hex": "0x" + preimage.hex(),
            "tx":           tx.to_dict(),
        }

    async def inft_postJobSigned(self, params) -> str:
        """
        Submit a wallet-signed job transaction.
        params: [tx_dict, signature]
          tx_dict   — unsigned tx dict from inft_buildJobTx
          signature — 0x-prefixed hex signature from personal_sign
        Returns: job_id
        """
        if len(params) < 2:
            raise ValueError("inft_postJobSigned requires [tx_dict, signature]")

        tx_dict, signature = params[0], params[1]

        from ..types import Transaction
        from ..crypto import verify_sig, encode_transaction

        tx = Transaction.from_dict({**tx_dict, "signature": signature})

        preimage = encode_transaction({
            **{k: v for k, v in tx.to_dict().items() if k not in ("signature", "tx_hash")},
            "chain_id": self._seq.chain_id,
        })
        if not verify_sig(preimage, signature, tx.sender):
            raise ValueError("Invalid signature: signer does not match sender")

        ok, reason = await self._seq.submit_transaction(tx)
        if not ok:
            raise ValueError(f"postJobSigned rejected: {reason}")

        payload = json.loads(tx.payload)
        return payload["job_id"]

    async def inft_postJobChain(self, params) -> list:
        """
        Post a sequence of dependent inference jobs in one call.

        Each step's output is injected as {prev_output} in the next step's
        prompt_template.  All JOB_POST transactions land in the mempool
        immediately; downstream steps start as WAITING in state and are
        dispatched automatically by the sequencer as each parent completes.

        params: [steps_array, private_key]
          steps_array — list of objects:
            { model_id, prompt_template, max_tokens?, shard_mode?, n_shards? }
          private_key — 0x-prefixed hex; funds all steps from one account

        Returns: [{step, job_id, ok, error?}, ...]
        """
        if len(params) < 2:
            raise ValueError(
                "inft_postJobChain requires [steps_array, private_key]"
            )

        steps   = params[0]
        privkey = params[1]

        import uuid
        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key

        sender = address_from_key(privkey)
        nonce  = self._seq.state().nonce(sender)
        pending = await self._seq.mempool.pending_for(sender)
        if pending:
            nonce = max(pending) + 1

        results      = []
        prev_job_id  = None

        for step_idx, step in enumerate(steps):
            if isinstance(step, (list, tuple)):
                model_id        = step[0] if len(step) > 0 else ""
                prompt_template = step[1] if len(step) > 1 else ""
                max_tokens      = int(step[2]) if len(step) > 2 else 128
                shard_mode      = step[3] if len(step) > 3 else "parallel_sample"
                n_shards        = int(step[4]) if len(step) > 4 else 1
            else:
                model_id        = step.get("model_id", step.get("model", ""))
                prompt_template = step.get("prompt_template", step.get("prompt", ""))
                max_tokens      = int(step.get("max_tokens", 128))
                shard_mode      = step.get("shard_mode", "parallel_sample")
                n_shards        = int(step.get("n_shards", 1))

            if not model_id or not prompt_template.strip():
                results.append({
                    "step": step_idx, "job_id": None, "ok": False,
                    "error": "model_id and prompt_template required",
                })
                prev_job_id = None
                continue

            job_id = str(uuid.uuid4())
            fee    = n_shards * 10

            # Inject context only for the first step; subsequent steps derive their
            # input from {prev_output} and don't need independent history context.
            if prev_job_id is None:
                ctx_text, ctx_hash, ctx_entries = self._seq.state().assemble_context(sender, model_id)
                raw_step0 = prompt_template.strip()
                step0_prompt = ctx_text + raw_step0 if ctx_text else raw_step0
            else:
                ctx_text, ctx_hash, ctx_entries = "", "", 0
                step0_prompt = ""

            payload: dict = {
                "job_id":          job_id,
                "model_id":        model_id,
                "prompt_template": prompt_template,
                # First step has no parent; subsequent steps have their prompt
                # resolved at dispatch time using {prev_output}.
                "prompt":          step0_prompt if prev_job_id is None else "",
                "original_prompt": prompt_template if prev_job_id is None else "",
                "max_tokens":      max_tokens,
                "shard_mode":      shard_mode,
                "n_shards":        n_shards,
                "fee_inft":        fee,
                "timeout_ms":      35_000,
                "chain_step":      step_idx,
                "context_hash":    ctx_hash,
                "context_entries": ctx_entries,
                "context_text":    ctx_text,
            }
            if prev_job_id is not None:
                payload["parent_job_id"] = prev_job_id

            try:
                tx = build_transaction(
                    tx_type=TxType.JOB_POST,
                    sender=sender,
                    nonce=nonce,
                    payload=payload,
                    chain_id=self._seq.chain_id,
                    private_key=privkey,
                    gas_price=1,
                )
                ok, reason = await self._seq.submit_transaction(tx)
                if ok:
                    nonce       += 1
                    prev_job_id  = job_id
                    results.append({"step": step_idx, "job_id": job_id, "ok": True})
                else:
                    results.append({
                        "step": step_idx, "job_id": None, "ok": False,
                        "error": reason,
                    })
                    prev_job_id = None   # break the chain on failure
            except Exception as exc:
                results.append({
                    "step": step_idx, "job_id": None, "ok": False,
                    "error": str(exc),
                })
                prev_job_id = None

        return results

    async def inft_awaitChain(self, params) -> list:
        """
        Long-poll until every job in a chain reaches a terminal state.

        params: [job_ids_array, timeout_seconds=300]
        Returns: [{job_id, step?, status, final_output, output_hash}, ...]
        """
        import asyncio
        job_ids = params[0] if params else []
        timeout = float(params[1]) if len(params) > 1 else 300.0
        elapsed = 0.0

        while elapsed < timeout:
            statuses = []
            all_done = True
            for job_id in job_ids:
                result = await self.inft_getJob([job_id])
                if result:
                    statuses.append(result)
                    if result.get("status") not in ("complete", "failed"):
                        all_done = False
                else:
                    all_done = False
                    statuses.append({"job_id": job_id, "status": "unknown"})
            if all_done:
                return statuses
            await asyncio.sleep(1.0)
            elapsed += 1.0

        raise TimeoutError(
            f"chain did not complete within {timeout}s — "
            f"last statuses: {[s.get('status') for s in statuses]}"
        )

    async def inft_getJob(self, params) -> Optional[dict]:
        """
        Get job status including per-shard state.
        params: [job_id]
        """
        job_id = params[0] if params else ""

        # First check the shard protocol (in-flight)
        if self._shards:
            job = self._shards.get_job(job_id)
            if job:
                return {
                    "job_id":           job.job_id,
                    "requester":        job.requester,
                    "model_id":         job.model_id,
                    "mode":             job.mode,
                    "n_shards":         job.n_shards,
                    "status":           job.status,
                    "shards":           {
                        str(i): {
                            "status":  job.shard_status.get(i, "unassigned"),
                            "miner":   job.specs.get(i, {}).assigned_miner if i in job.specs else None,
                            "output":  job.results[i].output if i in job.results else None,
                        }
                        for i in range(job.n_shards)
                    },
                    "final_output":       job.final_output,
                    "output_hash":        job.output_hash,
                    "parent_job_id":      job.parent_job_id,
                    "chain_step":         job.chain_step,
                    "parent_output_hash": job.parent_output_hash,
                    "context_entries":    job.context_entries,
                    "context_hash":       job.context_hash,
                    "original_prompt":    job.original_prompt,
                }

        # Fall back to confirmed state
        job = self._seq.state().job(job_id)
        if job:
            return {
                "job_id":             job.job_id,
                "status":             job.status,
                "final_output":       job.final_output,
                "output_hash":        job.output_hash,
                "parent_job_id":      job.parent_job_id,
                "chain_step":         job.chain_step,
                "parent_output_hash": job.parent_output_hash,
                "context_entries":    job.context_entries,
                "context_hash":       job.context_hash,
                "original_prompt":    job.original_prompt,
            }
        return None

    async def inft_awaitJob(self, params) -> Optional[dict]:
        """
        Long-poll until job completes or timeout.
        params: [job_id, timeout_seconds]
        """
        import asyncio
        job_id  = params[0] if params else ""
        timeout = float(params[1]) if len(params) > 1 else 120.0
        elapsed = 0.0

        while elapsed < timeout:
            result = await self.inft_getJob([job_id])
            if result and result.get("status") == "complete":
                return result
            await asyncio.sleep(0.5)
            elapsed += 0.5

        raise TimeoutError(f"job {job_id} did not complete within {timeout}s")

    async def inft_getHistory(self, params) -> list:
        """
        Return per-wallet inference history, newest first.
        params: [wallet, model_id?, limit?]
          wallet   — Ethereum address
          model_id — optional filter (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
          limit    — max entries to return (default 20, max 100)
        Returns: [{job_id, model_id, prompt, output, prompt_hash, output_hash, timestamp, block_number}, ...]
        """
        if not params:
            raise ValueError("inft_getHistory requires [wallet, model_id?, limit?]")
        wallet   = params[0]
        model_id = params[1] if len(params) > 1 else None
        limit    = min(int(params[2]) if len(params) > 2 else 20, 100)
        return self._seq.state().get_history(wallet, model_id=model_id, limit=limit)

    async def inft_getContextLoad(self, params) -> dict:
        """
        Return context load state for a job (Phase 3 introspection).
        params: [job_id]
        Returns: {miners, confirmed, all_cache_hit, context_hash, commits: [...]}
        """
        if not params:
            raise ValueError("inft_getContextLoad requires [job_id]")
        job_id = params[0]
        proto_state = {}
        if self._shards:
            proto_state = self._shards.get_context_load_state(job_id)
        on_chain = self._seq.state().get_context_loads(job_id)
        return {**proto_state, "commits": on_chain}

    async def inft_getAccount(self, params) -> dict:
        address = params[0] if params else ""
        acc     = self._seq.state().account(address)
        return {
            "address":      address,
            "balance_inft": acc.balance_inft,
            "stake_inft":   acc.stake_inft,
            "nonce":        acc.nonce,
            "reputation":   acc.reputation,
            "unlock_block": acc.unlock_block,
        }

    async def inft_getValidators(self, params) -> list:
        """Return all staked validators with full miner info. params: []"""
        state      = self._seq.state()
        validators = state.active_validators()
        active     = self._shards.active_jobs() if self._shards else []
        result     = []
        for addr, stake in validators:
            active_shards = 0
            for jid in active:
                job = self._shards.get_job(jid) if self._shards else None
                if job:
                    active_shards += sum(
                        1 for shard in job.specs.values()
                        if getattr(shard, "assigned_miner", None) and
                           shard.assigned_miner.lower() == addr.lower()
                    )
            acc = state.account(addr)
            result.append({
                "address":       addr,
                "stake_inft":    stake,
                "reputation":    acc.reputation,
                "balance_inft":  acc.balance_inft,
                "active_shards": active_shards,
                "models":        state.get_miner_models(addr),
                "unlock_block":  acc.unlock_block,
            })
        result.sort(key=lambda v: v["stake_inft"], reverse=True)
        return result

    async def inft_getActiveMiners(self, params) -> list:
        """
        Return all miners seen via P2P heartbeat in the last 5 minutes.
        Each entry: {address, backend, models, rpc_addr, last_seen, active, maxShards}

        This is the authoritative source for model discovery in the dashboard —
        it reflects what the sequencer actually knows from the P2P network.
        """
        if not self._shards:
            return []
        return self._shards.active_miners()

    async def inft_getMinerInfo(self, params) -> dict:
        address = params[0] if params else ""
        acc     = self._seq.state().account(address)
        active  = self._shards.active_jobs() if self._shards else []
        return {
            "address":      address,
            "stake_inft":   acc.stake_inft,
            "reputation":   acc.reputation,
            "active_shards": sum(
                1 for jid in active
                for shard in (self._shards.get_job(jid).specs.values() if self._shards.get_job(jid) else [])
                if shard.assigned_miner.lower() == address.lower()
            ),
        }

    async def inft_getChainInfo(self, params) -> dict:
        head = self._seq.head()
        return {
            "chain_id":          self._seq.chain_id,
            "block_number":      head.header.block_number,
            "block_hash":        head.block_hash,
            "state_root":        head.header.state_root,
            "tps":               round(self._seq.tps(), 2),
            "active_jobs":       len(self._shards.active_jobs()) if self._shards else 0,
            "validator_count":   len(self._seq.state().active_validators()),
            "sequencer_address": head.header.sequencer,
        }

    # ── Explorer methods ───────────────────────────────────────────────────────

    async def inft_getRecentBlocks(self, params) -> list:
        """Return last N blocks as summaries. params: [count=20]"""
        count = min(int(params[0]) if params else 20, 200)
        blocks = list(self._seq._block_history)[:count]
        result = []
        for b in blocks:
            result.append({
                "block_number": b.header.block_number,
                "block_hash":   b.block_hash,
                "parent_hash":  b.header.parent_hash,
                "timestamp":    b.header.timestamp,
                "sequencer":    b.header.sequencer,
                "tx_count":     len(b.transactions),
                "state_root":   b.header.state_root,
                "gas_used":     b.header.gas_used,
            })
        return result

    async def inft_getBlockDetail(self, params) -> Optional[dict]:
        """Return a full block with all transactions. params: [block_number]"""
        if not params:
            return None
        num = int(params[0])
        for b in self._seq._block_history:
            if b.header.block_number == num:
                return {
                    "block_number": b.header.block_number,
                    "block_hash":   b.block_hash,
                    "parent_hash":  b.header.parent_hash,
                    "timestamp":    b.header.timestamp,
                    "sequencer":    b.header.sequencer,
                    "state_root":   b.header.state_root,
                    "shard_root":   b.header.shard_root,
                    "tx_count":     len(b.transactions),
                    "gas_used":     b.header.gas_used,
                    "transactions": [tx.to_dict() for tx in b.transactions],
                }
        return None

    async def inft_getRecentJobs(self, params) -> list:
        """Return last N jobs from state, sorted newest first. params: [count=20]"""
        count = min(int(params[0]) if params else 20, 200)
        jobs  = self._seq.state().all_jobs()
        sorted_jobs = sorted(jobs.values(), key=lambda j: j.block_number, reverse=True)
        result = []
        for j in sorted_jobs[:count]:
            result.append({
                "job_id":          j.job_id,
                "model_id":        j.model_id,
                "mode":            j.mode,
                "n_shards":        j.n_shards,
                "requester":       j.requester,
                "status":          j.status,
                "block_number":    j.block_number,
                "fee_inft":        j.fee_inft,
                "final_output":    j.final_output,
                "output_hash":     j.output_hash,
                "shard_count":     len(j.results),
                "parent_job_id":   j.parent_job_id,
                "chain_step":      j.chain_step,
                "context_entries": j.context_entries,
                "context_hash":    j.context_hash,
            })
        return result

    async def inft_getRecentTransactions(self, params) -> list:
        """Return last N transactions across recent blocks. params: [count=50]"""
        count = min(int(params[0]) if params else 50, 500)
        txs = []
        for b in self._seq._block_history:
            for tx in b.transactions:
                txs.append({
                    "tx_hash":     tx.tx_hash,
                    "tx_type":     tx.tx_type,
                    "sender":      tx.sender,
                    "block_number":b.header.block_number,
                    "timestamp":   b.header.timestamp,
                    "gas_price":   tx.gas_price,
                    "payload_preview": tx.payload[:80] if tx.payload else "",
                })
                if len(txs) >= count:
                    return txs
        return txs

    async def inft_getTransaction(self, params) -> Optional[dict]:
        """Look up a transaction by hash. params: [tx_hash]"""
        if not params:
            return None
        tx_hash = params[0]
        for b in self._seq._block_history:
            for tx in b.transactions:
                if tx.tx_hash == tx_hash:
                    return {
                        **tx.to_dict(),
                        "block_number": b.header.block_number,
                        "block_hash":   b.block_hash,
                        "timestamp":    b.header.timestamp,
                        "payload_parsed": tx.payload_dict(),
                    }
        return None

    async def inft_getAddressHistory(self, params) -> dict:
        """Return account state + recent jobs/txs for an address. params: [address]"""
        if not params:
            return {}
        address = params[0].lower()
        acc     = self._seq.state().account(address)

        # Jobs where address is requester or miner
        all_jobs = self._seq.state().all_jobs()
        related_jobs = []
        for j in sorted(all_jobs.values(), key=lambda j: j.block_number, reverse=True):
            if j.requester.lower() == address or any(
                r.miner.lower() == address for r in j.results.values()
            ):
                related_jobs.append({
                    "job_id":   j.job_id,
                    "model_id": j.model_id,
                    "status":   j.status,
                    "role":     "requester" if j.requester.lower() == address else "miner",
                    "block_number": j.block_number,
                })

        # Recent txs from this address
        related_txs = []
        for b in self._seq._block_history:
            for tx in b.transactions:
                if tx.sender and tx.sender.lower() == address:
                    related_txs.append({
                        "tx_hash":     tx.tx_hash,
                        "tx_type":     tx.tx_type,
                        "block_number":b.header.block_number,
                        "timestamp":   b.header.timestamp,
                    })

        return {
            "address":       address,
            "balance_inft":  acc.balance_inft,
            "stake_inft":    acc.stake_inft,
            "nonce":         acc.nonce,
            "reputation":    acc.reputation,
            "unlock_block":  acc.unlock_block,
            "jobs":          related_jobs[:50],
            "transactions":  related_txs[:50],
        }

    async def inft_search(self, params) -> dict:
        """Universal search: block number, tx hash, job ID, or address. params: [query]"""
        query = (params[0] if params else "").strip()
        if not query:
            return {"type": "empty"}

        # Block number
        if query.isdigit():
            b = await self.inft_getBlockDetail([int(query)])
            if b:
                return {"type": "block", "data": b}

        # Job ID (UUID format)
        if "-" in query and len(query) > 8:
            job = await self.inft_getJob([query])
            if job:
                return {"type": "job", "data": job}

        # Transaction hash (0x + 64 hex)
        if query.startswith("0x") and len(query) == 66:
            tx = await self.inft_getTransaction([query])
            if tx:
                return {"type": "transaction", "data": tx}

        # Address (0x + 40 hex)
        if query.startswith("0x") and len(query) == 42:
            addr = await self.inft_getAddressHistory([query])
            return {"type": "address", "data": addr}

        return {"type": "not_found", "query": query}

    async def inft_getStats(self, params) -> dict:
        """Extended chain statistics for the explorer homepage."""
        state = self._seq.state()
        head  = self._seq.head()
        validators = state.active_validators()
        all_jobs   = state.all_jobs()

        total_jobs   = len(all_jobs)
        complete_jobs= sum(1 for j in all_jobs.values() if j.status == "complete")
        total_inft   = sum(acc.balance_inft + acc.stake_inft
                          for acc in state._accounts.values())

        return {
            "chain_id":         self._seq.chain_id,
            "block_number":     head.header.block_number,
            "block_hash":       head.block_hash,
            "tps":              round(self._seq.tps(), 2),
            "validator_count":  len(validators),
            "total_stake":      sum(s for _, s in validators),
            "total_jobs":       total_jobs,
            "complete_jobs":    complete_jobs,
            "active_jobs":      len(self._shards.active_jobs()) if self._shards else 0,
            "total_inft_supply":total_inft,
            "mempool_size":     await self._seq.mempool.size(),
        }

    async def inft_getPendingWithdrawals(self, params) -> list:
        """Return pending L2→L1 withdrawals (for bridge relayer)."""
        return self._seq.state().pending_withdrawals()

    # ── MINE token (proof-of-work) ─────────────────────────────────────────────

    async def mine_getInfo(self, params) -> dict:
        """Return current mining puzzle info — everything a miner needs to start hashing."""
        from ..mine import difficulty_display, MAX_SUPPLY, INITIAL_REWARD, HALVING_INTERVAL
        m = self._seq.state()._mine
        reward = m.current_reward()
        return {
            "challenge":        "0x" + m.challenge.hex(),
            "mining_target":    m.mining_target,
            "mining_target_hex":"0x" + m.mining_target.to_bytes(32, "big").hex(),
            "difficulty":       difficulty_display(m.mining_target),
            "current_reward":   reward,
            "current_reward_human": reward / 10**18,
            "solutions_found":  m.solutions_found,
            "total_minted":     m.total_minted,
            "total_minted_human": m.total_minted / 10**18,
            "max_supply":       MAX_SUPPLY,
            "epoch":            m.epoch_number(),
            "halvings_to_go":   (HALVING_INTERVAL - m.solutions_found % HALVING_INTERVAL),
            "last_reward_block":m.last_reward_block,
        }

    async def mine_getBalance(self, params) -> dict:
        """Return MINE balance for an address. params: [address]"""
        address = (params[0] if params else "").lower()
        m = self._seq.state()._mine
        return {
            "address": address,
            "balance": m.balances.get(address, 0),
            "balance_human": m.balances.get(address, 0) / 10**18,
        }

    async def mine_submit(self, params) -> str:
        """
        Submit a PoW solution. params: [nonce, challenge_digest, private_key]
        Returns the tx_hash of the submitted MINE_SUBMIT transaction.
        """
        if len(params) < 3:
            raise ValueError("mine_submit requires [nonce, challenge_digest, private_key]")
        nonce, challenge_digest, privkey = int(params[0]), params[1], params[2]

        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key

        sender = address_from_key(privkey)
        acc    = self._seq.state().account(sender)
        head   = self._seq.head()

        tx = build_transaction(
            tx_type = TxType.MINE_SUBMIT,
            sender  = sender,
            nonce   = acc.nonce,
            payload = {
                "nonce":            nonce,
                "challenge_digest": challenge_digest,
                "block_number":     head.header.block_number,
                "block_hash":       head.block_hash.lstrip("0x"),
            },
            chain_id    = self._seq.chain_id,
            private_key = privkey,
            gas_price   = 1,
        )
        ok, reason = await self._seq.submit_transaction(tx)
        if not ok:
            raise ValueError(f"mine_submit rejected: {reason}")
        return tx.tx_hash

    async def mine_bridgeWithdraw(self, params) -> str:
        """
        Burn MINE on L2 to receive wrapped MINE on L1.
        params: [amount_wei, l1_recipient_address, private_key]
        """
        if len(params) < 3:
            raise ValueError("mine_bridgeWithdraw requires [amount, l1_recipient, private_key]")
        amount, l1_recipient, privkey = int(params[0]), params[1], params[2]

        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key

        sender = address_from_key(privkey)
        acc    = self._seq.state().account(sender)

        tx = build_transaction(
            tx_type = TxType.MINE_BRIDGE,
            sender  = sender,
            nonce   = acc.nonce,
            payload = {"amount": amount, "l1_recipient": l1_recipient},
            chain_id    = self._seq.chain_id,
            private_key = privkey,
            gas_price   = 1,
        )
        ok, reason = await self._seq.submit_transaction(tx)
        if not ok:
            raise ValueError(reason)
        return tx.tx_hash

    async def inft_stake(self, params) -> str:
        """params: [amount_inft, private_key]"""
        if len(params) < 2:
            raise ValueError("inft_stake requires [amount, private_key]")
        amount, privkey = int(params[0]), params[1]
        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key
        sender = address_from_key(privkey)
        nonce  = self._seq.state().nonce(sender)
        tx = build_transaction(TxType.STAKE, sender, nonce,
                               {"amount": amount}, self._seq.chain_id, privkey)
        ok, reason = await self._seq.submit_transaction(tx)
        if not ok:
            raise ValueError(reason)
        return tx.tx_hash

    async def inft_registerModel(self, params) -> str:
        """
        Register a Merkle root committing the caller's model weights on-chain.

        params: [model_id, model_root, leaf_count, private_key]
          model_id   — HuggingFace model ID, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
          model_root — 0x-prefixed 32-byte Merkle root hex string
          leaf_count — number of leaves in the Merkle tree (>= 1)
          private_key — caller's hex private key

        Returns: tx_hash of the MODEL_REGISTER transaction.
        """
        if len(params) < 4:
            raise ValueError("inft_registerModel requires [model_id, model_root, leaf_count, private_key]")

        model_id, model_root, leaf_count, privkey = params[0], params[1], int(params[2]), params[3]

        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key

        sender = address_from_key(privkey)
        nonce  = self._seq.state().nonce(sender)
        tx = build_transaction(
            TxType.MODEL_REGISTER, sender, nonce,
            {"model_id": model_id, "model_root": model_root, "leaf_count": leaf_count},
            self._seq.chain_id, privkey,
        )
        ok, reason = await self._seq.submit_transaction(tx)
        if not ok:
            raise ValueError(f"registerModel rejected: {reason}")

        log.info("model_registered sender=%s model=%s root=%s leaves=%d",
                 sender[:10], model_id, model_root[:18], leaf_count)
        return tx.tx_hash

    async def inft_getModelRoot(self, params) -> dict:
        """
        Query a miner's registered model Merkle root.

        params: [address, model_id]
        Returns: {model_root, leaf_count, registered: bool, miner_models: [model_id, ...]}
        """
        if len(params) < 2:
            raise ValueError("inft_getModelRoot requires [address, model_id]")

        address, model_id = params[0], params[1]
        state = self._seq.state()

        root       = state.get_model_root(address, model_id)
        leaf_count = state.get_model_leaf_count(address, model_id)

        return {
            "address":      address,
            "model_id":     model_id,
            "model_root":   root,
            "leaf_count":   leaf_count,
            "registered":   root is not None,
            "miner_models": state.get_miner_models(address),
        }

    # ── Benchmark RPC methods ─────────────────────────────────────────────────

    async def inft_requestBenchmark(self, params) -> dict:
        """
        Trigger a benchmark for a miner.

        params: [miner_address, model_id, timeout_s (optional, default 120)]
        Returns: {miner, model_id, tokens_per_sec, n_tokens, elapsed_ms, nonce}

        The sequencer sends a nonce-seeded BenchmarkChallenge over P2P and
        measures wall-clock time to receive BenchmarkResponse.  The miner
        never self-reports its own time.
        """
        if len(params) < 2:
            raise ValueError("inft_requestBenchmark requires [miner_address, model_id]")

        miner_address = params[0]
        model_id      = params[1]
        timeout_s     = float(params[2]) if len(params) > 2 else 120.0

        if not self._benchmark:
            raise ValueError("BenchmarkRunner not configured on this node")

        try:
            score = await self._benchmark.request_benchmark(miner_address, model_id, timeout_s)
        except TimeoutError as exc:
            raise ValueError(str(exc)) from exc

        return score

    async def inft_getMinerScore(self, params) -> Optional[dict]:
        """
        Return the on-chain benchmark score for a miner/model pair.

        params: [miner_address, model_id]
        Returns: {tokens_per_sec, n_tokens, elapsed_ms, block_number, expires_at_block, nonce}
                 or null if no valid (non-expired) score exists.
        """
        if len(params) < 2:
            raise ValueError("inft_getMinerScore requires [miner_address, model_id]")
        return self._seq.state().get_miner_score(params[0], params[1])

    async def inft_getAllMinerScores(self, params) -> list:
        """
        Return all non-expired benchmark scores across all miners and models.

        params: []
        Returns: [{miner, model_id, tokens_per_sec, n_tokens, elapsed_ms,
                   block_number, expires_at_block, nonce}, ...]
        """
        return self._seq.state().all_miner_scores()

    async def inft_submitBenchmarkScore(self, params) -> dict:
        """
        Accept a self-reported benchmark score signed by the miner's private key.

        params: [{miner, model_id, tokens_per_sec, n_tokens, elapsed_ms, nonce, signature}]

        The sequencer verifies the ECDSA signature (miner cannot submit scores
        for another miner's address), then commits a BENCHMARK_COMMIT tx.

        Returns: {miner, model_id, tokens_per_sec, block_number, expires_at_block}
        """
        from ..crypto import verify_sig, keccak256_hex
        from ..types import TxType, Transaction

        if not params or not isinstance(params[0], dict):
            raise ValueError("inft_submitBenchmarkScore requires [{miner, model_id, ...}]")

        p = params[0]
        required = ("miner", "model_id", "tokens_per_sec", "n_tokens", "elapsed_ms", "nonce", "signature")
        for field in required:
            if field not in p:
                raise ValueError(f"missing field: {field}")

        miner    = p["miner"].lower()
        model_id = p["model_id"]
        sig      = p["signature"]

        # Reconstruct canonical payload and verify signature
        payload_obj = {
            "miner":          miner,
            "model_id":       model_id,
            "tokens_per_sec": float(p["tokens_per_sec"]),
            "n_tokens":       int(p["n_tokens"]),
            "elapsed_ms":     int(p["elapsed_ms"]),
            "nonce":          p["nonce"],
        }
        payload_bytes = json.dumps(payload_obj, sort_keys=True, separators=(",", ":")).encode()

        if not verify_sig(payload_bytes, sig, miner):
            raise ValueError(f"signature verification failed for miner {miner[:10]}")

        # Build and queue BENCHMARK_COMMIT tx (sender = miner, marks as self-reported)
        validity = 5760  # ~1 day at 1 block/s
        state    = self._seq.state()
        current_block = state.block_number

        score_payload = json.dumps({
            "miner":           miner,
            "model_id":        model_id,
            "tokens_per_sec":  float(p["tokens_per_sec"]),
            "n_tokens":        int(p["n_tokens"]),
            "elapsed_ms":      int(p["elapsed_ms"]),
            "nonce":           p["nonce"],
            "validity_blocks": validity,
            "self_reported":   True,
        }, separators=(",", ":"))

        tx_hash = keccak256_hex(
            ("BENCHMARK_COMMIT" + miner + model_id + p["nonce"]).encode()
        )
        tx = Transaction(
            tx_type=TxType.BENCHMARK_COMMIT,
            sender=miner,
            nonce=0,
            payload=score_payload,
            gas_price=0,
            signature=sig,
            tx_hash=tx_hash,
        )
        await self._seq.mempool.add(tx)

        log.info(
            "benchmark_self_reported miner=%s model=%s tps=%.2f",
            miner[:10], model_id, float(p["tokens_per_sec"]),
        )

        return {
            "miner":           miner,
            "model_id":        model_id,
            "tokens_per_sec":  float(p["tokens_per_sec"]),
            "n_tokens":        int(p["n_tokens"]),
            "elapsed_ms":      int(p["elapsed_ms"]),
            "block_number":    current_block,
            "expires_at_block": current_block + validity,
            "self_reported":   True,
        }

    async def inft_searchThoughts(self, params) -> list:
        """
        Search the distributed thought store (pg_inft) for inference history.

        params: [question, model_id (optional, "" = any), limit (optional, default 10)]
        Returns: [{id, job_id, miner_address, model_id, question_text,
                   thinking_text, answer_text, score}, ...]
        """
        if not params:
            raise ValueError("inft_searchThoughts requires [question, ...]")
        question = str(params[0])
        model_id = str(params[1]) if len(params) > 1 else ""
        limit    = int(params[2]) if len(params) > 2 else 10

        if self._thought_store is None:
            return []
        results = await self._thought_store.search(question, model_id, limit)
        return [
            {
                "id":            r.id,
                "job_id":        r.job_id,
                "miner_address": r.miner_address,
                "model_id":      r.model_id,
                "question_text": r.question_text,
                "thinking_text": r.thinking_text,
                "answer_text":   r.answer_text,
                "score":         r.score,
            }
            for r in results
        ]

    async def inft_getRecentThoughts(self, params) -> list:
        """
        Return the most recent inference records from the distributed thought store.

        params: [limit (optional, default 20), model_id (optional, "" = any)]
        Returns: same shape as inft_searchThoughts.
        """
        limit    = int(params[0]) if params else 20
        model_id = str(params[1]) if len(params) > 1 else ""

        if self._thought_store is None:
            return []
        # Recency view: read the log directly. An empty full-text query matches no
        # lexemes and returns nothing, so search("") cannot back the "recent" list.
        results = await self._thought_store.recent(model_id, limit)
        return [
            {
                "id":            r.id,
                "job_id":        r.job_id,
                "miner_address": r.miner_address,
                "model_id":      r.model_id,
                "question_text": r.question_text,
                "thinking_text": r.thinking_text,
                "answer_text":   r.answer_text,
                "score":         r.score,
            }
            for r in results
        ]

    # ── Memory rollup (semantic consolidation) ─────────────────────────────────

    async def inft_rollupMemory(self, params) -> dict:
        """
        Consolidate a cluster of semantically-similar thoughts into one distilled
        memory using the chain's own inference (map-reduce across miners), then
        store it in inft_rollups so future inferences can inject it.

        params: [query_embedding(list[768]), model_id="", k=30, n_shards=3, topic="", timeout_s=180]
        The CALLER supplies the 768-dim query embedding (the sequencer has no
        embed model). Returns: {rollup_id, topic, summary, source_count, source_job_ids}.
        """
        import uuid, asyncio
        from ..mempool import build_transaction
        from ..types import TxType
        from ..crypto import address_from_key, keccak256_hex

        if not params or not isinstance(params[0], list):
            raise ValueError("inft_rollupMemory requires [query_embedding, model_id?, k?, n_shards?, topic?]")
        query_emb = params[0]
        model_id  = str(params[1]) if len(params) > 1 else ""
        k         = int(params[2]) if len(params) > 2 else 30
        n_shards  = int(params[3]) if len(params) > 3 else 3
        topic     = str(params[4]) if len(params) > 4 else ""
        timeout_s = float(params[5]) if len(params) > 5 else 180.0

        if self._thought_store is None:
            raise ValueError("memory store (pg_inft) not configured on this node")
        if len(query_emb) != 768:
            raise ValueError(f"query_embedding must be 768-dim, got {len(query_emb)}")

        # 1. Semantic cluster: nearest thoughts to the query embedding.
        hits = await self._thought_store.search_semantic(query_emb, model_id, k)
        hits = [h for h in hits if (h.answer_text or "").strip()]
        if not hits:
            raise ValueError("no embedded thoughts match the query (run jobs first, or embeddings missing)")

        corpus = "\n\n".join(f"Q: {h.question_text}\nA: {h.answer_text}" for h in hits)
        source_job_ids = [h.job_id for h in hits]

        # Clustering may span models (model_id=""), but the summarization jobs need a
        # concrete model that staked, benchmarked miners actually run — fall back to
        # the top hit's model so the map/reduce jobs pass the liveness/benchmark gates.
        infer_model = model_id or hits[0].model_id
        if not infer_model:
            raise ValueError("could not determine a model for summarization; pass model_id")

        # Cap shard count to the number of live miners so context_split does not
        # drop chunks (a chunk per miner; extra chunks would go unassigned).
        live = len(self._shards.active_miners()) if self._shards else 1
        n_shards = max(1, min(n_shards, live or 1))

        privkey = self._seq._privkey
        if not privkey:
            raise ValueError("sequencer has no signing key configured")
        sender = address_from_key(privkey)
        nonce  = self._seq.state().nonce(sender)
        pending = await self._seq.mempool.pending_for(sender)
        if pending:
            nonce = max(pending) + 1

        map_job_id = str(uuid.uuid4())
        map_prompt = (
            "Summarize the key facts and answers in the following Q&A pairs as "
            "concise bullet points. Do not repeat the questions:\n\n" + corpus
        )
        map_mode = "context_split" if n_shards > 1 else "parallel_sample"

        map_tx = build_transaction(
            tx_type=TxType.JOB_POST, sender=sender, nonce=nonce,
            payload={
                "job_id": map_job_id, "model_id": infer_model, "prompt": map_prompt,
                "original_prompt": map_prompt, "max_tokens": 256,
                "shard_mode": map_mode, "n_shards": n_shards,
                "fee_inft": n_shards * 10, "timeout_ms": 60_000,
            },
            chain_id=self._seq.chain_id, private_key=privkey, gas_price=1,
        )
        ok, reason = await self._seq.submit_transaction(map_tx)
        if not ok:
            raise ValueError(f"rollup map job rejected: {reason}")

        # With multiple shards, the map produces concatenated partial summaries;
        # a chained REDUCE job merges them into one. With a single shard the map
        # already summarized the whole corpus, so skip the reduce.
        await_job_id = map_job_id
        if n_shards > 1:
            reduce_job_id = str(uuid.uuid4())
            reduce_tpl = (
                "Combine the following notes into one consolidated, non-redundant "
                "summary:\n\n{prev_output}"
            )
            reduce_tx = build_transaction(
                tx_type=TxType.JOB_POST, sender=sender, nonce=nonce + 1,
                payload={
                    "job_id": reduce_job_id, "model_id": infer_model,
                    "prompt_template": reduce_tpl, "prompt": "", "original_prompt": "",
                    "max_tokens": 320, "shard_mode": "parallel_sample", "n_shards": 1,
                    "fee_inft": 10, "timeout_ms": 60_000,
                    "parent_job_id": map_job_id, "chain_step": 1,
                },
                chain_id=self._seq.chain_id, private_key=privkey, gas_price=1,
            )
            ok, reason = await self._seq.submit_transaction(reduce_tx)
            if not ok:
                raise ValueError(f"rollup reduce job rejected: {reason}")
            await_job_id = reduce_job_id

        # Await the final job.
        elapsed, summary = 0.0, None
        while elapsed < timeout_s:
            r = await self.inft_getJob([await_job_id])
            if r and r.get("status") == "complete":
                summary = (r.get("final_output") or "").strip()
                break
            if r and r.get("status") == "failed":
                raise ValueError("rollup job failed (a summarization shard timed out)")
            await asyncio.sleep(1.0)
            elapsed += 1.0
        if not summary:
            raise TimeoutError(f"rollup did not complete within {timeout_s}s")

        rollup_id = await_job_id
        chash = keccak256_hex(summary.encode())
        await self._thought_store.upsert_rollup(
            rollup_id=rollup_id, topic=(topic or hits[0].question_text[:80]),
            model_id=model_id, summary=summary, source_count=len(hits),
            source_job_ids=source_job_ids, embedding=query_emb,
            content_hash=bytes.fromhex(chash[2:]),
        )
        # Distribute the rollup to all miners so every replica's pg_inft has it and
        # can inject it at inference time (memory "distributed across miners").
        if self._shards is not None and getattr(self._shards, "_p2p", None) is not None:
            try:
                await self._shards._p2p.broadcast("rollup_broadcast", {
                    "rollup_id":      rollup_id,
                    "topic":          topic or hits[0].question_text[:80],
                    "model_id":       infer_model,
                    "summary":        summary,
                    "source_count":   len(hits),
                    "source_job_ids": source_job_ids,
                    "embedding":      query_emb,
                })
            except Exception as exc:
                log.debug("rollup_broadcast_err id=%s err=%s", rollup_id[:12], exc)

        log.info(
            "rollup_created id=%s sources=%d shards=%d summary_chars=%d",
            rollup_id[:12], len(hits), n_shards, len(summary),
        )
        return {
            "rollup_id":      rollup_id,
            "topic":          topic or hits[0].question_text[:80],
            "summary":        summary,
            "source_count":   len(hits),
            "source_job_ids": source_job_ids,
        }

    async def inft_getRollups(self, params) -> list:
        """List recent consolidated rollup memories. params: [model_id="", limit=20]"""
        if self._thought_store is None:
            return []
        model_id = str(params[0]) if params else ""
        limit    = int(params[1]) if len(params) > 1 else 20
        return await self._thought_store.list_rollups(model_id, limit)
