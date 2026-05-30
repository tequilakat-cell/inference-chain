"""
ShardWorker — executes one assigned shard end-to-end.

Fetch prompt slice from offer → decrypt if needed → run inference via GPU pool
→ sign result → return ShardResult for P2P broadcast.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from chain.types import ShardResult
from chain.crypto import keccak256_hex, sign as crypto_sign

log = logging.getLogger("l2_miner.shard_worker")


class ShardWorker:
    def __init__(
        self,
        models,                    # GPUModelPool from L1 miner
        privkey: str,
        priv_pem: Optional[bytes] = None,
        encryption_enabled: bool  = False,
    ):
        self._models             = models
        self._privkey            = privkey
        self._priv_pem           = priv_pem
        self._encryption_enabled = encryption_enabled

    async def run(self, offer: dict) -> Optional[ShardResult]:
        """
        Execute the shard described in offer.
        Returns ShardResult on success, None on failure.
        """
        spec      = offer.get("spec", {})
        job_id    = offer.get("job_id", "")
        model_id  = offer.get("model_id", "")
        shard_idx = int(spec.get("shard_index", 0))
        max_tok   = int(spec.get("max_tokens", 256))
        prompt_slice = spec.get("prompt_slice", "")

        # Decrypt if needed (uses the same RSA key machinery as L1 miner)
        if self._encryption_enabled and self._priv_pem:
            try:
                import keys as keys_mod
                prompt_bytes = bytes.fromhex(prompt_slice) if prompt_slice.startswith("0x") \
                               else prompt_slice.encode("utf-8")
                prompt_slice = keys_mod.decrypt(self._priv_pem, prompt_bytes).decode("utf-8")
            except Exception as exc:
                log.warning("shard_decrypt_failed job=%s shard=%d err=%s", job_id, shard_idx, exc)

        log.info(
            "shard_inference_start job=%s shard=%d model=%s max_tok=%d",
            job_id, shard_idx, model_id, max_tok,
        )
        t0 = time.monotonic()

        try:
            output = await self._models.run(model_id, prompt_slice, max_tok)
        except Exception as exc:
            log.error("shard_inference_failed job=%s shard=%d err=%s", job_id, shard_idx, exc)
            return None

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.info(
            "shard_inference_done job=%s shard=%d elapsed=%dms len=%d",
            job_id, shard_idx, elapsed_ms, len(output),
        )

        # Sign result: keccak(shard_index || job_id || output)
        preimage  = (str(shard_idx) + job_id + output).encode("utf-8")
        signature = crypto_sign(self._privkey, preimage)

        from chain.crypto import address_from_key
        miner_address = address_from_key(self._privkey)

        return ShardResult(
            shard_index=shard_idx,
            miner=miner_address,
            output=output,
            latency_ms=elapsed_ms,
            signature=signature,
        )
