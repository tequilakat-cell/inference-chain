"""
Tests for the model weight Merkle registry — full stack.

Covers:
  1. miner/model_registry.py  — tree maths, proof round-trips, root computation
  2. chain/types.py           — TxType.MODEL_REGISTER presence
  3. chain/state.py           — _apply_model_register, query helpers, copy()
  4. chain/rpc/handlers.py    — inft_registerModel, inft_getModelRoot
  5. chain/shard/protocol.py  — miner selection filtered by model registration
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT.parent / "l1" / "miner"))

from chain.crypto import address_from_key
from chain.mempool import build_transaction
from chain.state import StateDB
from chain.types import AccountState, TxType
from miner.model_registry import (
    _keccak,
    _leaves_mock,
    compute_model_root,
    merkle_proof,
    merkle_root_hex,
    verify_proof,
)

# Deterministic test key / address
_PRIVKEY = "4da245a36de729dcbe5263060b146e570674a384a047394fe0491015cf72095f"
_ADDR    = address_from_key(_PRIVKEY).lower()
_CHAIN   = 2026

# A valid 32-byte Merkle root placeholder
_FAKE_ROOT = "0x" + "ab" * 32


# ═════════════════════════════════════════════════════════════════════════════
# 1 — model_registry: Merkle tree maths
# ═════════════════════════════════════════════════════════════════════════════

class TestMerkleTree(unittest.TestCase):

    def _leaves(self, n: int) -> list[bytes]:
        return [_keccak(f"leaf{i}".encode()) for i in range(n)]

    # ── root is deterministic ──────────────────────────────────────────────

    def test_root_is_deterministic(self):
        leaves = self._leaves(4)
        self.assertEqual(merkle_root_hex(leaves), merkle_root_hex(leaves))

    def test_different_leaves_different_root(self):
        a = [_keccak(b"a"), _keccak(b"b")]
        b = [_keccak(b"a"), _keccak(b"c")]
        self.assertNotEqual(merkle_root_hex(a), merkle_root_hex(b))

    def test_single_leaf(self):
        leaf = [_keccak(b"solo")]
        root = merkle_root_hex(leaf)
        self.assertTrue(root.startswith("0x"))
        self.assertEqual(len(root), 66)

    def test_empty_leaves_returns_zero_hash(self):
        root = merkle_root_hex([])
        self.assertEqual(root, "0x" + "00" * 32)

    # ── proof generation and verification ─────────────────────────────────

    def test_proof_valid_for_all_indices(self):
        for n in (1, 2, 3, 4, 5, 7, 8, 9, 16):
            leaves = self._leaves(n)
            root   = merkle_root_hex(leaves)
            for i in range(n):
                proof    = merkle_proof(leaves, i)
                leaf_hex = "0x" + leaves[i].hex()
                self.assertTrue(
                    verify_proof(root, leaf_hex, i, proof),
                    f"proof failed: n={n} i={i}",
                )

    def test_wrong_leaf_fails_verification(self):
        leaves   = self._leaves(4)
        root     = merkle_root_hex(leaves)
        proof    = merkle_proof(leaves, 0)
        bad_leaf = "0x" + _keccak(b"impostor").hex()
        self.assertFalse(verify_proof(root, bad_leaf, 0, proof))

    def test_wrong_index_fails_verification(self):
        leaves   = self._leaves(4)
        root     = merkle_root_hex(leaves)
        proof    = merkle_proof(leaves, 0)
        leaf_hex = "0x" + leaves[0].hex()
        self.assertFalse(verify_proof(root, leaf_hex, 1, proof))

    def test_tampered_proof_fails_verification(self):
        leaves   = self._leaves(4)
        root     = merkle_root_hex(leaves)
        proof    = merkle_proof(leaves, 2)
        leaf_hex = "0x" + leaves[2].hex()
        bad_proof = proof[:]
        bad_proof[0] = "0x" + "ff" * 32
        self.assertFalse(verify_proof(root, leaf_hex, 2, bad_proof))

    def test_order_matters_different_models_different_root(self):
        a = [_keccak(b"alpha"), _keccak(b"beta")]
        b = [_keccak(b"beta"), _keccak(b"alpha")]
        self.assertNotEqual(merkle_root_hex(a), merkle_root_hex(b))


# ═════════════════════════════════════════════════════════════════════════════
# 2 — model_registry: compute_model_root
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeModelRoot(unittest.TestCase):

    def test_mock_path_no_file(self):
        root, count = compute_model_root("some-model", "")
        self.assertTrue(root.startswith("0x"))
        self.assertEqual(len(root), 66)
        self.assertGreaterEqual(count, 1)

    def test_mock_returns_same_root_for_same_model(self):
        r1, c1 = compute_model_root("model-x", "")
        r2, c2 = compute_model_root("model-x", "")
        self.assertEqual(r1, r2)
        self.assertEqual(c1, c2)

    def test_different_model_ids_different_roots(self):
        r1, _ = compute_model_root("model-a", "")
        r2, _ = compute_model_root("model-b", "")
        self.assertNotEqual(r1, r2)

    def test_file_based_root(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world model weights" * 1000)
            tmp_path = f.name
        try:
            root, count = compute_model_root("local/model", tmp_path)
            self.assertTrue(root.startswith("0x"))
            self.assertEqual(len(root), 66)
            self.assertGreaterEqual(count, 1)
        finally:
            os.unlink(tmp_path)

    def test_file_root_is_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 512)
            tmp_path = f.name
        try:
            r1, c1 = compute_model_root("local/model", tmp_path)
            r2, c2 = compute_model_root("local/model", tmp_path)
            self.assertEqual(r1, r2)
            self.assertEqual(c1, c2)
        finally:
            os.unlink(tmp_path)

    def test_different_file_content_different_root(self):
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"aaaa" * 256)
            p1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"bbbb" * 256)
            p2 = f2.name
        try:
            r1, _ = compute_model_root("m", p1)
            r2, _ = compute_model_root("m", p2)
            self.assertNotEqual(r1, r2)
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_large_file_produces_multiple_leaves(self):
        # 1 MiB + 1 byte → 2 leaves
        from miner.model_registry import CHUNK_BYTES
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * (CHUNK_BYTES + 1))
            tmp_path = f.name
        try:
            _, count = compute_model_root("local/big", tmp_path)
            self.assertEqual(count, 2)
        finally:
            os.unlink(tmp_path)


# ═════════════════════════════════════════════════════════════════════════════
# 3 — chain/types.py
# ═════════════════════════════════════════════════════════════════════════════

class TestTxType(unittest.TestCase):

    def test_model_register_value(self):
        self.assertEqual(TxType.MODEL_REGISTER, 12)

    def test_no_collision_with_other_tx_types(self):
        values = [t.value for t in TxType]
        self.assertEqual(len(values), len(set(values)), "TxType values must be unique")


# ═════════════════════════════════════════════════════════════════════════════
# 4 — chain/state.py
# ═════════════════════════════════════════════════════════════════════════════

def _funded_state(address: str, balance: int = 10_000) -> StateDB:
    db = StateDB()
    db.set_account(address, AccountState(balance_inft=balance, nonce=0))
    return db


def _register_tx(model_id: str, root: str, leaf_count: int, nonce: int = 0) -> object:
    return build_transaction(
        TxType.MODEL_REGISTER, _ADDR, nonce,
        {"model_id": model_id, "model_root": root, "leaf_count": leaf_count},
        chain_id=_CHAIN, private_key=_PRIVKEY,
    )


class TestStateModelRegistry(unittest.TestCase):

    def setUp(self):
        self.db = _funded_state(_ADDR)

    # ── happy path ─────────────────────────────────────────────────────────

    def test_apply_model_register_stores_root(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 5)
        new = self.db.apply_transaction(tx)
        self.assertEqual(new.get_model_root(_ADDR, "Qwen/Test"), _FAKE_ROOT)

    def test_apply_model_register_stores_leaf_count(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 17)
        new = self.db.apply_transaction(tx)
        self.assertEqual(new.get_model_leaf_count(_ADDR, "Qwen/Test"), 17)

    def test_apply_model_register_bumps_nonce(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 1)
        new = self.db.apply_transaction(tx)
        self.assertEqual(new.nonce(_ADDR), 1)

    def test_apply_model_register_does_not_deduct_balance(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 1)
        new = self.db.apply_transaction(tx)
        self.assertEqual(new.balance(_ADDR), 10_000)

    def test_re_registration_updates_root(self):
        root_v1 = "0x" + "aa" * 32
        root_v2 = "0x" + "bb" * 32
        db1 = self.db.apply_transaction(_register_tx("Qwen/Test", root_v1, 3, nonce=0))
        db2 = db1.apply_transaction(_register_tx("Qwen/Test", root_v2, 7, nonce=1))
        self.assertEqual(db2.get_model_root(_ADDR, "Qwen/Test"), root_v2)
        self.assertEqual(db2.get_model_leaf_count(_ADDR, "Qwen/Test"), 7)

    def test_multiple_models_registered_independently(self):
        root_a = "0x" + "aa" * 32
        root_b = "0x" + "bb" * 32
        db1 = self.db.apply_transaction(_register_tx("Model/A", root_a, 4, nonce=0))
        db2 = db1.apply_transaction(_register_tx("Model/B", root_b, 8, nonce=1))
        self.assertEqual(db2.get_model_root(_ADDR, "Model/A"), root_a)
        self.assertEqual(db2.get_model_root(_ADDR, "Model/B"), root_b)
        self.assertCountEqual(db2.get_miner_models(_ADDR), ["Model/A", "Model/B"])

    # ── query helpers ──────────────────────────────────────────────────────

    def test_get_model_root_returns_none_for_unknown(self):
        self.assertIsNone(self.db.get_model_root(_ADDR, "unknown/model"))

    def test_get_model_root_unknown_address_returns_none(self):
        self.assertIsNone(self.db.get_model_root("0x" + "ff" * 20, "any/model"))

    def test_get_miner_models_empty_before_registration(self):
        self.assertEqual(self.db.get_miner_models(_ADDR), [])

    def test_miners_with_model_empty_before_registration(self):
        self.assertEqual(self.db.miners_with_model("Qwen/Test"), [])

    def test_miners_with_model_after_registration(self):
        tx  = _register_tx("Shared/Model", _FAKE_ROOT, 2)
        new = self.db.apply_transaction(tx)
        self.assertIn(_ADDR, new.miners_with_model("Shared/Model"))

    def test_miners_with_model_multiple_miners(self):
        privkey_b = "1" * 64
        addr_b    = address_from_key(privkey_b).lower()
        root_b    = "0x" + "cc" * 32

        db1 = _funded_state(_ADDR)
        db1.set_account(addr_b, AccountState(balance_inft=5_000, nonce=0))

        tx_a = _register_tx("Shared/M", _FAKE_ROOT, 3, nonce=0)
        tx_b = build_transaction(
            TxType.MODEL_REGISTER, addr_b, 0,
            {"model_id": "Shared/M", "model_root": root_b, "leaf_count": 5},
            chain_id=_CHAIN, private_key=privkey_b,
        )
        db2 = db1.apply_transaction(tx_a)
        db3 = db2.apply_transaction(tx_b)
        self.assertCountEqual(db3.miners_with_model("Shared/M"), [_ADDR, addr_b])

    # ── error cases ────────────────────────────────────────────────────────

    def test_empty_model_id_rejected(self):
        from chain.state import StateError
        tx = _register_tx("", _FAKE_ROOT, 1)
        with self.assertRaises(StateError):
            self.db.apply_transaction(tx)

    def test_bad_root_format_rejected(self):
        from chain.state import StateError
        tx = _register_tx("Qwen/Test", "not-a-hex-root", 1)
        with self.assertRaises(StateError):
            self.db.apply_transaction(tx)

    def test_short_root_rejected(self):
        from chain.state import StateError
        tx = _register_tx("Qwen/Test", "0x" + "ab" * 16, 1)  # 16 bytes, not 32
        with self.assertRaises(StateError):
            self.db.apply_transaction(tx)

    def test_zero_leaf_count_rejected(self):
        from chain.state import StateError
        tx = _register_tx("Qwen/Test", _FAKE_ROOT, 0)
        with self.assertRaises(StateError):
            self.db.apply_transaction(tx)

    def test_wrong_nonce_rejected(self):
        from chain.state import StateError
        tx = _register_tx("Qwen/Test", _FAKE_ROOT, 1, nonce=99)
        with self.assertRaises(StateError):
            self.db.apply_transaction(tx)

    # ── copy() carries model state ─────────────────────────────────────────

    def test_copy_carries_model_roots(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 3)
        new = self.db.apply_transaction(tx)
        cp  = new.copy()
        self.assertEqual(cp.get_model_root(_ADDR, "Qwen/Test"), _FAKE_ROOT)

    def test_copy_is_independent(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 3)
        new = self.db.apply_transaction(tx)
        cp  = new.copy()
        # Mutate copy directly
        cp._model_roots[_ADDR]["Qwen/Test"] = "0x" + "00" * 32
        # Original must be unchanged
        self.assertEqual(new.get_model_root(_ADDR, "Qwen/Test"), _FAKE_ROOT)

    def test_copy_carries_leaf_counts(self):
        tx  = _register_tx("Qwen/Test", _FAKE_ROOT, 99)
        new = self.db.apply_transaction(tx)
        cp  = new.copy()
        self.assertEqual(cp.get_model_leaf_count(_ADDR, "Qwen/Test"), 99)


# ═════════════════════════════════════════════════════════════════════════════
# 5 — chain/rpc/handlers.py
# ═════════════════════════════════════════════════════════════════════════════

def _make_mock_sequencer(db: StateDB | None = None) -> MagicMock:
    """Return a minimal mock sequencer accepted by RPCHandlers."""
    seq = MagicMock()
    seq.chain_id = _CHAIN
    seq.state.return_value = db or _funded_state(_ADDR)
    seq.submit_transaction = AsyncMock(return_value=(True, "ok"))
    return seq


class TestRPCHandlers(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        from chain.rpc.handlers import RPCHandlers
        self.db  = _funded_state(_ADDR)
        self.seq = _make_mock_sequencer(self.db)
        self.h   = RPCHandlers(self.seq, shard_protocol=None)

    # ── inft_registerModel ─────────────────────────────────────────────────

    async def test_register_model_returns_tx_hash(self):
        result = await self.h.inft_registerModel(
            [MODEL_ID := "Qwen/Test", _FAKE_ROOT, 5, _PRIVKEY]
        )
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("0x"))

    async def test_register_model_calls_submit_transaction(self):
        await self.h.inft_registerModel(["M/X", _FAKE_ROOT, 3, _PRIVKEY])
        self.seq.submit_transaction.assert_awaited_once()
        submitted_tx = self.seq.submit_transaction.call_args[0][0]
        self.assertEqual(submitted_tx.tx_type, TxType.MODEL_REGISTER)

    async def test_register_model_payload_fields(self):
        await self.h.inft_registerModel(["M/Y", _FAKE_ROOT, 7, _PRIVKEY])
        tx  = self.seq.submit_transaction.call_args[0][0]
        p   = tx.payload_dict()
        self.assertEqual(p["model_id"],   "M/Y")
        self.assertEqual(p["model_root"], _FAKE_ROOT)
        self.assertEqual(p["leaf_count"], 7)

    async def test_register_model_missing_params_raises(self):
        with self.assertRaises(ValueError):
            await self.h.inft_registerModel(["only_one_param"])

    async def test_register_model_sequencer_reject_raises(self):
        self.seq.submit_transaction = AsyncMock(return_value=(False, "bad nonce"))
        with self.assertRaises(ValueError, msg="bad nonce"):
            await self.h.inft_registerModel(["M/Z", _FAKE_ROOT, 1, _PRIVKEY])

    # ── inft_getModelRoot ──────────────────────────────────────────────────

    async def test_get_model_root_unregistered(self):
        result = await self.h.inft_getModelRoot([_ADDR, "unknown/model"])
        self.assertFalse(result["registered"])
        self.assertIsNone(result["model_root"])

    async def test_get_model_root_after_registration(self):
        # Apply a MODEL_REGISTER tx directly to the state
        tx = _register_tx("Qwen/Test", _FAKE_ROOT, 5)
        new_db = self.db.apply_transaction(tx)
        self.seq.state.return_value = new_db

        result = await self.h.inft_getModelRoot([_ADDR, "Qwen/Test"])
        self.assertTrue(result["registered"])
        self.assertEqual(result["model_root"], _FAKE_ROOT)
        self.assertEqual(result["leaf_count"], 5)

    async def test_get_model_root_lists_all_miner_models(self):
        db = self.db
        db = db.apply_transaction(_register_tx("M/A", _FAKE_ROOT, 1, nonce=0))
        db = db.apply_transaction(_register_tx("M/B", "0x" + "cc" * 32, 2, nonce=1))
        self.seq.state.return_value = db

        result = await self.h.inft_getModelRoot([_ADDR, "M/A"])
        self.assertCountEqual(result["miner_models"], ["M/A", "M/B"])

    async def test_get_model_root_missing_params_raises(self):
        with self.assertRaises((ValueError, IndexError)):
            await self.h.inft_getModelRoot([])


# ═════════════════════════════════════════════════════════════════════════════
# 6 — chain/shard/protocol.py — miner selection filter
# ═════════════════════════════════════════════════════════════════════════════

class TestShardProtocolModelFilter(unittest.IsolatedAsyncioTestCase):
    """
    Verify that dispatch_job only assigns shards to miners that have
    registered the requested model, with graceful fallback.
    """

    def _make_protocol(self, db: StateDB):
        from chain.shard.protocol import ShardProtocol

        seq = MagicMock()
        seq.chain_id = _CHAIN
        seq.state.return_value = db
        seq.mempool = MagicMock()
        seq.mempool.add = AsyncMock(return_value=True)

        p2p = MagicMock()
        p2p.broadcast = AsyncMock()

        return ShardProtocol(seq, p2p, cfg={}), seq, p2p

    def _make_block(self, parent_hash: str = "0x" + "ab" * 32):
        block = MagicMock()
        block.header.parent_hash  = parent_hash
        block.header.block_number = 1
        return block

    def _job_payload(self, model_id: str, n_shards: int = 1) -> dict:
        return {
            "job_id":     "test-job-001",
            "model_id":   model_id,
            "prompt":     "Hello",
            "n_shards":   n_shards,
            "max_tokens": 64,
            "shard_mode": "parallel_sample",
            "fee_inft":   10,
            "sender":     _ADDR,
        }

    async def test_registered_miner_receives_offer(self):
        """Miner with registered model gets a ShardOffer broadcast."""
        db = _funded_state(_ADDR)
        db.set_account(_ADDR, AccountState(balance_inft=1000, stake_inft=500, nonce=0))
        db = db.apply_transaction(_register_tx("M/Test", _FAKE_ROOT, 3))
        db.block_number = 1

        proto, seq, p2p = self._make_protocol(db)
        await proto.dispatch_job(self._job_payload("M/Test"), self._make_block(), db)

        p2p.broadcast.assert_awaited()
        call_args = p2p.broadcast.call_args_list
        offers = [a for a in call_args if a[0][0] == "shard_offers"]
        self.assertGreater(len(offers), 0, "expected at least one ShardOffer")
        offer_payload = offers[0][0][1]
        self.assertEqual(offer_payload["spec"]["assigned_miner"].lower(), _ADDR)

    async def test_unregistered_miner_excluded_when_registered_miner_exists(self):
        """
        Two staked miners.  Only one has registered the model.
        Every shard must be assigned to the registered miner.
        """
        privkey_b = "2" * 64
        addr_b    = address_from_key(privkey_b).lower()
        root_b    = "0x" + "cc" * 32

        db = StateDB()
        db.set_account(_ADDR,  AccountState(balance_inft=1000, stake_inft=500, nonce=0))
        db.set_account(addr_b, AccountState(balance_inft=1000, stake_inft=500, nonce=0))
        # Only miner B registers the model
        tx_b = build_transaction(
            TxType.MODEL_REGISTER, addr_b, 0,
            {"model_id": "M/Exclusive", "model_root": root_b, "leaf_count": 2},
            chain_id=_CHAIN, private_key=privkey_b,
        )
        db = db.apply_transaction(tx_b)
        db.block_number = 1

        proto, _, p2p = self._make_protocol(db)
        await proto.dispatch_job(
            self._job_payload("M/Exclusive", n_shards=1), self._make_block(), db
        )

        offers = [
            a[0][1] for a in p2p.broadcast.call_args_list
            if a[0][0] == "shard_offers"
        ]
        self.assertGreater(len(offers), 0)
        for offer in offers:
            assigned = offer["spec"]["assigned_miner"].lower()
            self.assertEqual(assigned, addr_b,
                             f"expected addr_b only, got {assigned}")

    async def test_fallback_to_all_when_no_registration(self):
        """
        No miner has registered the model yet.
        dispatch_job falls back to all staked validators.
        """
        db = _funded_state(_ADDR)
        db.set_account(_ADDR, AccountState(balance_inft=1000, stake_inft=500, nonce=0))
        db.block_number = 1

        proto, _, p2p = self._make_protocol(db)
        await proto.dispatch_job(
            self._job_payload("M/Unregistered"), self._make_block(), db
        )

        offers = [
            a[0][1] for a in p2p.broadcast.call_args_list
            if a[0][0] == "shard_offers"
        ]
        self.assertGreater(len(offers), 0, "should fall back to all validators")

    async def test_no_miners_at_all_no_broadcast(self):
        """Empty validator set — no offers should go out."""
        db = StateDB()   # no accounts, no stake
        db.block_number = 1

        proto, _, p2p = self._make_protocol(db)
        await proto.dispatch_job(
            self._job_payload("M/Any"), self._make_block(), db
        )

        offers = [
            a for a in p2p.broadcast.call_args_list
            if a[0][0] == "shard_offers"
        ]
        self.assertEqual(len(offers), 0)


# ═════════════════════════════════════════════════════════════════════════════
# 7 — miner: _register_model_roots integration
# ═════════════════════════════════════════════════════════════════════════════

class TestMinerRegistration(unittest.IsolatedAsyncioTestCase):
    """
    Test that the miner's _register_model_roots background task correctly
    computes roots and calls inft_registerModel via _rpc_call.
    """

    async def test_register_model_roots_calls_rpc(self):
        from miner.l2_miner import L2MinerConfig, L2Miner

        cfg = L2MinerConfig(
            private_key=_PRIVKEY,
            models={"M/Mock": ""},  # empty path → mock backend
            l2_rpc_url="http://127.0.0.1:19999",  # no real server
            backend="mock",
        )

        miner = L2Miner.__new__(L2Miner)
        miner.cfg     = cfg
        miner.address = _ADDR

        from backends.base import MockBackend
        from miner.model_registry import compute_model_root

        captured = []

        async def fake_rpc(method, params):
            captured.append({"method": method, "params": params})
            return "0x" + "aa" * 32   # fake tx_hash

        miner._rpc_call = fake_rpc

        await miner._register_model_roots()

        self.assertEqual(len(captured), 1)
        call = captured[0]
        self.assertEqual(call["method"], "inft_registerModel")
        model_id, root, leaf_count, privkey = call["params"]
        self.assertEqual(model_id, "M/Mock")
        self.assertTrue(root.startswith("0x"))
        self.assertEqual(len(root), 66)
        self.assertGreaterEqual(leaf_count, 1)
        self.assertEqual(privkey, _PRIVKEY)

    async def test_register_model_roots_continues_on_rpc_error(self):
        """A failing RPC call for one model must not stop registration of others."""
        from miner.l2_miner import L2MinerConfig, L2Miner

        cfg = L2MinerConfig(
            private_key=_PRIVKEY,
            models={"M/Fail": "", "M/OK": ""},
            l2_rpc_url="http://127.0.0.1:19999",
            backend="mock",
        )

        miner = L2Miner.__new__(L2Miner)
        miner.cfg     = cfg
        miner.address = _ADDR

        call_count = [0]

        async def flaky_rpc(method, params):
            call_count[0] += 1
            if params[0] == "M/Fail":
                raise RuntimeError("simulated rpc error")
            return "0xtxhash"

        miner._rpc_call = flaky_rpc

        # Should not raise
        await miner._register_model_roots()
        self.assertEqual(call_count[0], 2, "should have attempted both models")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
