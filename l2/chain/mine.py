"""
MINE token — proof-of-work mineable ERC20 on InferenceChain L2.

Mirrors Bitcoin's emission curve and 0xBTC's PoW puzzle, with the
same-block double-mint exploit patched.

EXPLOIT FIXED:
  0xBTC's _startNewMiningEpoch() sets challengeNumber = blockhash(N-1).
  If TX1 and TX2 land in the same block, TX1's epoch reset generates a
  challengeNumber the attacker already solved offline, letting TX2 claim
  a second reward for free.

  Fix: require(block_number != last_reward_block) — one mint per L2 block.

Tokenomics:
  Max supply:  21,000,000 MINE
  Reward:      50 MINE → halves every 210,000 solutions (like Bitcoin)
  Difficulty:  adjusts every 2,016 solutions to target 600 L2 blocks/solution
  Puzzle:      keccak256(abi.encodePacked(challenge, miner_address, nonce)) < target
"""

from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_SUPPLY          = 21_000_000 * 10**18          # 21M MINE in wei
INITIAL_REWARD      = 50 * 10**18                  # 50 MINE in wei
HALVING_INTERVAL    = 210_000                       # solutions between halvings
DIFFICULTY_INTERVAL = 2_016                         # solutions per difficulty epoch
TARGET_BLOCK_TIME   = 600                           # target L2 blocks per solution (~10 min)
MINIMUM_TARGET      = 2**16
MAXIMUM_TARGET      = 2**234                        # initial easy target


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class MineState:
    total_minted:             int   = 0
    solutions_found:          int   = 0             # total solutions ever accepted
    mining_target:            int   = MAXIMUM_TARGET
    challenge:                bytes = b"\x00" * 32
    last_reward_block:        int   = 0             # exploit fix: one mint per block
    diff_period_start_block:  int   = 0             # block number when current diff epoch started
    diff_period_start_solutions: int = 0            # solutions_found at start of diff epoch
    balances: dict[str, int]  = field(default_factory=dict)
    # Used solutions (challenge_digest → miner address) — prevents replay
    used_solutions: dict[str, str] = field(default_factory=dict)

    # ── Reward calculation ────────────────────────────────────────────────────

    def current_reward(self) -> int:
        halvings = self.solutions_found // HALVING_INTERVAL
        if halvings >= 40:
            return 0
        reward = INITIAL_REWARD >> halvings
        remaining = MAX_SUPPLY - self.total_minted
        return min(reward, remaining)

    def epoch_number(self) -> int:
        return self.solutions_found // HALVING_INTERVAL

    # ── Mining ────────────────────────────────────────────────────────────────

    def verify_solution(
        self, miner: str, nonce: int, challenge_digest: str
    ) -> tuple[bool, str]:
        """
        Validate a PoW solution.  Returns (valid, reason_if_invalid).

        Puzzle: keccak256(challenge || miner_address || nonce) < mining_target
        The digest must also match challenge_digest (commit-verify).
        """
        digest = compute_digest(self.challenge, miner, nonce)
        digest_hex = "0x" + digest.hex()

        if digest_hex != challenge_digest:
            return False, "digest mismatch"
        if int.from_bytes(digest, "big") >= self.mining_target:
            return False, "digest does not meet target"
        if challenge_digest in self.used_solutions:
            return False, "solution already used"
        return True, ""

    def apply_solution(
        self, miner: str, nonce: int, challenge_digest: str,
        block_number: int, block_hash: bytes
    ) -> int:
        """
        Accept a valid solution, mint reward, update challenge and difficulty.
        Returns the reward minted (in wei).
        Raises ValueError if invalid (call verify_solution first for details).
        """
        # ── Exploit fix: one mint per L2 block ───────────────────────────────
        # Allow first mint always; after that, one per block (0xBTC exploit fix)
        if self.solutions_found > 0 and block_number == self.last_reward_block:
            raise ValueError("already minted this block — same-block double-mint prevented")

        # ── Validate ─────────────────────────────────────────────────────────
        valid, reason = self.verify_solution(miner, nonce, challenge_digest)
        if not valid:
            raise ValueError(reason)

        reward = self.current_reward()
        if reward == 0:
            raise ValueError("max supply reached")

        # ── Commit ───────────────────────────────────────────────────────────
        self.used_solutions[challenge_digest] = miner
        self.last_reward_block = block_number
        self.total_minted += reward
        self.solutions_found += 1
        self.balances[miner] = self.balances.get(miner, 0) + reward

        # ── New challenge: keccak256(old_challenge || block_hash) ─────────────
        # Mixing in block_hash ties the next puzzle to the L2 chain,
        # making it unpredictable before the block is produced.
        self.challenge = _keccak(self.challenge + block_hash)

        # ── Difficulty adjustment every DIFFICULTY_INTERVAL solutions ─────────
        if self.solutions_found % DIFFICULTY_INTERVAL == 0 and self.solutions_found > 0:
            self._adjust_difficulty(block_number)

        return reward

    def _adjust_difficulty(self, current_block: int) -> None:
        blocks_elapsed = current_block - self.diff_period_start_block
        if blocks_elapsed == 0:
            blocks_elapsed = 1

        # How many solutions were found in this period
        solutions_in_period = self.solutions_found - self.diff_period_start_solutions
        if solutions_in_period == 0:
            solutions_in_period = 1

        # Ratio: actual blocks per solution vs target
        actual_rate   = blocks_elapsed / solutions_in_period  # blocks per solution
        target_rate   = TARGET_BLOCK_TIME                      # target blocks per solution

        # Increase target (easier) if too slow, decrease (harder) if too fast
        # Cap adjustment to 4× in either direction (like Bitcoin)
        ratio = target_rate / actual_rate
        ratio = max(0.25, min(4.0, ratio))

        new_target = int(self.mining_target * ratio)
        self.mining_target = max(MINIMUM_TARGET, min(MAXIMUM_TARGET, new_target))

        self.diff_period_start_block     = current_block
        self.diff_period_start_solutions = self.solutions_found

    # ── Bridge ────────────────────────────────────────────────────────────────

    def bridge_burn(self, miner: str, amount: int) -> None:
        """Burn MINE on L2 to initiate a bridge withdrawal to L1."""
        bal = self.balances.get(miner, 0)
        if bal < amount:
            raise ValueError(f"insufficient MINE: have {bal}, need {amount}")
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balances[miner] = bal - amount
        self.total_minted -= amount  # supply decreases on L2 (minted on L1 by relayer)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "total_minted":              self.total_minted,
            "solutions_found":           self.solutions_found,
            "mining_target":             self.mining_target,
            "challenge":                 self.challenge.hex(),
            "last_reward_block":         self.last_reward_block,
            "diff_period_start_block":   self.diff_period_start_block,
            "diff_period_start_solutions": self.diff_period_start_solutions,
            "balances":                  self.balances,
            "used_solutions":            self.used_solutions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MineState":
        s = cls(
            total_minted              = d["total_minted"],
            solutions_found           = d["solutions_found"],
            mining_target             = d["mining_target"],
            challenge                 = bytes.fromhex(d["challenge"]),
            last_reward_block         = d["last_reward_block"],
            diff_period_start_block   = d["diff_period_start_block"],
            diff_period_start_solutions = d["diff_period_start_solutions"],
            balances                  = d.get("balances", {}),
            used_solutions            = d.get("used_solutions", {}),
        )
        return s


# ── Puzzle helpers ────────────────────────────────────────────────────────────

def compute_digest(challenge: bytes, miner_address: str, nonce: int) -> bytes:
    """
    keccak256(challenge || miner_address_bytes || nonce_bytes_32)

    Uses abi.encodePacked-style packing: challenge is 32 bytes, address is
    20 bytes (lower-case hex stripped of 0x), nonce is 32-byte big-endian.
    This matches what Solidity's abi.encodePacked would produce.
    """
    addr_bytes  = bytes.fromhex(miner_address.lower().removeprefix("0x"))
    nonce_bytes = nonce.to_bytes(32, "big")
    packed      = challenge + addr_bytes + nonce_bytes
    return _keccak(packed)


def _keccak(data: bytes) -> bytes:
    from Crypto.Hash import keccak as _k
    h = _k.new(digest_bits=256)
    h.update(data)
    return h.digest()


def difficulty_display(target: int) -> str:
    """Human-readable difficulty (higher = harder)."""
    if target == 0:
        return "∞"
    return f"{MAXIMUM_TARGET // target:,}"


def hashrate_estimate(solutions_per_sec: float, target: int) -> str:
    """Rough H/s from observed solution rate and current target."""
    if solutions_per_sec <= 0 or target <= 0:
        return "—"
    # Expected hashes per solution = 2^256 / target
    hashes_per_solution = 2**256 / target
    hs = solutions_per_sec * hashes_per_solution
    for unit, threshold in [("TH/s", 1e12), ("GH/s", 1e9), ("MH/s", 1e6), ("kH/s", 1e3)]:
        if hs >= threshold:
            return f"{hs/threshold:.2f} {unit}"
    return f"{hs:.0f} H/s"
