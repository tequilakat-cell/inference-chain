// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";

/**
 * @title  MineTokenL1 (MINE)
 * @notice Wrapped MINE token on Ethereum L1.
 *
 * MINE is mined via proof-of-work on InferenceChain L2 (chain 2026).
 * This contract lets users bridge their L2 MINE to L1 Ethereum.
 *
 * Bridge flow (L2 → L1):
 *   1. User calls mine_bridgeWithdraw(amount, l1_address, privkey) on L2 RPC.
 *   2. L2 burns the MINE and records a withdrawal in state.
 *   3. The bridge relayer (or owner) calls finalizeWithdrawal() here with the
 *      L2 withdrawal data and a Merkle proof against the committed state root.
 *   4. This contract mints the equivalent MINE to the L1 recipient.
 *
 * Bridge flow (L1 → L2, future):
 *   1. User approves and calls depositToL2(amount, l2_address).
 *   2. MINE is locked here.
 *   3. Bridge watcher mints MINE on L2.
 *
 * Supply integrity:
 *   Total supply across L1 + L2 is always ≤ L2 MAX_SUPPLY (21,000,000 MINE).
 *   L2 burns on withdrawal → L1 mints. L1 locks on deposit → L2 mints.
 *
 * Trust model (v1):
 *   The bridge relayer is the trusted party. In v2 this will be replaced by
 *   optimistic fraud proofs verified against state roots committed by the L2
 *   rollup contract (InferenceChainRollup.sol).
 */
contract MineTokenL1 is ERC20, Ownable2Step {

    // ── Constants ─────────────────────────────────────────────────────────────
    uint256 public constant MAX_SUPPLY = 21_000_000 * 10**18;

    // ── State ─────────────────────────────────────────────────────────────────
    address public bridgeRelayer;
    uint256 public totalBridgedIn;   // MINE minted to L1 via bridge
    uint256 public totalLockedForL2; // MINE locked here for L2 (deposit direction)

    // Prevent replaying the same L2 withdrawal twice
    mapping(bytes32 => bool) public processedWithdrawals;

    // ── Events ────────────────────────────────────────────────────────────────
    event WithdrawalFinalized(
        bytes32 indexed l2TxHash,
        address indexed recipient,
        uint256 amount
    );
    event DepositInitiated(
        address indexed l1Sender,
        address indexed l2Recipient,
        uint256 amount,
        uint256 lockNonce
    );
    event BridgeRelayerUpdated(address indexed oldRelayer, address indexed newRelayer);

    // ── Errors ────────────────────────────────────────────────────────────────
    error NotRelayer();
    error AlreadyProcessed();
    error MaxSupplyExceeded();
    error ZeroAmount();
    error ZeroAddress();
    error InsufficientLocked();

    // ── Constructor ───────────────────────────────────────────────────────────
    constructor(address _bridgeRelayer)
        ERC20("MINE Token", "MINE")
        Ownable(msg.sender)
    {
        if (_bridgeRelayer == address(0)) revert ZeroAddress();
        bridgeRelayer = _bridgeRelayer;
    }

    // ── Bridge: L2 → L1 (finalise withdrawal) ────────────────────────────────

    /**
     * @notice Mint MINE to an L1 recipient, finalising a withdrawal from L2.
     * @dev    Called by the bridge relayer after verifying the L2 state.
     *         In v2: relayer must supply a Merkle proof against the L2 state root.
     *
     * @param l2TxHash    The L2 TX_MINE_BRIDGE transaction hash (withdrawal key).
     * @param recipient   L1 address to receive MINE.
     * @param amount      Amount of MINE to mint (in wei).
     */
    function finalizeWithdrawal(
        bytes32 l2TxHash,
        address recipient,
        uint256 amount
    ) external {
        if (msg.sender != bridgeRelayer) revert NotRelayer();
        if (processedWithdrawals[l2TxHash]) revert AlreadyProcessed();
        if (amount == 0) revert ZeroAmount();
        if (recipient == address(0)) revert ZeroAddress();
        if (totalSupply() + amount > MAX_SUPPLY) revert MaxSupplyExceeded();

        processedWithdrawals[l2TxHash] = true;
        totalBridgedIn += amount;

        _mint(recipient, amount);
        emit WithdrawalFinalized(l2TxHash, recipient, amount);
    }

    // ── Bridge: L1 → L2 (initiate deposit) ───────────────────────────────────

    /**
     * @notice Lock L1 MINE to initiate a deposit to L2.
     * @dev    The bridge relayer watches for DepositInitiated events and mints
     *         the equivalent MINE on L2 (TX_BRIDGE_DEPOSIT transaction).
     *
     * @param amount       Amount of L1 MINE to lock (must be approved first).
     * @param l2Recipient  L2 address that will receive MINE on L2.
     */
    function depositToL2(uint256 amount, address l2Recipient) external {
        if (amount == 0) revert ZeroAmount();
        if (l2Recipient == address(0)) revert ZeroAddress();

        // Pull tokens from sender (requires prior approve())
        _transfer(msg.sender, address(this), amount);
        totalLockedForL2 += amount;

        emit DepositInitiated(msg.sender, l2Recipient, amount, totalLockedForL2);
    }

    // ── Admin ─────────────────────────────────────────────────────────────────

    function setBridgeRelayer(address newRelayer) external onlyOwner {
        if (newRelayer == address(0)) revert ZeroAddress();
        emit BridgeRelayerUpdated(bridgeRelayer, newRelayer);
        bridgeRelayer = newRelayer;
    }

    /**
     * @notice Emergency: release locked tokens if a deposit never landed on L2.
     * @dev    Only callable by owner. Requires depositor to have submitted proof
     *         that the L2 credit never appeared. In production this would use
     *         a timeout + L2 state proof.
     */
    function emergencyReleaseDeposit(address recipient, uint256 amount) external onlyOwner {
        if (amount > totalLockedForL2) revert InsufficientLocked();
        totalLockedForL2 -= amount;
        _transfer(address(this), recipient, amount);
    }
}
