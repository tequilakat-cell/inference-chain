// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title  InferenceChainBridge
 * @notice Locks L1 INFT so the bridge watcher can mint an equal amount on L2.
 *         Releases L1 INFT when a L2 withdrawal is proven against a finalised state root.
 *
 * L1→L2 flow:
 *   1. User calls depositINFT(amount, l2Recipient) — INFT locked in this contract.
 *   2. DepositInitiated event emitted.
 *   3. Bridge watcher on L2 picks up the event and submits TX_BRIDGE_DEPOSIT.
 *   4. L2 sequencer credits l2Recipient on L2.
 *
 * L2→L1 flow:
 *   1. User submits TX_BRIDGE_WITHDRAW on L2 — INFT burned from L2 balance.
 *   2. After 7 days, state root containing the withdrawal is finalised on L1.
 *   3. Anyone calls finalizeWithdrawal() with a merkle proof → INFT released.
 *
 * @custom:security This contract never calls untrusted external contracts (only the
 *                  pre-configured INFT token and Rollup). Reentrancy guard on all
 *                  state-changing functions.
 */
contract InferenceChainBridge is Ownable2Step, ReentrancyGuard {

    // ── Errors ────────────────────────────────────────────────────────────────
    error ZeroAmount();
    error ZeroAddress();
    error AlreadyRelayed();
    error NotFinalized();
    error InvalidMerkleProof();
    error InsufficientLocked();
    error TransferFailed();

    // ── Structs ───────────────────────────────────────────────────────────────
    struct Deposit {
        address l1Sender;
        address l2Recipient;
        uint256 amount;
        uint256 l1BlockNumber;
        bool    relayed;
    }

    // ── State ─────────────────────────────────────────────────────────────────
    IERC20  public immutable l1InftToken;
    address public           rollup;          // InferenceChainRollup address

    uint256 public nextDepositNonce;
    uint256 public totalLocked;

    mapping(uint256 => Deposit)  private _deposits;           // depositNonce → Deposit
    mapping(bytes32 => bool)     private _usedWithdrawals;    // withdrawalHash → spent

    // ── Events ────────────────────────────────────────────────────────────────
    event DepositInitiated(
        uint256 indexed depositNonce,
        address indexed l1Sender,
        address indexed l2Recipient,
        uint256         amount,
        uint256         l1BlockNumber
    );
    event DepositRelayed(uint256 indexed depositNonce);
    event WithdrawalFinalized(
        bytes32 indexed withdrawalHash,
        address indexed l1Recipient,
        uint256         amount,
        uint256         l2BlockNumber
    );

    // ── Constructor ───────────────────────────────────────────────────────────
    constructor(address _l1InftToken, address _rollup) Ownable(msg.sender) {
        if (_l1InftToken == address(0) || _rollup == address(0)) revert ZeroAddress();
        l1InftToken = IERC20(_l1InftToken);
        rollup      = _rollup;
    }

    // ── L1→L2 deposit ─────────────────────────────────────────────────────────

    /**
     * @notice Lock L1 INFT and signal the bridge watcher to mint on L2.
     * @dev Caller must approve this contract first: l1InftToken.approve(bridge, amount).
     * @param amount       Amount of INFT to bridge (18 decimals).
     * @param l2Recipient  The L2 address that should receive the minted INFT.
     */
    function depositINFT(
        uint256 amount,
        address l2Recipient
    ) external nonReentrant returns (uint256 depositNonce) {
        if (amount == 0)          revert ZeroAmount();
        if (l2Recipient == address(0)) revert ZeroAddress();

        // Pull INFT from sender — requires prior approval
        bool ok = l1InftToken.transferFrom(msg.sender, address(this), amount);
        if (!ok) revert TransferFailed();

        depositNonce       = nextDepositNonce++;
        totalLocked       += amount;

        _deposits[depositNonce] = Deposit({
            l1Sender:     msg.sender,
            l2Recipient:  l2Recipient,
            amount:       amount,
            l1BlockNumber: block.number,
            relayed:      false,
        });

        emit DepositInitiated(depositNonce, msg.sender, l2Recipient, amount, block.number);
    }

    /**
     * @notice Called by the bridge relayer to mark a deposit as relayed (informational).
     *         Does not affect fund custody — just for UI / indexer clarity.
     */
    function markDepositRelayed(uint256 depositNonce) external onlyOwner {
        _deposits[depositNonce].relayed = true;
        emit DepositRelayed(depositNonce);
    }

    // ── L2→L1 withdrawal ──────────────────────────────────────────────────────

    /**
     * @notice Release locked INFT after the L2 withdrawal is proven against a finalised root.
     *
     * @param l2BlockNumber  The L2 block containing the TX_BRIDGE_WITHDRAW.
     * @param l1Recipient    The L1 address to receive the INFT.
     * @param amount         Amount to release.
     * @param l2TxHash       The L2 tx_hash of the withdrawal transaction.
     * @param merkleProof    Merkle proof that l2TxHash is in the l2BlockNumber state root.
     *                       Leaf = keccak256(l2TxHash || l1Recipient || amount).
     */
    function finalizeWithdrawal(
        uint256         l2BlockNumber,
        address         l1Recipient,
        uint256         amount,
        bytes32         l2TxHash,
        bytes32[] calldata merkleProof
    ) external nonReentrant {
        if (amount == 0)           revert ZeroAmount();
        if (l1Recipient == address(0)) revert ZeroAddress();

        // Verify the L2 block is finalised on L1
        (bool finalized) = _isFinalized(l2BlockNumber);
        if (!finalized) revert NotFinalized();

        // Build the withdrawal leaf
        bytes32 leaf = keccak256(abi.encodePacked(l2TxHash, l1Recipient, amount));
        if (_usedWithdrawals[leaf]) revert AlreadyRelayed();

        // Verify merkle proof against the committed state root
        bytes32 stateRoot = _getStateRoot(l2BlockNumber);
        if (!_verifyProof(stateRoot, leaf, merkleProof)) revert InvalidMerkleProof();

        _usedWithdrawals[leaf] = true;

        if (amount > totalLocked) revert InsufficientLocked();
        totalLocked -= amount;

        bool ok = l1InftToken.transfer(l1Recipient, amount);
        if (!ok) revert TransferFailed();

        emit WithdrawalFinalized(leaf, l1Recipient, amount, l2BlockNumber);
    }

    // ── Views ─────────────────────────────────────────────────────────────────

    function getDeposit(uint256 depositNonce) external view returns (Deposit memory) {
        return _deposits[depositNonce];
    }

    function isWithdrawalUsed(bytes32 leaf) external view returns (bool) {
        return _usedWithdrawals[leaf];
    }

    // ── Admin ─────────────────────────────────────────────────────────────────

    function setRollup(address _rollup) external onlyOwner {
        rollup = _rollup;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    function _isFinalized(uint256 l2BlockNumber) internal view returns (bool) {
        // Call the rollup contract
        (bool success, bytes memory data) = rollup.staticcall(
            abi.encodeWithSignature("isFinalized(uint256)", l2BlockNumber)
        );
        if (!success || data.length == 0) return false;
        return abi.decode(data, (bool));
    }

    function _getStateRoot(uint256 l2BlockNumber) internal view returns (bytes32) {
        (bool success, bytes memory data) = rollup.staticcall(
            abi.encodeWithSignature("getCommitment(uint256)", l2BlockNumber)
        );
        if (!success || data.length == 0) return bytes32(0);
        (bytes32 stateRoot,,,,, ) = abi.decode(data, (bytes32, bytes32, uint64, bool, bool, address));
        return stateRoot;
    }

    function _verifyProof(
        bytes32   root,
        bytes32   leaf,
        bytes32[] calldata proof
    ) internal pure returns (bool) {
        bytes32 current = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 sibling = proof[i];
            if (current <= sibling) {
                current = keccak256(abi.encodePacked(current, sibling));
            } else {
                current = keccak256(abi.encodePacked(sibling, current));
            }
        }
        return current == root;
    }
}
