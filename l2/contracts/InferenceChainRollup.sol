// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title  InferenceChainRollup
 * @notice Optimistic rollup anchor for InferenceChain L2.
 *
 * The L2 sequencer posts a keccak256 state root every STATE_ROOT_INTERVAL
 * L2 blocks. The root covers all account balances, stakes, and completed job
 * output hashes. After FRAUD_WINDOW seconds without a successful challenge,
 * the root is finalised and L2→L1 withdrawals that reference it can be
 * processed by InferenceChainBridge.
 *
 * Fraud proofs (v1): owner-arbitrated. A valid challenge with a merkle proof
 * of an invalid state transition slashes the sequencer bond and reverts the
 * bad root. zkML-based proof verification replaces this in v2.
 *
 * Bond accounting: the sequencer deposits ETH as a slashable bond via
 * depositSequencerBond(). A successful fraud proof drains the bond to the
 * challenger.
 */
contract InferenceChainRollup is Ownable2Step, ReentrancyGuard {

    // ── Errors ────────────────────────────────────────────────────────────────
    error NotSequencer();
    error AlreadyCommitted();
    error BlockGap();
    error AlreadyFinalized();
    error AlreadyChallenged();
    error FraudWindowClosed();
    error FraudWindowOpen();
    error InvalidSequencerSig();
    error InsufficientBond();
    error TransferFailed();
    error InvalidProof();

    // ── Constants ─────────────────────────────────────────────────────────────
    uint256 public constant FRAUD_WINDOW      = 7 days;
    uint256 public constant CHALLENGE_BOND    = 0.01 ether;
    uint256 public constant STATE_ROOT_INTERVAL = 100;   // L2 blocks per commitment

    // ── Structs ───────────────────────────────────────────────────────────────
    struct Commitment {
        bytes32 stateRoot;
        bytes32 txBatchHash;
        uint64  committedAt;      // L1 block.timestamp
        bool    finalized;
        bool    challenged;
        address challenger;
    }

    // ── State ─────────────────────────────────────────────────────────────────
    address public sequencer;
    uint256 public sequencerBond;

    uint256 public latestCommittedBlock;
    uint256 public latestFinalizedBlock;

    mapping(uint256 => Commitment) private _commitments;  // l2BlockNumber → Commitment

    // ── Events ────────────────────────────────────────────────────────────────
    event StateRootCommitted(
        uint256 indexed l2BlockNumber,
        bytes32         stateRoot,
        bytes32         txBatchHash,
        address indexed sequencer,
        uint64          l1Timestamp
    );
    event StateRootChallenged(
        uint256 indexed l2BlockNumber,
        address indexed challenger,
        uint256         bond
    );
    event StateRootFinalized(uint256 indexed l2BlockNumber, bytes32 stateRoot);
    event FraudProofAccepted(
        uint256 indexed l2BlockNumber,
        address indexed challenger,
        uint256         sequencerSlashAmount
    );
    event SequencerUpdated(address indexed oldSeq, address indexed newSeq);
    event BondDeposited(address indexed by, uint256 amount);

    // ── Constructor ───────────────────────────────────────────────────────────
    constructor(address _sequencer) Ownable(msg.sender) {
        sequencer = _sequencer;
    }

    // ── Sequencer management ──────────────────────────────────────────────────

    function depositSequencerBond() external payable {
        sequencerBond += msg.value;
        emit BondDeposited(msg.sender, msg.value);
    }

    function setSequencer(address newSeq) external onlyOwner {
        emit SequencerUpdated(sequencer, newSeq);
        sequencer = newSeq;
    }

    // ── State root commitment ─────────────────────────────────────────────────

    /**
     * @notice Post an L2 state root. Only callable by the current sequencer.
     * @param l2BlockNumber  Must be exactly STATE_ROOT_INTERVAL ahead of last committed.
     * @param stateRoot      Merkle root of all L2 account states.
     * @param txBatchHash    keccak256 of all tx hashes in the batch.
     * @param sequencerSig   ECDSA over keccak256(l2BlockNumber || stateRoot || txBatchHash).
     */
    function commitStateRoot(
        uint256 l2BlockNumber,
        bytes32 stateRoot,
        bytes32 txBatchHash,
        bytes calldata sequencerSig
    ) external {
        if (msg.sender != sequencer) revert NotSequencer();
        if (_commitments[l2BlockNumber].committedAt != 0) revert AlreadyCommitted();
        if (latestCommittedBlock != 0 &&
            l2BlockNumber != latestCommittedBlock + STATE_ROOT_INTERVAL)
            revert BlockGap();

        // Verify sequencer signature over the commitment data
        bytes32 preimage = keccak256(abi.encodePacked(l2BlockNumber, stateRoot, txBatchHash));
        bytes32 ethHash  = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", preimage));
        address recovered = _recoverSigner(ethHash, sequencerSig);
        if (recovered != sequencer) revert InvalidSequencerSig();

        _commitments[l2BlockNumber] = Commitment({
            stateRoot:   stateRoot,
            txBatchHash: txBatchHash,
            committedAt: uint64(block.timestamp),
            finalized:   false,
            challenged:  false,
            challenger:  address(0),
        });

        latestCommittedBlock = l2BlockNumber;

        emit StateRootCommitted(l2BlockNumber, stateRoot, txBatchHash, sequencer, uint64(block.timestamp));
    }

    // ── Challenge ─────────────────────────────────────────────────────────────

    /**
     * @notice Challenge a committed state root within the fraud window.
     *         In v1, challenges are owner-arbitrated. In v2, this accepts a
     *         ZK proof directly.
     * @param l2BlockNumber  The block whose state root is being challenged.
     */
    function challengeStateRoot(uint256 l2BlockNumber) external payable nonReentrant {
        Commitment storage c = _commitments[l2BlockNumber];
        if (c.committedAt == 0)                                          revert AlreadyCommitted();
        if (c.finalized)                                                  revert AlreadyFinalized();
        if (c.challenged)                                                 revert AlreadyChallenged();
        if (block.timestamp > c.committedAt + FRAUD_WINDOW)              revert FraudWindowClosed();
        if (msg.value < CHALLENGE_BOND)                                   revert InsufficientBond();

        c.challenged  = true;
        c.challenger  = msg.sender;

        emit StateRootChallenged(l2BlockNumber, msg.sender, msg.value);
    }

    /**
     * @notice Owner resolves a challenge. If the challenger is correct, the
     *         sequencer bond is slashed and paid to the challenger.
     *         If the challenge is invalid, the challenger bond is forfeited to the protocol.
     */
    function resolveChallenge(
        uint256 l2BlockNumber,
        bool    challengerIsCorrect
    ) external onlyOwner nonReentrant {
        Commitment storage c = _commitments[l2BlockNumber];
        if (!c.challenged) revert InvalidProof();

        if (challengerIsCorrect) {
            // Slash sequencer bond; pay challenger
            uint256 slash = sequencerBond;
            sequencerBond = 0;
            delete _commitments[l2BlockNumber];

            // Remove from chain — latestCommittedBlock rolls back
            if (latestCommittedBlock == l2BlockNumber) {
                latestCommittedBlock = l2BlockNumber > STATE_ROOT_INTERVAL
                    ? l2BlockNumber - STATE_ROOT_INTERVAL
                    : 0;
            }

            _safeTransferETH(c.challenger, slash + CHALLENGE_BOND);
            emit FraudProofAccepted(l2BlockNumber, c.challenger, slash);
        } else {
            // Invalid challenge: keep challenger bond in contract (goes to treasury via withdrawProtocolFunds)
            c.challenged = false;
            c.challenger = address(0);
        }
    }

    // ── Finalization ──────────────────────────────────────────────────────────

    /**
     * @notice Finalise a state root after the fraud window passes unchallenged.
     *         Anyone can call this.
     */
    function finalizeStateRoot(uint256 l2BlockNumber) external {
        Commitment storage c = _commitments[l2BlockNumber];
        if (c.committedAt == 0)                               revert AlreadyCommitted();
        if (c.finalized)                                       revert AlreadyFinalized();
        if (c.challenged)                                      revert AlreadyChallenged();
        if (block.timestamp <= c.committedAt + FRAUD_WINDOW)  revert FraudWindowOpen();

        c.finalized = true;
        if (l2BlockNumber > latestFinalizedBlock) {
            latestFinalizedBlock = l2BlockNumber;
        }

        emit StateRootFinalized(l2BlockNumber, c.stateRoot);
    }

    // ── Views ─────────────────────────────────────────────────────────────────

    function getCommitment(uint256 l2BlockNumber) external view
        returns (
            bytes32 stateRoot,
            bytes32 txBatchHash,
            uint64  committedAt,
            bool    finalized,
            bool    challenged,
            address challenger
        )
    {
        Commitment storage c = _commitments[l2BlockNumber];
        return (c.stateRoot, c.txBatchHash, c.committedAt, c.finalized, c.challenged, c.challenger);
    }

    function isFinalized(uint256 l2BlockNumber) external view returns (bool) {
        return _commitments[l2BlockNumber].finalized;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    function _recoverSigner(bytes32 hash, bytes calldata sig) internal pure returns (address) {
        require(sig.length == 65, "bad sig length");
        bytes32 r;
        bytes32 s;
        uint8   v;
        assembly {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        if (v < 27) v += 27;
        return ecrecover(hash, v, r, s);
    }

    function _safeTransferETH(address to, uint256 amount) internal {
        if (amount == 0) return;
        (bool ok,) = payable(to).call{value: amount}("");
        if (!ok) revert TransferFailed();
    }

    function withdrawProtocolFunds(address to) external onlyOwner nonReentrant {
        uint256 bal = address(this).balance - sequencerBond;
        _safeTransferETH(to, bal);
    }

    receive() external payable {
        sequencerBond += msg.value;
    }
}
