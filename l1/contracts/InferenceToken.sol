// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/Ownable2Step.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title  InferenceToken (INFT)
 * @notice Proof-of-Inference ERC20. Tokens are minted when a miner completes
 *         an off-chain AI inference job and survives a 10-minute challenge window.
 *
 * Storage design (gas efficiency):
 *   Job prompts and outputs are passed as calldata and stored only in EVENT LOGS,
 *   not in contract storage. Only a keccak256 commitment is kept on-chain.
 *   This cuts per-job storage cost by ~10x versus storing bytes in state.
 *
 * Dispute resolution (v1):
 *   Owner-arbitrated. If the owner does not act within DISPUTE_TIMEOUT (48 h),
 *   anyone can call resolveExpiredDispute() to auto-resolve in the miner's favor,
 *   preventing indefinite bond lockup. Will be replaced by zkML in v2.
 *
 * Token emission:
 *   Tiered by output size, halves every HALVING_INTERVAL completed jobs.
 *   Hard cap: 21,000,000 INFT (Bitcoin-style).
 *
 * Struct packing (5 slots before string):
 *   slot 0: requester(20) + maxOutputTokens(7) + status(1)  = 28 bytes
 *   slot 1: miner(20)     + postedAt(8)                     = 28 bytes
 *   slot 2: challenger(20) + submittedAt(8)                 = 28 bytes
 *   slot 3: inputRef(32)
 *   slot 4: outputRef(32)
 *   slot 5+: modelId (string)
 *
 * @custom:security-contact security@inferencetoken.xyz
 */
contract InferenceToken is ERC20Permit, Ownable2Step, Pausable, ReentrancyGuard {

    // ── Custom errors ─────────────────────────────────────────────────────
    error NotOpen();
    error NotClaimed();
    error NotSubmitted();
    error NotDisputed();
    error NotYourJob();
    error ClaimWindowClosed();
    error SubmitDeadlinePassed();
    error ChallengeWindowOpen();
    error ChallengeWindowClosed();
    error DisputeTimeoutNotReached();
    error BondTooLow();
    error FeeTooLow();
    error ZeroOutputTokens();
    error ZeroAddress();
    error AlreadyRegistered();
    error NotExpired();
    error SupplyCapReached();
    error InvalidFeeBps();
    error TransferFailed();

    // ── Emission ──────────────────────────────────────────────────────────
    uint256 public constant MAX_SUPPLY         = 21_000_000e18;
    uint256 public constant HALVING_INTERVAL   = 1_000_000; // jobs
    uint256 public constant BASE_REWARD_SMALL  = 10e18;     // < 512 output tokens
    uint256 public constant BASE_REWARD_MEDIUM = 25e18;     // 512 – 2047
    uint256 public constant BASE_REWARD_LARGE  = 50e18;     // 2048+
    uint256 public totalMinted;
    uint256 public jobsCompleted;

    // ── Timing ───────────────────────────────────────────────────────────
    uint256 public constant CLAIM_WINDOW     =  5 minutes;
    uint256 public constant SUBMIT_DEADLINE  =  3 minutes;
    uint256 public constant CHALLENGE_PERIOD = 10 minutes;
    uint256 public constant DISPUTE_TIMEOUT  = 48 hours;

    // ── Fees ─────────────────────────────────────────────────────────────
    uint256 public constant JOB_FEE    = 0.001 ether;
    uint256 public constant MINER_BOND = 0.005 ether;
    uint256 public protocolFeeBps = 300; // 3 %, max 1000 (10 %)
    uint256 public protocolFees;         // accumulated ETH, withdrawable by owner

    // ── Job status constants ──────────────────────────────────────────────
    uint8 private constant S_OPEN      = 0;
    uint8 private constant S_CLAIMED   = 1;
    uint8 private constant S_SUBMITTED = 2;
    uint8 private constant S_COMPLETE  = 3;
    uint8 private constant S_DISPUTED  = 4;
    uint8 private constant S_EXPIRED   = 5;

    // ── Job (packed) ──────────────────────────────────────────────────────
    struct Job {
        address requester;       // slot 0 ─┐ 20 + 7 + 1 = 28 bytes
        uint56  maxOutputTokens; //          │
        uint8   status;          //          ┘
        address miner;           // slot 1 ─┐ 20 + 8 = 28 bytes
        uint64  postedAt;        //          ┘
        address challenger;      // slot 2 ─┐ 20 + 8 = 28 bytes
        uint64  submittedAt;     //          ┘
        bytes32 inputRef;        // slot 3   keccak256(encryptedInput)
        bytes32 outputRef;       // slot 4   keccak256(encryptedOutput)
        string  modelId;         // slot 5+
    }

    uint256 public nextJobId;
    mapping(uint256 => Job) public jobs;

    // Per-miner stats
    mapping(address => uint256) public minerReputation;
    mapping(address => uint256) public pendingBonds;
    mapping(address => uint256) public totalEarned;
    mapping(address => uint256) public totalCreatorEarned;

    // ── Model registry ────────────────────────────────────────────────────

    struct ModelInfo {
        address creator;
        uint16  royaltyBps;   // basis points, max 5000 = 50 %
        bool    exists;
    }

    mapping(string => ModelInfo) public modelRegistry;
    string[] public modelList;

    // ── Miner registry ────────────────────────────────────────────────────
    struct MinerProfile {
        string[] supportedModels;
        bytes    publicKey;     // RSA-2048 public key (DER-encoded)
        bool     active;
        uint64   registeredAt;
    }

    mapping(address => MinerProfile) private _minerProfiles;
    address[] public registeredMiners;

    // ── Events ────────────────────────────────────────────────────────────
    // NOTE: encryptedInput / encryptedOutput live in event LOGS, not storage.
    event JobPosted(
        uint256 indexed jobId,
        address indexed requester,
        string          modelId,
        bytes32         inputRef,
        bytes           encryptedInput,
        uint56          maxOutputTokens
    );
    event JobClaimed(uint256 indexed jobId, address indexed miner);
    event ResultSubmitted(
        uint256 indexed jobId,
        address indexed miner,
        bytes32         outputRef,
        bytes           encryptedOutput
    );
    event JobFinalized(uint256 indexed jobId, address indexed miner, uint256 tokensEarned);
    event JobDisputed(uint256 indexed jobId, address indexed challenger);
    event DisputeResolved(uint256 indexed jobId, bool minerWon, address indexed resolver);
    event JobExpired(uint256 indexed jobId);
    event ModelRegistered(string modelId, address indexed creator, uint16 royaltyBps);
    event MinerRegistered(address indexed miner, string[] models);
    event MinerDeactivated(address indexed miner);
    event CreatorEarned(uint256 indexed jobId, address indexed creator, uint256 amount);
    event ProtocolFeeBpsUpdated(uint256 oldBps, uint256 newBps);
    event ProtocolFeesWithdrawn(address indexed to, uint256 amount);

    // ── Constructor ───────────────────────────────────────────────────────
    constructor()
        ERC20("InferenceToken", "INFT")
        ERC20Permit("InferenceToken")
        Ownable(msg.sender)
    {}

    // ── Admin ─────────────────────────────────────────────────────────────

    function pause()   external onlyOwner { _pause(); }
    function unpause() external onlyOwner { _unpause(); }

    function setProtocolFeeBps(uint256 bps) external onlyOwner {
        if (bps > 1000) revert InvalidFeeBps();
        emit ProtocolFeeBpsUpdated(protocolFeeBps, bps);
        protocolFeeBps = bps;
    }

    function withdrawProtocolFees(address to) external onlyOwner nonReentrant {
        if (to == address(0)) revert ZeroAddress();
        uint256 amount = protocolFees;
        protocolFees = 0;
        _safeTransferETH(to, amount);
        emit ProtocolFeesWithdrawn(to, amount);
    }

    // ── Model registry ────────────────────────────────────────────────────

    function registerModel(
        string calldata modelId,
        uint16  royaltyBps
    ) external {
        if (modelRegistry[modelId].exists) revert AlreadyRegistered();
        if (royaltyBps > 5000) revert InvalidFeeBps();
        modelRegistry[modelId] = ModelInfo({
            creator:    msg.sender,
            royaltyBps: royaltyBps,
            exists:     true
        });
        modelList.push(modelId);
        emit ModelRegistered(modelId, msg.sender, royaltyBps);
    }

    function getModelInfo(string calldata modelId) external view returns (ModelInfo memory) {
        return modelRegistry[modelId];
    }

    function getModelCount() external view returns (uint256) {
        return modelList.length;
    }

    // ── Miner registry ────────────────────────────────────────────────────

    function registerMiner(
        string[] calldata models,
        bytes    calldata pubKey
    ) external {
        MinerProfile storage p = _minerProfiles[msg.sender];
        bool isNew = !p.active && p.registeredAt == 0;

        delete p.supportedModels;
        for (uint256 i; i < models.length; ++i) p.supportedModels.push(models[i]);
        p.publicKey    = pubKey;
        p.active       = true;
        p.registeredAt = uint64(block.timestamp);

        if (isNew) registeredMiners.push(msg.sender);
        emit MinerRegistered(msg.sender, models);
    }

    function deactivateMiner() external {
        _minerProfiles[msg.sender].active = false;
        emit MinerDeactivated(msg.sender);
    }

    function getMinerProfile(address miner) external view returns (
        string[] memory models,
        bytes    memory pubKey,
        bool            active,
        uint64          registeredAt
    ) {
        MinerProfile storage p = _minerProfiles[miner];
        return (p.supportedModels, p.publicKey, p.active, p.registeredAt);
    }

    function getRegisteredMinerCount() external view returns (uint256) {
        return registeredMiners.length;
    }

    function minersForModel(string calldata modelId, uint256 limit)
        external view returns (address[] memory result)
    {
        result = new address[](limit);
        uint256 found;
        for (uint256 i; i < registeredMiners.length && found < limit; ++i) {
            address m = registeredMiners[i];
            if (!_minerProfiles[m].active) continue;
            string[] storage ms = _minerProfiles[m].supportedModels;
            for (uint256 j; j < ms.length; ++j) {
                if (keccak256(bytes(ms[j])) == keccak256(bytes(modelId))) {
                    result[found++] = m;
                    break;
                }
            }
        }
        assembly { mstore(result, found) }
    }

    // ── Post job ──────────────────────────────────────────────────────────

    /**
     * @notice Post an inference job.
     *
     * The full `encryptedInput` is emitted in the JobPosted event log and
     * can be retrieved cheaply off-chain. Only its keccak256 hash is stored
     * in contract storage to enable on-chain integrity proofs.
     *
     * @param modelId          HuggingFace model path, e.g. "mistralai/Mistral-7B-Instruct-v0.3"
     * @param encryptedInput   Prompt bytes. In MVP mode pass raw UTF-8; in production
     *                         encrypt with the target miner's RSA-2048 public key.
     * @param maxOutputTokens  Upper bound on expected output length; determines reward tier.
     */
    function postJob(
        string  calldata modelId,
        bytes   calldata encryptedInput,
        uint56  maxOutputTokens
    ) external payable whenNotPaused returns (uint256 jobId) {
        if (msg.value < JOB_FEE)        revert FeeTooLow();
        if (maxOutputTokens == 0)        revert ZeroOutputTokens();

        jobId = nextJobId++;
        bytes32 inputRef = keccak256(encryptedInput);

        Job storage j = jobs[jobId];
        j.requester       = msg.sender;
        j.maxOutputTokens = maxOutputTokens;
        j.status          = S_OPEN;
        j.postedAt        = uint64(block.timestamp);
        j.inputRef        = inputRef;
        j.modelId         = modelId;

        emit JobPosted(jobId, msg.sender, modelId, inputRef, encryptedInput, maxOutputTokens);
    }

    // ── Claim ─────────────────────────────────────────────────────────────

    function claimJob(uint256 jobId) external payable whenNotPaused nonReentrant {
        Job storage job = jobs[jobId];
        if (job.status != S_OPEN)                                  revert NotOpen();
        if (block.timestamp >= job.postedAt + CLAIM_WINDOW)        revert ClaimWindowClosed();
        if (msg.value < MINER_BOND)                                revert BondTooLow();

        job.miner  = msg.sender;
        job.status = S_CLAIMED;
        pendingBonds[msg.sender] += msg.value;

        emit JobClaimed(jobId, msg.sender);
    }

    // ── Submit result ─────────────────────────────────────────────────────

    /**
     * @notice Submit inference output. Full `encryptedOutput` lives in event logs only.
     */
    function submitResult(
        uint256 jobId,
        bytes   calldata encryptedOutput
    ) external whenNotPaused {
        Job storage job = jobs[jobId];
        if (job.miner != msg.sender)                                              revert NotYourJob();
        if (job.status != S_CLAIMED)                                              revert NotClaimed();
        if (block.timestamp >= job.postedAt + CLAIM_WINDOW + SUBMIT_DEADLINE)    revert SubmitDeadlinePassed();

        job.outputRef   = keccak256(encryptedOutput);
        job.submittedAt = uint64(block.timestamp);
        job.status      = S_SUBMITTED;

        emit ResultSubmitted(jobId, msg.sender, job.outputRef, encryptedOutput);
    }

    // ── Finalize ──────────────────────────────────────────────────────────

    function finalizeJob(uint256 jobId) external nonReentrant {
        Job storage job = jobs[jobId];
        if (job.status != S_SUBMITTED)                                     revert NotSubmitted();
        if (block.timestamp <= job.submittedAt + CHALLENGE_PERIOD)         revert ChallengeWindowOpen();

        _completeJob(jobId, job.miner);
    }

    // ── Challenge ─────────────────────────────────────────────────────────

    function challengeResult(uint256 jobId) external payable whenNotPaused nonReentrant {
        Job storage job = jobs[jobId];
        if (job.status != S_SUBMITTED)                                     revert NotSubmitted();
        if (block.timestamp >= job.submittedAt + CHALLENGE_PERIOD)         revert ChallengeWindowClosed();
        if (msg.value < MINER_BOND)                                        revert BondTooLow();

        job.challenger = msg.sender;
        job.status     = S_DISPUTED;
        pendingBonds[msg.sender] += msg.value;

        emit JobDisputed(jobId, msg.sender);
    }

    // ── Dispute resolution ────────────────────────────────────────────────

    /// @notice Owner resolves a disputed job. (v1 manual arbitration — v2: zkML)
    function resolveDispute(uint256 jobId, bool minerWon) external onlyOwner nonReentrant {
        Job storage job = jobs[jobId];
        if (job.status != S_DISPUTED) revert NotDisputed();
        _settleDispute(jobId, job, minerWon, msg.sender);
    }

    /**
     * @notice If the owner hasn't resolved within DISPUTE_TIMEOUT, anyone can call
     *         this to auto-resolve in the miner's favour. Prevents bond lockup.
     */
    function resolveExpiredDispute(uint256 jobId) external nonReentrant {
        Job storage job = jobs[jobId];
        if (job.status != S_DISPUTED) revert NotDisputed();
        if (block.timestamp <= job.submittedAt + CHALLENGE_PERIOD + DISPUTE_TIMEOUT)
            revert DisputeTimeoutNotReached();
        _settleDispute(jobId, job, true, address(0));
    }

    // ── Reclaim expired jobs ──────────────────────────────────────────────

    function reclaimExpiredJob(uint256 jobId) external nonReentrant {
        Job storage job = jobs[jobId];
        if (job.requester != msg.sender) revert NotYourJob();

        bool claimExpired  = job.status == S_OPEN &&
            block.timestamp > job.postedAt + CLAIM_WINDOW;
        bool submitExpired = job.status == S_CLAIMED &&
            block.timestamp > job.postedAt + CLAIM_WINDOW + SUBMIT_DEADLINE;

        if (!claimExpired && !submitExpired) revert NotExpired();

        if (submitExpired) {
            address miner = job.miner;
            uint256 bond  = pendingBonds[miner];
            pendingBonds[miner] = 0;
            // Half penalty to requester; half to protocol
            uint256 half = bond / 2;
            protocolFees += bond - half;
            _safeTransferETH(msg.sender, half);
            if (minerReputation[miner] > 0) --minerReputation[miner];
        }

        // Reopen
        job.status   = S_OPEN;
        job.miner    = address(0);
        job.postedAt = uint64(block.timestamp);

        emit JobExpired(jobId);
    }

    // ── Views ─────────────────────────────────────────────────────────────

    function getJob(uint256 jobId) external view returns (Job memory) {
        return jobs[jobId];
    }

    function currentReward(uint256 outputTokens) external view returns (uint256) {
        return _calculateReward(outputTokens);
    }

    // ── Internal ─────────────────────────────────────────────────────────

    function _completeJob(uint256 jobId, address miner) internal {
        Job storage job = jobs[jobId];
        job.status = S_COMPLETE;
        ++jobsCompleted;

        // Protocol fee on the job fee
        uint256 fee = (JOB_FEE * protocolFeeBps) / 10_000;
        protocolFees += fee;

        // Mint tokens
        uint256 reward = _calculateReward(job.maxOutputTokens);
        if (totalMinted + reward > MAX_SUPPLY) reward = MAX_SUPPLY - totalMinted;
        if (reward > 0) {
            totalMinted        += reward;
            totalEarned[miner] += reward;
            _mint(miner, reward);
        }

        // ── Creator royalty ────────────────────────────────────────────
        {
            ModelInfo storage model = modelRegistry[job.modelId];
            if (model.creator != address(0) && model.royaltyBps > 0) {
                uint256 creatorReward = reward * model.royaltyBps / 10_000;
                if (creatorReward > 0) {
                    if (totalMinted + creatorReward > MAX_SUPPLY) {
                        creatorReward = MAX_SUPPLY - totalMinted;
                    }
                    if (creatorReward > 0) {
                        totalMinted += creatorReward;
                        totalCreatorEarned[model.creator] += creatorReward;
                        _mint(model.creator, creatorReward);
                        emit CreatorEarned(jobId, model.creator, creatorReward);
                    }
                }
            }
        }
        // ───────────────────────────────────────────────────────────────

        // Return miner bond
        uint256 bond = pendingBonds[miner];
        pendingBonds[miner] = 0;
        if (bond > 0) _safeTransferETH(miner, bond);

        // Return net job fee to requester
        _safeTransferETH(job.requester, JOB_FEE - fee);

        ++minerReputation[miner];
        emit JobFinalized(jobId, miner, reward);
    }

    function _settleDispute(
        uint256 jobId,
        Job storage job,
        bool minerWon,
        address resolver
    ) internal {
        address miner      = job.miner;
        address challenger = job.challenger;

        if (minerWon) {
            _completeJob(jobId, miner);
            // Challenger bond → protocol (penalty for bad-faith challenge)
            uint256 chalBond = pendingBonds[challenger];
            pendingBonds[challenger] = 0;
            protocolFees += chalBond;
        } else {
            job.status = S_EXPIRED;

            uint256 minerBond = pendingBonds[miner];
            uint256 chalBond  = pendingBonds[challenger];
            pendingBonds[miner]      = 0;
            pendingBonds[challenger] = 0;

            // Challenger gets both bonds as reward for catching a cheating miner
            _safeTransferETH(challenger, minerBond + chalBond);

            // Refund net job fee to requester
            uint256 fee = (JOB_FEE * protocolFeeBps) / 10_000;
            protocolFees += fee;
            _safeTransferETH(job.requester, JOB_FEE - fee);

            if (minerReputation[miner] > 0) --minerReputation[miner];
        }

        emit DisputeResolved(jobId, minerWon, resolver);
    }

    function _calculateReward(uint256 outputTokens) internal view returns (uint256) {
        uint256 base;
        if      (outputTokens < 512)  base = BASE_REWARD_SMALL;
        else if (outputTokens < 2048) base = BASE_REWARD_MEDIUM;
        else                           base = BASE_REWARD_LARGE;

        uint256 halvings = jobsCompleted / HALVING_INTERVAL;
        if (halvings > 20) halvings = 20;
        return base >> halvings;
    }

    function _safeTransferETH(address to, uint256 amount) internal {
        if (amount == 0) return;
        (bool ok,) = payable(to).call{value: amount}("");
        if (!ok) revert TransferFailed();
    }
}
