const { expect }  = require("chai");
const { ethers }  = require("hardhat");
const { time, loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

// ── Fixture ───────────────────────────────────────────────────────────────────

async function deployFixture() {
  const [owner, requester, miner, challenger, other, creator] = await ethers.getSigners();
  const Factory = await ethers.getContractFactory("InferenceToken");
  const token   = await Factory.deploy();

  // Register common test models so postJob calls don't fail
  await token.registerModel("mistralai/Mistral-7B-Instruct-v0.3", 0);
  await token.registerModel("any/model", 0);

  return { token, owner, requester, miner, challenger, other, creator };
}

const JOB_FEE    = ethers.parseEther("0.001");
const MINER_BOND = ethers.parseEther("0.005");

const INPUT  = ethers.toUtf8Bytes("Explain transformers in 3 sentences.");
const OUTPUT = ethers.toUtf8Bytes("Transformers use attention mechanisms…");

async function postJob(token, requester, tokens = 512, modelId = "mistralai/Mistral-7B-Instruct-v0.3") {
  const tx = await token.connect(requester).postJob(
    modelId,
    INPUT,
    tokens,
    { value: JOB_FEE }
  );
  const receipt = await tx.wait();
  for (const log of receipt.logs) {
    try {
      const evt = token.interface.parseLog(log);
      if (evt.name === "JobPosted") return evt.args.jobId;
    } catch {}
  }
  throw new Error("JobPosted event not found");
}

// ── Deployment ────────────────────────────────────────────────────────────────

describe("Deployment", () => {
  it("sets name, symbol, and owner", async () => {
    const { token, owner } = await loadFixture(deployFixture);
    expect(await token.name()).to.equal("InferenceToken");
    expect(await token.symbol()).to.equal("INFT");
    expect(await token.owner()).to.equal(owner.address);
  });

  it("starts with zero supply and zero jobs", async () => {
    const { token } = await loadFixture(deployFixture);
    expect(await token.totalSupply()).to.equal(0n);
    expect(await token.jobsCompleted()).to.equal(0n);
  });

  it("exposes correct constants", async () => {
    const { token } = await loadFixture(deployFixture);
    expect(await token.MAX_SUPPLY()).to.equal(ethers.parseEther("21000000"));
    expect(await token.protocolFeeBps()).to.equal(300n);
  });
});

// ── Model registry ────────────────────────────────────────────────────────────

describe("Model registry", () => {
  it("registers and deduplicates models", async () => {
    const { token, owner } = await loadFixture(deployFixture);
    // Use a model NOT registered by the fixture
    const modelId = "test/registry-test-model";
    await token.registerModel(modelId, 2000);
    const info = await token.getModelInfo(modelId);
    expect(info.creator).to.equal(owner.address);
    expect(info.royaltyBps).to.equal(2000n);
    expect(info.exists).to.be.true;
    // model count = 2 from fixture + 1 new = 3
    expect(await token.getModelCount()).to.equal(3n);
    await expect(token.registerModel(modelId, 2000))
      .to.be.revertedWithCustomError(token, "AlreadyRegistered");
  });

  it("rejects royalty above 50%", async () => {
    const { token } = await loadFixture(deployFixture);
    // Use a model NOT registered by the fixture
    await expect(token.registerModel("test/royalty-cap-model", 5001))
      .to.be.revertedWithCustomError(token, "InvalidFeeBps");
    // 50 % is OK
    await token.registerModel("test/royalty-cap-model", 5000);
    const info = await token.getModelInfo("test/royalty-cap-model");
    expect(info.royaltyBps).to.equal(5000n);
  });

  it("ModelRegistered event includes creator and royalty", async () => {
    const { token, other } = await loadFixture(deployFixture);
    await expect(token.connect(other).registerModel("other/model", 1500))
      .to.emit(token, "ModelRegistered")
      .withArgs("other/model", other.address, 1500);
  });
});

// ── Miner registry ────────────────────────────────────────────────────────────

describe("Miner registry", () => {
  it("registers miner with models and pubkey", async () => {
    const { token, miner } = await loadFixture(deployFixture);
    const models = ["mistralai/Mistral-7B-Instruct-v0.3"];
    const pubKey = ethers.toUtf8Bytes("fake-rsa-pubkey");

    await token.connect(miner).registerMiner(models, pubKey);
    const [ms,, active] = await token.getMinerProfile(miner.address);
    expect(ms).to.deep.equal(models);
    expect(active).to.be.true;
    expect(await token.getRegisteredMinerCount()).to.equal(1n);
  });

  it("returns miners for a given model", async () => {
    const { token, miner } = await loadFixture(deployFixture);
    await token.connect(miner).registerMiner(["mistralai/Mistral-7B-Instruct-v0.3"], "0x");
    const result = await token.minersForModel("mistralai/Mistral-7B-Instruct-v0.3", 10);
    expect(result).to.include(miner.address);
  });

  it("deactivating excludes miner from results", async () => {
    const { token, miner } = await loadFixture(deployFixture);
    await token.connect(miner).registerMiner(["any/model"], "0x");
    await token.connect(miner).deactivateMiner();
    const [,, active] = await token.getMinerProfile(miner.address);
    expect(active).to.be.false;
    const result = await token.minersForModel("any/model", 10);
    expect(result).to.not.include(miner.address);
  });
});

// ── postJob ───────────────────────────────────────────────────────────────────

describe("postJob", () => {
  it("emits JobPosted with full input in log", async () => {
    const { token, requester } = await loadFixture(deployFixture);
    await expect(
      token.connect(requester).postJob("any/model", INPUT, 512, { value: JOB_FEE })
    ).to.emit(token, "JobPosted")
     .withArgs(0n, requester.address, "any/model",
               ethers.keccak256(INPUT), INPUT, 512n);
  });

  it("stores only the input hash (not the bytes) in state", async () => {
    const { token, requester } = await loadFixture(deployFixture);
    const jobId = await postJob(token, requester);
    const job   = await token.getJob(jobId);
    expect(job.inputRef).to.equal(ethers.keccak256(INPUT));
    // encryptedInput is NOT a field in the Job struct — it lives in the event log
  });

  it("reverts without sufficient fee", async () => {
    const { token } = await loadFixture(deployFixture);
    await expect(
      token.postJob("any/model", INPUT, 512, { value: 0 })
    ).to.be.revertedWithCustomError(token, "FeeTooLow");
  });

  it("reverts with zero maxOutputTokens", async () => {
    const { token } = await loadFixture(deployFixture);
    await expect(
      token.postJob("any/model", INPUT, 0, { value: JOB_FEE })
    ).to.be.revertedWithCustomError(token, "ZeroOutputTokens");
  });

  it("reverts when paused", async () => {
    const { token, owner } = await loadFixture(deployFixture);
    await token.connect(owner).pause();
    await expect(
      token.postJob("any/model", INPUT, 512, { value: JOB_FEE })
    ).to.be.revertedWithCustomError(token, "EnforcedPause");
  });
});

// ── Full happy-path lifecycle ─────────────────────────────────────────────────

describe("Happy-path lifecycle", () => {
  it("mints tokens and returns ETH after challenge window", async () => {
    const { token, requester, miner } = await loadFixture(deployFixture);
    const jobId = await postJob(token, requester, 512);

    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);

    // Still in challenge window — cannot finalize
    await expect(token.connect(miner).finalizeJob(jobId))
      .to.be.revertedWithCustomError(token, "ChallengeWindowOpen");

    await time.increase(10 * 60 + 1);

    const minerBefore    = await token.balanceOf(miner.address);
    const requesterEth   = await ethers.provider.getBalance(requester.address);
    await token.connect(miner).finalizeJob(jobId);
    const minerAfter     = await token.balanceOf(miner.address);
    const requesterEthAfter = await ethers.provider.getBalance(requester.address);

    // 512 tokens → 25 INFT (MEDIUM tier, first era)
    expect(minerAfter - minerBefore).to.equal(ethers.parseEther("25"));
    // Requester gets back JOB_FEE minus 3% protocol fee
    const expectedRefund = JOB_FEE - (JOB_FEE * 300n / 10_000n);
    expect(requesterEthAfter - requesterEth).to.equal(expectedRefund);

    expect(await token.jobsCompleted()).to.equal(1n);
    expect(await token.minerReputation(miner.address)).to.equal(1n);
    expect(await token.totalEarned(miner.address)).to.equal(ethers.parseEther("25"));
  });

  it("ResultSubmitted event contains full output bytes", async () => {
    const { token, requester, miner } = await loadFixture(deployFixture);
    const jobId = await postJob(token, requester);
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });

    await expect(token.connect(miner).submitResult(jobId, OUTPUT))
      .to.emit(token, "ResultSubmitted")
      .withArgs(jobId, miner.address, ethers.keccak256(OUTPUT), OUTPUT);

    // outputRef in state = keccak256 of output (NOT the raw bytes)
    const job = await token.getJob(jobId);
    expect(job.outputRef).to.equal(ethers.keccak256(OUTPUT));
  });

  it("accumulates protocol fees", async () => {
    const { token, requester, miner } = await loadFixture(deployFixture);
    const jobId = await postJob(token, requester, 512);
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await time.increase(10 * 60 + 1);
    await token.connect(miner).finalizeJob(jobId);

    const expected = JOB_FEE * 300n / 10_000n; // 3%
    expect(await token.protocolFees()).to.equal(expected);
  });

  it("awards creator royalty on job completion", async () => {
    const { token, requester, miner, creator } = await loadFixture(deployFixture);

    // creator registers a model with 20 % royalty
    const modelId = "creator/custom-model";
    await token.connect(creator).registerModel(modelId, 2000);

    const jobId = await postJob(token, requester, 512, modelId);
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await time.increase(10 * 60 + 1);

    const creatorBefore = await token.balanceOf(creator.address);
    await token.connect(miner).finalizeJob(jobId);
    const creatorAfter  = await token.balanceOf(creator.address);

    // 512 tokens → 25 INFT base reward, 20 % royalty → 5 INFT to creator
    expect(creatorAfter - creatorBefore).to.equal(ethers.parseEther("5"));

    // totalCreatorEarned tracking
    expect(await token.totalCreatorEarned(creator.address)).to.equal(ethers.parseEther("5"));
  });

  it("CreatorEarned event emitted with correct args", async () => {
    const { token, requester, miner, creator } = await loadFixture(deployFixture);

    await token.connect(creator).registerModel("creator/model-v2", 1000); // 10 %
    const jobId = await postJob(token, requester, 2048, "creator/model-v2");
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await time.increase(10 * 60 + 1);

    // 2048 tokens → 50 INFT, 10 % → 5 INFT creator reward
    await expect(token.connect(miner).finalizeJob(jobId))
      .to.emit(token, "CreatorEarned")
      .withArgs(jobId, creator.address, ethers.parseEther("5"));
  });

  it("zero-royalty model does not emit CreatorEarned", async () => {
    const { token, requester, miner } = await loadFixture(deployFixture);
    // "any/model" already registered with 0 royalty in the fixture
    const jobId = await postJob(token, requester, 512, "any/model");
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await time.increase(10 * 60 + 1);

    // Should NOT emit CreatorEarned
    const tx = await token.connect(miner).finalizeJob(jobId);
    const receipt = await tx.wait();
    let creatorEarnedFound = false;
    for (const log of receipt.logs) {
      try {
        const evt = token.interface.parseLog(log);
        if (evt.name === "CreatorEarned") creatorEarnedFound = true;
      } catch {}
    }
    expect(creatorEarnedFound).to.be.false;
  });
});

// ── Reward tiers ──────────────────────────────────────────────────────────────

describe("Reward tiers", () => {
  it("returns correct base rewards before any halving", async () => {
    const { token } = await loadFixture(deployFixture);
    expect(await token.currentReward(100)).to.equal(ethers.parseEther("10"));
    expect(await token.currentReward(512)).to.equal(ethers.parseEther("25"));
    expect(await token.currentReward(2048)).to.equal(ethers.parseEther("50"));
  });
});

// ── Dispute resolution ────────────────────────────────────────────────────────

describe("Dispute resolution", () => {
  async function disputedJob({ token, requester, miner, challenger }) {
    const jobId = await postJob(token, requester, 512);
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await token.connect(challenger).challengeResult(jobId, { value: MINER_BOND });
    return jobId;
  }

  it("owner resolves in miner's favour — miner earns tokens", async () => {
    const fix   = await loadFixture(deployFixture);
    const jobId = await disputedJob(fix);
    const { token, miner, owner } = fix;

    const before = await token.balanceOf(miner.address);
    await token.connect(owner).resolveDispute(jobId, true);
    const after  = await token.balanceOf(miner.address);

    expect(after).to.be.gt(before);
    expect((await token.getJob(jobId)).status).to.equal(3n); // Complete
    // Challenger bond goes to protocol
    expect(await token.protocolFees()).to.be.gt(0n);
  });

  it("owner resolves against miner — challenger gets both bonds", async () => {
    const fix   = await loadFixture(deployFixture);
    const jobId = await disputedJob(fix);
    const { token, challenger, owner } = fix;

    const ethBefore = await ethers.provider.getBalance(challenger.address);
    await token.connect(owner).resolveDispute(jobId, false);
    const ethAfter  = await ethers.provider.getBalance(challenger.address);

    // Challenger receives both bonds (miner + challenger = 0.01 ETH)
    expect(ethAfter - ethBefore).to.equal(MINER_BOND * 2n);
    expect(await token.balanceOf(fix.miner.address)).to.equal(0n);
  });

  it("non-owner cannot resolve", async () => {
    const fix   = await loadFixture(deployFixture);
    const jobId = await disputedJob(fix);
    await expect(fix.token.connect(fix.requester).resolveDispute(jobId, true))
      .to.be.revertedWithCustomError(fix.token, "OwnableUnauthorizedAccount");
  });

  it("anyone can resolve after DISPUTE_TIMEOUT", async () => {
    const fix   = await loadFixture(deployFixture);
    const jobId = await disputedJob(fix);
    const { token, other } = fix;

    // Before timeout
    await expect(token.connect(other).resolveExpiredDispute(jobId))
      .to.be.revertedWithCustomError(token, "DisputeTimeoutNotReached");

    // 10 min challenge + 48 h timeout
    await time.increase(10 * 60 + 48 * 3600 + 1);
    await expect(token.connect(other).resolveExpiredDispute(jobId))
      .to.emit(token, "DisputeResolved")
      .withArgs(jobId, true, ethers.ZeroAddress);
  });
});

// ── Admin / governance ────────────────────────────────────────────────────────

describe("Admin", () => {
  it("owner can pause and unpause", async () => {
    const { token, owner, requester } = await loadFixture(deployFixture);
    await token.connect(owner).pause();
    await expect(
      token.connect(requester).postJob("m", INPUT, 100, { value: JOB_FEE })
    ).to.be.revertedWithCustomError(token, "EnforcedPause");
    await token.connect(owner).unpause();
    await expect(
      token.connect(requester).postJob("m", INPUT, 100, { value: JOB_FEE })
    ).to.not.be.reverted;
  });

  it("owner can update protocol fee (max 10%)", async () => {
    const { token, owner } = await loadFixture(deployFixture);
    await token.connect(owner).setProtocolFeeBps(500);
    expect(await token.protocolFeeBps()).to.equal(500n);
    await expect(token.connect(owner).setProtocolFeeBps(1001))
      .to.be.revertedWithCustomError(token, "InvalidFeeBps");
  });

  it("owner can withdraw protocol fees", async () => {
    const { token, owner, requester, miner } = await loadFixture(deployFixture);
    const jobId = await postJob(token, requester, 512);
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await time.increase(10 * 60 + 1);
    await token.connect(miner).finalizeJob(jobId);

    const fees = await token.protocolFees();
    expect(fees).to.be.gt(0n);

    const before = await ethers.provider.getBalance(owner.address);
    const tx     = await token.connect(owner).withdrawProtocolFees(owner.address);
    const receipt = await tx.wait();
    const gasUsed  = receipt.gasUsed * tx.gasPrice;
    const after  = await ethers.provider.getBalance(owner.address);

    expect(await token.protocolFees()).to.equal(0n);
    expect(after + gasUsed - before).to.equal(fees);
  });

  it("uses Ownable2Step for safe ownership transfer", async () => {
    const { token, owner, other } = await loadFixture(deployFixture);
    await token.connect(owner).transferOwnership(other.address);
    // Pending — not yet transferred
    expect(await token.owner()).to.equal(owner.address);
    await token.connect(other).acceptOwnership();
    expect(await token.owner()).to.equal(other.address);
  });
});

// ── EIP-2612 permit ───────────────────────────────────────────────────────────

describe("ERC20Permit (EIP-2612)", () => {
  it("supports permit for gasless approvals", async () => {
    const { token, requester, miner } = await loadFixture(deployFixture);
    // Mint some tokens to requester first via a completed job
    const jobId = await postJob(token, requester, 512);
    await token.connect(miner).claimJob(jobId, { value: MINER_BOND });
    await token.connect(miner).submitResult(jobId, OUTPUT);
    await time.increase(10 * 60 + 1);
    await token.connect(miner).finalizeJob(jobId);

    // Build permit signature
    const amount   = ethers.parseEther("10");
    const deadline = Math.floor(Date.now() / 1000) + 3600;
    const nonce    = await token.nonces(miner.address);
    const domain   = {
      name: "InferenceToken",
      version: "1",
      chainId: (await ethers.provider.getNetwork()).chainId,
      verifyingContract: await token.getAddress(),
    };
    const types = {
      Permit: [
        { name:"owner",    type:"address" },
        { name:"spender",  type:"address" },
        { name:"value",    type:"uint256" },
        { name:"nonce",    type:"uint256" },
        { name:"deadline", type:"uint256" },
      ],
    };
    const sig = await miner.signTypedData(domain, types, {
      owner: miner.address, spender: requester.address,
      value: amount, nonce, deadline,
    });
    const { v, r, s } = ethers.Signature.from(sig);

    await token.permit(miner.address, requester.address, amount, deadline, v, r, s);
    expect(await token.allowance(miner.address, requester.address)).to.equal(amount);
  });
});
