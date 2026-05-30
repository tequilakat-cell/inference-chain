/**
 * Quick smoke test — post a single inference job to a local hardhat node.
 * Run after: npm run node  (in one terminal)
 *            npm run deploy:local  (in another)
 *
 * Usage: npx hardhat run scripts/post_test_job.js --network localhost
 */
const hre = require("hardhat");
const fs  = require("fs");

async function main() {
  const [, user] = await hre.ethers.getSigners();  // use second account as user
  const dep = JSON.parse(fs.readFileSync("deployment.json"));
  const token = await hre.ethers.getContractAt("InferenceToken", dep.address, user);

  const modelId   = "mistralai/Mistral-7B-Instruct-v0.3";
  const prompt    = "Explain what a transformer neural network is in 3 sentences.";
  const encInput  = hre.ethers.toUtf8Bytes(prompt);
  const maxTokens = 512;

  // Ensure model is registered (idempotent via try/catch)
  try { await token.registerModel(modelId, 0); } catch (_) {}

  console.log("Posting job...");
  const tx = await token.postJob(
    modelId,
    encInput,
    maxTokens,
    { value: hre.ethers.parseEther("0.001") }
  );
  const receipt = await tx.wait();

  let jobId;
  for (const log of receipt.logs) {
    try {
      const evt = token.interface.parseLog(log);
      if (evt.name === "JobPosted") { jobId = evt.args.jobId; break; }
    } catch (_) {}
  }

  console.log(`Job #${jobId} posted!`);
  console.log("Model:      mistralai/Mistral-7B-Instruct-v0.3");
  console.log("Prompt:    ", prompt);
  console.log("MaxTokens: ", maxTokens);
  console.log("Fee paid:   0.001 ETH");
  console.log("\nNow start the miner to pick it up:");
  console.log("  cd miner && python miner.py --config config.json");
}

main().catch(console.error);
