const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  console.log(
    "Balance:",
    hre.ethers.formatEther(await hre.ethers.provider.getBalance(deployer.address)),
    "ETH"
  );

  const InferenceToken = await hre.ethers.getContractFactory("InferenceToken");
  const token = await InferenceToken.deploy();
  await token.waitForDeployment();

  const address = await token.getAddress();
  console.log("\nInferenceToken deployed to:", address);

  // Register a few well-known models (0 royalty by default for seed models)
  const models = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
  ];
  for (const model of models) {
    await token.registerModel(model, 0);
    console.log("Registered model:", model);
  }

  // Save deployment artifact for SDK / frontend
  const rpcUrl =
    hre.network.config.url ||
    process.env.BASE_SEPOLIA_RPC ||
    process.env.BASE_RPC ||
    "http://127.0.0.1:8545";

  const artifact = {
    network: hre.network.name,
    chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
    rpc_url: rpcUrl,
    address,
    abi: JSON.parse(
      fs.readFileSync(
        path.join(__dirname, "../artifacts/contracts/InferenceToken.sol/InferenceToken.json")
      )
    ).abi,
    deployedAt: new Date().toISOString(),
  };

  const outPath = path.join(__dirname, "../deployment.json");
  fs.writeFileSync(outPath, JSON.stringify(artifact, null, 2));
  console.log("\nDeployment artifact saved to deployment.json");
  console.log("Contract address:", address);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
