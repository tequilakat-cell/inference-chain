/**
 * Deploy InferenceChainRollup + InferenceChainBridge to Sepolia.
 *
 * Usage:
 *   npx hardhat run scripts/deploy_l1.js --network sepolia
 *
 * Reads:
 *   DEPLOYER_PRIVATE_KEY — Sepolia deployer wallet
 *   SEQUENCER_ADDRESS    — address that will call commitStateRoot (can equal deployer)
 *   L1_INFT_ADDRESS      — already-deployed InferenceToken on Sepolia (0x0C6EFB51...)
 *
 * Writes:
 *   l1_deployment.json   — read by rollup_poster.py and bridge/watcher.py
 */

const hre    = require("hardhat");
const fs     = require("fs");
const path   = require("path");

const L1_INFT = process.env.L1_INFT_ADDRESS || "0x0C6EFB51aed8A830AFaABe74D4C721180aB285DB";
const SEQ     = process.env.SEQUENCER_ADDRESS || (async () => {
    const [d] = await hre.ethers.getSigners();
    return d.address;
})();

async function main() {
    const [deployer] = await hre.ethers.getSigners();
    const sequencer  = process.env.SEQUENCER_ADDRESS || deployer.address;

    console.log("Deployer  :", deployer.address);
    console.log("Sequencer :", sequencer);
    console.log("L1 INFT   :", L1_INFT);
    console.log("Network   :", hre.network.name);
    console.log("Balance   :", hre.ethers.formatEther(
        await hre.ethers.provider.getBalance(deployer.address)
    ), "ETH\n");

    // 1. Deploy Rollup
    console.log("Deploying InferenceChainRollup...");
    const Rollup  = await hre.ethers.getContractFactory("InferenceChainRollup");
    const rollup  = await Rollup.deploy(sequencer);
    await rollup.waitForDeployment();
    const rollupAddr = await rollup.getAddress();
    console.log("  InferenceChainRollup →", rollupAddr);

    // 2. Deposit sequencer bond (0.01 ETH)
    const bondTx = await rollup.depositSequencerBond({ value: hre.ethers.parseEther("0.01") });
    await bondTx.wait();
    console.log("  Sequencer bond deposited (0.01 ETH)");

    // 3. Deploy Bridge
    console.log("\nDeploying InferenceChainBridge...");
    const Bridge = await hre.ethers.getContractFactory("InferenceChainBridge");
    const bridge = await Bridge.deploy(L1_INFT, rollupAddr);
    await bridge.waitForDeployment();
    const bridgeAddr = await bridge.getAddress();
    console.log("  InferenceChainBridge →", bridgeAddr);

    // 4. Save deployment artifact
    const artifact = {
        network:        hre.network.name,
        chainId:        (await hre.ethers.provider.getNetwork()).chainId.toString(),
        deployedAt:     new Date().toISOString(),
        sequencer:      sequencer,
        l1_inft:        L1_INFT,
        rollup: {
            address: rollupAddr,
            abi:     JSON.parse(
                fs.readFileSync(
                    path.join(__dirname, "../artifacts/contracts/InferenceChainRollup.sol/InferenceChainRollup.json")
                )
            ).abi,
        },
        bridge: {
            address: bridgeAddr,
            abi:     JSON.parse(
                fs.readFileSync(
                    path.join(__dirname, "../artifacts/contracts/InferenceChainBridge.sol/InferenceChainBridge.json")
                )
            ).abi,
        },
    };

    const outPath = path.join(__dirname, "../l1_deployment.json");
    fs.writeFileSync(outPath, JSON.stringify(artifact, null, 2));

    console.log("\n✓ l1_deployment.json written");
    console.log("  Rollup :", rollupAddr);
    console.log("  Bridge :", bridgeAddr);
    console.log("\nNext steps:");
    console.log("  1. Set L2_ROLLUP_ADDRESS=" + rollupAddr + " in .env");
    console.log("  2. Set L2_BRIDGE_ADDRESS=" + bridgeAddr + " in .env");
    console.log("  3. python -m chain.sequencer --genesis genesis.json");
    console.log("  4. python -m bridge.watcher");
    console.log("  5. python -m miner.l2_miner --config config_l2.json");
}

main().catch(err => { console.error(err); process.exit(1); });
