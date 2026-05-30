"""
Submit a 2-shard inference job to the local chain and watch it get split.

Run this AFTER the chain node and both miners have started.
Usage: python3 submit_job.py
"""
import json, sys, time, urllib.request

RPC = "http://127.0.0.1:18545"
SEQUENCER_KEY = "4da245a36de729dcbe5263060b146e570674a384a047394fe0491015cf72095f"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Explain the RK3588 SoC in one sentence."
MAX_TOKENS = 128
N_SHARDS = 2
SHARD_MODE = "parallel_sample"


def rpc(method, params):
    req = json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": 1}).encode()
    r = urllib.request.urlopen(
        urllib.request.Request(RPC, data=req, headers={"Content-Type": "application/json"}),
        timeout=30
    )
    resp = json.loads(r.read())
    if "error" in resp:
        raise RuntimeError(f"{method} error: {resp['error']}")
    return resp["result"]


def main():
    # Check chain is up
    info = rpc("inft_getChainInfo", [])
    print(f"\n[chain] block={info['block_number']}  validators={info['validator_count']}  active_jobs={info['active_jobs']}")

    if info["validator_count"] < 2:
        print("[WARN] Only 1 validator — waiting for Khadas miner to stake & connect (up to 30s)...")
        for _ in range(30):
            time.sleep(1)
            info = rpc("inft_getChainInfo", [])
            if info["validator_count"] >= 2:
                print(f"[chain] Now {info['validator_count']} validators — ready!")
                break
        else:
            print("[WARN] Still only 1 validator — the job may assign both shards to the same miner.")

    # Post 2-shard job using sequencer key (already has INFT)
    print(f"\n[job] Posting {N_SHARDS}-shard job: model={MODEL}")
    print(f"[job] Prompt: \"{PROMPT}\"")
    job_id = rpc("inft_postJob", [MODEL, PROMPT, MAX_TOKENS, SHARD_MODE, N_SHARDS, SEQUENCER_KEY])
    print(f"[job] job_id = {job_id}")

    # Poll until complete
    print(f"\n[poll] Waiting for job to complete (timeout 90s)...")
    start = time.monotonic()
    while time.monotonic() - start < 90:
        status = rpc("inft_getJob", [job_id])
        if not status:
            time.sleep(0.5)
            continue

        s = status["status"]
        shards = status.get("shards", {})
        shard_info = " | ".join(
            f"shard{i}: {shards[str(i)]['status']} miner={shards[str(i)].get('miner','?')[:12]}..."
            for i in range(len(shards))
        ) if shards else "(none yet)"

        elapsed = time.monotonic() - start
        print(f"\r[{elapsed:.1f}s] status={s:12s}  {shard_info}", end="", flush=True)

        if s == "complete":
            print()
            print(f"\n[DONE] Job assembled in {elapsed:.1f}s")
            print(f"[output_hash] {status.get('output_hash', 'N/A')}")
            print(f"\n[final output]\n{status.get('final_output', '')}")

            # Show which miners handled each shard
            print("\n[shard assignment]")
            for i in sorted(shards.keys(), key=int):
                sh = shards[i]
                print(f"  Shard {i}: miner={sh.get('miner','?')}  status={sh['status']}")
                if sh.get("output"):
                    print(f"    output: {sh['output'][:80]}...")
            return 0
        elif s == "failed":
            print()
            print("[FAIL] Job failed")
            return 1

        time.sleep(0.5)

    print("\n[TIMEOUT] Job did not complete within 90s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
