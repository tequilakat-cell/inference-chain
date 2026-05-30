const { useState, useEffect, useRef, useCallback } = React;

// ── Defaults ─────────────────────────────────────────────────────────────────
const _host = window.location.hostname;
const _origin = window.location.origin;
// When loaded via file:// or an unknown origin, fall back to the chain's direct address.
const DEFAULT_L2_RPC =
  (_host === "127.0.0.1" || _host === "localhost") ? "http://127.0.0.1:8545" :
  (_origin && _origin !== "null")                  ? _origin + "/rpc" :
                                                     "http://192.168.198.48:18545";
const DEFAULT_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct";

// ── Pipeline step definitions ─────────────────────────────────────────────────
const PIPE_STEPS = [
  { key: "post",     n: "01", name: "POST",       sub: "Submitted to L2 mempool",                   icon: "↑" },
  { key: "block",    n: "02", name: "BLOCK",       sub: "Included in L2 block",                      icon: "⬡" },
  { key: "vrf",      n: "03", name: "VRF ASSIGN",  sub: "Miners selected deterministically",         icon: "⚄" },
  { key: "context",  n: "04", name: "CTX LOAD",    sub: "Context chunks pre-loaded across miners",   icon: "◎" },
  { key: "offer",    n: "05", name: "SHARD OFFER", sub: "Offers broadcast over P2P",                 icon: "→" },
  { key: "infer",    n: "06", name: "INFERENCE",   sub: "Parallel CPU/GPU running",                  icon: "⚙" },
  { key: "result",   n: "07", name: "RESULT",      sub: "Output submitted on-chain",                 icon: "✓" },
  { key: "assemble", n: "08", name: "ASSEMBLE",    sub: "Shards merged, answer ready",               icon: "◈" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────
const shortAddr = (a = "", n = 6) => a ? `${a.slice(0, 2 + n)}…${a.slice(-4)}` : "—";
const fmtMs     = ms => ms < 1000 ? `${ms}ms` : `${(ms/1000).toFixed(1)}s`;
const now       = () => Date.now();

function useInterval(fn, delay) {
  const cb = useRef(fn);
  useEffect(() => { cb.current = fn; }, [fn]);
  useEffect(() => {
    if (delay == null) return;
    const id = setInterval(() => cb.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

// ── JSON-RPC call ─────────────────────────────────────────────────────────────
async function rpc(url, method, params = []) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", method, params, id: 1 }),
  });
  const d = await r.json();
  if (d.error) throw new Error(d.error.message || JSON.stringify(d.error));
  return d.result;
}

// ── Top bar ───────────────────────────────────────────────────────────────────
function TopBar({ chain, connected, onSettings, walletAddr, onConnect }) {
  return (
    <div className="topbar">
      <div className="tb-brand">
        <div className="tb-diamond" />
        InferenceChain
      </div>

      <div className="tb-stat">
        <div className={`dot ${connected ? "" : "bad"}`} style={{ background: connected ? "var(--good)" : "var(--bad)" }}/>
        <span>{connected ? "LIVE" : "OFFLINE"}</span>
      </div>

      {chain && <>
        <div className="tb-stat">Block <b>#{chain.block_number}</b></div>
        <div className="tb-stat">Chain <b>{chain.chain_id}</b></div>
        <div className="tb-stat">TPS <b>{chain.tps}</b></div>
        <div className="tb-stat">Jobs <b>{chain.active_jobs}</b></div>
        <div className="tb-stat">Validators <b>{chain.validator_count}</b></div>
      </>}

      <a className="tb-stat" href="miners.html" style={{ textDecoration: "none", color: "inherit" }}>
        Miners
      </a>
      <a className="tb-stat" href="index.html" style={{ textDecoration: "none", color: "inherit" }}>
        ← Landing
      </a>

      <div className="tb-spacer" />

      <button
        className={`tb-btn wallet-btn ${walletAddr ? "wallet-connected" : ""}`}
        onClick={onConnect}
        title={walletAddr ? walletAddr : "Connect Wallet"}
        style={{ width: "auto", padding: "0 10px", fontSize: 10, fontFamily: "var(--mono)",
                 letterSpacing: ".04em", gap: 6, display: "flex", alignItems: "center" }}
      >
        <span style={{ fontSize: 12 }}>◈</span>
        {walletAddr ? shortAddr(walletAddr, 4) : "Connect Wallet"}
      </button>

      <button className="tb-btn" onClick={onSettings} title="Settings" style={{ marginLeft: 4 }}>⚙</button>
    </div>
  );
}

// ── Miner card ────────────────────────────────────────────────────────────────
function MinerCard({ miner, activeJobCount }) {
  // Accept models as either a JSON string (OrbitDB) or a plain array (RPC)
  const models = (() => {
    if (Array.isArray(miner.models)) return miner.models;
    try { return JSON.parse(miner.models || "[]"); } catch { return []; }
  })();
  const backendClass = {
    cpu: "badge-cpu", cuda: "badge-cuda", mlx: "badge-mlx", vulkan: "badge-vulkan"
  }[miner.backend] || "badge-cpu";

  const isActive = activeJobCount > 0;
  const repPct   = Math.min(100, ((miner.reputation || 500) / 1000) * 100);
  const secAgo   = Math.floor((now() - (miner.lastSeen || now())) / 1000);
  const online   = secAgo < 120;

  return (
    <div className={`miner-card ${isActive ? "active" : ""}`}>
      <div className="miner-addr">{shortAddr(miner.address)}</div>
      <div className="miner-meta">
        <span className={`backend-badge ${backendClass}`}>{(miner.backend||"cpu").toUpperCase()}</span>
        <span className="tag">{miner.maxShards || 4} shards</span>
      </div>
      <div className="rep-bar">
        <div className="rep-fill" style={{ width: `${repPct}%` }} />
      </div>
      <div className="miner-status">
        <div className={`status-dot ${isActive ? "working" : online ? "idle" : ""}`} />
        <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)" }}>
          {isActive ? `${activeJobCount} shard${activeJobCount > 1 ? "s" : ""} active`
                    : online ? "idle" : `${secAgo}s ago`}
        </span>
      </div>
      {models.slice(0, 1).map(m => (
        <div key={m} style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--ink-3)", marginTop: 4 }}>
          {m.split("/")[1] || m}
        </div>
      ))}
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ miners, chainStats, activeMiners }) {
  return (
    <div className="sidebar">
      <div className="sb-section">Miners · {miners.length}</div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {miners.length === 0 && (
          <div style={{ padding: "20px 14px", fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)", textAlign: "center" }}>
            No miners online.<br/>Start a miner node first.
          </div>
        )}
        {miners.map(m => (
          <MinerCard key={m.address} miner={m} activeJobCount={activeMiners[m.address] || 0} />
        ))}
      </div>

      <hr className="sb-divider" />
      <div className="sb-section">Chain Stats</div>
      {chainStats && <>
        <div className="sb-stat-row"><span>INFT minted</span><b>{chainStats.totalMinted || "—"}</b></div>
        <div className="sb-stat-row"><span>Jobs done</span><b>{chainStats.jobsCompleted || "—"}</b></div>
      </>}
      <div style={{ height: 12 }} />
    </div>
  );
}

// ── Per-shard breakdown ───────────────────────────────────────────────────────
function ShardGrid({ shards, nShards, jobDone, mode }) {
  const entries = Array.from({ length: nShards }, (_, i) => ({
    idx: i, ...(shards?.[String(i)] || {}),
  }));

  const eff = (s) => (s.status === "offered" && jobDone) ? "superseded" : (s.status || "unassigned");

  const icon  = s => ({ complete:"✓", submitted:"✓", computing:"●", offered:"→", assigned:"○", superseded:"—" }[s] || "·");
  const color = s => ({ complete:"var(--good)", submitted:"var(--good)", computing:"var(--warn)",
                         offered:"var(--warn)", assigned:"var(--ink-2)", superseded:"var(--ink-3)" }[s] || "var(--ink-3)");

  const hasSuperseded = jobDone && entries.some(e => eff(e) === "superseded");

  return (
    <div className="shard-grid">
      {entries.map(s => {
        const e = eff(s);
        return (
          <div key={s.idx} className="shard-row">
            <span style={{ color: color(e), fontFamily:"var(--mono)", fontSize:10, width:14, textAlign:"center" }}>{icon(e)}</span>
            <span style={{ fontFamily:"var(--mono)", fontSize:9, color:"var(--ink-3)", minWidth:16 }}>S{s.idx}</span>
            <span style={{ fontFamily:"var(--mono)", fontSize:9, color: s.miner ? "var(--ink-2)" : "var(--ink-3)", flex:1 }}>
              {s.miner ? shortAddr(s.miner) : "waiting…"}
            </span>
            <span style={{ fontFamily:"var(--mono)", fontSize:9, color: color(e) }}>{e}</span>
          </div>
        );
      })}
      {hasSuperseded && (
        <div style={{ fontFamily:"var(--mono)", fontSize:8, color:"var(--ink-3)", marginTop:4 }}>
          {mode === "parallel_sample" ? "parallel_sample: first result wins" : "superseded"}
        </div>
      )}
    </div>
  );
}

// ── Inline pipeline (inside a chat bubble) ────────────────────────────────────
function InlinePipeline({ steps }) {
  return (
    <div className="inline-pipe">
      <div className="pipe-steps">
        {PIPE_STEPS.map(def => {
          const s   = steps[def.key] || {};
          const cls = s.done ? "done" : s.active ? "active" : s.failed ? "failed" : "";
          return (
            <div key={def.key} className={`pipe-step ${cls}`}>
              <div className="ps-dot" style={s.failed ? { background: "var(--bad)" } : {}} />
              <div className="ps-n">{def.n}</div>
              <div className="ps-name" style={s.failed ? { color: "var(--bad)" } : {}}>{def.name}</div>
              {s.detail && <div className="ps-detail" style={s.failed ? { color: "var(--bad)" } : {}}>{s.detail}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Chat message ──────────────────────────────────────────────────────────────
function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`msg-row ${isUser ? "user" : ""}`}>
      <div className={`msg-avatar ${isUser ? "avatar-user" : "avatar-ai"}`}>
        {isUser ? "you" : "IC"}
      </div>
      <div className="msg-body">
        <div className="msg-meta">
          {isUser ? "YOU" : "INFERENCECHAIN"}
          {" · "}{new Date(msg.ts).toLocaleTimeString()}
          {msg.jobId && ` · Job #${msg.jobId.slice(0, 8)}`}
          {msg.openMode && <span style={{ marginLeft: 6, color: "var(--warn)", fontFamily: "var(--mono)", fontSize: 9 }}>OPEN</span>}
        </div>
        <div className={`bubble ${isUser ? "bubble-user" : msg.thinking ? "bubble-thinking" : "bubble-ai"}`}>
          {!isUser && msg.steps && (
            <InlinePipeline steps={msg.steps} />
          )}
          {msg.text
            ? <div className="msg-text">{msg.text}</div>
            : msg.thinking
            ? <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>
                {msg.statusText || "Processing…"}
              </div>
            : null
          }
          {msg.done && msg.latencyMs && (
            <div className="msg-footer">
              <span className="good">✓ Complete</span>
              <span>{fmtMs(msg.latencyMs)}</span>
              {msg.minersUsed && <span>{msg.minersUsed} miner{msg.minersUsed > 1 ? "s" : ""}</span>}
              {msg.mode && <span>{msg.mode}</span>}
              {msg.contextEntries > 0 && (
                <span style={{ color: "var(--accent)", fontFamily: "var(--mono)", fontSize: 9,
                               background: "rgba(212,113,42,0.12)", padding: "1px 5px", borderRadius: 3 }}>
                  ctx:{msg.contextEntries}
                </span>
              )}
              {msg.allCacheHit && (
                <span style={{ color: "var(--good)", fontFamily: "var(--mono)", fontSize: 9,
                               background: "rgba(68,200,100,0.12)", padding: "1px 5px", borderRadius: 3 }}>
                  ⚡ KV cache
                </span>
              )}
            </div>
          )}
          {msg.error && (
            <div style={{ color: "var(--bad)", fontFamily: "var(--mono)", fontSize: 11, marginTop: 8 }}>
              ✗ {msg.error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── History entry ─────────────────────────────────────────────────────────────
function HistoryEntry({ entry }) {
  const [open, setOpen] = useState(false);
  const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : "?";
  const model = entry.model_id?.split("/")?.[1] || entry.model_id || "?";
  return (
    <div style={{ marginBottom: 12, borderBottom: "1px solid var(--border)", paddingBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center",
                    cursor: "pointer", marginBottom: 4 }}
           onClick={() => setOpen(o => !o)}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--ink-3)" }}>{model} · {ts}</span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--ink-3)" }}>{open ? "▲" : "▼"}</span>
      </div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--warn)", lineHeight: 1.5 }}>
        Q: {entry.prompt?.slice(0, open ? 9999 : 80)}{!open && (entry.prompt?.length || 0) > 80 ? "…" : ""}
      </div>
      {open && (
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-2)",
                      lineHeight: 1.5, marginTop: 4, whiteSpace: "pre-wrap" }}>
          A: {entry.output}
        </div>
      )}
      {!open && (
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)", lineHeight: 1.5 }}>
          A: {entry.output?.slice(0, 80)}{(entry.output?.length || 0) > 80 ? "…" : ""}
        </div>
      )}
    </div>
  );
}

// ── Right pipeline / history column ──────────────────────────────────────────
function PipelineColumn({ currentJob, historyAddr, l2Rpc, model }) {
  const [pipeTab,  setPipeTab]  = useState("pipeline");
  const [history,  setHistory]  = useState([]);
  const [histLoad, setHistLoad] = useState(false);

  const loadHistory = useCallback(async () => {
    if (!historyAddr) return;
    setHistLoad(true);
    try {
      const h = await rpc(l2Rpc, "inft_getHistory", [historyAddr, null, 20]);
      setHistory(Array.isArray(h) ? h : []);
    } catch { setHistory([]); }
    finally { setHistLoad(false); }
  }, [historyAddr, l2Rpc]);

  useEffect(() => {
    if (pipeTab === "history") loadHistory();
  }, [pipeTab, historyAddr]);

  // Reload history after each job completes
  useEffect(() => {
    if (currentJob?.done && pipeTab === "history") loadHistory();
  }, [currentJob?.done]);

  const steps  = currentJob?.steps  || {};
  const shards = currentJob?.shards || null;
  const nShards = currentJob?.nShards || 1;
  const jobDone = currentJob?.done   || false;
  const mode    = currentJob?.mode   || "parallel_sample";

  const tabStyle = (active) => ({
    fontFamily: "var(--mono)", fontSize: 9, letterSpacing: ".06em", cursor: "pointer",
    padding: "3px 10px", border: "1px solid",
    borderColor: active ? "var(--accent)" : "var(--border)",
    color: active ? "var(--accent)" : "var(--ink-3)",
    background: active ? "rgba(212,113,42,0.08)" : "transparent",
    borderRadius: 3,
  });

  return (
    <div className="pipeline-col">
      <div className="pc-header" style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button style={tabStyle(pipeTab === "pipeline")} onClick={() => setPipeTab("pipeline")}>Pipeline</button>
        <button style={tabStyle(pipeTab === "history")} onClick={() => setPipeTab("history")}>History</button>
        {nShards > 1 && pipeTab === "pipeline" && (
          <span style={{ marginLeft: "auto", color: "var(--accent)", fontSize: 9 }}>{nShards} SHARDS</span>
        )}
        {pipeTab === "history" && historyAddr && (
          <button onClick={loadHistory}
                  style={{ marginLeft: "auto", fontFamily: "var(--mono)", fontSize: 9, cursor: "pointer",
                           background: "none", border: "none", color: "var(--ink-3)", padding: 0 }}>
            ↺
          </button>
        )}
      </div>

      {pipeTab === "pipeline" && (
        <>
          {PIPE_STEPS.map(def => {
            const s   = steps[def.key] || {};
            const cls = s.done ? "done" : s.active ? "active" : s.failed ? "failed" : "";
            return (
              <div key={def.key} className={`pc-step ${cls}`}>
                <div className="pc-icon" style={s.failed ? { color: "var(--bad)" } : {}}>
                  {s.done ? "✓" : s.active ? "●" : s.failed ? "✗" : def.icon}
                </div>
                <div className="pc-content">
                  <div className="pc-name" style={s.failed ? { color: "var(--bad)" } : {}}>{def.n} · {def.name}</div>
                  <div className="pc-sub" style={s.failed ? { color: "var(--bad)" } : {}}>{s.detail || def.sub}</div>
                  {shards && nShards > 1 && (def.key === "vrf" || def.key === "infer") && (s.done || s.active) && (
                    <ShardGrid shards={shards} nShards={nShards} jobDone={jobDone} mode={mode} />
                  )}
                  {s.ts != null && <div className="pc-time">+{fmtMs(s.ts)}</div>}
                </div>
              </div>
            );
          })}
          {!currentJob && (
            <div style={{ padding: 20, fontFamily: "var(--mono)", fontSize: 10,
                          color: "var(--ink-3)", textAlign: "center", lineHeight: 1.7 }}>
              Submit a prompt to see<br/>the live job pipeline
            </div>
          )}
        </>
      )}

      {pipeTab === "history" && (
        <div style={{ flex: 1, overflowY: "auto", padding: "8px 12px" }}>
          {!historyAddr ? (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)",
                          textAlign: "center", padding: 20, lineHeight: 1.7 }}>
              Connect wallet or post in<br/>open mode to see history
            </div>
          ) : histLoad ? (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)",
                          textAlign: "center", padding: 20 }}>
              Loading…
            </div>
          ) : history.length === 0 ? (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--ink-3)",
                          textAlign: "center", padding: 20, lineHeight: 1.7 }}>
              No history yet.<br/>Post a job to start.
            </div>
          ) : (
            <>
              <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--ink-3)",
                            marginBottom: 10, letterSpacing: ".04em" }}>
                {history.length} EXCHANGE{history.length !== 1 ? "S" : ""} · {shortAddr(historyAddr)}
              </div>
              {history.map((entry, i) => (
                <HistoryEntry key={entry.job_id || i} entry={entry} />
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Settings modal ────────────────────────────────────────────────────────────
function SettingsModal({ settings, onChange, onClose }) {
  const [local, setLocal] = useState(settings);
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h2>Connection Settings</h2>

        <div className="form-field">
          <label className="form-label">L2 JSON-RPC URL</label>
          <input className="form-input" value={local.l2Rpc}
            onChange={e => setLocal(s => ({...s, l2Rpc: e.target.value}))} />
          <div className="form-hint">InferenceChain sequencer RPC (default: 127.0.0.1:8545)</div>
        </div>

        <div className="modal-btns">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => { onChange(local); onClose(); }}>Save</button>
        </div>
      </div>
    </div>
  );
}

// ── Root app ──────────────────────────────────────────────────────────────────
function App() {
  const savedSettings = (() => {
    try { return JSON.parse(localStorage.getItem("ic-dashboard-settings") || "{}"); } catch { return {}; }
  })();

  // Discard a saved RPC URL if it points to a different host than the page —
  // it's stale from a previous deployment and will appear "unreachable".
  const savedRpc = savedSettings.l2Rpc;
  const savedRpcSameOrigin = savedRpc && (_origin === "null" || savedRpc.startsWith(_origin));
  const [settings, setSettings] = useState({
    l2Rpc: (savedRpcSameOrigin ? savedRpc : null) || DEFAULT_L2_RPC,
  });
  const [showSettings, setShowSettings] = useState(false);

  // Wallet state
  const [walletAddr, setWalletAddr] = useState(null);

  useEffect(() => {
    if (!window.ethereum) return;
    window.ethereum.request({ method: "eth_accounts" })
      .then(accounts => { if (accounts[0]) setWalletAddr(accounts[0]); })
      .catch(() => {});
    window.ethereum.on("accountsChanged", accounts => setWalletAddr(accounts[0] || null));
  }, []);

  const connectWallet = useCallback(async () => {
    if (!window.ethereum) {
      alert("No wallet detected. Please install MetaMask or a compatible browser wallet.");
      return;
    }
    try {
      const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
      setWalletAddr(accounts[0] || null);
    } catch (err) {
      console.error("Wallet connect failed:", err);
    }
  }, []);

  // Form state
  const [model,     setModel]     = useState(DEFAULT_MODEL);
  const [nShards,   setNShards]   = useState(1);
  const [shardMode, setShardMode] = useState("parallel_sample");
  const [prompt,    setPrompt]    = useState("");
  const [posting,   setPosting]   = useState(false);

  // Chain / miner state
  const [chain,        setChain]        = useState(null);
  const [chainStats,   setChainStats]   = useState(null);
  const [connected,    setConnected]    = useState(false);
  const [miners,       setMiners]       = useState([]);
  const [models,       setModels]       = useState([DEFAULT_MODEL]);
  const [minersByModel,setMinersByModel]= useState({});  // model_id → miner count
  const [activeMiners, setActiveMiners] = useState({});

  // Context count state
  const [ctxCount, setCtxCount] = useState(0);

  // Sequencer address (from chain info) — used as history address in open mode
  const seqAddr = chain?.sequencer_address || null;

  // Effective history address: wallet if connected, else sequencer for open-mode jobs
  const historyAddr = walletAddr || seqAddr;

  // Chat state
  const [messages, setMessages] = useState([
    {
      id: 0, role: "assistant", ts: Date.now(),
      text: "Welcome to InferenceChain. Connect your wallet and post a prompt, or use Quick Send ⚡ to submit without a wallet (open mode).",
      done: true,
    }
  ]);
  const [currentJob, setCurrentJob] = useState(null);
  const chatRef = useRef(null);
  const msgId   = useRef(1);

  const scrollChat = useCallback(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, []);
  useEffect(scrollChat, [messages]);

  const saveSettings = useCallback(s => {
    setSettings(s);
    localStorage.setItem("ic-dashboard-settings", JSON.stringify(s));
  }, []);

  // Poll chain info
  const loadChain = useCallback(async () => {
    try {
      const info = await rpc(settings.l2Rpc, "inft_getChainInfo", []);
      setChain(info);
      setConnected(true);
    } catch { setConnected(false); }
  }, [settings.l2Rpc]);

  const loadMiners = useCallback(async () => {
    let minersArr = [];
    try {
      // Primary: sequencer RPC — authoritative heartbeat data
      const active = await rpc(settings.l2Rpc, "inft_getActiveMiners", []);
      if (Array.isArray(active) && active.length > 0) minersArr = active;
    } catch {}

    setMiners(minersArr);

    // Rebuild model list and per-model miner count from live miner data
    const allModels = new Set([DEFAULT_MODEL]);
    const modelCount = {};
    minersArr.forEach(m => {
      const mods = Array.isArray(m.models) ? m.models
        : (() => { try { return JSON.parse(m.models || "[]"); } catch { return []; } })();
      mods.forEach(mid => {
        allModels.add(mid);
        modelCount[mid] = (modelCount[mid] || 0) + 1;
      });
    });
    setModels([...allModels]);
    setMinersByModel(modelCount);
  }, [settings.l2Rpc]);

  // Refresh context count for input area chip
  const refreshCtxCount = useCallback(async () => {
    if (!historyAddr) { setCtxCount(0); return; }
    try {
      const h = await rpc(settings.l2Rpc, "inft_getHistory", [historyAddr, model, 20]);
      setCtxCount(Array.isArray(h) ? h.length : 0);
    } catch { setCtxCount(0); }
  }, [historyAddr, model, settings.l2Rpc]);

  useInterval(loadChain,       2000);
  useInterval(loadMiners,      5000);
  useInterval(refreshCtxCount, 10000);
  useEffect(() => { loadChain(); loadMiners(); }, [settings]);
  useEffect(() => { refreshCtxCount(); }, [historyAddr, model]);

  // Auto-update nShards when model changes or miner availability changes
  useEffect(() => {
    const count = minersByModel[model];
    if (count > 0) setNShards(count);
  }, [model, minersByModel]);

  // Poll active job
  const pollJob = useCallback(async (jobId, msgIdTarget, startMs) => {
    const poll = async () => {
      try {
        const job = await rpc(settings.l2Rpc, "inft_getJob", [jobId]);
        if (!job) { setTimeout(poll, 1500); return; }

        const status     = job.status || "pending";
        const elapsed    = now() - startMs;
        const shardsData = job.shards  || {};
        const nShards    = job.n_shards || 1;
        const jobMode    = job.mode    || "parallel_sample";
        const ctxEntries = job.context_entries || 0;

        let ctxLoad = null;
        try { ctxLoad = await rpc(settings.l2Rpc, "inft_getContextLoad", [jobId]); } catch {}

        const assignedMiners = [...new Set(
          Object.values(shardsData).map(s => s.miner).filter(Boolean)
        )];
        const minerLabel = assignedMiners.length
          ? assignedMiners.map(a => shortAddr(a)).join(" · ")
          : `${nShards} shard${nShards > 1 ? "s" : ""}`;

        const liveMinerMap = {};
        Object.values(shardsData).forEach(s => {
          if (s.miner && s.status !== "complete" && s.status !== "submitted") {
            const k = s.miner.toLowerCase();
            liveMinerMap[k] = (liveMinerMap[k] || 0) + 1;
          }
        });
        setActiveMiners(liveMinerMap);

        const ctxDetail = (() => {
          if (ctxLoad && ctxLoad.miners && ctxLoad.miners.length > 0) {
            const confirmed = ctxLoad.confirmed?.length || 0;
            const total     = ctxLoad.miners.length;
            let d = `${confirmed}/${total} miners`;
            if (ctxEntries > 0) d += ` · ${ctxEntries} exchange${ctxEntries !== 1 ? "s" : ""}`;
            if (ctxLoad.all_cache_hit) d += " · ⚡ cached";
            return d;
          }
          return ctxEntries > 0
            ? `${ctxEntries} exchange${ctxEntries !== 1 ? "s" : ""} loaded`
            : "No prior history";
        })();

        const buildSteps = () => {
          const done   = (detail = null, ts = null) => ({ done: true,  active: false, detail, ts });
          const active = (detail = null)             => ({ done: false, active: true,  detail });
          const pend   = ()                          => ({ done: false, active: false });
          const fail   = (detail = null)             => ({ done: false, active: false, failed: true, detail });

          if (status === "failed") {
            const timedOut = Object.values(shardsData).filter(s =>
              s.status === "timeout" || s.status === "slashed").length;
            return {
              post:     done(`Job ${jobId.slice(0, 8)}`, 0),
              block:    done(`Block #${job.block_number || "?"}`, 10),
              vrf:      done(minerLabel, 20),
              context:  done(ctxDetail, 25),
              offer:    done("Sent", 30),
              infer:    fail(`${timedOut} shard${timedOut !== 1 ? "s" : ""} timed out`),
              result:   fail("No result"),
              assemble: fail("Job failed"),
            };
          }

          if (status === "complete") {
            const doneShards = Object.values(shardsData).filter(s =>
              s.status === "complete" || s.status === "submitted").length;
            return {
              post:     done(`Job ${jobId.slice(0, 8)}`, 0),
              block:    done(`Block #${job.block_number || "?"}`, 10),
              vrf:      done(minerLabel, 20),
              context:  done(ctxDetail, 25),
              offer:    done(`${nShards} offer${nShards > 1 ? "s" : ""} sent`, 30),
              infer:    done(`${doneShards}/${nShards} shards · ${jobMode}`, elapsed - 100),
              result:   done("Output on-chain", elapsed - 50),
              assemble: done("Result assembled", elapsed),
            };
          }
          if (status === "partial" || status === "assembling") {
            const doneShards = Object.values(shardsData).filter(s =>
              s.status === "complete" || s.status === "submitted").length;
            return {
              post:     done(`Job ${jobId.slice(0, 8)}`, 0),
              block:    done("Confirmed", 10),
              vrf:      done(minerLabel, 20),
              context:  done(ctxDetail, 25),
              offer:    done("Sent", 30),
              infer:    active(`${doneShards}/${nShards} shards done · ${fmtMs(elapsed)}`),
              result:   pend(), assemble: pend(),
            };
          }
          // "pending" = job is IN a block, shards are being worked on.
          // Derive step states from actual shard data rather than the coarse status string.
          const hasShards = Object.keys(shardsData).length > 0;
          const doneShards = Object.values(shardsData).filter(s =>
            s.status === "complete" || s.status === "submitted").length;
          return {
            post:     done(`Job ${jobId.slice(0, 8)}`, 0),
            block:    done(`Block #${job.block_number || "?"}`, 10),
            vrf:      hasShards ? done(minerLabel, 20) : active("Selecting miners…"),
            context:  hasShards ? done(ctxDetail, 25) : pend(),
            offer:    hasShards ? done(`${nShards} offer${nShards > 1 ? "s" : ""} sent`, 30) : pend(),
            infer:    hasShards ? active(`${doneShards}/${nShards} shards · ${fmtMs(elapsed)}`) : pend(),
            result:   pend(),
            assemble: pend(),
          };
        };

        const steps = buildSteps();

        if (status === "failed") {
          setActiveMiners({});
          setCurrentJob({ jobId, steps, done: true, shards: shardsData, nShards, mode: jobMode });
          const timedOut = Object.values(shardsData).filter(s =>
            s.status === "timeout" || s.status === "slashed").length;
          const reason = timedOut > 0
            ? `${timedOut} shard${timedOut !== 1 ? "s" : ""} timed out — miner couldn't complete inference (model may be too large)`
            : "all miners exhausted";
          setMessages(msgs => msgs.map(m => m.id === msgIdTarget ? {
            ...m, thinking: false, done: true, text: null, error: reason, steps,
          } : m));
          return;
        }

        if (status === "complete") {
          setActiveMiners({});
          setCurrentJob({ jobId, steps, done: true, shards: shardsData, nShards, mode: jobMode });
          setMessages(msgs => msgs.map(m => m.id === msgIdTarget ? {
            ...m,
            thinking:       false,
            done:           true,
            text:           job.final_output || "(no output)",
            steps,
            statusText:     null,
            latencyMs:      elapsed,
            minersUsed:     assignedMiners.length || nShards,
            minerAddrs:     assignedMiners,
            mode:           jobMode,
            contextEntries: ctxEntries,
            contextHash:    job.context_hash,
            originalPrompt: job.original_prompt,
            allCacheHit:    ctxLoad?.all_cache_hit || false,
          } : m));
          setTimeout(refreshCtxCount, 1500);
          return;
        }

        setCurrentJob({ jobId, steps, done: false, shards: shardsData, nShards, mode: jobMode });
        setMessages(msgs => msgs.map(m => m.id === msgIdTarget ? {
          ...m, steps,
          statusText: status === "pending" ? "Waiting for block…"
                    : status === "partial"  ? "Miners running inference…"
                    : "Assembling result…",
        } : m));

        setTimeout(poll, 1500);
      } catch { setTimeout(poll, 2000); }
    };
    poll();
  }, [settings.l2Rpc, refreshCtxCount]);

  // Shared initial steps object
  const blankSteps = () => ({
    post: { done: false }, block: { done: false }, vrf: { done: false },
    context: { done: false }, offer: { done: false },
    infer: { done: false }, result: { done: false }, assemble: { done: false },
  });

  // Submit with wallet signature
  const submit = useCallback(async () => {
    if (!prompt.trim() || posting || !walletAddr) return;

    const userMsg = { id: msgId.current++, role: "user", ts: now(), text: prompt.trim(), done: true };
    const aiMsg   = {
      id: msgId.current++, role: "assistant", ts: now(),
      thinking: true, statusText: "Waiting for wallet signature…",
      steps: { ...blankSteps(), post: { done: false, active: true, detail: "Sign in wallet…" } },
    };

    setMessages(msgs => [...msgs, userMsg, aiMsg]);
    setPrompt("");
    setPosting(true);
    const startMs = now();

    try {
      const { job_id: jobId, preimage_hex, tx } = await rpc(settings.l2Rpc, "inft_buildJobTx", [
        model, prompt.trim(), 128, shardMode, nShards, walletAddr
      ]);

      const signature = await window.ethereum.request({
        method: "personal_sign",
        params: [preimage_hex, walletAddr],
      });

      await rpc(settings.l2Rpc, "inft_postJobSigned", [tx, signature]);

      const postSteps = {
        ...aiMsg.steps,
        post:  { done: true, active: false, detail: `Job ${jobId.slice(0, 8)}`, ts: now() - startMs },
        block: { done: false, active: true,  detail: "Waiting for block…" },
      };
      setMessages(msgs => msgs.map(m => m.id === aiMsg.id ? {
        ...m, jobId, statusText: "Included in block…", steps: postSteps
      } : m));
      setCurrentJob({ jobId, steps: postSteps, done: false });

      pollJob(jobId, aiMsg.id, startMs);
    } catch (err) {
      setMessages(msgs => msgs.map(m => m.id === aiMsg.id ? {
        ...m, thinking: false, error: err.message || String(err),
        steps: { ...aiMsg.steps, post: { done: false, active: false, detail: "Failed" } }
      } : m));
    } finally {
      setPosting(false);
    }
  }, [prompt, posting, settings, walletAddr, model, nShards, shardMode, pollJob]);

  // Submit without wallet (open mode — sequencer signs)
  const submitDirect = useCallback(async () => {
    if (!prompt.trim() || posting) return;

    const userMsg = { id: msgId.current++, role: "user", ts: now(), text: prompt.trim(), done: true };
    const aiMsg   = {
      id: msgId.current++, role: "assistant", ts: now(), openMode: true,
      thinking: true, statusText: "Submitting (open mode)…",
      steps: { ...blankSteps(), post: { done: false, active: true, detail: "Open submit…" } },
    };

    setMessages(msgs => [...msgs, userMsg, aiMsg]);
    setPrompt("");
    setPosting(true);
    const startMs = now();

    try {
      // Returns job_id string directly
      const jobId = await rpc(settings.l2Rpc, "inft_postJobOpen", [
        model, prompt.trim(), 128, shardMode, nShards
      ]);

      const postSteps = {
        ...aiMsg.steps,
        post:  { done: true, active: false, detail: `Job ${jobId.slice(0, 8)}`, ts: now() - startMs },
        block: { done: false, active: true, detail: "Waiting for block…" },
      };
      setMessages(msgs => msgs.map(m => m.id === aiMsg.id ? {
        ...m, jobId, statusText: "In block…", steps: postSteps
      } : m));
      setCurrentJob({ jobId, steps: postSteps, done: false });

      pollJob(jobId, aiMsg.id, startMs);
    } catch (err) {
      setMessages(msgs => msgs.map(m => m.id === aiMsg.id ? {
        ...m, thinking: false, error: err.message || String(err),
        steps: { ...aiMsg.steps, post: { done: false, active: false, detail: "Failed" } }
      } : m));
    } finally {
      setPosting(false);
    }
  }, [prompt, posting, settings, model, nShards, shardMode, pollJob]);

  const handleKey = useCallback(e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (walletAddr) submit(); else submitDirect();
    }
  }, [submit, submitDirect, walletAddr]);

  return (
    <div className="shell">
      <TopBar chain={chain} connected={connected} onSettings={() => setShowSettings(true)}
              walletAddr={walletAddr} onConnect={connectWallet} />

      <div className="body">
        <Sidebar miners={miners} chainStats={chainStats} activeMiners={activeMiners} />

        <div className="main">
          <div className="chat-wrap" ref={chatRef}>
            {messages.map(msg => <Message key={msg.id} msg={msg} />)}
          </div>

          <div className="input-area">
            {!walletAddr && (
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--warn)",
                            marginBottom: 8, letterSpacing: ".06em", cursor: "pointer" }}
                   onClick={connectWallet}>
                ⚠ No wallet — connect above for signed jobs, or use Quick Send ⚡ below
              </div>
            )}
            <div className="input-controls">
              <select className="ctrl-select" value={model} onChange={e => setModel(e.target.value)}>
                {models.map(m => <option key={m} value={m}>{m.split("/")[1] || m}</option>)}
              </select>
              <select className="ctrl-select" value={shardMode} onChange={e => setShardMode(e.target.value)}>
                <option value="parallel_sample">⚡ Parallel sample</option>
                <option value="pipeline_parallel">⛓ Pipeline parallel</option>
                <option value="context_split">⊟ Context split</option>
                <option value="speculative">◈ Speculative</option>
              </select>
              <select className="ctrl-select" value={nShards} onChange={e => setNShards(Number(e.target.value))}>
                {[1,2,3,4,6,8].map(n => (
                  <option key={n} value={n}>{n} shard{n > 1 ? "s" : ""}</option>
                ))}
              </select>
              {ctxCount > 0 && (
                <span style={{ fontFamily: "var(--mono)", fontSize: 9, letterSpacing: ".04em",
                               color: "var(--accent)", background: "rgba(212,113,42,0.12)",
                               padding: "2px 7px", borderRadius: 3, alignSelf: "center",
                               border: "1px solid rgba(212,113,42,0.25)" }}>
                  ctx:{ctxCount}
                </span>
              )}
                      {(() => {
                const mc = minersByModel[model] || 0;
                return (
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, alignSelf: "center",
                                 marginLeft: 4, color: mc > 0 ? "var(--ink-2)" : "var(--warn)" }}>
                    {mc > 0
                      ? `${mc} miner${mc !== 1 ? "s" : ""} · ${mc} shard${mc !== 1 ? "s" : ""} suggested`
                      : "no miners for this model"}
                  </span>
                );
              })()}
            </div>
            <div className="input-row">
              <textarea
                className="chat-input"
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={handleKey}
                placeholder={walletAddr
                  ? "Ask anything… (Enter to send, Shift+Enter for newline)"
                  : "Ask anything… (Enter or Quick ⚡ to send without wallet)"}
                disabled={posting}
                rows={1}
              />
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <button className="send-btn" onClick={submit}
                        disabled={posting || !walletAddr || !prompt.trim()}
                        title={!walletAddr ? "Connect wallet to use signed submit" : ""}>
                  {posting ? "…" : "Send →"}
                </button>
                <button onClick={submitDirect}
                        disabled={posting || !prompt.trim()}
                        title="Submit without wallet (sequencer signs)"
                        style={{ fontFamily: "var(--mono)", fontSize: 9, cursor: "pointer",
                                 padding: "4px 8px", borderRadius: 3,
                                 background: posting || !prompt.trim() ? "var(--ink-4)" : "rgba(212,113,42,0.15)",
                                 color: posting || !prompt.trim() ? "var(--ink-3)" : "var(--accent)",
                                 border: "1px solid rgba(212,113,42,0.3)", letterSpacing: ".04em" }}>
                  Quick ⚡
                </button>
              </div>
            </div>
          </div>
        </div>

        <PipelineColumn
          currentJob={currentJob}
          historyAddr={historyAddr}
          l2Rpc={settings.l2Rpc}
          model={model}
        />
      </div>

      {showSettings && (
        <SettingsModal
          settings={settings}
          onChange={saveSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
