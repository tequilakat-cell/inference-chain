const { useState, useEffect, useCallback, useRef } = React;

// ── Config ────────────────────────────────────────────────────────────────────
const _host   = window.location.hostname;
const _origin = window.location.origin;
const _defaultRpc =
  (_host === "127.0.0.1" || _host === "localhost") ? "http://127.0.0.1:18545" :
  (_origin && _origin !== "null")                  ? _origin :
                                                     "http://192.168.198.48:18545";
const _savedRpc = localStorage.getItem("ic-rpc");
const L2_RPC = (_savedRpc && (_origin === "null" || _savedRpc.startsWith(_origin)) ? _savedRpc : null) || _defaultRpc;

const STORAGE_KEY = "ic-miners-extra";

async function rpc(method, params = []) {
  const r = await fetch(L2_RPC, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ jsonrpc: "2.0", method, params, id: 1 }),
  });
  const d = await r.json();
  if (d.error) throw new Error(d.error.message || String(d.error));
  return d.result;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const shortAddr = (a = "", n = 6) => a ? `${a.slice(0, 2 + n)}…${a.slice(-4)}` : "—";
const fmtNum    = n => (n ?? 0).toLocaleString();

function useInterval(fn, ms) {
  const cb = useRef(fn);
  useEffect(() => { cb.current = fn; }, [fn]);
  useEffect(() => {
    const id = setInterval(() => cb.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

// Try to fetch a miner's health endpoint (may be blocked by CORS in some browsers)
async function fetchHealth(url) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) return { ok: false, error: `HTTP ${r.status}` };
    const data = await r.json();
    return { ok: true, data };
  } catch (e) {
    return { ok: false, error: e.message || String(e) };
  }
}

// ── Nav ───────────────────────────────────────────────────────────────────────
function NavBar() {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <div className="brand">
          <div className="diamond"/>
          InferenceChain
          <span className="nav-tag">MINERS</span>
        </div>
        <div className="nav-links">
          <a href="explorer.html">Explorer</a>
          <a href="dashboard.html">Dashboard</a>
          <a href="miners.html" className="active">Miners</a>
          <a href="memory.html">Memory</a>
          <a href="index.html">← Site</a>
        </div>
      </div>
    </nav>
  );
}

// ── Diagnostic panel ──────────────────────────────────────────────────────────
function DiagnosticPanel({ validators, jobs, healthResults }) {
  const totalValidators  = validators.length;
  const stakedCount      = validators.filter(v => v.stake_inft > 0).length;
  const modelsCount      = validators.filter(v => v.models && v.models.length > 0).length;
  const onlineCount      = Object.values(healthResults).filter(h => h.ok).length;
  const recentJobs       = jobs.slice(0, 20);
  const splitJobs        = recentJobs.filter(j => j.n_shards > 1).length;
  const allJobs          = recentJobs.length;

  // For the last 5 multi-shard jobs, check if shards went to >1 unique miner
  const recentSplit = jobs.filter(j => j.n_shards > 1).slice(0, 5);

  const checks = [
    {
      label: "Validators staked",
      ok:    stakedCount >= 2,
      value: `${stakedCount} / ${totalValidators}`,
      sub:   stakedCount < 2
        ? "Less than 2 staked validators — VRF can only select 1 miner"
        : "Both validators have stake — eligible for VRF selection",
    },
    {
      label: "Models registered",
      ok:    modelsCount >= 2,
      value: `${modelsCount} / ${totalValidators}`,
      sub:   modelsCount < 2
        ? "Miners without registered models are skipped for model-specific jobs"
        : "Both miners have models registered on-chain",
    },
    {
      label: "Miners reachable",
      ok:    onlineCount >= 2,
      warn:  onlineCount === 1,
      value: `${onlineCount} / ${Object.keys(healthResults).length || totalValidators}`,
      sub:   onlineCount < 2
        ? "Some miners unreachable — check health URLs below"
        : "All configured miners responded to health check",
    },
    {
      label: "Multi-shard jobs",
      ok:    splitJobs > 0,
      warn:  splitJobs === 0 && allJobs > 0,
      value: `${splitJobs} / ${allJobs} recent`,
      sub:   splitJobs === 0
        ? "No multi-shard jobs found — set n_shards ≥ 2 when posting jobs"
        : `${splitJobs} recent jobs used multiple shards`,
    },
  ];

  return (
    <div className="diag-grid">
      {checks.map(c => {
        const cls = c.ok ? "diag-ok" : c.warn ? "diag-warn" : "diag-bad";
        const icon = c.ok ? "✓" : c.warn ? "△" : "✗";
        return (
          <div key={c.label} className="diag-card">
            <div className={`diag-icon ${cls}`}>{icon}</div>
            <div className="diag-body">
              <div className="diag-label">{c.label}</div>
              <div className={`diag-value ${cls}`}>{c.value}</div>
              <div className="diag-sub">{c.sub}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Miner card ────────────────────────────────────────────────────────────────
function MinerCard({ validator, healthUrl, onHealthUrlChange, healthResult, onCheckHealth }) {
  const [urlDraft, setUrlDraft] = useState(healthUrl || "");
  const [showRaw,  setShowRaw]  = useState(false);

  const repPct = Math.min(100, ((validator.reputation || 500) / 1000) * 100);
  const stakeOk = validator.stake_inft > 0;

  const statusCls = healthResult === null ? "unknown"
                  : healthResult.checking  ? "checking"
                  : healthResult.ok        ? "online"
                  : "offline";

  const cardCls   = healthResult?.ok ? "online"
                  : healthResult && !healthResult.ok ? "offline"
                  : "unknown";

  const backendFromHealth = healthResult?.data?.backend || null;
  const modelsFromHealth  = healthResult?.data?.models  || null;
  const activeShards      = healthResult?.data?.active_shards ?? validator.active_shards;
  const maxShards         = healthResult?.data?.max_shards ?? 4;

  const backend = backendFromHealth || "—";
  const backendCls = { cpu:"badge-cpu", cuda:"badge-cuda", mlx:"badge-mlx",
                       vulkan:"badge-vulkan", llama:"badge-llama" }[backend] || "badge-cpu";

  const modelList = modelsFromHealth || validator.models || [];

  return (
    <div className={`miner-card ${cardCls}`}>
      <div className="mc-head">
        <div className={`mc-status ${statusCls}`}/>
        <div className="mc-addr" title={validator.address}>{validator.address}</div>
        {backend !== "—" && <span className={`mc-badge ${backendCls}`}>{backend.toUpperCase()}</span>}
      </div>

      <div className="mc-body">
        <div className="mc-kv">
          <span className="mc-key">Stake</span>
          <span className={`mc-val ${stakeOk ? "good" : "bad"}`}>
            {fmtNum(validator.stake_inft)} INFT{!stakeOk ? " — NOT STAKED" : ""}
          </span>

          <span className="mc-key">Balance</span>
          <span className="mc-val">{fmtNum(validator.balance_inft)} INFT</span>

          <span className="mc-key">Reputation</span>
          <span className="mc-val" style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
            <span style={{color: validator.reputation >= 500 ? "var(--good)" : "var(--bad)"}}>
              {validator.reputation}/1000
            </span>
            <div className="rep-bar" style={{flex:1,minWidth:60}}>
              <div className="rep-fill" style={{width:`${repPct}%`}}/>
            </div>
          </span>

          <span className="mc-key">Active shards</span>
          <span className={`mc-val ${activeShards > 0 ? "warn" : "good"}`}>
            {activeShards} / {maxShards}
          </span>

          {validator.unlock_block > 0 && <>
            <span className="mc-key">Unlock block</span>
            <span className="mc-val warn">#{fmtNum(validator.unlock_block)}</span>
          </>}
        </div>

        {/* Registered models */}
        <div className="mc-models">
          {modelList.length === 0
            ? <span className="model-tag" style={{color:"var(--bad)",borderColor:"rgba(224,90,69,.3)"}}>
                no models registered
              </span>
            : modelList.map(m => (
                <span key={m} className="model-tag registered"
                      title={m}>{m.split("/")[1] || m}</span>
              ))
          }
        </div>

        {/* Health URL config */}
        <div className="mc-health-url">
          <input
            className="url-input"
            placeholder="http://ip:port/health"
            value={urlDraft}
            onChange={e => setUrlDraft(e.target.value)}
            onKeyDown={e => e.key === "Enter" && onHealthUrlChange(urlDraft)}
          />
          <button className="url-btn" onClick={() => { onHealthUrlChange(urlDraft); onCheckHealth(urlDraft); }}>
            Check
          </button>
          {healthResult?.data && (
            <button className="url-btn" onClick={() => setShowRaw(v => !v)}>
              {showRaw ? "Hide" : "Raw"}
            </button>
          )}
        </div>

        {/* Health status / raw response */}
        {healthResult && (
          <div className={`health-raw ${healthResult.ok ? "ok" : "err"}`}>
            {healthResult.checking
              ? "Checking…"
              : healthResult.ok
              ? JSON.stringify(healthResult.data, null, 2)
              : `Error: ${healthResult.error}`}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Shard distribution table ──────────────────────────────────────────────────
function ShardDistribution({ jobs, validators }) {
  const addrSet = new Set(validators.map(v => v.address.toLowerCase()));

  // Build miner → count mapping from recent jobs with shard data
  const minerShardCounts = {};
  validators.forEach(v => { minerShardCounts[v.address.toLowerCase()] = 0; });

  const jobRows = jobs.slice(0, 30).map(j => {
    const shards = j.shards || {};
    const assignments = Object.entries(shards).map(([idx, s]) => ({
      idx: parseInt(idx),
      miner: s.miner || null,
      status: s.status || "unassigned",
      output: s.output,
    }));

    assignments.forEach(a => {
      if (a.miner) {
        const k = a.miner.toLowerCase();
        if (k in minerShardCounts) minerShardCounts[k]++;
        else minerShardCounts[k] = (minerShardCounts[k] || 0) + 1;
      }
    });

    const uniqueMiners = new Set(assignments.map(a => a.miner).filter(Boolean));
    return { ...j, assignments, uniqueMiners };
  });

  // Pie chart: miner → total shards completed
  const totalShards = Object.values(minerShardCounts).reduce((a, b) => a + b, 0);

  // Colour by whether miner is known validator
  function shardPillCls(miner) {
    if (!miner) return "unset";
    const idx = validators.findIndex(v => v.address.toLowerCase() === miner.toLowerCase());
    if (idx === 0) return "khadas";
    if (idx === 1) return "mac";
    return "other";
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">Recent Job Shard Assignments</span>
        <span style={{fontFamily:"var(--mono)",fontSize:10,color:"var(--ink-3)"}}>
          last 30 jobs
        </span>
      </div>

      {/* Shard count summary per miner */}
      {totalShards > 0 && (
        <div style={{padding:"12px 16px",borderBottom:"1px solid var(--border)",
                     display:"flex",gap:24,flexWrap:"wrap"}}>
          {Object.entries(minerShardCounts).filter(([,c]) => c > 0).map(([addr, cnt]) => {
            const pct = Math.round((cnt / totalShards) * 100);
            const v   = validators.find(v => v.address.toLowerCase() === addr.toLowerCase());
            const cls = shardPillCls(addr);
            return (
              <div key={addr} style={{display:"flex",flexDirection:"column",gap:4}}>
                <span style={{fontFamily:"var(--mono)",fontSize:9,color:"var(--ink-3)"}}>
                  {v ? shortAddr(v.address) : shortAddr(addr)}
                </span>
                <div style={{display:"flex",alignItems:"center",gap:6}}>
                  <div style={{
                    width: Math.max(4, pct * 1.4),
                    height: 6,
                    borderRadius: 3,
                    background: cls === "khadas" ? "var(--accent)"
                              : cls === "mac"    ? "var(--blue)"
                              : "var(--ink-3)",
                  }}/>
                  <span style={{fontFamily:"var(--mono)",fontSize:10,
                                color:"var(--ink-2)"}}>
                    {cnt} shard{cnt !== 1 ? "s" : ""} ({pct}%)
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <table className="data-table">
        <thead>
          <tr>
            <th>Job ID</th>
            <th>Mode</th>
            <th>Shards</th>
            <th>Assignments</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {jobRows.length === 0 && (
            <tr><td colSpan={5} className="empty">No jobs found</td></tr>
          )}
          {jobRows.map(j => {
            const splitOk = j.uniqueMiners.size > 1;
            return (
              <tr key={j.job_id}>
                <td className="hash">
                  <a href={`explorer.html#/job/${j.job_id}`}
                     style={{color:"var(--accent)"}}>
                    {j.job_id.slice(0, 8)}…
                  </a>
                </td>
                <td>
                  <span className="badge" style={{background:"var(--bg-3)",
                        border:"1px solid var(--border)",color:"var(--ink-3)"}}>
                    {(j.mode || "?").replace(/_/g," ").toUpperCase()}
                  </span>
                </td>
                <td className="hash">
                  <span style={{color: j.n_shards > 1 ? "var(--ink-2)" : "var(--ink-3)"}}>
                    {j.n_shards}
                  </span>
                </td>
                <td>
                  {j.n_shards === 1 && j.assignments.length === 0 ? (
                    <span style={{fontFamily:"var(--mono)",fontSize:9,
                                  color:"var(--ink-3)"}}>single shard</span>
                  ) : j.assignments.length === 0 ? (
                    <span style={{fontFamily:"var(--mono)",fontSize:9,
                                  color:"var(--warn)"}}>no shards assigned yet</span>
                  ) : (
                    <div className="shard-pills">
                      {j.assignments.map(a => (
                        <span key={a.idx}
                              className={`shard-pill ${shardPillCls(a.miner)}`}
                              title={a.miner || "unassigned"}>
                          S{a.idx}:{a.miner ? shortAddr(a.miner, 4) : "—"}
                        </span>
                      ))}
                      {j.n_shards > 1 && j.assignments.length > 0 && (
                        <span style={{fontFamily:"var(--mono)",fontSize:9,
                              color: splitOk ? "var(--good)" : "var(--warn)",
                              marginLeft:2}}>
                          {splitOk ? "✓ split" : "⚠ same miner"}
                        </span>
                      )}
                    </div>
                  )}
                </td>
                <td>
                  <span className={`badge badge-${j.status}`}>
                    {(j.status || "pending").toUpperCase()}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Benchmark scores panel ────────────────────────────────────────────────────
function BenchmarkPanel({ scores, validators, currentBlock }) {
  if (scores.length === 0) {
    return (
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">Miner Benchmark Scores</span>
          <span style={{fontFamily:"var(--mono)",fontSize:10,color:"var(--ink-3)"}}>
            run <code>python -m miner.run_benchmark</code> on each miner
          </span>
        </div>
        <div className="empty">No benchmark scores on-chain yet.</div>
      </div>
    );
  }

  // Max TPS across all scores for bar scaling
  const maxTps = Math.max(...scores.map(s => s.tokens_per_sec || 0), 1);

  function tpsColor(tps) {
    if (tps >= maxTps * 0.7) return "var(--good)";
    if (tps >= maxTps * 0.3) return "var(--warn)";
    return "var(--bad)";
  }

  function layerPct(tps) {
    // Mirrors _compute_tensor_split logic: proportional, clamped [5%, 80%]
    if (scores.length < 2) return 100;
    const total = scores.reduce((s, r) => s + (r.tokens_per_sec || 0), 0) || 1;
    const raw   = (tps / total) * 100;
    return Math.max(5, Math.min(80, raw));
  }

  function blocksLeft(expires) {
    const left = (expires || 0) - (currentBlock || 0);
    if (left <= 0) return <span style={{color:"var(--bad)"}}>Expired</span>;
    const hrs = Math.round(left / 3600);
    return <span style={{color:"var(--ink-3)"}}>~{hrs}h ({left.toLocaleString()} blocks)</span>;
  }

  // Merge validator labels
  function minerLabel(addr) {
    const v = validators.find(v => v.address.toLowerCase() === addr.toLowerCase());
    return v?._label || shortAddr(addr);
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-title">Miner Benchmark Scores</span>
        <span style={{fontFamily:"var(--mono)",fontSize:10,color:"var(--ink-3)"}}>
          {scores.length} score{scores.length !== 1 ? "s" : ""}
        </span>
      </div>

      <table className="data-table">
        <thead>
          <tr>
            <th>Miner</th>
            <th>Model</th>
            <th>Tokens / sec</th>
            <th>Layer share</th>
            <th>Source</th>
            <th>Expires</th>
          </tr>
        </thead>
        <tbody>
          {scores
            .slice()
            .sort((a, b) => (b.tokens_per_sec || 0) - (a.tokens_per_sec || 0))
            .map((s, i) => {
              const tps  = s.tokens_per_sec || 0;
              const frac = layerPct(tps);
              const col  = tpsColor(tps);
              return (
                <tr key={`${s.miner}:${s.model_id}:${i}`}>
                  <td className="hash" title={s.miner}>
                    {minerLabel(s.miner)}
                  </td>
                  <td>
                    <span className="model-tag registered" style={{margin:0}}>
                      {(s.model_id || "").split("/")[1] || s.model_id}
                    </span>
                  </td>
                  <td>
                    <div style={{display:"flex",alignItems:"center",gap:8}}>
                      <div style={{
                        width: Math.max(4, Math.round((tps / maxTps) * 80)),
                        height: 6,
                        borderRadius: 3,
                        background: col,
                        flexShrink: 0,
                      }}/>
                      <span style={{fontFamily:"var(--mono)",fontSize:11,color: col}}>
                        {tps.toFixed(1)} t/s
                      </span>
                    </div>
                  </td>
                  <td>
                    <span style={{fontFamily:"var(--mono)",fontSize:11,color:"var(--ink-2)"}}>
                      {frac.toFixed(0)}%
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${s.self_reported ? "badge-partial" : "badge-complete"}`}>
                      {s.self_reported ? "self-reported" : "sequencer"}
                    </span>
                  </td>
                  <td className="hash">
                    {blocksLeft(s.expires_at_block)}
                  </td>
                </tr>
              );
            })}
        </tbody>
      </table>
    </div>
  );
}

// ── Add miner modal ───────────────────────────────────────────────────────────
function AddMinerModal({ onAdd, onClose }) {
  const [addr,    setAddr]    = useState("");
  const [healthUrl, setHealthUrl] = useState("");
  const [label,   setLabel]   = useState("");

  const save = () => {
    if (!addr.trim()) return;
    onAdd({ address: addr.trim(), healthUrl: healthUrl.trim(), label: label.trim() });
    onClose();
  };

  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={e => e.stopPropagation()}>
        <h2>Add Remote Miner</h2>
        <div className="form-field">
          <label className="form-label">Wallet Address</label>
          <input className="form-input" placeholder="0x…" value={addr}
                 onChange={e => setAddr(e.target.value)}/>
          <div className="form-hint">The miner's L2 wallet address (must be staked)</div>
        </div>
        <div className="form-field">
          <label className="form-label">Health URL</label>
          <input className="form-input" placeholder="http://192.168.1.x:19001/health"
                 value={healthUrl} onChange={e => setHealthUrl(e.target.value)}/>
          <div className="form-hint">HTTP health endpoint — used to ping the miner</div>
        </div>
        <div className="form-field">
          <label className="form-label">Label (optional)</label>
          <input className="form-input" placeholder="Macbook, Khadas…"
                 value={label} onChange={e => setLabel(e.target.value)}/>
        </div>
        <div className="modal-btns">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={save}>Add Miner</button>
        </div>
      </div>
    </div>
  );
}

// ── Root app ──────────────────────────────────────────────────────────────────
function App() {
  const [validators,    setValidators]    = useState([]);
  const [jobs,          setJobs]          = useState([]);
  const [jobDetails,    setJobDetails]    = useState({});
  const [chainConnected,setChainConnected]= useState(null);
  const [lastRefresh,   setLastRefresh]   = useState(null);
  const [healthResults, setHealthResults] = useState({});
  const [benchScores,   setBenchScores]   = useState([]);
  const [currentBlock,  setCurrentBlock]  = useState(0);
  const [healthUrls,    setHealthUrls]    = useState(() => {
    try { return JSON.parse(localStorage.getItem("ic-health-urls") || "{}"); } catch { return {}; }
  });
  const [extraMiners, setExtraMiners] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; }
  });
  const [showAddModal, setShowAddModal] = useState(false);

  const saveHealthUrl = useCallback((addr, url) => {
    setHealthUrls(prev => {
      const next = { ...prev, [addr.toLowerCase()]: url };
      localStorage.setItem("ic-health-urls", JSON.stringify(next));
      return next;
    });
  }, []);

  const checkHealth = useCallback(async (addr, url) => {
    if (!url) return;
    setHealthResults(prev => ({ ...prev, [addr.toLowerCase()]: { checking: true } }));
    const result = await fetchHealth(url);
    setHealthResults(prev => ({ ...prev, [addr.toLowerCase()]: result }));
  }, []);

  // Load chain data
  const loadAll = useCallback(async () => {
    try {
      const [vs, js, scores, info] = await Promise.all([
        rpc("inft_getValidators", []),
        rpc("inft_getRecentJobs", [30]),
        rpc("inft_getAllMinerScores", []).catch(() => []),
        rpc("inft_getChainInfo",   []).catch(() => null),
      ]);
      setValidators(vs || []);
      setBenchScores(scores || []);
      if (info?.block_number) setCurrentBlock(info.block_number);
      setChainConnected(true);

      // Merge extra (manually added) miners that aren't already in validators
      const knownAddrs = new Set((vs || []).map(v => v.address.toLowerCase()));
      const extras = (extraMiners || []).filter(m => !knownAddrs.has(m.address.toLowerCase()));

      // For recent multi-shard jobs, fetch full detail to get shard assignments
      const multiShardJobs = (js || []).filter(j => j.n_shards > 1).slice(0, 10);
      const details = {};
      await Promise.all(multiShardJobs.map(async j => {
        try {
          const d = await rpc("inft_getJob", [j.job_id]);
          if (d) details[j.job_id] = d;
        } catch {}
      }));
      setJobDetails(details);

      // Merge shard data into job list
      const enriched = (js || []).map(j => ({
        ...j,
        shards: details[j.job_id]?.shards || {},
      }));
      setJobs(enriched);
      setLastRefresh(new Date());
    } catch {
      setChainConnected(false);
    }
  }, [extraMiners]);

  // Auto health-check all validators using stored URLs on first load
  const initialChecked = useRef(false);
  useEffect(() => {
    if (initialChecked.current || validators.length === 0) return;
    initialChecked.current = true;
    validators.forEach(v => {
      const url = healthUrls[v.address.toLowerCase()];
      if (url) checkHealth(v.address, url);
    });
  }, [validators, healthUrls, checkHealth]);

  useEffect(() => { loadAll(); }, []);
  useInterval(loadAll, 5000);

  // All validators to show (chain validators + manually added extras not already in list)
  const knownAddrs = new Set(validators.map(v => v.address.toLowerCase()));
  const allMiners = [
    ...validators,
    ...extraMiners
      .filter(m => !knownAddrs.has(m.address.toLowerCase()))
      .map(m => ({
        address:      m.address,
        stake_inft:   0,
        balance_inft: 0,
        reputation:   500,
        active_shards:0,
        models:       [],
        unlock_block: 0,
        _extra:       true,
        _label:       m.label,
      })),
  ];

  const addExtraMiner = useCallback(({ address, healthUrl, label }) => {
    const updated = [...extraMiners.filter(m => m.address.toLowerCase() !== address.toLowerCase()),
                     { address, healthUrl, label }];
    setExtraMiners(updated);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    if (healthUrl) saveHealthUrl(address, healthUrl);
    if (healthUrl) checkHealth(address, healthUrl);
  }, [extraMiners, saveHealthUrl, checkHealth]);

  const removeExtraMiner = useCallback(addr => {
    const updated = extraMiners.filter(m => m.address.toLowerCase() !== addr.toLowerCase());
    setExtraMiners(updated);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  }, [extraMiners]);

  return (
    <>
      <NavBar />
      <div className="page">

        {/* Refresh bar */}
        <div className="refresh-bar">
          <div className={`live-dot ${chainConnected === false ? "bad" : ""}`}/>
          <span className="refresh-label">
            {chainConnected === null ? "Connecting to chain…"
           : chainConnected === false ? "Chain unreachable — check L2 RPC"
           : `Chain live · ${validators.length} validator${validators.length !== 1 ? "s" : ""}`}
          </span>
          {lastRefresh && (
            <span className="refresh-label" style={{marginLeft:8}}>
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button className="refresh-btn ml-auto" onClick={loadAll}>Refresh</button>
          <button className="refresh-btn" onClick={() => setShowAddModal(true)}>
            + Add Remote Miner
          </button>
        </div>

        {/* Diagnostics */}
        <div className="section-head">
          <span className="section-title">Split Diagnostics</span>
        </div>
        <DiagnosticPanel
          validators={validators}
          jobs={jobs}
          healthResults={healthResults}
        />

        {/* Miner cards */}
        <div className="section-head">
          <span className="section-title">
            Connected Miners · {allMiners.length}
          </span>
        </div>

        {allMiners.length === 0 && chainConnected !== false && (
          <div className="empty loading-dot">Loading validators</div>
        )}
        {allMiners.length === 0 && chainConnected === false && (
          <div className="empty">
            Cannot reach chain at {L2_RPC}<br/>
            <span style={{fontSize:10,marginTop:6,display:"block"}}>
              Start the chain node with run_chain.sh, then refresh.
            </span>
          </div>
        )}

        <div className="miner-grid">
          {allMiners.map((v, i) => {
            const addrLower = v.address.toLowerCase();
            const storedUrl  = healthUrls[addrLower] || "";
            const extraEntry = extraMiners.find(m => m.address.toLowerCase() === addrLower);
            const autoUrl    = extraEntry?.healthUrl || storedUrl;

            return (
              <MinerCard
                key={v.address}
                validator={v}
                healthUrl={autoUrl}
                onHealthUrlChange={url => saveHealthUrl(v.address, url)}
                healthResult={healthResults[addrLower] || null}
                onCheckHealth={url => checkHealth(v.address, url)}
              />
            );
          })}
        </div>

        {/* Benchmark scores */}
        <div className="section-head" style={{marginTop:20}}>
          <span className="section-title">Benchmark Scores</span>
          <span style={{fontFamily:"var(--mono)",fontSize:10,color:"var(--ink-3)"}}>
            determines layer share in pipeline-parallel jobs
          </span>
        </div>
        <BenchmarkPanel
          scores={benchScores}
          validators={allMiners}
          currentBlock={currentBlock}
        />

        {/* Shard distribution */}
        <div className="section-head" style={{marginTop:20}}>
          <span className="section-title">Shard Distribution</span>
        </div>
        <ShardDistribution jobs={jobs} validators={allMiners} />

      </div>

      {showAddModal && (
        <AddMinerModal
          onAdd={addExtraMiner}
          onClose={() => setShowAddModal(false)}
        />
      )}
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
