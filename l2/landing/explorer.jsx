const { useState, useEffect, useCallback, useRef } = React;

// ── Config ────────────────────────────────────────────────────────────────────
const _host   = window.location.hostname;
const _origin = window.location.origin;
const _defaultRpc =
  (_host === "127.0.0.1" || _host === "localhost") ? "http://127.0.0.1:8545" :
  (_origin && _origin !== "null")                  ? _origin + "/rpc" :
                                                     "http://192.168.198.48:18545";
const _savedRpc = localStorage.getItem("ic-rpc");
const L2_RPC = (_savedRpc && (_origin === "null" || _savedRpc.startsWith(_origin)) ? _savedRpc : null) || _defaultRpc;

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
const shortHash = (h = "", n = 8) => h ? `${h.slice(0, 2 + n)}…` : "—";
const shortAddr = (a = "", n = 6) => a ? `${a.slice(0, 2 + n)}…${a.slice(-4)}` : "—";
const ageMs   = ts => {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60)   return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
};
const fmtNum  = n => (n ?? 0).toLocaleString();
const TX_TYPES = {
  1: "JOB_POST", 2: "SHARD_COMMIT", 3: "STAKE", 4: "UNSTAKE",
  5: "TRANSFER", 6: "BRIDGE_DEPOSIT", 7: "BRIDGE_WITHDRAW",
  8: "SLASH", 9: "SLASH_HARD", 13: "HISTORY_COMMIT",
};

function StatusBadge({ status }) {
  const cls = status === "complete" ? "badge-complete"
            : status === "partial"  ? "badge-partial"
            : status === "failed"   ? "badge-failed"
            : "badge-pending";
  return <span className={`badge ${cls}`}>{(status||"pending").toUpperCase()}</span>;
}
function TxTypeBadge({ type }) {
  return <span className="badge badge-tx">{TX_TYPES[type] || `TYPE_${type}`}</span>;
}

function useInterval(fn, ms) {
  const cb = useRef(fn);
  useEffect(() => { cb.current = fn; }, [fn]);
  useEffect(() => {
    const id = setInterval(() => cb.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

// ── Router ────────────────────────────────────────────────────────────────────
function useRoute() {
  const [route, setRoute] = useState(location.hash.slice(1) || "/");
  useEffect(() => {
    const onHash = () => setRoute(location.hash.slice(1) || "/");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return route;
}
function nav(path) { location.hash = path; }
function Link({ to, children, style }) {
  return <a href={`#${to}`} style={style}>{children}</a>;
}

// ── Nav bar ───────────────────────────────────────────────────────────────────
function NavBar({ onSearch }) {
  const [q, setQ] = useState("");
  const submit = e => { e.preventDefault(); if (q.trim()) onSearch(q.trim()); };
  return (
    <nav className="nav">
      <div className="nav-inner">
        <div className="brand">
          <div className="diamond"/>
          InferenceChain
          <span className="nav-tag">EXPLORER</span>
        </div>
        <form className="search-bar" onSubmit={submit}>
          <input className="search-input" value={q} onChange={e => setQ(e.target.value)}
            placeholder="Search block · tx hash · job ID · address" />
          <button className="search-btn" type="submit">⌕</button>
        </form>
        <div className="nav-links">
          <Link to="/">Home</Link>
          <Link to="/blocks">Blocks</Link>
          <Link to="/jobs">Jobs</Link>
          <Link to="/transactions">Txns</Link>
          <a href="dashboard.html">Dashboard</a>
          <a href="miners.html">Miners</a>
          <a href="memory.html">Memory</a>
          <a href="index.html">← Site</a>
        </div>
      </div>
    </nav>
  );
}

// ── Stats strip ───────────────────────────────────────────────────────────────
function StatsStrip({ stats }) {
  if (!stats) return <div className="stats-strip"><div className="stat-cell"><div className="loading-dot" style={{fontFamily:"var(--mono)",fontSize:11,color:"var(--ink-3)"}}>Loading</div></div></div>;
  const cells = [
    { label:"L2 Block",     value: fmtNum(stats.block_number),  delta: `${stats.tps} TPS` },
    { label:"Total Jobs",   value: fmtNum(stats.total_jobs),     delta: `${stats.complete_jobs} complete` },
    { label:"Active Jobs",  value: fmtNum(stats.active_jobs),    delta: "in-flight" },
    { label:"Validators",   value: fmtNum(stats.validator_count),delta: `${(stats.total_stake||0).toLocaleString()} staked` },
    { label:"Mempool",      value: fmtNum(stats.mempool_size),   delta: "pending txs" },
    { label:"INFT Supply",  value: fmtNum(Math.round(stats.total_inft_supply||0)), delta: "total" },
  ];
  return (
    <div className="stats-strip">
      {cells.map(c => (
        <div key={c.label} className="stat-cell">
          <div className="stat-label">{c.label}</div>
          <div className="stat-value">{c.value}</div>
          <div className="stat-delta">{c.delta}</div>
        </div>
      ))}
    </div>
  );
}

// ── Home page ─────────────────────────────────────────────────────────────────
function HomePage() {
  const [stats,  setStats]  = useState(null);
  const [blocks, setBlocks] = useState([]);
  const [jobs,   setJobs]   = useState([]);

  const load = useCallback(async () => {
    try {
      const [s, b, j] = await Promise.all([
        rpc("inft_getStats"),
        rpc("inft_getRecentBlocks", [10]),
        rpc("inft_getRecentJobs",   [10]),
      ]);
      setStats(s); setBlocks(b || []); setJobs(j || []);
    } catch(e) {}
  }, []);

  useEffect(() => { load(); }, []);
  useInterval(load, 3000);

  return (
    <div className="page">
      <StatsStrip stats={stats} />

      <div className="home-grid">
        {/* Recent blocks */}
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Recent Blocks</span>
            <Link to="/blocks" className="panel-more">View all →</Link>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Block</th><th>Age</th><th>Txns</th><th>State Root</th>
              </tr>
            </thead>
            <tbody>
              {blocks.length === 0 && (
                <tr><td colSpan={4} className="empty">No blocks yet</td></tr>
              )}
              {blocks.map(b => (
                <tr key={b.block_number}>
                  <td>
                    <span className="badge badge-block" style={{marginRight:6}}>⬡</span>
                    <Link to={`/block/${b.block_number}`}>{fmtNum(b.block_number)}</Link>
                  </td>
                  <td className="age">{ageMs(b.timestamp)}</td>
                  <td className="hash">{b.tx_count}</td>
                  <td className="hash" style={{color:"var(--ink-3)"}}>{shortHash(b.state_root, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Recent jobs */}
        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Recent Inference Jobs</span>
            <Link to="/jobs" className="panel-more">View all →</Link>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Job ID</th><th>Model</th><th>Mode</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {jobs.length === 0 && (
                <tr><td colSpan={4} className="empty">No jobs yet</td></tr>
              )}
              {jobs.map(j => (
                <tr key={j.job_id}>
                  <td className="hash">
                    <Link to={`/job/${j.job_id}`}>{j.job_id.slice(0,8)}…</Link>
                  </td>
                  <td style={{fontSize:12,color:"var(--ink-2)"}}>{j.model_id?.split("/")[1]||"—"}</td>
                  <td><span className="badge badge-tx">{j.mode?.replace("_"," ").toUpperCase()}</span></td>
                  <td><StatusBadge status={j.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ── Blocks page ───────────────────────────────────────────────────────────────
function BlocksPage() {
  const [blocks, setBlocks] = useState([]);
  const [page,   setPage]   = useState(0);
  const PER = 25;

  const load = useCallback(async () => {
    try {
      const b = await rpc("inft_getRecentBlocks", [200]);
      setBlocks(b || []);
    } catch(e) {}
  }, []);

  useEffect(() => { load(); }, []);
  useInterval(load, 5000);

  const slice = blocks.slice(page * PER, (page + 1) * PER);

  return (
    <div className="page">
      <div className="detail-head">
        <div className="detail-title">Blocks</div>
        <div className="detail-sub">{fmtNum(blocks.length)} blocks in history (last 1,000 stored)</div>
      </div>
      <div className="panel">
        <table className="data-table">
          <thead>
            <tr>
              <th>Block</th><th>Age</th><th>Txns</th>
              <th>Sequencer</th><th>Gas used</th><th>State root</th>
            </tr>
          </thead>
          <tbody>
            {slice.map(b => (
              <tr key={b.block_number}>
                <td>
                  <span className="badge badge-block" style={{marginRight:6}}>⬡</span>
                  <Link to={`/block/${b.block_number}`}>{fmtNum(b.block_number)}</Link>
                </td>
                <td className="age">{ageMs(b.timestamp)}</td>
                <td className="hash">{b.tx_count}</td>
                <td className="hash"><Link to={`/address/${b.sequencer}`}>{shortAddr(b.sequencer)}</Link></td>
                <td className="hash">{fmtNum(b.gas_used)}</td>
                <td className="hash" style={{color:"var(--ink-3)"}}>{shortHash(b.state_root)}</td>
              </tr>
            ))}
            {blocks.length === 0 && <tr><td colSpan={6} className="empty">No blocks yet — post a job first</td></tr>}
          </tbody>
        </table>
        <div className="pager">
          <button className="pager-btn" disabled={page === 0} onClick={() => setPage(p => p-1)}>← Prev</button>
          <span className="pager-info">&nbsp;Page {page+1} of {Math.max(1, Math.ceil(blocks.length/PER))}&nbsp;</span>
          <button className="pager-btn" disabled={(page+1)*PER >= blocks.length} onClick={() => setPage(p => p+1)}>Next →</button>
        </div>
      </div>
    </div>
  );
}

// ── Block detail page ─────────────────────────────────────────────────────────
function BlockDetailPage({ num }) {
  const [block, setBlock] = useState(null);
  const [err,   setErr]   = useState(null);

  useEffect(() => {
    rpc("inft_getBlockDetail", [parseInt(num)])
      .then(b => { if (!b) setErr("Block not found"); else setBlock(b); })
      .catch(e => setErr(e.message));
  }, [num]);

  if (err)   return <div className="page"><div className="empty">{err}</div></div>;
  if (!block) return <div className="page"><div className="empty loading-dot">Loading</div></div>;

  return (
    <div className="page">
      <div className="crumb">
        <Link to="/blocks">Blocks</Link>
        <span className="crumb-sep">/</span>
        <span>Block #{fmtNum(block.block_number)}</span>
      </div>
      <div className="detail-head">
        <div className="detail-title">⬡ Block #{fmtNum(block.block_number)}</div>
        <div className="detail-sub">{block.tx_count} transactions · {ageMs(block.timestamp)}</div>
      </div>

      <div className="kv-grid" style={{marginBottom:24}}>
        <div className="kv-row"><div className="kv-key">Block number</div><div className="kv-val">{fmtNum(block.block_number)}</div></div>
        <div className="kv-row"><div className="kv-key">Block hash</div><div className="kv-val" style={{color:"var(--accent)"}}>{block.block_hash}</div></div>
        <div className="kv-row"><div className="kv-key">Parent hash</div><div className="kv-val">{block.parent_hash}</div></div>
        <div className="kv-row"><div className="kv-key">Timestamp</div><div className="kv-val">{new Date(block.timestamp).toLocaleString()} ({ageMs(block.timestamp)})</div></div>
        <div className="kv-row"><div className="kv-key">Sequencer</div><div className="kv-val"><Link to={`/address/${block.sequencer}`}>{block.sequencer}</Link></div></div>
        <div className="kv-row"><div className="kv-key">State root</div><div className="kv-val">{block.state_root}</div></div>
        <div className="kv-row"><div className="kv-key">Shard root</div><div className="kv-val">{block.shard_root}</div></div>
        <div className="kv-row"><div className="kv-key">Gas used</div><div className="kv-val">{fmtNum(block.gas_used)}</div></div>
        <div className="kv-row"><div className="kv-key">Transactions</div><div className="kv-val">{block.tx_count}</div></div>
      </div>

      {block.transactions?.length > 0 && (
        <div className="panel">
          <div className="panel-head"><span className="panel-title">Transactions ({block.tx_count})</span></div>
          <table className="data-table">
            <thead>
              <tr><th>Tx Hash</th><th>Type</th><th>From</th><th>Gas price</th></tr>
            </thead>
            <tbody>
              {block.transactions.map(tx => (
                <tr key={tx.tx_hash}>
                  <td className="hash"><Link to={`/tx/${tx.tx_hash}`}>{shortHash(tx.tx_hash)}</Link></td>
                  <td><TxTypeBadge type={tx.tx_type} /></td>
                  <td className="hash">{tx.sender ? <Link to={`/address/${tx.sender}`}>{shortAddr(tx.sender)}</Link> : <span style={{color:"var(--ink-3)"}}>—</span>}</td>
                  <td className="hash">{tx.gas_price}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Transaction detail page ───────────────────────────────────────────────────
function TxDetailPage({ hash }) {
  const [tx,  setTx]  = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    rpc("inft_getTransaction", [hash])
      .then(t => { if (!t) setErr("Transaction not found"); else setTx(t); })
      .catch(e => setErr(e.message));
  }, [hash]);

  if (err)  return <div className="page"><div className="empty">{err}</div></div>;
  if (!tx)  return <div className="page"><div className="empty loading-dot">Loading</div></div>;

  const payload = tx.payload_parsed || {};

  return (
    <div className="page">
      <div className="crumb">
        <span>Transaction</span>
      </div>
      <div className="detail-head">
        <div className="detail-title">Transaction</div>
        <div className="detail-sub">{shortHash(tx.tx_hash, 20)}</div>
      </div>
      <div className="kv-grid" style={{marginBottom:24}}>
        <div className="kv-row"><div className="kv-key">Tx hash</div><div className="kv-val" style={{color:"var(--accent)"}}>{tx.tx_hash}</div></div>
        <div className="kv-row"><div className="kv-key">Type</div><div className="kv-val"><TxTypeBadge type={tx.tx_type}/></div></div>
        <div className="kv-row"><div className="kv-key">Block</div><div className="kv-val"><Link to={`/block/${tx.block_number}`}>#{fmtNum(tx.block_number)}</Link></div></div>
        <div className="kv-row"><div className="kv-key">Timestamp</div><div className="kv-val">{tx.timestamp ? new Date(tx.timestamp).toLocaleString() : "—"}</div></div>
        <div className="kv-row"><div className="kv-key">From</div><div className="kv-val">{tx.sender ? <Link to={`/address/${tx.sender}`}>{tx.sender}</Link> : <span style={{color:"var(--ink-3)"}}>Sequencer-synthesized</span>}</div></div>
        <div className="kv-row"><div className="kv-key">Gas price</div><div className="kv-val">{tx.gas_price}</div></div>
        <div className="kv-row"><div className="kv-key">Nonce</div><div className="kv-val">{tx.nonce}</div></div>
      </div>

      {/* Payload detail */}
      {Object.keys(payload).length > 0 && (
        <div className="panel" style={{marginBottom:24}}>
          <div className="panel-head"><span className="panel-title">Payload</span></div>
          {Object.entries(payload).map(([k, v]) => (
            <div key={k} className="kv-row">
              <div className="kv-key">{k}</div>
              <div className="kv-val">
                {typeof v === "string" && v.startsWith("0x") && v.length === 66
                  ? <Link to={`/tx/${v}`}>{v}</Link>
                  : String(v).length > 120
                  ? <span style={{wordBreak:"break-all",fontSize:11}}>{String(v).slice(0,200)}…</span>
                  : String(v)
                }
              </div>
            </div>
          ))}
        </div>
      )}

      {/* If it's a job post, show the job */}
      {tx.tx_type === 1 && payload.job_id && (
        <div style={{fontFamily:"var(--mono)",fontSize:11,color:"var(--ink-3)"}}>
          View inference job: <Link to={`/job/${payload.job_id}`}>{payload.job_id}</Link>
        </div>
      )}
    </div>
  );
}

// ── Jobs page ─────────────────────────────────────────────────────────────────
function JobsPage() {
  const [jobs,   setJobs]  = useState([]);
  const [filter, setFilter]= useState("all");
  const [page,   setPage]  = useState(0);
  const PER = 25;

  const load = useCallback(async () => {
    try { setJobs(await rpc("inft_getRecentJobs", [200]) || []); } catch {}
  }, []);
  useEffect(() => { load(); }, []);
  useInterval(load, 5000);

  const filtered = filter === "all" ? jobs : jobs.filter(j => j.status === filter);
  const slice    = filtered.slice(page * PER, (page + 1) * PER);

  return (
    <div className="page">
      <div className="detail-head">
        <div className="detail-title">Inference Jobs</div>
        <div className="detail-sub">{fmtNum(jobs.length)} total · live view</div>
      </div>
      <div className="panel">
        <div className="tab-bar">
          {["all","complete","partial","pending","failed"].map(s => (
            <div key={s} className={`tab ${filter === s ? "active" : ""}`}
                 onClick={() => { setFilter(s); setPage(0); }}>
              {s.toUpperCase()} ({s === "all" ? jobs.length : jobs.filter(j => j.status === s).length})
            </div>
          ))}
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Job ID</th><th>Model</th><th>Mode</th>
              <th>Shards</th><th>Ctx</th><th>Block</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {slice.map(j => (
              <tr key={j.job_id}>
                <td className="hash"><Link to={`/job/${j.job_id}`}>{j.job_id.slice(0,8)}…</Link></td>
                <td style={{fontSize:12}}>{j.model_id?.split("/")[1]||j.model_id||"—"}</td>
                <td><span className="badge badge-tx">{(j.mode||"?").replace("_"," ").toUpperCase()}</span></td>
                <td className="hash">{j.n_shards}</td>
                <td className="hash" style={{color: j.context_entries > 0 ? "var(--good)" : "var(--ink-3)"}}>
                  {j.context_entries > 0 ? j.context_entries : "—"}
                </td>
                <td className="hash"><Link to={`/block/${j.block_number}`}>#{fmtNum(j.block_number)}</Link></td>
                <td><StatusBadge status={j.status}/></td>
              </tr>
            ))}
            {filtered.length === 0 && <tr><td colSpan={7} className="empty">No jobs</td></tr>}
          </tbody>
        </table>
        <div className="pager">
          <button className="pager-btn" disabled={page===0} onClick={() => setPage(p=>p-1)}>← Prev</button>
          <span className="pager-info">&nbsp;Page {page+1} of {Math.max(1,Math.ceil(filtered.length/PER))}&nbsp;</span>
          <button className="pager-btn" disabled={(page+1)*PER>=filtered.length} onClick={() => setPage(p=>p+1)}>Next →</button>
        </div>
      </div>
    </div>
  );
}

// ── Job detail page ───────────────────────────────────────────────────────────
const JOB_PIPE_STEPS = [
  {key:"post",     n:"01", name:"POST",     sub:"Tx confirmed in block"},
  {key:"block",    n:"02", name:"BLOCK",    sub:"L2 block included"},
  {key:"vrf",      n:"03", name:"VRF",      sub:"Miners assigned"},
  {key:"context",  n:"04", name:"CONTEXT",  sub:"Prior exchanges loaded"},
  {key:"infer",    n:"05", name:"INFER",    sub:"Parallel inference"},
  {key:"assemble", n:"06", name:"ASSEMBLE", sub:"Result on-chain"},
];

function JobDetailPage({ id }) {
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);

  const load = useCallback(async () => {
    try {
      const j = await rpc("inft_getJob", [id]);
      if (!j) setErr("Job not found");
      else setJob(j);
    } catch(e) { setErr(e.message); }
  }, [id]);

  useEffect(() => { load(); }, [id]);
  useInterval(() => { if (job?.status !== "complete") load(); }, 2000);

  if (err)  return <div className="page"><div className="empty">{err}</div></div>;
  if (!job) return <div className="page"><div className="empty loading-dot">Loading</div></div>;

  const status = job.status || "pending";
  const ctxEntries = job.context_entries || 0;
  const steps  = {
    post:     { done: true },
    block:    { done: status !== "pending" },
    vrf:      { done: ["partial","assembling","complete"].includes(status),
                active: status === "pending" && job.block_number },
    context:  { done: ["partial","assembling","complete"].includes(status),
                detail: ctxEntries > 0 ? `${ctxEntries} exchange${ctxEntries!==1?"s":""} loaded` : "No prior history" },
    infer:    { done: status === "complete",
                active: status === "partial" || status === "assembling" },
    assemble: { done: status === "complete", active: status === "assembling" },
  };

  return (
    <div className="page">
      <div className="crumb">
        <Link to="/jobs">Jobs</Link>
        <span className="crumb-sep">/</span>
        <span>{id.slice(0,8)}…</span>
      </div>
      <div className="detail-head">
        <div className="detail-title" style={{display:"flex",alignItems:"center",gap:14}}>
          Inference Job
          <StatusBadge status={status}/>
        </div>
        <div className="detail-sub" style={{marginTop:4}}>{id}</div>
      </div>

      {/* Pipeline */}
      <div className="job-pipe">
        {JOB_PIPE_STEPS.map(s => {
          const st = steps[s.key] || {};
          const cls = st.done ? "done" : st.active ? "active" : "";
          return (
            <div key={s.key} className={`jp-step ${cls}`}>
              <div className="jp-dot"/>
              <div className="jp-n">{s.n}</div>
              <div className="jp-name">{s.name}</div>
              <div className="jp-sub">{s.sub}</div>
            </div>
          );
        })}
      </div>

      {/* Key-value detail */}
      <div className="kv-grid" style={{marginBottom:24}}>
        <div className="kv-row"><div className="kv-key">Job ID</div><div className="kv-val">{id}</div></div>
        <div className="kv-row"><div className="kv-key">Requester</div><div className="kv-val"><Link to={`/address/${job.requester}`}>{job.requester||"—"}</Link></div></div>
        <div className="kv-row"><div className="kv-key">Model</div><div className="kv-val">{job.model_id||"—"}</div></div>
        <div className="kv-row"><div className="kv-key">Shard mode</div><div className="kv-val">{(job.mode||"—").replace(/_/g," ")}</div></div>
        <div className="kv-row"><div className="kv-key">Shards</div><div className="kv-val">{job.n_shards||1} ({job.shard_count||0} submitted)</div></div>
        <div className="kv-row"><div className="kv-key">Block</div><div className="kv-val"><Link to={`/block/${job.block_number}`}>#{fmtNum(job.block_number)}</Link></div></div>
        <div className="kv-row"><div className="kv-key">Fee</div><div className="kv-val">{job.fee_inft||0} INFT</div></div>
        <div className="kv-row"><div className="kv-key">Status</div><div className="kv-val"><StatusBadge status={status}/></div></div>
        {job.output_hash && (
          <div className="kv-row"><div className="kv-key">Output hash</div><div className="kv-val">{job.output_hash}</div></div>
        )}
        <div className="kv-row">
          <div className="kv-key">Context exchanges</div>
          <div className="kv-val">
            {ctxEntries > 0
              ? <span style={{color:"var(--good)"}}>{ctxEntries} loaded</span>
              : <span style={{color:"var(--ink-3)"}}>none</span>}
          </div>
        </div>
        {ctxEntries > 0 && job.context_hash && (
          <div className="kv-row"><div className="kv-key">Context hash</div><div className="kv-val" style={{wordBreak:"break-all",fontSize:11}}>{job.context_hash}</div></div>
        )}
        {job.original_prompt && job.original_prompt !== job.prompt && (
          <div className="kv-row">
            <div className="kv-key">Original prompt</div>
            <div className="kv-val" style={{fontSize:12,color:"var(--ink-2)"}}>{job.original_prompt.slice(0,200)}{job.original_prompt.length>200?"…":""}</div>
          </div>
        )}
      </div>

      {/* Output */}
      {job.final_output && (
        <div className="output-box">
          <div className="output-label">✓ INFERENCE OUTPUT</div>
          <div style={{whiteSpace:"pre-wrap",lineHeight:1.7}}>{job.final_output}</div>
        </div>
      )}

      {/* Shard breakdown */}
      {job.shards && Object.keys(job.shards).length > 0 && (
        <div className="panel" style={{marginTop:24}}>
          <div className="panel-head"><span className="panel-title">Shards</span></div>
          <table className="data-table">
            <thead>
              <tr><th>#</th><th>Status</th><th>Miner</th><th>Output</th></tr>
            </thead>
            <tbody>
              {Object.entries(job.shards).map(([i, s]) => (
                <tr key={i}>
                  <td className="hash">{i}</td>
                  <td><StatusBadge status={s.status}/></td>
                  <td className="hash">{s.miner ? <Link to={`/address/${s.miner}`}>{shortAddr(s.miner)}</Link> : "—"}</td>
                  <td style={{fontSize:12,color:"var(--ink-2)",maxWidth:300}}>
                    {s.output ? s.output.slice(0,80)+"…" : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Transactions page ─────────────────────────────────────────────────────────
function TransactionsPage() {
  const [txs,  setTxs]  = useState([]);
  const [page, setPage] = useState(0);
  const PER = 25;

  const load = useCallback(async () => {
    try { setTxs(await rpc("inft_getRecentTransactions", [200]) || []); } catch {}
  }, []);
  useEffect(() => { load(); }, []);
  useInterval(load, 5000);

  const slice = txs.slice(page * PER, (page + 1) * PER);

  return (
    <div className="page">
      <div className="detail-head">
        <div className="detail-title">Transactions</div>
        <div className="detail-sub">{fmtNum(txs.length)} recent transactions</div>
      </div>
      <div className="panel">
        <table className="data-table">
          <thead>
            <tr><th>Tx Hash</th><th>Type</th><th>Block</th><th>From</th><th>Age</th></tr>
          </thead>
          <tbody>
            {slice.map(tx => (
              <tr key={tx.tx_hash}>
                <td className="hash"><Link to={`/tx/${tx.tx_hash}`}>{shortHash(tx.tx_hash)}</Link></td>
                <td><TxTypeBadge type={tx.tx_type}/></td>
                <td className="hash"><Link to={`/block/${tx.block_number}`}>#{fmtNum(tx.block_number)}</Link></td>
                <td className="hash">{tx.sender ? <Link to={`/address/${tx.sender}`}>{shortAddr(tx.sender)}</Link> : <span style={{color:"var(--ink-3)"}}>system</span>}</td>
                <td className="age">{tx.timestamp ? ageMs(tx.timestamp) : "—"}</td>
              </tr>
            ))}
            {txs.length === 0 && <tr><td colSpan={5} className="empty">No transactions yet</td></tr>}
          </tbody>
        </table>
        <div className="pager">
          <button className="pager-btn" disabled={page===0} onClick={() => setPage(p=>p-1)}>← Prev</button>
          <span className="pager-info">&nbsp;Page {page+1} of {Math.max(1,Math.ceil(txs.length/PER))}&nbsp;</span>
          <button className="pager-btn" disabled={(page+1)*PER>=txs.length} onClick={() => setPage(p=>p+1)}>Next →</button>
        </div>
      </div>
    </div>
  );
}

// ── Address detail page ───────────────────────────────────────────────────────
function AddressPage({ addr }) {
  const [data,    setData]    = useState(null);
  const [tab,     setTab]     = useState("jobs");
  const [err,     setErr]     = useState(null);
  const [history, setHistory] = useState(null);

  useEffect(() => {
    rpc("inft_getAddressHistory", [addr])
      .then(d => { if (!d || !d.address) setErr("Address not found"); else setData(d); })
      .catch(e => setErr(e.message));
  }, [addr]);

  useEffect(() => {
    if (tab !== "history") return;
    rpc("inft_getHistory", [addr, null, 50])
      .then(h => setHistory(Array.isArray(h) ? h : []))
      .catch(() => setHistory([]));
  }, [tab, addr]);

  if (err)  return <div className="page"><div className="empty">{err}</div></div>;
  if (!data) return <div className="page"><div className="empty loading-dot">Loading</div></div>;

  return (
    <div className="page">
      <div className="crumb"><span>Address</span></div>
      <div className="detail-head">
        <div className="detail-title" style={{fontFamily:"var(--mono)",fontSize:18,wordBreak:"break-all"}}>{addr}</div>
        <div className="detail-sub">InferenceChain L2 · Chain {localStorage.getItem("ic-chain")||2026}</div>
      </div>

      <div className="addr-hero">
        <div className="addr-stat">
          <div className="stat-label">INFT Balance</div>
          <div className="stat-value">{fmtNum(data.balance_inft)} <span style={{fontSize:14,color:"var(--ink-3)"}}>INFT</span></div>
        </div>
        <div className="addr-stat">
          <div className="stat-label">Staked</div>
          <div className="stat-value">{fmtNum(data.stake_inft)} <span style={{fontSize:14,color:"var(--ink-3)"}}>INFT</span></div>
        </div>
        <div className="addr-stat">
          <div className="stat-label">Reputation</div>
          <div className="stat-value" style={{color: data.reputation >= 500 ? "var(--good)" : "var(--bad)"}}>{data.reputation}<span style={{fontSize:14,color:"var(--ink-3)"}}>/1000</span></div>
        </div>
        <div className="addr-stat">
          <div className="stat-label">Nonce</div>
          <div className="stat-value">{data.nonce}</div>
        </div>
        {data.unlock_block > 0 && (
          <div className="addr-stat">
            <div className="stat-label">Unlock block</div>
            <div className="stat-value" style={{fontSize:18}}>{fmtNum(data.unlock_block)}</div>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="tab-bar">
          <div className={`tab ${tab==="jobs"?"active":""}`} onClick={() => setTab("jobs")}>
            Jobs ({data.jobs?.length||0})
          </div>
          <div className={`tab ${tab==="txs"?"active":""}`} onClick={() => setTab("txs")}>
            Transactions ({data.transactions?.length||0})
          </div>
          <div className={`tab ${tab==="history"?"active":""}`} onClick={() => setTab("history")}>
            AI History
          </div>
        </div>

        {tab === "jobs" && (
          <table className="data-table">
            <thead>
              <tr><th>Job ID</th><th>Model</th><th>Role</th><th>Block</th><th>Status</th></tr>
            </thead>
            <tbody>
              {(data.jobs||[]).map(j => (
                <tr key={j.job_id}>
                  <td className="hash"><Link to={`/job/${j.job_id}`}>{j.job_id.slice(0,8)}…</Link></td>
                  <td style={{fontSize:12}}>{j.model_id?.split("/")?.[1]||"—"}</td>
                  <td><span className={`badge ${j.role==="miner"?"badge-partial":"badge-tx"}`}>{j.role?.toUpperCase()}</span></td>
                  <td className="hash"><Link to={`/block/${j.block_number}`}>#{fmtNum(j.block_number)}</Link></td>
                  <td><StatusBadge status={j.status}/></td>
                </tr>
              ))}
              {(!data.jobs||data.jobs.length===0) && <tr><td colSpan={5} className="empty">No jobs</td></tr>}
            </tbody>
          </table>
        )}

        {tab === "txs" && (
          <table className="data-table">
            <thead>
              <tr><th>Tx Hash</th><th>Type</th><th>Block</th><th>Age</th></tr>
            </thead>
            <tbody>
              {(data.transactions||[]).map(tx => (
                <tr key={tx.tx_hash}>
                  <td className="hash"><Link to={`/tx/${tx.tx_hash}`}>{shortHash(tx.tx_hash)}</Link></td>
                  <td><TxTypeBadge type={tx.tx_type}/></td>
                  <td className="hash"><Link to={`/block/${tx.block_number}`}>#{fmtNum(tx.block_number)}</Link></td>
                  <td className="age">{tx.timestamp ? ageMs(tx.timestamp) : "—"}</td>
                </tr>
              ))}
              {(!data.transactions||data.transactions.length===0) && <tr><td colSpan={4} className="empty">No transactions</td></tr>}
            </tbody>
          </table>
        )}

        {tab === "history" && (
          <div style={{padding:"12px 16px"}}>
            {history === null && (
              <div className="empty" style={{padding:20}}>Loading…</div>
            )}
            {history !== null && history.length === 0 && (
              <div className="empty" style={{padding:20}}>No inference history for this address</div>
            )}
            {(history||[]).map((entry, i) => (
              <div key={entry.job_id || i} style={{
                marginBottom:16, padding:"12px 0",
                borderBottom:"1px solid var(--border)"
              }}>
                <div style={{display:"flex",gap:12,alignItems:"center",marginBottom:6,flexWrap:"wrap"}}>
                  <span className="badge badge-tx">{entry.model_id?.split("/")?.[1]||entry.model_id||"?"}</span>
                  <span className="age">{entry.timestamp ? ageMs(entry.timestamp) : "?"}</span>
                  {entry.job_id && (
                    <Link to={`/job/${entry.job_id}`} style={{fontFamily:"var(--mono)",fontSize:11,color:"var(--accent)"}}>
                      {entry.job_id.slice(0,8)}…
                    </Link>
                  )}
                </div>
                <div style={{fontFamily:"var(--mono)",fontSize:12,color:"var(--warn)",marginBottom:4,lineHeight:1.5}}>
                  Q: {entry.prompt}
                </div>
                <div style={{fontFamily:"var(--mono)",fontSize:12,color:"var(--ink-2)",lineHeight:1.5,whiteSpace:"pre-wrap"}}>
                  A: {entry.output}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Search result page ────────────────────────────────────────────────────────
function SearchPage({ query }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    rpc("inft_search", [query])
      .then(r => { setResult(r); setLoading(false); })
      .catch(() => { setResult({ type: "not_found" }); setLoading(false); });
  }, [query]);

  useEffect(() => {
    if (!result) return;
    if (result.type === "block")       nav(`/block/${result.data.block_number}`);
    if (result.type === "job")         nav(`/job/${result.data.job_id}`);
    if (result.type === "transaction") nav(`/tx/${result.data.tx_hash}`);
    if (result.type === "address")     nav(`/address/${query}`);
  }, [result]);

  if (loading) return <div className="page"><div className="empty loading-dot">Searching</div></div>;
  return (
    <div className="page">
      <div className="empty">
        <div style={{fontSize:32,marginBottom:12}}>🔍</div>
        <div>Nothing found for <code style={{color:"var(--accent)"}}>{query}</code></div>
        <div style={{marginTop:8,fontSize:12}}>Try a block number, job ID, transaction hash, or address</div>
      </div>
    </div>
  );
}

// ── Root app ──────────────────────────────────────────────────────────────────
function App() {
  const route = useRoute();
  const [searchQuery, setSearchQuery] = useState(null);

  const handleSearch = useCallback(q => {
    setSearchQuery(q);
    nav(`/search/${encodeURIComponent(q)}`);
  }, []);

  const renderPage = () => {
    if (route === "/" || route === "") return <HomePage />;
    if (route === "/blocks")          return <BlocksPage />;
    if (route === "/jobs")            return <JobsPage />;
    if (route === "/transactions")    return <TransactionsPage />;

    const blockMatch = route.match(/^\/block\/(\d+)$/);
    if (blockMatch) return <BlockDetailPage num={blockMatch[1]} />;

    const txMatch = route.match(/^\/tx\/(.+)$/);
    if (txMatch) return <TxDetailPage hash={txMatch[1]} />;

    const jobMatch = route.match(/^\/job\/(.+)$/);
    if (jobMatch) return <JobDetailPage id={jobMatch[1]} />;

    const addrMatch = route.match(/^\/address\/(.+)$/);
    if (addrMatch) return <AddressPage addr={addrMatch[1]} />;

    const searchMatch = route.match(/^\/search\/(.+)$/);
    if (searchMatch) return <SearchPage query={decodeURIComponent(searchMatch[1])} />;

    return <div className="page"><div className="empty">Page not found: {route}</div></div>;
  };

  return (
    <>
      <NavBar onSearch={handleSearch} />
      {renderPage()}
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
