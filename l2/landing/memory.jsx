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

const PAGE_SIZE = 15;

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

const shortAddr = (a = "", n = 6) => a ? `${a.slice(0, 2 + n)}…${a.slice(-4)}` : "—";

function useInterval(fn, ms) {
  const cb = useRef(fn);
  useEffect(() => { cb.current = fn; }, [fn]);
  useEffect(() => {
    const id = setInterval(() => cb.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}

// ── Nav ───────────────────────────────────────────────────────────────────────
function NavBar() {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <div className="brand">
          <div className="diamond"/>
          InferenceChain
          <span className="nav-tag">MEMORY</span>
        </div>
        <div className="nav-links">
          <a href="explorer.html">Explorer</a>
          <a href="dashboard.html">Dashboard</a>
          <a href="miners.html">Miners</a>
          <a href="memory.html" className="active">Memory</a>
          <a href="index.html">← Site</a>
        </div>
      </div>
    </nav>
  );
}

// ── Single thought card ───────────────────────────────────────────────────────
function ThoughtCard({ thought }) {
  const [showThink, setShowThink] = useState(false);
  const hasThinking = thought.thinking_text && thought.thinking_text.trim().length > 0;

  const modelShort = (thought.model_id || "").split("/")[1] || thought.model_id || "unknown";

  return (
    <div className="thought-card">
      <div className="thought-head">
        <span className="thought-model">{modelShort}</span>
        <span className="thought-miner" title={thought.miner_address}>
          {shortAddr(thought.miner_address)}
        </span>
        {thought.job_id && (
          <a className="thought-job"
             href={`explorer.html#/job/${thought.job_id}`}
             title={thought.job_id}>
            job:{thought.job_id.slice(0, 10)}…
          </a>
        )}
        {thought.score > 0 && (
          <span className="thought-score">score:{thought.score.toFixed(3)}</span>
        )}
      </div>
      <div className="thought-body">
        {/* Question */}
        <div className="thought-q">{thought.question_text || "—"}</div>

        {/* Thinking (collapsible) */}
        {hasThinking && (
          <>
            <button className="thought-think-toggle" onClick={() => setShowThink(v => !v)}>
              <span className={`toggle-arrow ${showThink ? "open" : ""}`}>▶</span>
              Chain of thought {showThink ? "(hide)" : `(${thought.thinking_text.length} chars)`}
            </button>
            {showThink && (
              <div className="thought-think">{thought.thinking_text}</div>
            )}
          </>
        )}

        {/* Answer */}
        {thought.answer_text && (
          <>
            <div className="thought-a-label">Answer</div>
            <div className="thought-a">{thought.answer_text}</div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Root app ──────────────────────────────────────────────────────────────────
function App() {
  const [thoughts,      setThoughts]      = useState([]);
  const [query,         setQuery]         = useState("");
  const [draftQuery,    setDraftQuery]    = useState("");
  const [modelFilter,   setModelFilter]   = useState("");
  const [models,        setModels]        = useState([]);
  const [loading,       setLoading]       = useState(false);
  const [connected,     setConnected]     = useState(null);
  const [lastRefresh,   setLastRefresh]   = useState(null);
  const [page,          setPage]          = useState(0);
  const [totalIngested, setTotalIngested] = useState(0);

  // Fetch thoughts (search or recent)
  const fetchThoughts = useCallback(async (q, model, silent = false) => {
    if (!silent) setLoading(true);
    try {
      let results;
      if (q.trim()) {
        results = await rpc("inft_searchThoughts", [q.trim(), model, 50]);
      } else {
        results = await rpc("inft_getRecentThoughts", [50, model]);
      }
      setThoughts(results || []);
      setConnected(true);
      setLastRefresh(new Date());
      setPage(0);

      // Collect unique models for filter dropdown
      const modelSet = new Set((results || []).map(t => t.model_id).filter(Boolean));
      setModels(prev => {
        const merged = new Set([...prev, ...modelSet]);
        return [...merged].sort();
      });
    } catch {
      setConnected(false);
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  // Initial load + periodic refresh of recent thoughts
  useEffect(() => { fetchThoughts("", ""); }, []);
  useInterval(() => {
    if (!query) fetchThoughts("", modelFilter, true);
  }, 10000);

  // Count total ingested (approximate from results)
  useEffect(() => {
    if (thoughts.length > 0) {
      setTotalIngested(prev => Math.max(prev, thoughts.length));
    }
  }, [thoughts]);

  const handleSearch = () => {
    setQuery(draftQuery);
    fetchThoughts(draftQuery, modelFilter);
  };

  const handleModelChange = (e) => {
    const m = e.target.value;
    setModelFilter(m);
    fetchThoughts(query, m);
  };

  // Paginate
  const pageStart  = page * PAGE_SIZE;
  const pageEnd    = pageStart + PAGE_SIZE;
  const pageThoughts = thoughts.slice(pageStart, pageEnd);
  const totalPages = Math.max(1, Math.ceil(thoughts.length / PAGE_SIZE));

  const uniqueMiners = new Set(thoughts.map(t => t.miner_address).filter(Boolean)).size;
  const uniqueModels = new Set(thoughts.map(t => t.model_id).filter(Boolean)).size;

  return (
    <>
      <NavBar/>
      <div className="page">

        {/* Status bar */}
        <div className="refresh-bar">
          <div className={`live-dot ${connected === false ? "bad" : ""}`}/>
          <span className="refresh-label">
            {connected === null ? "Connecting…"
           : connected === false ? "Chain unreachable"
           : `Memory live · pg_inft distributed store`}
          </span>
          {lastRefresh && (
            <span className="refresh-label" style={{marginLeft:8}}>
              Updated {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button className="refresh-btn ml-auto" onClick={() => fetchThoughts(query, modelFilter)}>
            Refresh
          </button>
        </div>

        {/* Stats */}
        <div className="stats-bar">
          <div className="stat-item">
            <span className="stat-val">{thoughts.length}</span>
            <span className="stat-label">Results</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">{uniqueMiners}</span>
            <span className="stat-label">Miners</span>
          </div>
          <div className="stat-item">
            <span className="stat-val">{uniqueModels}</span>
            <span className="stat-label">Models</span>
          </div>
          <div className="stat-item">
            <span className="stat-val" style={{color:"var(--purple)"}}>
              {thoughts.filter(t => t.thinking_text?.trim()).length}
            </span>
            <span className="stat-label">With Thinking</span>
          </div>
        </div>

        {/* Search */}
        <div className="search-bar">
          <input
            className="search-input"
            placeholder="Search questions and answers… (BM25 + trigram)"
            value={draftQuery}
            onChange={e => setDraftQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSearch()}
          />
          <select className="model-filter" value={modelFilter} onChange={handleModelChange}>
            <option value="">All models</option>
            {models.map(m => (
              <option key={m} value={m}>{m.split("/")[1] || m}</option>
            ))}
          </select>
          <button className="search-btn" onClick={handleSearch} disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        {/* Thought list */}
        {connected === false ? (
          <div className="empty">
            Cannot reach chain at {L2_RPC}<br/>
            <span style={{fontSize:10,marginTop:6,display:"block"}}>
              The chain node needs pg_dsn configured for memory search to work.
            </span>
          </div>
        ) : loading ? (
          <div className="empty loading-dot">Searching memory</div>
        ) : thoughts.length === 0 ? (
          <div className="empty">
            {query
              ? `No results for "${query}"`
              : "No inference history yet — run jobs to populate the distributed memory store."}
            <div style={{marginTop:12,fontSize:10,color:"var(--ink-3)"}}>
              Thoughts are automatically ingested after each inference job and gossiped
              across all nodes via P2P. They are searchable using BM25 full-text + trigram ranking.
            </div>
          </div>
        ) : (
          <>
            <div className="thought-list">
              {pageThoughts.map((t, i) => (
                <ThoughtCard key={t.id ?? `${t.job_id}-${i}`} thought={t}/>
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="pagination">
                <button
                  className="page-btn"
                  disabled={page === 0}
                  onClick={() => setPage(p => Math.max(0, p - 1))}>
                  ← Prev
                </button>
                {Array.from({length: totalPages}, (_, i) => (
                  <button
                    key={i}
                    className={`page-btn ${i === page ? "active" : ""}`}
                    onClick={() => setPage(i)}>
                    {i + 1}
                  </button>
                ))}
                <button
                  className="page-btn"
                  disabled={page === totalPages - 1}
                  onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}>
                  Next →
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
