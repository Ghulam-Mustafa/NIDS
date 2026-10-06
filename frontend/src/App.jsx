import { useEffect, useRef, useState } from "react";

const STEPS = { pcap: ["Upload", "Convert to flows", "Classify", "Report"], csv: ["Upload", "Classify", "Report"] };
const ORDER = { queued: 0, converting: 1, classifying: 2, done: 3 };
const tone = (c) => (c === "BENIGN" ? "ok" : c === "ANOMALY" ? "odd" : "bad");

export default function App() {
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("alerts");
  const [page, setPage] = useState(1);
  const [data, setData] = useState({ rows: [], total: 0 });
  const [drag, setDrag] = useState(false);
  const input = useRef();
  const SIZE = 15;

  async function upload(file) {
    if (!file) return;
    setError(""); setJob(null); setPage(1); setFilter("alerts");
    const body = new FormData(); body.append("file", file);
    const r = await fetch("/api/analyze", { method: "POST", body });
    const j = await r.json();
    if (!r.ok) return setError(j.error);
    setJob({ id: j.id, name: file.name, kind: /\.csv$/i.test(file.name) ? "csv" : "pcap", status: "queued" });
  }

  useEffect(() => {
    if (!job || ["done", "error"].includes(job.status)) return;
    const t = setInterval(async () => setJob(await (await fetch(`/api/jobs/${job.id}`)).json()), 1500);
    return () => clearInterval(t);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (job?.status !== "done") return;
    fetch(`/api/jobs/${job.id}/flows?cls=${filter}&page=${page}&size=${SIZE}`).then((r) => r.json()).then(setData);
  }, [job?.status, filter, page]);

  const s = job?.summary;
  const classes = s ? Object.entries(s.counts).sort((a, b) => b[1] - a[1]) : [];
  const max = classes[0]?.[1] || 1;
  const alerts = s ? s.rows - (s.counts.BENIGN || 0) : 0;
  const steps = STEPS[job?.kind || "pcap"];
  const at = job ? (job.kind === "csv" && job.status === "classifying" ? 1 : job.kind === "csv" && job.status === "done" ? 2 : ORDER[job.status]) : -1;

  return (
    <main>
      <header>
        <h1>Flow Watch</h1>
        <p>Upload captured traffic and see which network flows look like attacks.</p>
      </header>

      <section
        className={"drop" + (drag ? " over" : "")}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}
      >
        <p><strong>Drop a capture here</strong>, or <button onClick={() => input.current.click()}>choose a file</button></p>
        <small>.pcap and .pcapng are converted to flows first. A flow .csv from CICFlowMeter is classified directly.</small>
        <input ref={input} type="file" hidden accept=".pcap,.pcapng,.cap,.csv" onChange={(e) => upload(e.target.files[0])} />
      </section>

      {error && <p className="msg bad">{error}</p>}

      {job && (
        <ol className="steps" aria-label="Progress">
          {steps.map((label, i) => (
            <li key={label} className={job.status === "error" && i === Math.max(at, 0) ? "fail" : i < at ? "done" : i === at ? "now" : ""}>{label}</li>
          ))}
        </ol>
      )}
      {job?.status === "error" && <p className="msg bad">{job.error}</p>}

      {s && (
        <>
          {s.warnings.map((w) => <p key={w} className="msg warn">{w}</p>)}
          <section className="summary">
            <div className="totals">
              <div><b>{s.rows.toLocaleString()}</b><span>flows analysed</span></div>
              <div className={alerts ? "bad" : "ok"}><b>{alerts.toLocaleString()}</b><span>flagged as attack or anomaly</span></div>
              <a className="btn" href={`/api/jobs/${job.id}/download`}>Download predictions</a>
            </div>
            <ul className="bars">
              {classes.map(([c, n]) => (
                <li key={c}>
                  <span>{c}</span>
                  <i className={tone(c)} style={{ width: `${Math.max(2, (n / max) * 100)}%` }} />
                  <em>{n.toLocaleString()}</em>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <div className="bar">
              <h2>Flows</h2>
              <select value={filter} onChange={(e) => { setFilter(e.target.value); setPage(1); }}>
                <option value="alerts">Alerts only</option>
                <option value="all">All flows</option>
                {classes.map(([c]) => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="scroll">
              <table>
                <thead><tr><th>Source</th><th>Destination</th><th>Port</th><th>Verdict</th><th>Attack score</th></tr></thead>
                <tbody>
                  {data.rows.map((r, i) => (
                    <tr key={i}>
                      <td>{r["Src IP"] || "-"}</td><td>{r["Dst IP"] || "-"}</td><td>{r["Destination Port"] ? Number(r["Destination Port"]) : "-"}</td>
                      <td><span className={"tag " + tone(r.prediction)}>{r.prediction}</span></td>
                      <td>{(Number(r.attack_probability) * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                  {!data.rows.length && <tr><td colSpan="5" className="empty">No flows match this filter.</td></tr>}
                </tbody>
              </table>
            </div>
            <div className="pager">
              <button disabled={page === 1} onClick={() => setPage(page - 1)}>Previous</button>
              <span>Page {page} of {Math.max(1, Math.ceil(data.total / SIZE))}</span>
              <button disabled={page * SIZE >= data.total} onClick={() => setPage(page + 1)}>Next</button>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
