import "dotenv/config";
import express from "express";
import multer from "multer";
import cors from "cors";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { parse } from "csv-parse";

const env = process.env;
const PORT = env.PORT || 5000;
const PYTHON = env.PYTHON_BIN || "python";
const NIDS_DIR = path.resolve(env.NIDS_DIR || "../ML");
const CONVERTER = env.CONVERTER || "cfm";
const CFM_PATH = env.CFM_PATH || "cfm";
const JOBS_DIR = path.resolve("jobs");
const jobs = new Map();
fs.mkdirSync(JOBS_DIR, { recursive: true });

const modelsReady = () => fs.existsSync(path.join(NIDS_DIR, "models", "metadata.json"));
if (!fs.existsSync(path.join(NIDS_DIR, "nids"))) console.warn(`! ML folder not found at ${NIDS_DIR} - set NIDS_DIR in .env`);
else if (!modelsReady()) console.warn("! No trained models yet. In the ML folder run: python -m nids run-all");

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, f, cb) => { const d = path.join(JOBS_DIR, (req.jobId ??= randomUUID())); fs.mkdirSync(d, { recursive: true }); cb(null, d); },
    filename: (req, f, cb) => cb(null, "input" + path.extname(f.originalname).toLowerCase()),
  }),
  limits: { fileSize: Number(env.MAX_UPLOAD_MB || 1024) * 1024 * 1024 },
  fileFilter: (req, f, cb) => cb(/\.(pcap|pcapng|cap|csv)$/i.test(f.originalname) ? null : new Error("Upload a .pcap, .pcapng or flow .csv file"), true),
});

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const win = process.platform === "win32";           // cmd.exe needs quoted paths (spaces in folder names)
    const q = (x) => (win ? `"${x}"` : x);
    const p = spawn(q(cmd), args.map(q), { shell: win, ...opts });
    let err = "";
    p.stdout.on("data", () => {});
    p.stderr.on("data", (d) => (err = (err + d).slice(-2000)));
    p.on("error", reject);
    p.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited with ${code}: ${err.trim()}`))));
  });
}

async function convert(job) {
  const d = job.dir, flows = path.join(d, "flows.csv");
  if (CONVERTER === "python") {
    const wrapper = path.join(import.meta.dirname, "cicflowmeter-wrapper.py");
    await run(PYTHON, [wrapper, "-f", job.input, "-c", flows]);
  } else {
    const inDir = path.join(d, "pcap_in"), outDir = path.join(d, "flows_out");
    fs.mkdirSync(inDir, { recursive: true }); fs.mkdirSync(outDir, { recursive: true });
    fs.copyFileSync(job.input, path.join(inDir, path.basename(job.input)));
    await run(CFM_PATH, [inDir, outDir]);
    const csv = fs.readdirSync(outDir).find((f) => f.endsWith(".csv"));
    if (!csv) throw new Error("CICFlowMeter produced no flows (is the pcap empty or unsupported?)");
    fs.renameSync(path.join(outDir, csv), flows);
  }
  if (!fs.existsSync(flows)) throw new Error("Converter finished but flows.csv was not created");
  job.flows = flows;
}

async function process_(job) {
  try {
    if (job.kind === "pcap") { job.status = "converting"; await convert(job); }
    job.status = "classifying";
    const pred = path.join(job.dir, "predictions.csv"), sum = path.join(job.dir, "summary.json");
    await run(PYTHON, ["-m", "nids", "predict", "--input", job.flows, "--output", pred, "--summary", sum], { cwd: NIDS_DIR });
    job.pred = pred;
    job.summary = JSON.parse(fs.readFileSync(sum, "utf8"));
    job.status = "done";
  } catch (e) { job.status = "error"; job.error = e.message; }
}

const app = express();
app.use(cors());

app.get("/api/health", (req, res) => res.json({ ok: true, converter: CONVERTER, mlDir: NIDS_DIR, modelsReady: modelsReady() }));

app.post("/api/analyze", (req, res, next) =>
  modelsReady() ? next() : res.status(503).json({ error: "The ML models are not trained yet. Run 'python -m nids run-all' in the ML folder." }),
  upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  const kind = /\.csv$/i.test(req.file.originalname) ? "csv" : "pcap";
  const job = { id: req.jobId, kind, status: "queued", dir: path.dirname(req.file.path), input: req.file.path,
                flows: kind === "csv" ? req.file.path : null, name: req.file.originalname };
  jobs.set(job.id, job);
  process_(job);
  res.json({ id: job.id });
});

const view = (j) => ({ id: j.id, name: j.name, kind: j.kind, status: j.status, error: j.error, summary: j.summary });
const find = (req, res) => { const j = jobs.get(req.params.id); if (!j) res.status(404).json({ error: "Unknown job" }); return j; };

app.get("/api/jobs/:id", (req, res) => { const j = find(req, res); if (j) res.json(view(j)); });

app.get("/api/jobs/:id/flows", async (req, res) => {
  const j = find(req, res); if (!j) return;
  if (j.status !== "done") return res.status(409).json({ error: "Job not finished" });
  const { cls = "alerts", page = 1, size = 15 } = req.query;
  const start = (page - 1) * size, rows = []; let total = 0;
  const parser = fs.createReadStream(j.pred).pipe(parse({ columns: true }));
  for await (const r of parser) {
    const ok = cls === "all" || (cls === "alerts" ? r.prediction !== "BENIGN" : r.prediction === cls);
    if (!ok) continue;
    if (total >= start && rows.length < size) rows.push(r);
    total++;
  }
  res.json({ rows, total });
});

app.get("/api/jobs/:id/download", (req, res) => { const j = find(req, res); if (j?.pred) res.download(j.pred, "predictions.csv"); });
app.use((err, req, res, next) => res.status(400).json({ error: err.message }));
app.listen(PORT, () => console.log(`NIDS backend on http://localhost:${PORT} (converter: ${CONVERTER})`));
