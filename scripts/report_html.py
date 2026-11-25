"""One self-contained HTML page of an ablation comparison (scripts/compare_saes.py), for a meeting.

    python scripts/report_html.py ablations/sae_brain.tsv     # -> <OUTPUT_ROOT>/compare/sae_brain/report.html

Reads the comparison CSVs and match tables, every run's training log (``log.jsonl``) and the
store's split. The page loads nothing from the web, so it opens offline, and holds only counts and
metrics, no slide or patient ids. Sections: data and split, key numbers of the ``main`` runs,
reconstruction of every run, training curves, shared concepts, probes and the full metrics table.
"""
import argparse, csv, datetime, json, math, os, pathlib, re, socket, sys

import polars as pl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", type=pathlib.Path, help="the ablation table given to compare_saes.py")
    p.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path(os.environ.get("DATA_ROOT", "data")))
    p.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path(os.environ.get("OUTPUT_ROOT", "outputs")))
    p.add_argument("--compare-dir", type=pathlib.Path, default=None, help="default: <output-root>/compare/<table name>")
    p.add_argument("--out", type=pathlib.Path, default=None, help="default: <compare-dir>/report.html")
    p.add_argument("--title", default="Virchow2 vs UNI2 sparse autoencoders")
    return p.parse_args()


def number(v: str):
    if v in ("", None):
        return None
    try:
        x = float(v)
    except ValueError:
        return v
    return x if math.isfinite(x) else None


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fd:
        return [{k: number(v) for k, v in r.items()} for r in csv.DictReader(fd)]


def pretty(run: str) -> str:
    """virchow2_cls_x4_k30_seed42 -> Virchow2 · 4× · k 30 · seed 42."""
    s = run.replace("virchow2", "Virchow2").replace("uni2", "UNI2")
    s = re.sub(r"_cls(?=_|$)", "", s)
    s = re.sub(r"_meanpatch(?=_|$)", " · mean patch", s)
    s = re.sub(r"_x(\d+)", r" · \1×", s)
    s = re.sub(r"_k(\d+)", r" · k \1", s)
    s = re.sub(r"_mat(\d+)", r" · matryoshka \1", s)
    s = re.sub(r"_norm(?=_|$)", " · z-scored", s)
    s = re.sub(r"_ep(\d+)", r" · \1 epochs", s)
    s = re.sub(r"_seed(\d+)", r" · seed \1", s)
    return s.replace("_", " ")


def curve(run_dir: pathlib.Path, bins: int = 250) -> dict:
    """Training explained variance per batch, averaged into at most ``bins`` points; validation per epoch."""
    steps, ev, val = [], [], []
    path = run_dir / "log.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in r and r.get("ev_batch") is not None:
            steps.append(r["step"]), ev.append(r["ev_batch"])
        elif "step" not in r and r.get("explained_variance") is not None:
            val.append(r["explained_variance"])
    pts = []
    per = max(1, math.ceil(len(steps) / bins))
    for i in range(0, len(steps), per):
        s, e = steps[i:i + per], ev[i:i + per]
        pts.append([s[-1], round(sum(e) / len(e), 5)])
    return {"train": pts, "val": [round(v, 5) for v in val]}


def dataset(store_dir: pathlib.Path | None, data_root: pathlib.Path) -> dict:
    """Slides, patients and tiles per project and split (counts only)."""
    out = {}
    record = data_root / "trident_import.json"
    if record.exists():
        r = json.loads(record.read_text())
        out["import"] = {k: r.get(k) for k in ("n_slides", "n_tiles", "projects", "max_slides_per_project", "max_tiles_per_slide")}
    summary = data_root / "tiles" / "slides_summary.parquet"
    splits = store_dir / "splits.json" if store_dir else None
    if not summary.exists():
        return out
    s = pl.read_parquet(summary)
    split_of = {}
    if splits and splits.exists():
        for name, ids in json.loads(splits.read_text()).items():
            split_of |= {i: name for i in ids}
    s = s.with_columns(pl.col("slide_id").replace_strict(split_of, default="not in store").alias("split"))
    has_patient = "patient" in s.columns
    rows = []
    for (proj, split), g in s.group_by(["project", "split"]):
        rows.append({"project": proj, "split": split, "slides": g.height, "tiles": int(g["n_tiles"].sum()),
                     "patients": g["patient"].drop_nulls().n_unique() if has_patient else None})
    out["by_split"] = sorted(rows, key=lambda r: (r["project"], ["train", "val", "test"].index(r["split"]) if r["split"] in ("train", "val", "test") else 9))
    out["n_patients"] = s["patient"].drop_nulls().n_unique() if has_patient else None
    return out


def histograms(compare_dir: pathlib.Path, encoder_of: dict) -> list[dict]:
    """Best-match correlation of every concept, per run pair, as fractions in 20 bins over [0, 1]."""
    out = []
    for f in sorted(compare_dir.glob("matches_*__*.parquet")):
        a, b = f.stem.removeprefix("matches_").split("__")
        corr = pl.read_parquet(f, columns=["corr"])["corr"].clip(0, 0.99999).to_list()
        if not corr:
            continue
        counts = [0] * 20
        for c in corr:
            counts[int(c * 20)] += 1
        out.append({"a": a, "b": b, "enc_a": encoder_of.get(a), "enc_b": encoder_of.get(b), "n": len(corr),
                    "frac": [round(c / len(corr), 5) for c in counts]})
    return out


def main() -> None:
    a = parse_args()
    cdir = a.compare_dir or a.output_root / "compare" / a.table.stem
    metrics = read_csv(cdir / "metrics.csv")
    if not metrics:
        sys.exit(f"no {cdir / 'metrics.csv'}: run scripts/compare_saes.py {a.table} first (or pass --compare-dir)")
    for r in metrics:
        r["label"] = pretty(r["run_name"])
    encoder_of = {r["run_name"]: r["encoder"] for r in metrics}

    store_dir = None
    runs = {}
    for r in metrics:
        rdir = a.output_root / "sae" / r["run_name"]
        runs[r["run_name"]] = curve(rdir)
        cfg = rdir / "config.json"
        if store_dir is None and r.get("groups", "").split(",")[0] == "main" and cfg.exists():
            store_dir = pathlib.Path(json.loads(cfg.read_text())["store_dir"])
    if store_dir is None or not store_dir.exists():
        store_dir = a.data_root / "store" / str(metrics[0].get("store") or metrics[0]["encoder"])

    data = {
        "title": a.title, "table": a.table.stem,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "metrics": metrics, "main_summary": read_csv(cdir / "main_summary.csv"),
        "matching": read_csv(cdir / "matching.csv"), "probes": read_csv(cdir / "probes.csv"),
        "histograms": histograms(cdir, encoder_of), "curves": runs, "dataset": dataset(store_dir, a.data_root),
    }
    out = a.out or cdir / "report.html"
    page = TEMPLATE.replace("__TITLE__", a.title).replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out.write_text(page)
    fqdn = socket.getfqdn()
    host = next((h for h in ("vista", "ls6", "frontera", "stampede3") if f".{h}." in fqdn), None)
    print(f"-> {out} ({out.stat().st_size / 1e3:.0f} KB). Open it in a browser"
          + (f"; from your laptop: scp {os.environ.get('USER', '<user>')}@{host}.tacc.utexas.edu:'{out.resolve()}' ." if host else ""))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --ref: #52514e;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
    --ref: #c3c2b7;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --ref: #c3c2b7;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 8px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 19px; margin: 0 0 4px; font-weight: 650; }
.sub, .note { color: var(--ink-2); margin: 0; }
.note { font-size: 13.5px; margin-top: 8px; max-width: 72ch; }
section { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 20px; margin-top: 20px; }
.head { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.head p { color: var(--ink-2); margin: 0; max-width: 64ch; font-size: 14px; }
select, button { font: inherit; font-size: 14px; color: var(--ink); background: var(--surface); border: 1px solid var(--axis); border-radius: 8px; padding: 5px 10px; }
button { cursor: pointer; }
ul.findings { margin: 8px 0 0; padding-left: 20px; }
ul.findings li { margin: 6px 0; max-width: 80ch; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.tile { border: 1px solid var(--ring); border-radius: 10px; padding: 12px 14px; }
.tile .l { color: var(--ink-2); font-size: 13px; }
.tile .v { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile .d { color: var(--muted); font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.card { border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; }
.card h3 { margin: 0 0 8px; font-size: 16px; display: flex; align-items: center; gap: 8px; }
.card dl { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; margin: 0; font-size: 14px; }
.card dt { color: var(--ink-2); } .card dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 13px; color: var(--ink-2); margin: 0 0 6px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend .line { width: 16px; height: 2px; border-radius: 1px; }
.legend .tick { width: 2px; height: 12px; background: var(--ref); }
.chart { width: 100%; }
.chart svg { display: block; overflow: visible; }
.chart text { font: 12.5px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink-2); }
.chart text.m { fill: var(--muted); font-variant-numeric: tabular-nums; }
.chart text.g { fill: var(--ink); font-weight: 600; }
.chart text.v { fill: var(--ink); font-variant-numeric: tabular-nums; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 24px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13.5px; width: 100%; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; } th:first-child, td:first-child { text-align: left; }
#tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
  box-shadow: 0 4px 16px rgba(0,0,0,.12); border-radius: 8px; padding: 8px 10px; font-size: 13px; line-height: 1.45; white-space: pre; display: none; }
.empty { color: var(--muted); font-size: 14px; }
@media (max-width: 600px) { .two { grid-template-columns: 1fr; } h1 { font-size: 23px; } }
@media print { button, select { display: none; } section { break-inside: avoid; } }
</style>
</head>
<body>
<main>
  <header>
    <div><h1 id="title"></h1><p class="sub" id="subtitle"></p></div>
    <button id="theme" type="button" aria-label="Switch light or dark">Dark</button>
  </header>

  <section id="s-findings"><h2>Summary</h2><ul class="findings" id="findings"></ul>
    <p class="note">Written from the numbers below by the script. The seed spread is the standard deviation over the seeds of one encoder: a gap between encoders counts only if it is clearly larger.</p></section>

  <section id="s-data"><div class="head"><div><h2>Data</h2><p>Tiles from the TRIDENT import, split by patient into train, validation and test. Every encoder sees the same tiles and the same split.</p></div></div>
    <div class="tiles" id="data-tiles"></div><div class="scroll" style="margin-top:16px" id="data-table"></div></section>

  <section id="s-main"><div class="head"><div><h2>Paper recipe, per encoder</h2><p>The <i>main</i> runs: mean ± standard deviation over seeds, on the test tiles.</p></div></div>
    <div class="cards" id="cards"></div>
    <p class="note">Explained variance: share of the embedding's variance the SAE reconstructs (1 is perfect); per tile, the same averaged over tiles. Cosine: similarity of each tile's reconstruction to its embedding. Dead concepts never fire on the test tiles; active ones fire on at least 0.1% of them. The encoders have different widths and scales, so compare these, not raw MSE.</p></section>

  <section id="s-recon"><div class="head"><div><h2>Reconstruction, every run</h2><p id="recon-desc"></p></div>
    <label>Metric <select id="recon-metric"></select></label></div>
    <div class="legend" id="recon-legend"></div><div class="chart" id="recon"></div></section>

  <section id="s-train"><div class="head"><div><h2>Training</h2><p>Explained variance of each training batch, averaged over windows of steps. A curve still rising at the end means more epochs would help.</p></div>
    <label>Group <select id="train-group"></select></label></div>
    <div class="legend" id="train-legend"></div><div class="chart" id="train"></div></section>

  <section id="s-concepts"><div class="head"><div><h2>Shared concepts</h2><p>For each concept of SAE A, its best correlation with any concept of SAE B on the same test tiles. Two seeds of one encoder set the baseline for "the same concept"; the encoder pairs show how much Virchow2 and UNI2 share.</p></div>
    <label>Measure <select id="match-metric"></select></label></div>
    <div class="two"><div><div class="legend" id="match-legend"></div><div class="chart" id="match"></div></div>
    <div><div class="legend" id="hist-legend"></div><div class="chart" id="hist"></div>
    <p class="note">Share of concepts by best-match correlation (A → B), averaged over the pairs of each kind.</p></div></div></section>

  <section id="s-probes"><div class="head"><div><h2>Probes</h2><p id="probe-desc">A logistic regression predicts each tile's project label from the raw embedding or from the SAE concepts. Tile AUC scores tiles; slide AUC scores the mean tile probability of each slide.</p></div>
    <label>Score <select id="probe-metric"><option value="slide_auc">Slide AUC</option><option value="tile_auc">Tile AUC</option></select></label></div>
    <div class="legend" id="probe-legend"></div><div class="chart" id="probes"></div><p class="note" id="probe-note"></p></section>

  <section id="s-table"><div class="head"><div><h2>All runs</h2><p>Test metrics of every run, as in <code>metrics.csv</code>.</p></div></div><div class="scroll" id="table"></div></section>
</main>
<div id="tip" role="tooltip"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (v, d = 3) => v == null || !isFinite(v) ? "–" : Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (Number.isInteger(v) && Math.abs(v) >= 10 ? v.toLocaleString() : v.toFixed(d));
const pct = (v, d = 1) => v == null ? "–" : (100 * v).toFixed(d) + "%";
const ENC_NAME = e => ({virchow2: "Virchow2", uni2: "UNI2"}[e] || e);
// Fixed colour per encoder (never by rank); slot 3 is kept for "encoder vs encoder".
const encoders = [...new Set(D.metrics.map(r => r.encoder))].sort((a, b) => (a === "virchow2" ? -1 : b === "virchow2" ? 1 : a === "uni2" ? -1 : b === "uni2" ? 1 : a < b ? -1 : 1));
const SLOTS = ["--s1", "--s2", "--s4", "--s5", "--s6", "--s7", "--s8"];
const encVar = e => `var(${SLOTS[Math.max(0, encoders.indexOf(e)) % SLOTS.length]})`;
const CROSS = "var(--s3)";

// ---------- tooltip ----------
const tip = $("tip");
function showTip(text, e) {
  tip.textContent = text; tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
document.addEventListener("pointermove", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) showTip(t.getAttribute("data-tip"), e);
  else if (!e.target.closest || !e.target.closest(".xh")) tip.style.display = "none";
});

// ---------- scales ----------
function niceTicks(lo, hi, n) {
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const t = [];
  for (let v = Math.floor(lo / step + 1e-9) * step; v <= Math.ceil(hi / step - 1e-9) * step + step / 2; v += step) t.push(+v.toFixed(10));
  return t;
}
const tickFmt = (v, span) => Math.abs(v) >= 10000 ? (v / 1000).toFixed(0) + "K" : span < 0.1 ? v.toFixed(3) : span < 2 ? v.toFixed(2).replace(/0$/, "") : v.toLocaleString();
function legend(el, items) {
  el.innerHTML = items.map(i => `<span>${i.kind === "line" ? `<i class="line" style="background:${i.color}"></i>` : i.kind === "tick" ? `<i class="tick"></i>` :
    i.kind === "diamond" ? `<svg width="12" height="12" viewBox="-6 -6 12 12"><path d="M0,-5.5L5.5,0L0,5.5L-5.5,0Z" style="fill:${i.color}"/></svg>` :
    `<i class="key" style="background:${i.color}"></i>`}${esc(i.name)}</span>`).join("");
}

// ---------- dot plot: one row per item, grouped, value printed at the right ----------
function dotPlot(el, rows, o = {}) {
  const W = Math.max(320, el.clientWidth), narrow = W < 600;
  const longest = Math.max(...rows.map(r => String(r.label).length));
  const LW = Math.min(narrow ? 160 : 300, Math.max(90, longest * 6.9 + 20)), VW = 64, rowH = 26, gh = 28, T = 4, B = 30;
  let y = T, last;
  const items = [];
  for (const r of rows) {
    if (r.group != null && r.group !== last) { items.push({head: r.group, y: y + 18}); y += gh; last = r.group; }
    items.push(Object.assign({}, r, {y: y + rowH / 2})); y += rowH;
  }
  const H = y + B;
  const vals = rows.flatMap(r => [r.v, r.ref]).filter(v => v != null && isFinite(v));
  if (!vals.length) { el.innerHTML = `<p class="empty">No values.</p>`; return; }
  let lo = o.min != null ? Math.min(o.min, ...vals) : Math.min(0, ...vals), hi = o.max != null ? Math.max(o.max, ...vals) : Math.max(...vals);
  const ticks = niceTicks(lo, hi, narrow ? 4 : 6); lo = ticks[0]; hi = ticks[ticks.length - 1];
  const x0 = LW, x1 = W - VW, x = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
  let s = `<svg width="${W}" height="${H}" role="img" aria-label="${esc(o.aria || "")}">`;
  for (const t of ticks) {
    s += `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${y}" style="stroke:${t === 0 && lo < 0 ? "var(--axis)" : "var(--grid)"}" stroke-width="1"/>`;
    s += `<text class="m" x="${x(t)}" y="${y + 18}" text-anchor="middle">${tickFmt(t, hi - lo)}</text>`;
  }
  if (o.unit) s += `<text class="m" x="${x1}" y="${y + 18}" dx="8">${esc(o.unit)}</text>`;
  for (const it of items) {
    if (it.head != null) { s += `<text class="g" x="0" y="${it.y}">${esc(it.head)}</text>`; continue; }
    const cy = it.y;
    s += `<line x1="${x0}" x2="${x1}" y1="${cy}" y2="${cy}" style="stroke:var(--grid)" stroke-width="1" stroke-dasharray="0"/>`;
    s += `<text x="${LW - 12}" y="${cy + 4}" text-anchor="end">${esc(it.label)}</text>`;
    if (it.ref != null && isFinite(it.ref)) s += `<line x1="${x(it.ref)}" x2="${x(it.ref)}" y1="${cy - 8}" y2="${cy + 8}" style="stroke:var(--ref)" stroke-width="2" stroke-linecap="round"/>`;
    if (it.v != null && isFinite(it.v)) {
      const cx = x(it.v);
      s += it.shape === "diamond"
        ? `<path d="M${cx},${cy - 7}L${cx + 7},${cy}L${cx},${cy + 7}L${cx - 7},${cy}Z" style="fill:${it.color};stroke:var(--surface)" stroke-width="2"/>`
        : `<circle cx="${cx}" cy="${cy}" r="6" style="fill:${it.color};stroke:var(--surface)" stroke-width="2"/>`;
      s += `<text class="v" x="${W}" y="${cy + 4}" text-anchor="end">${esc(o.valFmt ? o.valFmt(it.v) : fmt(it.v))}</text>`;
    }
    s += `<rect x="0" y="${cy - rowH / 2}" width="${W}" height="${rowH}" fill="transparent" data-tip="${esc(it.tip || it.label)}"/>`;
  }
  el.innerHTML = s + "</svg>";
}

// ---------- line chart with a crosshair ----------
function lineChart(el, series, o = {}) {
  const W = Math.max(320, el.clientWidth), H = o.h || 280, L = 52, R = 16, T = 10, B = 40;
  const pts = series.flatMap(s => s.pts);
  if (!pts.length) { el.innerHTML = `<p class="empty">No ${esc(o.what || "data")}.</p>`; return; }
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const xt = niceTicks(o.xmin != null ? o.xmin : Math.min(...xs), o.xmax != null ? o.xmax : Math.max(...xs), W < 600 ? 4 : 6);
  const yt = niceTicks(o.ymin != null ? Math.min(o.ymin, ...ys) : Math.min(...ys), Math.max(...ys), 5);
  const xlo = o.xmin != null ? o.xmin : xt[0], xhi = o.xmax != null ? o.xmax : xt[xt.length - 1], ylo = yt[0], yhi = yt[yt.length - 1];
  const x = v => L + (v - xlo) / (xhi - xlo) * (W - L - R), y = v => T + (1 - (v - ylo) / (yhi - ylo)) * (H - T - B);
  let s = `<svg width="${W}" height="${H}" role="img" aria-label="${esc(o.aria || "")}">`;
  for (const t of yt) s += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--grid)"/><text class="m" x="${L - 8}" y="${y(t) + 4}" text-anchor="end">${tickFmt(t, yhi - ylo)}</text>`;
  for (const t of xt) if (t >= xlo && t <= xhi) s += `<text class="m" x="${x(t)}" y="${H - B + 18}" text-anchor="middle">${tickFmt(t, xhi - xlo)}</text>`;
  s += `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" style="stroke:var(--axis)"/>`;
  if (o.xlabel) s += `<text class="m" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(o.xlabel)}</text>`;
  for (const sr of series) {
    if (!sr.pts.length) continue;
    s += `<path d="${sr.pts.map((p, i) => (i ? "L" : "M") + x(p[0]).toFixed(1) + "," + y(p[1]).toFixed(1)).join("")}" fill="none" style="stroke:${sr.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"${sr.dash ? ` stroke-dasharray="${sr.dash}"` : ""}/>`;
  }
  s += `<line class="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" style="stroke:var(--axis);display:none"/><g class="dots"></g>`;
  s += `<rect class="xh" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/></svg>`;
  el.innerHTML = s;
  const svg = el.querySelector("svg"), ov = el.querySelector(".xh"), cross = el.querySelector(".cross"), dots = el.querySelector(".dots");
  const hide = () => { cross.style.display = "none"; dots.innerHTML = ""; tip.style.display = "none"; };
  ov.addEventListener("pointerleave", hide);
  ov.addEventListener("pointermove", e => {
    const px = e.clientX - svg.getBoundingClientRect().left, xv = xlo + (px - L) / (W - L - R) * (xhi - xlo);
    let lines = [], d = "";
    for (const sr of series) {
      if (!sr.pts.length) continue;
      let best = sr.pts[0];
      for (const p of sr.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      lines.push([best[1], `${sr.name}: ${(o.yFmt || fmt)(best[1])}`]);
      d += `<circle cx="${x(best[0])}" cy="${y(best[1])}" r="4.5" style="fill:${sr.color};stroke:var(--surface)" stroke-width="2"/>`;
    }
    lines.sort((a, b) => b[0] - a[0]);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.style.display = "";
    dots.innerHTML = d;
    showTip(`${o.xname || "x"} ${(o.xFmt || (v => Math.round(v).toLocaleString()))(xv)}\n` + lines.map(l => l[1]).join("\n"), e);
  });
}

// ---------- sections ----------
const M = D.metrics, mainRuns = M.filter(r => String(r.groups).split(",").includes("main"));
const S = Object.fromEntries(D.main_summary.map(r => [r.encoder, r]));
const projects = [...new Set((D.dataset.by_split || []).map(r => r.project))];

$("title").textContent = D.title;
document.title = D.title;
$("subtitle").textContent = `Ablation table ${D.table} · ${M.length} runs · generated ${D.generated}`;

function findings() {
  const out = [], enc = encoders.filter(e => S[e]);
  if (enc.length >= 2) {
    const [a, b] = enc, ev = e => S[e].explained_variance_mean, sd = e => S[e].explained_variance_std || 0;
    const gap = Math.abs(ev(a) - ev(b)), spread = Math.max(sd(a), sd(b));
    const hiE = ev(a) >= ev(b) ? a : b, loE = hiE === a ? b : a;
    out.push(`<b>${ENC_NAME(hiE)}</b> SAEs reconstruct more of their embeddings: ${pct(ev(hiE))} of the variance vs ${pct(ev(loE))} for ${ENC_NAME(loE)} (${S[a].n_seeds} seeds each). ` +
      (spread > 0 ? `The gap, ${(100 * gap).toFixed(1)} points, is ${(gap / spread).toFixed(0)}× the seed spread (${(100 * spread).toFixed(2)} points).` : `Only one seed per encoder, so there is no seed spread to compare with.`));
    const da = S[a].n_dead_mean, db = S[b].n_dead_mean, aa = S[a]["n_active_1e-3_mean"], ab = S[b]["n_active_1e-3_mean"];
    if (da != null && db != null) out.push(`Dead concepts: ${fmt(da)} for ${ENC_NAME(a)}, ${fmt(db)} for ${ENC_NAME(b)}; concepts active on ≥ 0.1% of tiles: ${fmt(aa)} vs ${fmt(ab)}.`);
  }
  const mm = r => (r.ab_mean + r.ba_mean) / 2;
  const seeds = D.matching.filter(r => r.kind === "seed"), cross = D.matching.filter(r => r.kind === "encoder");
  if (seeds.length && cross.length) {
    const within = encoders.map(e => [e, seeds.filter(r => M.find(m => m.run_name === r.a)?.encoder === e)]).filter(x => x[1].length)
      .map(([e, rs]) => `${ENC_NAME(e)} ${fmt(rs.reduce((s, r) => s + mm(r), 0) / rs.length, 2)}`);
    out.push(`Concepts: the mean best-match correlation between two seeds of one encoder is ${within.join(", ")}; between Virchow2 and UNI2 it is ${fmt(cross.reduce((s, r) => s + mm(r), 0) / cross.length, 2)}. ` +
      `A cross-encoder value near the seed baseline means the encoders learn largely the same concepts.`);
  }
  const P = D.probes.filter(r => r.slide_auc != null);
  if (P.length) {
    const parts = encoders.map(e => {
      const raw = P.find(r => r.input === "raw:" + e), sae = P.filter(r => r.input.startsWith("sae:") && M.find(m => "sae:" + m.run_name === r.input)?.encoder === e);
      if (!raw && !sae.length) return null;
      return `${ENC_NAME(e)}: raw ${fmt(raw?.slide_auc, 2)}` + (sae.length ? `, SAE ${fmt(sae.reduce((s, r) => s + r.slide_auc, 0) / sae.length, 2)}` : "");
    }).filter(Boolean);
    out.push(`Project probes (slide AUC) — ${parts.join("; ")}. An SAE close to its raw embedding keeps the label information in its concepts.`);
  }
  if (mixedDatasets()) out.push(`The data mixes datasets (${[...new Set(projects.map(p => p.split("-")[0]))].join(", ")}), so the probes partly learn which dataset a tile came from, not only the diagnosis.`);
  if (D.dataset.import && D.dataset.import.max_slides_per_project) out.push(`This is a small test import (${D.dataset.import.max_slides_per_project} slides per project, ≤ ${D.dataset.import.max_tiles_per_slide} tiles each), so these numbers check the pipeline, not the science.`);
  $("findings").innerHTML = out.length ? out.map(t => `<li>${t}</li>`).join("") : `<li class="empty">Not enough runs for a summary.</li>`;
}
function mixedDatasets() { return new Set(projects.map(p => p.split("-")[0])).size > 1; }

function dataSection() {
  const ds = D.dataset, rows = ds.by_split || [];
  if (!rows.length) { $("s-data").querySelector(".tiles").innerHTML = `<p class="empty">No import summary found under the data root.</p>`; return; }
  const sum = (k, f = () => true) => rows.filter(f).reduce((s, r) => s + (r[k] || 0), 0);
  const tiles = [["Slides", sum("slides").toLocaleString()], ["Patients", ds.n_patients != null ? ds.n_patients.toLocaleString() : "–"],
    ["Tiles", sum("tiles").toLocaleString()], ["Projects", projects.length]];
  for (const sp of ["train", "val", "test"]) tiles.push([`${sp === "val" ? "Validation" : sp[0].toUpperCase() + sp.slice(1)} tiles`, sum("tiles", r => r.split === sp).toLocaleString(), pct(sum("tiles", r => r.split === sp) / sum("tiles"), 0)]);
  $("data-tiles").innerHTML = tiles.map(t => `<div class="tile"><div class="l">${t[0]}</div><div class="v">${t[1]}</div>${t[2] ? `<div class="d">${t[2]} of tiles</div>` : ""}</div>`).join("");
  const splits = ["train", "val", "test"].filter(sp => rows.some(r => r.split === sp)).concat(rows.some(r => r.split === "not in store") ? ["not in store"] : []);
  const hasPat = rows.some(r => r.patients != null);
  let t = `<table><thead><tr><th>Project</th>${splits.map(sp => `<th colspan="${hasPat ? 3 : 2}" style="text-align:center">${sp}</th>`).join("")}</tr>` +
    `<tr><th></th>${splits.map(() => (hasPat ? "<th>patients</th>" : "") + "<th>slides</th><th>tiles</th>").join("")}</tr></thead><tbody>`;
  for (const p of projects) {
    t += `<tr><td>${esc(p)}</td>`;
    for (const sp of splits) { const r = rows.find(r => r.project === p && r.split === sp) || {}; t += (hasPat ? `<td>${fmt(r.patients)}</td>` : "") + `<td>${fmt(r.slides || 0)}</td><td>${(r.tiles || 0).toLocaleString()}</td>`; }
    t += "</tr>";
  }
  $("data-table").innerHTML = t + "</tbody></table>";
}

function cards() {
  const list = [["Explained variance", "explained_variance", pct], ["Explained variance per tile", "explained_variance_per_sample", pct], ["Cosine", "cosine", v => fmt(v, 3)],
    ["Dead concepts", "n_dead", v => fmt(v, 0)], ["Active concepts (≥ 0.1%)", "n_active_1e-3", v => fmt(v, 0)], ["Dense concepts", "n_dense", v => fmt(v, 0)]];
  const enc = encoders.filter(e => S[e]);
  $("cards").innerHTML = enc.length ? enc.map(e => `<div class="card"><h3><i class="key" style="background:${encVar(e)}"></i>${ENC_NAME(e)} <span class="sub" style="font-weight:400;font-size:13px">${S[e].n_seeds} seed${S[e].n_seeds > 1 ? "s" : ""}</span></h3><dl>` +
    list.map(([l, k, f]) => `<dt>${l}</dt><dd>${f(S[e][k + "_mean"])}${S[e][k + "_std"] != null && S[e].n_seeds > 1 ? ` <span class="sub">± ${k.startsWith("n_") ? fmt(S[e][k + "_std"], 0) : (k.startsWith("explained") ? (100 * S[e][k + "_std"]).toFixed(2) : fmt(S[e][k + "_std"], 3))}</span>` : ""}</dd>`).join("") + "</dl></div>").join("")
    : `<p class="empty">No main runs in main_summary.csv.</p>`;
}

const baseKey = Object.keys(M[0]).find(k => k.startsWith("baseline_pca_"));
const RECON = [
  ["explained_variance", "Explained variance", "Higher is better." + (baseKey ? ` The grey tick is PCA with ${baseKey.split("_").pop()} components on the same tiles, a linear baseline.` : ""), true],
  ["explained_variance_per_sample", "Explained variance per tile", "Higher is better. Explained variance of each tile, averaged.", false],
  ["cosine", "Cosine similarity", "Higher is better. Cosine between each tile's embedding and its reconstruction.", false],
  ["n_dead", "Dead concepts", "Lower is better. Concepts that never fire on the test tiles.", false],
  ["n_active_1e-3", "Active concepts (≥ 0.1% of tiles)", "Higher means more of the dictionary is in use.", false],
  ["n_dense", "Dense concepts", "Concepts that fire on a large share of tiles; few is better.", false],
  ["normalized_mse", "Normalized MSE", "Lower is better. MSE divided by the variance of the input.", false],
].filter(m => M.some(r => r[m[0]] != null));
$("recon-metric").innerHTML = RECON.map(m => `<option value="${m[0]}">${m[1]}</option>`).join("");
const groupsOrder = [...new Set(M.map(r => String(r.groups).split(",")[0]))].sort((a, b) => (a === "main" ? -1 : b === "main" ? 1 : 0));
function recon() {
  const m = RECON.find(m => m[0] === $("recon-metric").value) || RECON[0];
  $("recon-desc").textContent = m[2];
  const rows = [];
  for (const g of groupsOrder) for (const r of M.filter(r => String(r.groups).split(",")[0] === g))
    rows.push({group: g, label: r.label, v: r[m[0]], color: encVar(r.encoder), ref: m[3] && baseKey ? r[baseKey] : null,
      tip: `${r.run_name}\n${m[1]}: ${fmt(r[m[0]], 4)}` + (m[3] && baseKey ? `\nPCA baseline: ${fmt(r[baseKey], 4)}` : "") + `\nGroups: ${r.groups}`});
  legend($("recon-legend"), encoders.map(e => ({name: ENC_NAME(e), color: encVar(e)})).concat(m[3] && baseKey ? [{name: "PCA baseline", kind: "tick"}] : []));
  dotPlot($("recon"), rows, {aria: m[1] + " of every run", valFmt: m[0].startsWith("n_") ? v => fmt(v, 0) : v => fmt(v, 3)});
}

const curveGroups = groupsOrder.filter(g => M.some(r => String(r.groups).split(",")[0] === g && D.curves[r.run_name] && (D.curves[r.run_name].train || []).length));
$("train-group").innerHTML = ["all"].concat(curveGroups).map(g => `<option value="${g}">${g === "all" ? "All runs" : g}</option>`).join("");
if (curveGroups.includes("main")) $("train-group").value = "main";
function train() {
  const g = $("train-group").value, runs = M.filter(r => g === "all" || String(r.groups).split(",")[0] === g);
  const seen = {};
  const series = runs.map(r => { const k = r.encoder; seen[k] = (seen[k] || 0) + 1;
    return {name: r.label, color: encVar(r.encoder), dash: g !== "main" && seen[k] > 1 ? "6 4" : null, pts: (D.curves[r.run_name] || {}).train || []}; });
  legend($("train-legend"), encoders.filter(e => runs.some(r => r.encoder === e)).map(e => ({name: ENC_NAME(e), color: encVar(e), kind: "line"})));
  lineChart($("train"), series, {aria: "Training explained variance by step", xname: "step", xlabel: "training step", yFmt: v => pct(v, 1), what: "training logs"});
}

const MATCH = [["mean", "Mean best match"], ["median", "Median best match"], ["frac_gt_0.5", "Share of concepts matched > 0.5"], ["frac_gt_0.7", "Share of concepts matched > 0.7"]];
$("match-metric").innerHTML = MATCH.map(m => `<option value="${m[0]}">${m[1]}</option>`).join("");
const encOf = n => (M.find(r => r.run_name === n) || {}).encoder;
const parts = n => (M.find(r => r.run_name === n) || {label: n}).label.split(" · ").slice(1);
function pairLabel(a, b) {
  const ta = parts(a), tb = parts(b), da = ta.filter(t => !tb.includes(t)), db = tb.filter(t => !ta.includes(t));
  const seed = t => t.filter(x => x.startsWith("seed")).join(" · ") || t.join(" · ");
  return `${da.join(" · ") || seed(ta)} → ${db.join(" · ") || seed(tb)}`;
}
function concepts() {
  const k = $("match-metric").value, name = MATCH.find(m => m[0] === k)[1], rows = [];
  const kindOf = (a, b) => encOf(a) === encOf(b) ? `${ENC_NAME(encOf(a))} seed vs seed` : `${ENC_NAME(encOf(a))} → ${ENC_NAME(encOf(b))}`;
  const colorOf = (a, b) => encOf(a) === encOf(b) ? encVar(encOf(a)) : CROSS;
  for (const r of D.matching) for (const [a, b, p, na] of [[r.a, r.b, "ab_", r.n_concepts_a], [r.b, r.a, "ba_", r.n_concepts_b]])
    rows.push({group: kindOf(a, b), label: pairLabel(a, b), v: r[p + k], color: colorOf(a, b),
      rank: encOf(a) === encOf(b) ? encoders.indexOf(encOf(a)) : encoders.length + encoders.indexOf(encOf(a)),
      tip: `${a}\n  → ${b}\n${name}: ${fmt(r[p + k], 3)}\nConcepts of A matched: ${fmt(na, 0)}`});
  rows.sort((x, y) => x.rank - y.rank);
  legend($("match-legend"), encoders.filter(e => rows.some(r => r.group === `${ENC_NAME(e)} seed vs seed`)).map(e => ({name: `${ENC_NAME(e)} seeds`, color: encVar(e)}))
    .concat(rows.some(r => r.color === CROSS) ? [{name: "Across encoders", color: CROSS}] : []));
  dotPlot($("match"), rows, {aria: name + " per SAE pair", min: 0, max: k.startsWith("frac") ? 1 : undefined, valFmt: k.startsWith("frac") ? v => pct(v, 0) : v => fmt(v, 2)});
  // histogram, averaged per kind of pair
  const kinds = {};
  for (const h of D.histograms) {
    const key = h.enc_a === h.enc_b ? h.enc_a : "cross:" + h.enc_a + ">" + h.enc_b;
    (kinds[key] = kinds[key] || []).push(h.frac);
  }
  const series = Object.entries(kinds).map(([key, fr]) => {
    const cross = key.startsWith("cross:"), [ea, eb] = cross ? key.slice(6).split(">") : [key, key];
    return {name: cross ? `${ENC_NAME(ea)} → ${ENC_NAME(eb)}` : `${ENC_NAME(ea)} seeds`, color: cross ? CROSS : encVar(ea),
      rank: (cross ? encoders.length : 0) + encoders.indexOf(ea),
      pts: fr[0].map((_, i) => [(i + 0.5) / 20, fr.reduce((s, f) => s + f[i], 0) / fr.length])};
  }).sort((a, b) => a.rank - b.rank);
  legend($("hist-legend"), series.map(s => ({name: s.name, color: s.color, kind: "line"})));
  lineChart($("hist"), series, {aria: "Distribution of best-match correlation", xmin: 0, xmax: 1, ymin: 0, xname: "correlation", xlabel: "best-match correlation", xFmt: v => v.toFixed(2), yFmt: v => pct(v, 1), what: "match tables", h: 260});
}

function probes() {
  const k = $("probe-metric").value, P = D.probes;
  if (!P.length) { $("probes").innerHTML = `<p class="empty">No probes.csv (the training tiles may hold one project only).</p>`; return; }
  const rows = [];
  for (const e of encoders) for (const r of P) {
    const isRaw = r.input === "raw:" + e, run = r.input.startsWith("sae:") && M.find(m => "sae:" + m.run_name === r.input);
    if (!isRaw && !(run && run.encoder === e)) continue;
    rows.push({group: ENC_NAME(e), enc: e, label: isRaw ? "raw embedding" : "SAE · " + run.label.replace(/^\S+ · /, ""), v: r[k], color: encVar(e), shape: isRaw ? "diamond" : null, order: isRaw ? 0 : 1,
      tip: `${r.input}\nTile AUC: ${fmt(r.tile_auc, 3)}\nSlide AUC: ${fmt(r.slide_auc, 3)}\nTest slides: ${fmt(r.n_test_slides, 0)}` + (r.n_classes_scored != null ? `\nProjects scored: ${fmt(r.n_classes_scored, 0)}` : "")});
  }
  rows.sort((a, b) => encoders.indexOf(a.enc) - encoders.indexOf(b.enc) || a.order - b.order);
  legend($("probe-legend"), encoders.filter(e => rows.some(r => r.group === ENC_NAME(e))).map(e => ({name: ENC_NAME(e), color: encVar(e)}))
    .concat([{name: "raw embedding", kind: "diamond", color: "var(--muted)"}, {name: "SAE concepts", color: "var(--muted)"}]));
  dotPlot($("probes"), rows, {aria: "Probe AUC", min: 0.5, max: 1, valFmt: v => fmt(v, 3)});
  const nc = Math.max(...P.map(r => r.n_classes_scored || 2)), nt = Math.max(...P.map(r => r.n_test_slides || 0));
  $("probe-note").textContent = `AUC 0.5 is chance, 1 is perfect. ${nc > 2 ? `${nc} projects: the mean one-vs-rest AUC.` : "Two projects: binary AUC."} ${nt ? `${nt} test slides; ` : ""}with few test slides the slide AUC moves in large steps.` +
    (mixedDatasets() ? " The projects come from different datasets, so part of what the probe learns is the dataset." : "");
}

function table() {
  const cols = [["label", "Run"], ["groups", "Group"], ["explained_variance", "Expl. var."], ["explained_variance_per_sample", "Per tile"], ["cosine", "Cosine"], ["normalized_mse", "Norm. MSE"],
    ["l0", "L0"], ["n_dead", "Dead"], ["n_dense", "Dense"], ["n_active_1e-3", "Active ≥0.1%"], ["n_active_1e-5", "Active ≥0.001%"]].concat(Object.keys(M[0]).filter(k => k.startsWith("baseline_")).map(k => [k, k.replace("baseline_", "").replace("_", " ").replace(/^pca/, "PCA").replace(/^kmeans/, "k-means")]));
  $("table").innerHTML = `<table><thead><tr>${cols.map(c => `<th>${esc(c[1])}</th>`).join("")}</tr></thead><tbody>` +
    M.map(r => `<tr>${cols.map(([k]) => `<td>${k === "label" ? `<i class="key" style="background:${encVar(r.encoder)};margin-right:8px"></i>${esc(r.label)}` : typeof r[k] === "number" ? (k.startsWith("n_") || k === "l0" ? fmt(r[k], 0) : fmt(r[k], 3)) : esc(r[k] ?? "–")}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
}

function drawCharts() { recon(); train(); concepts(); probes(); }
findings(); dataSection(); cards(); table(); drawCharts();
$("recon-metric").onchange = recon; $("train-group").onchange = train; $("match-metric").onchange = concepts; $("probe-metric").onchange = probes;
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawCharts, 150); });
// Theme: follows the system; the button forces the other one.
const darkNow = () => document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
const label = () => { $("theme").textContent = darkNow() ? "Light" : "Dark"; };
$("theme").onclick = () => { document.documentElement.dataset.theme = darkNow() ? "light" : "dark"; label(); };
label();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
