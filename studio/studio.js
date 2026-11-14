'use strict';
// ParaBPE Studio viewer (B7). Plain JS, no build step. Data comes from studio/serve.py.

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SVGNS = 'http://www.w3.org/2000/svg';

const KINDS = ['CHUNK', 'PRETOK_UNIT', 'SHARD_REDUCE', 'SELECT', 'SERIAL_APPLY', 'OWNER_MERGE',
  'ZERO_SWEEP', 'ORDER_UPDATE'];
// Swimlanes are coloured by what the work is, in palette-slot order (validated as a set).
const GROUP_OF = { CHUNK: 0, PRETOK_UNIT: 0, SHARD_REDUCE: 1, SERIAL_APPLY: 2, OWNER_MERGE: 2,
  SELECT: 3, ORDER_UPDATE: 3, ZERO_SWEEP: 4 };
const GROUPS = ['Phase 1: chunk + pre-tokenize', 'Reduce shards', 'Rewrite words',
  'Select + update order (master)', 'Sweep zero pairs'];
const GROUP_VARS = ['--g0', '--g1', '--g2', '--g3', '--g4'];
const ENGINES = [
  { merge: 'incr-serial', label: 'IncrSerial', color: '--e-serial', marker: 'circle' },
  { merge: 'incr-par', label: 'IncrPar', color: '--e-par', marker: 'triangle' },
  { merge: 'incr-batch', label: 'incr-batch', color: '--e-batch', marker: 'hexagon' },
];
const P_CORES = 5;  // this Mac: threads 1-5 fit on the P-cores
const GUTTER = 96, LANE_H = 20, LANE_GAP = 7, TOP = 22, AXIS_H = 26, RIGHT = 12;

const S = {
  run: null, lanes: [], rounds: [], merges: [], totalMerges: 0, view: [0, 1], t: 0,
  playing: false, last: 0, p3: [0, 0], end: 0, landed: -1, coreLabels: false, colors: null,
};

// URL hash options for stills and rehearsals, e.g. #end&p3&light.
const HASH = new Set(location.hash.slice(1).split('&').filter(Boolean));
if (HASH.has('light') || HASH.has('dark')) {
  document.documentElement.dataset.theme = HASH.has('light') ? 'light' : 'dark';
}

// ---------- helpers ----------
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
  return body;
}
async function getText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.text();
}
function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
function svg(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function fmtTime(us) {
  if (us < 1000) return `${us.toFixed(1)} µs`;
  if (us < 1e6) return `${(us / 1000).toFixed(2)} ms`;
  return `${(us / 1e6).toFixed(3)} s`;
}
function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
function ticks(max, n = 5) {
  const raw = max / n, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function setStatus(id, text, err = false) {
  const s = $(id);
  s.textContent = text;
  s.classList.toggle('err', err);
}
function readColors() {
  S.colors = {
    groups: GROUP_VARS.map(cssVar), ink: cssVar('--ink'), ink2: cssVar('--ink-2'),
    grid: cssVar('--grid'), axis: cssVar('--axis'), strip: cssVar('--e-strip'),
    region: cssVar('--region'), batch: cssVar('--e-batch'), muted: cssVar('--muted'),
    surface: cssVar('--surface'),
  };
}

// ---------- run + trace loading ----------
async function init() {
  readColors();
  renderLegend();
  if (HASH.has('p3')) $('view').value = 'p3';
  for (let t = 1; t <= 16; ++t) $('threads').append(el('option', { value: t }, String(t)));
  $('threads').value = '11';
  try {
    const info = await getJSON('/api/inputs');
    for (const i of info.inputs) {
      $('input').append(el('option', { value: i.name }, `${i.name} (${(i.bytes / 1e6).toFixed(0)} MB)`));
    }
    if (info.inputs.some((i) => i.name === 'wikitext.full.txt')) $('input').value = 'wikitext.full.txt';
    if (!info.binary) setStatus('run-status', 'build/parabpe not found: build the project first.', true);
    if (!info.inputs.length) setStatus('run-status', 'No corpora in data/: run scripts/get_data.sh.', true);
    const latest = await getJSON('/api/runs/latest');
    if (latest.run) await loadRun(latest.run);
  } catch (e) {
    setStatus('run-status', `Server error: ${e.message}`, true);
  }
  loadResults();
  initRace();
}

$('run-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const body = {
    input: $('input').value, vocab_size: +$('vocab').value, threads: +$('threads').value,
    merge: $('merge').value, full_rounds: $('full-rounds').checked,
  };
  $('run-btn').disabled = true;
  setStatus('run-status', 'Training with --trace… (IncrSerial runs once per input and vocab for the reference)');
  try {
    const run = await getJSON('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    await loadRun(run);
  } catch (e) {
    setStatus('run-status', e.message, true);
  } finally {
    $('run-btn').disabled = false;
  }
});

async function loadRun(run) {
  stop();
  setStatus('run-status', 'Loading trace…');
  const [trace, mergesTxt] = await Promise.all([
    getJSON(`/api/runs/${run.id}/trace.json`), getText(`/api/runs/${run.id}/merges.txt`)]);
  S.run = run;
  S.merges = mergesTxt.split('\n').slice(1).filter((l) => l.length);
  S.totalMerges = S.merges.length;
  const d = parseTrace(trace, run);
  Object.assign(S, { lanes: d.lanes, rounds: d.rounds, p3: d.p3, end: d.end, coreLabels: d.coreLabels });

  const tm = run.timings.timings;
  const label = `${run.input}, vocab ${fmt(run.vocab_size)}, ${run.threads} threads, ${run.merge}` +
    (run.full_rounds ? ' (every round batched)' : '') +
    (run.merge_threads ? ` (merges on ${run.merge_threads_used} threads)` : '') +
    ` · total ${tm.total.toFixed(2)} s, phase 3 ${tm.merge.toFixed(3)} s`;
  $('run-label').textContent = label;
  setStatus('run-status', `Run ${run.id} loaded.`);
  for (const id of ['play', 'restart']) $(id).disabled = false;
  $('hf-btn').disabled = false;
  setView();
  buildKindTable();
  buildSpan();
  setBadge('pending');
  loadWork(run);
  // Stills: #end for the finished run, #at=0.4 for 40% of the way through the view.
  const at = [...HASH].find((h) => h.startsWith('at='));
  if (HASH.has('end')) { S.t = S.view[1]; render(); }
  else if (at) { S.t = S.view[0] + Math.min(1, Math.max(0, +at.slice(3) || 0)) * (S.view[1] - S.view[0]); render(); }
}

// A Chrome-trace file as per-thread lanes, incr-batch rounds and the phase windows (µs from the
// start of the run). Shared by the single-run replay and the race.
function parseTrace(trace, run) {
  const tm = run.timings.timings;
  const other = trace.otherData || {};
  const lanes = Array.from({ length: other.threads }, (_, tid) => ({ tid, ev: [] }));
  let tmax = 0, maxCpu = -1;
  for (const e of trace.traceEvents) {
    if (e.ph !== 'X') continue;
    const k = KINDS.indexOf(e.name);
    if (k < 0 || !lanes[e.tid]) continue;
    const args = e.args || {};
    const argName = Object.keys(args)[0];
    lanes[e.tid].ev.push({ t0: e.ts, t1: e.ts + e.dur, k, g: GROUP_OF[e.name], argName,
      arg: args[argName], cpu: args.cpu, e: args.core === 'E', unit: args.unit });
    tmax = Math.max(tmax, e.ts + e.dur);
    if (args.cpu !== undefined) maxCpu = Math.max(maxCpu, args.cpu);
  }
  for (const lane of lanes) lane.ev.sort((a, b) => a.t0 - b.t0 || b.t1 - a.t1);

  // Rounds (incr-batch): SELECT i picks round i's merges; they land when its ORDER_UPDATE ends.
  // Between the two the words are rewritten: by owners in parallel, or by the master alone.
  const master = lanes[0].ev;
  const selects = master.filter((e) => KINDS[e.k] === 'SELECT');
  const orders = master.filter((e) => KINDS[e.k] === 'ORDER_UPDATE');
  const rounds = [];
  let done = 0;
  selects.forEach((s, i) => {
    if (!s.arg) return;
    rounds.push({ k: s.arg, start: done, t: orders[i] ? orders[i].t1 : s.t1, rewrite: [s.t1,
      orders[i] ? orders[i].t0 : s.t1] });
    done += s.arg;
  });

  // Phase 1 ends with the last traced phase-1 event, or with the measured chunk + pretok time when
  // the serial pre-tokenizer (untraced per unit) ran. Phase 3 starts at the first selection, else
  // after the untraced phase 2.
  let p1End = 0;
  for (const lane of lanes) {
    for (const e of lane.ev) if (e.g === 0) p1End = Math.max(p1End, e.t1);
  }
  p1End = Math.max(p1End, (tm.chunk + tm.pretok) * 1e6);
  const firstP3 = Math.min(...lanes.flatMap((l) => l.ev.filter((e) => e.g >= 2).map((e) => e.t0)),
    Infinity);
  const p3Start = Number.isFinite(firstP3) && run.merge === 'incr-batch' ? firstP3
    : p1End + (tm.build + tm.pairs) * 1e6;
  const p3 = [p3Start, Math.max(p3Start + tm.merge * 1e6, run.merge === 'incr-batch' ? tmax : 0)];
  return { lanes, rounds, p1End, p3, end: Math.max(tmax, p3[1]), coreLabels: 'e_cores' in other,
    eCores: 'e_cores' in other ? other.e_cores : -1, cpus: Math.max(maxCpu + 1, other.threads) };
}

function setView() {
  S.view = $('view').value === 'p3' ? [S.p3[0], S.p3[1]] : [0, S.end];
  S.t = S.view[0];
  S.landed = -1;
  layoutLanes();
  render();
}

// ---------- swimlanes ----------
function layoutLanes() {
  const canvas = $('lanes');
  const w = canvas.parentElement.clientWidth;
  const h = TOP + S.lanes.length * (LANE_H + LANE_GAP) + AXIS_H;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  S.plotW = Math.max(50, w - GUTTER - RIGHT);
  aggregate();
}

// Per pixel column and lane: the group covering most of it, how full it is, and whether most of
// it was sampled on an E-core. Redrawn per frame up to the playhead.
function aggregate() {
  const W = Math.floor(S.plotW), [v0, v1] = S.view, scale = W / (v1 - v0);
  for (const lane of S.lanes) {
    const cov = new Float32Array(W * 5), ecov = new Float32Array(W);
    let busy = 0, eBusy = 0;
    for (const e of lane.ev) {
      if (e.t1 < v0 || e.t0 > v1) continue;
      const a = Math.max(e.t0, v0), b = Math.min(e.t1, v1);
      busy += b - a;
      if (e.e) eBusy += b - a;
      const x0 = (a - v0) * scale, x1 = Math.min(W, (b - v0) * scale);
      for (let c = Math.floor(x0); c < Math.ceil(x1) && c < W; ++c) {
        const o = Math.min(x1, c + 1) - Math.max(x0, c);
        cov[c * 5 + e.g] += o;
        if (e.e) ecov[c] += o;
      }
    }
    lane.dom = new Int8Array(W).fill(-1);
    lane.alpha = new Float32Array(W);
    lane.estrip = new Uint8Array(W);
    for (let c = 0; c < W; ++c) {
      let best = -1, bv = 0, tot = 0;
      for (let g = 0; g < 5; ++g) {
        const v = cov[c * 5 + g];
        tot += v;
        if (v > bv) { bv = v; best = g; }
      }
      lane.dom[c] = best;
      lane.alpha[c] = Math.round(Math.min(1, Math.max(0.35, tot)) * 4) / 4;
      lane.estrip[c] = ecov[c] > 0.5 * tot && tot > 0 ? 1 : 0;
    }
    lane.eShare = busy > 0 ? eBusy / busy : null;
  }
}

function draw() {
  const canvas = $('lanes'), ctx = canvas.getContext('2d'), C = S.colors;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.clearRect(0, 0, w, h);
  if (!S.lanes.length) return;
  const [v0, v1] = S.view, W = Math.floor(S.plotW), xOf = (t) => GUTTER + (t - v0) / (v1 - v0) * W;
  const playX = Math.min(W, Math.max(0, Math.floor((S.t - v0) / (v1 - v0) * W)));
  const lanesBottom = TOP + S.lanes.length * (LANE_H + LANE_GAP);
  ctx.font = '11.5px system-ui, -apple-system, sans-serif';
  ctx.textBaseline = 'middle';

  // Phase 3 boundary.
  if (S.p3[0] > v0 && S.p3[0] < v1) {
    const x = Math.round(xOf(S.p3[0])) + 0.5;
    ctx.strokeStyle = C.axis;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, 4); ctx.lineTo(x, lanesBottom); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.ink2;
    ctx.fillText('phase 3: merges', x + 6, 10);
    ctx.fillText('phase 1: pre-tokenize', GUTTER + 4, 10);
  } else {
    ctx.fillStyle = C.ink2;
    ctx.fillText(v0 >= S.p3[0] ? 'phase 3: merges' : 'phase 1: pre-tokenize', GUTTER + 4, 10);
  }

  S.lanes.forEach((lane, i) => {
    const y = TOP + i * (LANE_H + LANE_GAP);
    ctx.fillStyle = C.ink;
    ctx.textAlign = 'left';
    ctx.fillText(`t${lane.tid}`, 4, y + LANE_H / 2);
    if (lane.eShare !== null && S.coreLabels) {
      const pShare = 1 - lane.eShare, text = pShare >= 0.5 ? `P ${Math.round(pShare * 100)}%`
        : `E ${Math.round(lane.eShare * 100)}%`;
      ctx.fillStyle = pShare >= 0.5 ? C.ink : C.surface;
      const bx = 30, bw = 58;
      ctx.strokeStyle = C.axis;
      ctx.beginPath(); ctx.roundRect(bx, y + 2, bw, LANE_H - 4, 8);
      if (pShare >= 0.5) { ctx.stroke(); } else { ctx.fillStyle = C.strip; ctx.fill(); ctx.fillStyle = C.surface; }
      ctx.textAlign = 'center';
      ctx.fillText(text, bx + bw / 2, y + LANE_H / 2);
      ctx.textAlign = 'left';
    }
    // Lane track, then the columns up to the playhead, run-length encoded by colour and alpha.
    ctx.fillStyle = C.grid;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(GUTTER, y, W, LANE_H - 4);
    ctx.globalAlpha = 1;
    let c = 0;
    while (c < playX) {
      const g = lane.dom[c];
      if (g < 0) { ++c; continue; }
      const a = lane.alpha[c];
      let e = c + 1;
      while (e < playX && lane.dom[e] === g && lane.alpha[e] === a) ++e;
      ctx.globalAlpha = a;
      ctx.fillStyle = C.groups[g];
      ctx.fillRect(GUTTER + c, y, e - c, LANE_H - 4);
      c = e;
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = C.strip;
    for (let x = 0; x < playX; ++x) if (lane.estrip[x]) ctx.fillRect(GUTTER + x, y + LANE_H - 2, 1, 2);
  });

  // Time axis.
  const span = v1 - v0, tk = ticks(span / 1000, 6);  // in ms
  ctx.fillStyle = C.ink2;
  ctx.strokeStyle = C.axis;
  ctx.textAlign = 'center';
  ctx.beginPath(); ctx.moveTo(GUTTER, lanesBottom + 0.5); ctx.lineTo(GUTTER + W, lanesBottom + 0.5); ctx.stroke();
  for (const ms of tk) {
    const x = GUTTER + ms * 1000 / span * W;
    if (x > GUTTER + W + 1) continue;
    ctx.beginPath(); ctx.moveTo(x + 0.5, lanesBottom); ctx.lineTo(x + 0.5, lanesBottom + 4); ctx.stroke();
    const t = v0 / 1000 + ms;
    ctx.fillText(t >= 1000 ? `${(t / 1000).toFixed(2)} s` : `${t.toFixed(0)} ms`, x, lanesBottom + 14);
  }
  // Playhead.
  if (playX > 0 && playX < W) {
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(GUTTER + playX + 0.5, TOP - 6); ctx.lineTo(GUTTER + playX + 0.5, lanesBottom); ctx.stroke();
    ctx.lineWidth = 1;
  }
  ctx.textAlign = 'left';
}

function renderLegend() {
  const lg = $('legend');
  lg.replaceChildren();
  GROUPS.forEach((name, g) => {
    const s = el('span');
    const sw = el('i');
    sw.style.background = `var(${GROUP_VARS[g]})`;
    s.append(sw, name);
    lg.append(s);
  });
  const s = el('span'), sw = el('i', { class: 'line' });
  sw.style.background = 'var(--e-strip)';
  s.append(sw, 'sampled on an E-core');
  lg.append(s);
}

// Hover: the innermost event under the cursor.
$('lanes').addEventListener('mousemove', (ev) => {
  const tip = $('tip');
  if (!S.lanes.length) return;
  const r = ev.target.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const i = Math.floor((y - TOP) / (LANE_H + LANE_GAP));
  const lane = S.lanes[i];
  if (!lane || x < GUTTER || x > GUTTER + S.plotW) { tip.hidden = true; return; }
  const [v0, v1] = S.view, perPx = (v1 - v0) / S.plotW, t = v0 + (x - GUTTER) * perPx;
  let lo = 0, hi = lane.ev.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (lane.ev[m].t0 <= t + perPx) lo = m + 1; else hi = m; }
  let hit = null;
  for (let j = lo - 1, n = 0; j >= 0 && n < 400; --j, ++n) {
    const e = lane.ev[j];
    if (e.t1 >= t - perPx && (!hit || e.t1 - e.t0 < hit.t1 - hit.t0)) hit = e;
  }
  if (!hit) { tip.hidden = true; return; }
  const core = S.coreLabels ? ` (${hit.e ? 'E' : 'P'}-core)` : '';
  tip.innerHTML = '';
  tip.append(el('b', {}, KINDS[hit.k]), ` · ${fmtTime(hit.t1 - hit.t0)}`, el('br'),
    `${hit.argName}: ${fmt(hit.arg)} · thread ${lane.tid} · cpu ${hit.cpu}${core}`, el('br'),
    `starts at ${fmtTime(hit.t0)}`);
  tip.hidden = false;
  const wrap = ev.target.parentElement.clientWidth;
  tip.style.left = `${Math.min(x + 14, wrap - tip.offsetWidth - 4)}px`;
  tip.style.top = `${y + 14}px`;
});
$('lanes').addEventListener('mouseleave', () => { $('tip').hidden = true; });

function buildKindTable() {
  const rows = KINDS.map(() => [0, 0, 0]);
  for (const lane of S.lanes) {
    for (const e of lane.ev) {
      rows[e.k][0] += 1;
      rows[e.k][1] += e.t1 - e.t0;
      if (e.e) rows[e.k][2] += e.t1 - e.t0;
    }
  }
  const t = el('table');
  const head = el('tr');
  for (const h of ['event', 'count', 'total ms', 'mean µs', 'E-core share']) head.append(el('th', {}, h));
  t.append(head);
  KINDS.forEach((k, i) => {
    const [n, us, e] = rows[i];
    if (!n) return;
    const tr = el('tr');
    for (const v of [k, fmt(n), (us / 1000).toFixed(1), (us / n).toFixed(1),
      S.coreLabels ? `${Math.round(100 * e / us)}%` : 'n/a']) tr.append(el('td', {}, v));
    t.append(tr);
  });
  $('kind-table').replaceChildren(t);
}

// ---------- replay ----------
function speed() {
  const v = $('speed').value;
  return v === 'auto' ? (S.view[1] - S.view[0]) / 1e6 / 20 : +v;
}
function play() {
  if (!S.lanes.length) return;
  if (S.t >= S.view[1]) setView();
  S.playing = true;
  S.last = performance.now();
  $('play').textContent = 'Pause';
  requestAnimationFrame(frame);
}
function stop() {
  S.playing = false;
  $('play').textContent = 'Play';
}
function frame(ts) {
  if (!S.playing) return;
  const dt = Math.min(0.1, (ts - S.last) / 1000);
  S.last = ts;
  S.t = Math.min(S.view[1], S.t + dt * speed() * 1e6);
  render();
  if (S.t >= S.view[1]) { stop(); return; }
  requestAnimationFrame(frame);
}
$('play').addEventListener('click', () => (S.playing ? stop() : play()));
$('restart').addEventListener('click', () => { stop(); setView(); });
$('view').addEventListener('change', () => { if (S.lanes.length) { stop(); setView(); } });

function landedAt(t) {
  let lo = 0, hi = S.rounds.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (S.rounds[m].t <= t) lo = m + 1; else hi = m; }
  return lo;
}
function render() {
  draw();
  if (!S.run) return;
  $('clock').textContent = `t = ${fmtTime(S.t)} of ${fmtTime(S.end)}`;
  const batch = S.rounds.length > 0;
  const landed = batch ? landedAt(S.t) : 0;
  const finished = S.t >= S.p3[1] - 1;
  if (batch) {
    const done = landed ? S.rounds[landed - 1].start + S.rounds[landed - 1].k : 0;
    $('t-merges').textContent = `${fmt(done)} / ${fmt(S.totalMerges)}`;
    $('t-rounds').textContent = fmt(landed);
    const k = landed ? S.rounds[landed - 1].k : 0;
    $('t-round').textContent = landed ? `${k} merge${k > 1 ? 's' : ''}` : '–';
  } else {
    $('t-merges').textContent = finished ? fmt(S.totalMerges) : `– / ${fmt(S.totalMerges)}`;
    $('t-rounds').textContent = 'one per merge';
    $('t-round').textContent = '1 merge';
  }
  if (landed !== S.landed) {
    S.landed = landed;
    renderFeed(landed);
    updateSpan(landed);
    renderWords(landed);
  }
  drawUnits();
  setBadge(finished ? (S.run.identical ? 'ok' : 'bad') : 'pending');
}

function setBadge(state) {
  const b = $('badge');
  if (b.dataset.state === state && S.run && b.dataset.run === S.run.id) return;
  b.dataset.state = state;
  b.dataset.run = S.run ? S.run.id : '';
  b.className = `badge ${state}`;
  const [icon, text] = { pending: ['…', 'checking at the end of the run'],
    ok: ['✓', 'identical to IncrSerial'], bad: ['✗', 'differs from IncrSerial'] }[state];
  b.querySelector('.badge-icon').textContent = icon;
  b.querySelector('.badge-text').textContent = text;
  $('badge-detail').textContent = state === 'pending' || !S.run ? ''
    : `merges.txt sha256 ${S.run.sha256.slice(0, 16)}… vs reference ${S.run.reference_sha256.slice(0, 16)}…`;
}

function renderFeed(landed) {
  const feed = $('feed');
  feed.replaceChildren();
  if (!S.rounds.length) {
    feed.append(el('p', { class: 'muted' }, S.run && S.run.merge !== 'incr-batch'
      ? `${S.run.merge} merges one pair per step, so there are no rounds to show. Train incr-batch to see batches.`
      : 'Press Play.'));
    return;
  }
  if (!landed) { feed.append(el('p', { class: 'muted' }, 'No round has landed yet.')); return; }
  for (let r = landed - 1, shown = 0; r >= 0 && shown < 7; --r, ++shown) {
    const round = S.rounds[r];
    const box = el('div', { class: `round${round.k >= 8 ? ' big' : ''}${shown ? ' old' : ''}` });
    const head = el('div', { class: 'round-head' });
    head.append(el('b', {}, `Round ${fmt(r + 1)}`), ` · ${round.k} merge${round.k > 1 ? 's' : ''}`);
    const chips = el('div', { class: 'chips' });
    const limit = shown ? 10 : 80;
    for (let i = 0; i < Math.min(round.k, limit); ++i) {
      const line = S.merges[round.start + i] || '';
      const [a, b] = line.split(' ');
      chips.append(el('span', { class: 'chip', title: `${a} + ${b}` }, `${a}${b}`));
    }
    if (round.k > limit) chips.append(el('span', { class: 'muted small' }, ` +${round.k - limit} more`));
    box.append(head, chips);
    feed.append(box);
  }
}

// ---------- span curve ----------
function buildSpan() {
  const host = $('span');
  host.replaceChildren();
  S.spanParts = null;
  if (!S.rounds.length) {
    host.append(el('p', { class: 'muted' }, 'The span curve needs an incr-batch run.'));
    return;
  }
  const R = S.rounds.length, M = S.totalMerges;
  const w = Math.max(280, host.clientWidth), h = 260, m = { l: 64, r: 16, t: 14, b: 36 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const xMax = niceMax(R), yMax = niceMax(M);
  const X = (v) => m.l + v / xMax * pw, Y = (v) => m.t + ph - v / yMax * ph;
  const root = svg('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img',
    'aria-label': `${fmt(M)} merges in ${fmt(R)} rounds` }, host);
  for (const v of ticks(yMax, 4)) {
    svg('line', { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: 'gridline' }, root);
    svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, root).textContent = fmt(v);
  }
  for (const v of ticks(xMax, 4)) {
    svg('text', { x: X(v), y: m.t + ph + 16, 'text-anchor': 'middle' }, root).textContent = fmt(v);
  }
  svg('line', { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, class: 'axis-line' }, root);
  svg('text', { x: m.l + pw / 2, y: h - 4, 'text-anchor': 'middle' }, root).textContent = 'rounds';
  svg('text', { x: 12, y: m.t + ph / 2, transform: `rotate(-90 12 ${m.t + ph / 2})`,
    'text-anchor': 'middle' }, root).textContent = 'merges done';
  const d = Math.min(xMax, yMax);
  svg('line', { x1: X(0), y1: Y(0), x2: X(d), y2: Y(d), class: 'ref' }, root);
  svg('text', { x: X(d * 0.55) + 6, y: Y(d * 0.55) - 6, 'text-anchor': 'start' }, root)
    .textContent = 'one merge per step';
  const step = Math.max(1, Math.ceil(R / 1500));
  let path = `M${X(0)},${Y(0)}`;
  for (let i = 0; i < R; i += step) path += `L${X(i + 1).toFixed(1)},${Y(S.rounds[i].start + S.rounds[i].k).toFixed(1)}`;
  path += `L${X(R).toFixed(1)},${Y(M).toFixed(1)}`;
  const clipId = `clip-${S.run.id}`;
  const clip = svg('clipPath', { id: clipId }, svg('defs', {}, root));
  const clipRect = svg('rect', { x: 0, y: 0, width: 0, height: h }, clip);
  svg('path', { d: path, fill: 'none', style: 'stroke: var(--axis)', 'stroke-width': 2 }, root);
  svg('path', { d: path, fill: 'none', style: 'stroke: var(--e-batch)', 'stroke-width': 2.5,
    'clip-path': `url(#${clipId})` }, root);
  const dot = svg('circle', { r: 5, cx: X(0), cy: Y(0), style: 'fill: var(--e-batch); stroke: var(--surface)',
    'stroke-width': 2 }, root);
  const label = svg('text', { x: m.l + 8, y: m.t + 12, class: 'end-label' }, root);
  const note = el('p', { class: 'muted small' });
  host.append(note);
  S.spanParts = { X, Y, clipRect, dot, label, note, R, M };
}
function updateSpan(landed) {
  const p = S.spanParts;
  if (!p) return;
  const done = landed ? S.rounds[landed - 1].start + S.rounds[landed - 1].k : 0;
  p.clipRect.setAttribute('width', p.X(landed) + 1);
  p.dot.setAttribute('cx', p.X(landed));
  p.dot.setAttribute('cy', p.Y(done));
  p.label.textContent = landed === p.R
    ? `${fmt(p.M)} merges in ${fmt(p.R)} rounds: ${(p.M / p.R).toFixed(1)}× fewer steps`
    : `${fmt(done)} merges in ${fmt(landed)} rounds`;
  p.note.textContent = S.run.full_rounds ? '' : 'Default mode: rounds too small to go parallel merge one '
    + 'pair. Tick "batch every round" for the span figure.';
}

// ---------- benchmark charts ----------
const BENCH = { rows: [], load: [] };
function parseCSV(text) {
  const lines = text.replace(/\r/g, '').split('\n').filter((l) => l.length);
  const head = lines[0].split(',');
  return lines.slice(1).map((l) => {
    const cells = l.split(','), o = {};
    head.forEach((h, i) => { const v = cells[i]; o[h] = v !== '' && !Number.isNaN(+v) ? +v : v; });
    return o;
  });
}
const usable = (r) => !r.warmup && !(r.total_s > 2 * r.cpu_s + 0.05);  // plot.py drops stalls too
const nameOf = (r) => String(r.input).split('/').pop().replace(/\.txt$/, '');

async function loadResults() {
  try {
    const { files } = await getJSON('/api/results');
    if (files.includes('batch.csv')) BENCH.rows = parseCSV(await getText('/api/results/batch.csv')).filter(usable);
    if (files.includes('owner_load.csv')) {
      BENCH.load = parseCSV(await getText('/api/results/owner_load.csv')).filter((r) => !r.warmup);
    }
  } catch (e) {
    $('c-amdahl').textContent = `Could not load results: ${e.message}`;
    return;
  }
  const names = [...new Set(BENCH.rows.map(nameOf))].sort();
  for (const n of names) $('bench-input').append(el('option', { value: n }, n));
  if (names.includes('wikitext.full')) $('bench-input').value = 'wikitext.full';
  const vocabs = [...new Set(BENCH.rows.map((r) => r.vocab_size))].sort((a, b) => a - b);
  for (const v of vocabs) $('bench-vocab').append(el('option', { value: v }, fmt(v)));
  drawBench();
}
$('bench-input').addEventListener('change', drawBench);
$('bench-vocab').addEventListener('change', drawBench);

function drawBench() {
  const name = $('bench-input').value, vocab = +$('bench-vocab').value;
  const rows = BENCH.rows.filter((r) => nameOf(r) === name && r.vocab_size === vocab);
  const base = rows.filter((r) => r.pretok === 'serial' && r.merge === 'incr-serial' && r.threads === 1);
  if (!base.length) {
    for (const id of ['c-amdahl', 'c-phase3']) $(id).textContent = 'No IncrSerial baseline in batch.csv for this choice.';
  } else {
    const bP1 = median(base.map((r) => r.chunk_s + r.pretok_s)), bM = median(base.map((r) => r.merge_s));
    const by = (merge, f) => {
      const g = new Map();
      for (const r of rows) {
        if (r.pretok !== 'omp' || r.merge !== merge) continue;
        if (!g.has(r.threads)) g.set(r.threads, []);
        g.get(r.threads).push(f(r));
      }
      return [...g.entries()].sort((a, b) => a[0] - b[0]).map(([t, v]) => [t, median(v)]);
    };
    lineChart($('c-amdahl'), {
      series: [
        { name: 'phase 1: pre-tokenize', color: '--s1', marker: 'circle',
          pts: by('incr-serial', (r) => r.chunk_s + r.pretok_s).map(([t, v]) => [t, bP1 / v]) },
        { name: 'phase 3: merges (IncrSerial)', color: '--s2', marker: 'square',
          pts: by('incr-serial', (r) => r.merge_s).map(([t, v]) => [t, bM / v]) },
      ],
      ideal: true, ref: null, yLabel: 'speedup',
    });
    lineChart($('c-phase3'), {
      series: ENGINES.map((e) => ({ name: e.label, color: e.color, marker: e.marker,
        pts: by(e.merge, (r) => r.merge_s).map(([t, v]) => [t, bM / v]) })),
      ideal: false, ref: { y: 1, label: 'IncrSerial' }, yLabel: 'phase 3 speedup',
    });
  }
  const load = BENCH.load.filter((r) => nameOf(r) === name && r.vocab_size === vocab);
  barChart($('c-load'), load);
}

function marker(kind, x, y, color, parent) {
  const style = `fill: var(${color}); stroke: var(--surface)`;
  if (kind === 'circle') return svg('circle', { cx: x, cy: y, r: 4.5, style, 'stroke-width': 2 }, parent);
  if (kind === 'square') return svg('rect', { x: x - 4.5, y: y - 4.5, width: 9, height: 9, rx: 1.5, style, 'stroke-width': 2 }, parent);
  const n = kind === 'triangle' ? 3 : 6, r = kind === 'triangle' ? 6 : 5.5;
  const pts = Array.from({ length: n }, (_, i) => {
    const a = -Math.PI / 2 + i * 2 * Math.PI / n;
    return `${(x + r * Math.cos(a)).toFixed(1)},${(y + r * Math.sin(a)).toFixed(1)}`;
  }).join(' ');
  return svg('polygon', { points: pts, style, 'stroke-width': 2 }, parent);
}

function legendFor(series, extra = []) {
  const lg = el('div', { class: 'legend' });
  for (const s of series) {
    const span = el('span'), sw = el('i', { class: 'line' });
    sw.style.background = `var(${s.color})`;
    span.append(sw, s.name);
    lg.append(span);
  }
  for (const [label, cls] of extra) {
    const span = el('span'), sw = el('i', { class: 'line' });
    sw.style.background = 'var(--muted)';
    sw.className = cls;
    span.append(sw, label);
    lg.append(span);
  }
  return lg;
}

function tableFor(headers, rows) {
  const d = el('details', { class: 'table-view' });
  d.append(el('summary', {}, 'Table'));
  const t = el('table'), tr = el('tr');
  for (const h of headers) tr.append(el('th', {}, h));
  t.append(tr);
  for (const r of rows) {
    const row = el('tr');
    for (const v of r) row.append(el('td', {}, v));
    t.append(row);
  }
  d.append(t);
  return d;
}

function lineChart(host, o) {
  host.replaceChildren();
  const series = o.series.filter((s) => s.pts.length);
  if (!series.length) { host.textContent = 'No data.'; return; }
  host.append(legendFor(series, o.ideal ? [['ideal (linear)', 'line']] : []));
  const wrap = el('div', { class: 'chart' });
  host.append(wrap);
  const xs = [...new Set(series.flatMap((s) => s.pts.map((p) => p[0])))].sort((a, b) => a - b);
  const maxT = Math.max(...xs);
  const dataMax = Math.max(...series.flatMap((s) => s.pts.map((p) => p[1])), o.ref ? o.ref.y : 0);
  const yMax = niceMax(dataMax * 1.12);
  const w = Math.max(260, host.clientWidth), h = 250, m = { l: 40, r: 50, t: 10, b: 34 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const X = (t) => m.l + (t - 1) / Math.max(1, maxT - 1) * pw, Y = (v) => m.t + ph - v / yMax * ph;
  const root = svg('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img',
    'aria-label': series.map((s) => `${s.name}: up to ${Math.max(...s.pts.map((p) => p[1])).toFixed(2)}×`).join('; ') }, wrap);
  if (maxT > P_CORES) {
    svg('rect', { x: X(P_CORES + 0.5), y: m.t, width: X(maxT) - X(P_CORES + 0.5) + 6, height: ph, class: 'region' }, root);
    svg('text', { x: X(P_CORES + 0.5) + 4, y: m.t + 12 }, root).textContent = 'E-cores join';
  }
  for (const v of ticks(yMax, 4)) {
    svg('line', { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: 'gridline' }, root);
    svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, root).textContent = `${v}×`;
  }
  for (const t of xs) svg('text', { x: X(t), y: m.t + ph + 16, 'text-anchor': 'middle' }, root).textContent = t;
  svg('text', { x: m.l + pw / 2, y: h - 3, 'text-anchor': 'middle' }, root).textContent = 'threads';
  svg('line', { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, class: 'axis-line' }, root);
  if (o.ideal) {
    const top = Math.min(maxT, yMax);
    svg('line', { x1: X(1), y1: Y(1), x2: X(top), y2: Y(top), class: 'ref' }, root);
  }
  if (o.ref) {
    svg('line', { x1: m.l, x2: m.l + pw, y1: Y(o.ref.y), y2: Y(o.ref.y), class: 'ref' }, root);
  }
  for (const s of series) {
    const d = s.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    svg('path', { d, fill: 'none', style: `stroke: var(${s.color})`, 'stroke-width': 2 }, root);
    for (const p of s.pts) marker(s.marker, X(p[0]), Y(p[1]), s.color, root);
  }
  // End labels, pushed apart so close values stay readable.
  const ends = series.map((s) => ({ v: s.pts[s.pts.length - 1][1], t: s.pts[s.pts.length - 1][0] }))
    .map((e) => ({ ...e, y: Y(e.v) })).sort((a, b) => b.y - a.y);
  for (let i = 1; i < ends.length; ++i) ends[i].y = Math.min(ends[i].y, ends[i - 1].y - 14);
  for (const e of ends) {
    svg('text', { x: X(e.t) + 9, y: e.y + 4, class: 'end-label' }, root).textContent = `${e.v.toFixed(2)}×`;
  }
  // Hover: a crosshair at the nearest thread count and every series' value there.
  const hover = svg('line', { y1: m.t, y2: m.t + ph, class: 'hover-line', visibility: 'hidden' }, root);
  const tip = el('div', { class: 'tip', hidden: '' });
  wrap.append(tip);
  const hit = svg('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent' }, root);
  hit.addEventListener('mousemove', (ev) => {
    const r = root.getBoundingClientRect(), x = ev.clientX - r.left;
    const t = xs.reduce((a, b) => (Math.abs(X(b) - x) < Math.abs(X(a) - x) ? b : a));
    hover.setAttribute('x1', X(t)); hover.setAttribute('x2', X(t)); hover.setAttribute('visibility', 'visible');
    tip.replaceChildren(el('b', {}, `${t} thread${t > 1 ? 's' : ''}`));
    for (const s of series) {
      const p = s.pts.find((q) => q[0] === t);
      if (p) tip.append(el('br'), `${s.name}: ${p[1].toFixed(2)}×`);
    }
    tip.hidden = false;
    tip.style.left = `${Math.min(X(t) + 12, w - tip.offsetWidth - 4)}px`;
    tip.style.top = `${m.t + 8}px`;
  });
  hit.addEventListener('mouseleave', () => { tip.hidden = true; hover.setAttribute('visibility', 'hidden'); });
  host.append(tableFor(['threads', ...series.map((s) => s.name)],
    xs.map((t) => [t, ...series.map((s) => { const p = s.pts.find((q) => q[0] === t); return p ? `${p[1].toFixed(2)}×` : '–'; })])));
}

function barChart(host, rows) {
  host.replaceChildren();
  const maps = [['contiguous', '--s1'], ['hashed', '--s2']];
  const threads = [...new Set(rows.map((r) => r.threads))].sort((a, b) => a - b);
  if (!threads.length) { host.textContent = 'No owner-load data for this corpus and vocab.'; return; }
  const val = (t, map) => {
    const v = rows.filter((r) => r.threads === t && r.owner_map === map).map((r) => r.busiest_mean);
    return v.length ? median(v) : null;
  };
  host.append(legendFor(maps.map(([name, color]) => ({ name, color })), [['perfect balance (1.0)', 'line']]));
  const wrap = el('div', { class: 'chart' });
  host.append(wrap);
  const yMax = niceMax(Math.max(...threads.flatMap((t) => maps.map(([m]) => val(t, m) || 0))) * 1.15);
  const w = Math.max(260, host.clientWidth), h = 250, m = { l: 40, r: 12, t: 10, b: 34 };
  const pw = w - m.l - m.r, ph = h - m.t - m.b, Y = (v) => m.t + ph - v / yMax * ph;
  const root = svg('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img',
    'aria-label': 'Busiest owner over mean owner load, contiguous vs hashed owners' }, wrap);
  for (const v of ticks(yMax, 4)) {
    svg('line', { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: 'gridline' }, root);
    svg('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, root).textContent = v;
  }
  const gw = pw / threads.length, bw = Math.min(46, (gw - 24) / 2);
  const tip = el('div', { class: 'tip', hidden: '' });
  wrap.append(tip);
  threads.forEach((t, i) => {
    const cx = m.l + gw * (i + 0.5);
    svg('text', { x: cx, y: m.t + ph + 16, 'text-anchor': 'middle' }, root).textContent = `${t} threads`;
    maps.forEach(([map, color], j) => {
      const v = val(t, map);
      if (v === null) return;
      const x = cx + (j ? 1 : -bw - 1), y = Y(v), base = Y(0), r = Math.min(4, bw / 2);
      const d = `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${base}Z`;
      const bar = svg('path', { d, style: `fill: var(${color})` }, root);
      svg('text', { x: x + bw / 2, y: y - 5, 'text-anchor': 'middle', class: 'end-label' }, root).textContent = v.toFixed(2);
      bar.addEventListener('mousemove', (ev) => {
        const rr = root.getBoundingClientRect();
        tip.replaceChildren(el('b', {}, `${t} threads, ${map} owners`), el('br'), `busiest ÷ mean: ${v.toFixed(3)}`);
        tip.hidden = false;
        tip.style.left = `${Math.min(ev.clientX - rr.left + 12, w - tip.offsetWidth - 4)}px`;
        tip.style.top = `${ev.clientY - rr.top + 12}px`;
      });
      bar.addEventListener('mouseleave', () => { tip.hidden = true; });
    });
  });
  svg('line', { x1: m.l, x2: m.l + pw, y1: Y(1), y2: Y(1), class: 'ref' }, root);
  svg('line', { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, class: 'axis-line' }, root);
  host.append(tableFor(['threads', ...maps.map(([m]) => m)],
    threads.map((t) => [t, ...maps.map(([m]) => { const v = val(t, m); return v === null ? '–' : v.toFixed(3); })])));
}

// ---------- Hugging Face check ----------
$('hf-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  if (!S.run) return;
  $('hf-btn').disabled = true;
  setStatus('hf-status', 'Encoding with tokenizers…');
  try {
    const r = await getJSON('/api/encode', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: S.run.id, text: $('hf-text').value }) });
    setStatus('hf-status', r.ok
      ? `✓ ${r.ids.length} tokens. Hugging Face gives the same ids as the Python reference encoder.`
      : '✗ Hugging Face and the reference encoder disagree.', !r.ok);
    const chips = $('hf-tokens');
    chips.replaceChildren();
    r.tokens.forEach((tok, i) => {
      const c = el('span', { class: 'chip' }, tok);
      c.append(el('small', {}, String(r.ids[i])));
      chips.append(c);
    });
  } catch (e) {
    setStatus('hf-status', e.message, true);
  } finally {
    $('hf-btn').disabled = false;
  }
});

// ---------- the work (B10) ----------
// The replay clock shown on the data. Phase 1: the corpus as its work units, coloured by where
// each was pre-tokenized. Phase 3: real words of this run (--trace-words) rewritten round by round,
// grouped by the thread that ran their owner. Word states come from replaying merges.txt on the
// words' bytes: that is exactly what training does to every word, one merge at a time.
const W = { units: [], words: null, rounds: [], note: '', shown: -1, cells: null };
const WALL_WORDS = 60;

// GPT-2's byte <-> printable character map (merges.txt is written in it).
const BYTE_CHAR = (() => {
  const bs = [];
  for (let b = 33; b <= 126; ++b) bs.push(b);
  for (let b = 161; b <= 172; ++b) bs.push(b);
  for (let b = 174; b <= 255; ++b) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; ++b) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const m = new Array(256);
  bs.forEach((b, i) => { m[b] = String.fromCodePoint(cs[i]); });
  return m;
})();
const CHAR_BYTE = new Map(BYTE_CHAR.map((c, b) => [c, b]));
const UTF8 = new TextDecoder('utf-8', { fatal: true });
// A token as readable text: its bytes decoded as UTF-8 (a leading space shown as ␣), or the GPT-2
// characters when the token ends inside a multi-byte character.
function tokenText(tok) {
  try {
    const s = UTF8.decode(Uint8Array.from([...tok].map((c) => CHAR_BYTE.get(c))));
    return s.replace(/^ /, '␣').replace(/\n/g, '⏎');
  } catch {
    return tok;
  }
}

async function loadWork(run) {
  W.units = [];
  W.words = null;
  W.rounds = [];
  W.shown = -1;
  for (const lane of S.lanes) {
    for (const e of lane.ev) {
      if (KINDS[e.k] === 'PRETOK_UNIT' && e.unit !== undefined) W.units.push({ ...e, tid: lane.tid });
    }
  }
  W.units.sort((a, b) => a.unit - b.unit);
  W.note = W.units.length ? unitSummary(W.units) : run.threads > 1
    ? 'This trace predates per-unit indices: train the run again to see its work units.'
    : 'One thread with the serial pre-tokenizer: the corpus is read in one pass, not traced per unit.';
  layoutUnits();
  renderUnitsPerThread();
  if (run.merge !== 'incr-batch' || !S.rounds.length) {
    W.wordsNote = 'The word view needs an incr-batch run: its rounds and owners are traced.';
  } else {
    try {
      const data = await getJSON(`/api/runs/${run.id}/words.json`);
      if (S.run !== run) return;  // another run was loaded meanwhile
      buildWords(data);
      W.wordsNote = '';
    } catch {
      W.wordsNote = 'This run has no words.json (trained before B10): train it again to see its words.';
    }
  }
  S.landed = -1;
  render();
}

// Replays merges.txt on the sampled words and records, per round, which words changed and on
// which thread (the thread that ran the word's owner that round; the master in a serial round).
function buildWords(data) {
  const words = data.words.map((d) => ({
    ...d, tok: d.hex.match(/../g).map((h) => BYTE_CHAR[parseInt(h, 16)]), hist: [] }));
  for (const w of words) w.tok0 = w.tok;
  const index = new Map();
  const pairs = (i, f) => {
    const t = words[i].tok;
    for (let j = 0; j + 1 < t.length; ++j) f(`${t[j]} ${t[j + 1]}`);
  };
  const add = (i) => pairs(i, (k) => { let s = index.get(k); if (!s) index.set(k, (s = new Set())); s.add(i); });
  const del = (i) => pairs(i, (k) => { const s = index.get(k); if (s) s.delete(i); });
  words.forEach((_, i) => add(i));
  S.merges.forEach((line, m) => {
    const hit = index.get(line);
    if (!hit || !hit.size) return;
    const sp = line.indexOf(' '), a = line.slice(0, sp), b = line.slice(sp + 1);
    for (const i of [...hit]) {
      const before = words[i].tok, out = [];
      for (let j = 0; j < before.length;) {
        if (j + 1 < before.length && before[j] === a && before[j + 1] === b) { out.push(a + b); j += 2; }
        else out.push(before[j++]);
      }
      del(i);
      words[i].tok = out;
      add(i);
      words[i].hist.push({ m, tok: out });
    }
  });

  // Owner -> thread per round, from the OWNER_MERGE events inside the round's rewrite window.
  const ownerEv = [];
  for (const lane of S.lanes) {
    for (const e of lane.ev) if (KINDS[e.k] === 'OWNER_MERGE') ownerEv.push({ ...e, tid: lane.tid });
  }
  ownerEv.sort((x, y) => x.t0 - y.t0);
  const master = S.lanes[0].ev.filter((e) => KINDS[e.k] === 'SERIAL_APPLY');
  const firstAtOrAfter = (arr, t) => {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const md = (lo + hi) >> 1; if (arr[md].t0 < t) lo = md + 1; else hi = md; }
    return lo;
  };
  const roundOf = (m) => {
    let lo = 0, hi = S.rounds.length;
    while (lo < hi) { const md = (lo + hi) >> 1; if (S.rounds[md].start <= m) lo = md + 1; else hi = md; }
    return lo - 1;
  };
  const rounds = new Map();
  words.forEach((w, i) => {
    let prev = w.tok0;
    for (const h of w.hist) {
      const r = roundOf(h.m);
      if (!rounds.has(r)) rounds.set(r, new Map());
      const ch = rounds.get(r);
      if (!ch.has(i)) ch.set(i, { i, before: prev });
      ch.get(i).after = h.tok;
      prev = h.tok;
    }
  });
  W.rounds = [...rounds.keys()].sort((x, y) => x - y).map((r) => {
    const [t0, t1] = S.rounds[r].rewrite;
    const byOwner = new Map();
    for (let j = firstAtOrAfter(ownerEv, t0); j < ownerEv.length && ownerEv[j].t0 <= t1; ++j) {
      byOwner.set(ownerEv[j].arg, ownerEv[j]);
    }
    const serial = byOwner.size === 0;
    const sa = serial ? master[Math.max(0, firstAtOrAfter(master, t0))] : null;
    const changes = [...rounds.get(r).values()].map((c) => {
      const ev = serial ? sa : byOwner.get(words[c.i].owner);
      return { ...c, tid: serial ? 0 : ev ? ev.tid : null, cpu: ev ? ev.cpu : undefined, e: ev ? ev.e : false };
    });
    return { r, serial, changes };
  });
  W.words = words;
  W.mergeThreads = data.merge_threads;
}

// The latest round with a word change that has landed by `landed` rounds.
function shownRound(landed) {
  let lo = 0, hi = W.rounds.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (W.rounds[m].r < landed) lo = m + 1; else hi = m; }
  return lo - 1;
}
function tokensAt(w, mergesDone) {
  let tok = w.tok0;
  for (const h of w.hist) { if (h.m >= mergesDone) break; tok = h.tok; }
  return tok;
}
// A word as token boxes. With `prev`, tokens that did not exist at the same place in prev (the
// ones this round created) are marked.
function chipsFor(tok, cls = '', prev = null) {
  const word = el('span', { class: `word ${cls}`.trim() });
  const old = new Set();
  if (prev) { let o = 0; for (const t of prev) { old.add(`${o}:${t}`); o += t.length; } }
  let o = 0;
  for (const t of tok) {
    word.append(el('span', { class: prev && !old.has(`${o}:${t}`) ? 'tok new' : 'tok' }, tokenText(t)));
    o += t.length;
  }
  return word;
}

function renderWords(landed) {
  const lanesHost = $('round-lanes'), wall = $('word-wall'), label = $('round-label');
  if (!W.words) {
    lanesHost.replaceChildren(el('p', { class: 'muted' }, W.wordsNote || 'Loading words…'));
    wall.replaceChildren();
    label.textContent = '';
    for (const id of ['round-prev', 'round-next']) $(id).disabled = true;
    return;
  }
  const mergesDone = landed ? S.rounds[landed - 1].start + S.rounds[landed - 1].k : 0;
  const si = shownRound(landed);
  W.shown = si;
  $('round-prev').disabled = si <= 0;
  $('round-next').disabled = si + 1 >= W.rounds.length;
  const shown = si >= 0 ? W.rounds[si] : null;
  const hitTid = new Map();

  // This round, one row per thread that rewrote a sampled word.
  lanesHost.replaceChildren();
  if (!shown) {
    lanesHost.append(el('p', { class: 'muted' }, 'No sampled word has been rewritten yet. Press Play, or step to the next round.'));
    label.textContent = '';
  } else {
    const R = S.rounds[shown.r];
    label.textContent = `Round ${fmt(shown.r + 1)} of ${fmt(S.rounds.length)} · ${R.k} merge${R.k > 1 ? 's' : ''} · `
      + (shown.serial ? 'small round: the master rewrote every word alone'
        : `parallel: owners rewritten on ${W.mergeThreads} threads`);
    const byTid = new Map();
    for (const c of shown.changes) {
      hitTid.set(c.i, c.tid);
      if (!byTid.has(c.tid)) byTid.set(c.tid, []);
      byTid.get(c.tid).push(c);
    }
    for (const tid of [...byTid.keys()].sort((a, b) => a - b)) {
      const list = byTid.get(tid), row = el('div', { class: 'round-lane' });
      const core = S.coreLabels && list[0].cpu !== undefined ? ` · ${list[0].e ? 'E' : 'P'}-core` : '';
      row.append(el('span', { class: 'lane-label' }, tid === null ? '?' : `t${tid}${core}`));
      const items = el('div', { class: 'lane-items' });
      for (const c of list.slice(0, 6)) {
        const item = el('span', { class: 'rewrite' });
        item.append(chipsFor(c.before), el('span', { class: 'arrow', 'aria-hidden': 'true' }, '→'), chipsFor(c.after, 'after', c.before));
        items.append(item);
      }
      if (list.length > 6) items.append(el('span', { class: 'muted small' }, `+${list.length - 6} more`));
      row.append(items);
      lanesHost.append(row);
    }
  }

  // The wall: the most frequent words as they stand now; this round's changes marked.
  wall.replaceChildren();
  for (let i = 0; i < Math.min(WALL_WORDS, W.words.length); ++i) {
    const w = W.words[i];
    const tok = tokensAt(w, mergesDone);
    const hit = hitTid.has(i);
    const box = chipsFor(tok, `${hit ? 'hit' : ''} ${tok.length === 1 ? 'whole' : ''}`);
    box.title = `${fmt(w.count)} occurrences · ${tok.length} token${tok.length > 1 ? 's' : ''}`
      + (w.owner >= 0 ? ` · owner ${w.owner}` : '');
    if (hit) box.append(el('small', { class: 'by' }, hitTid.get(i) === null ? '?' : `t${hitTid.get(i)}`));
    wall.append(box);
  }
  const whole = W.words.slice(0, WALL_WORDS).filter((w) => tokensAt(w, mergesDone).length === 1).length;
  $('wall-count').textContent = `${whole} of ${Math.min(WALL_WORDS, W.words.length)} are single tokens`;
}

function stepRound(dir) {
  if (!W.words) return;
  const target = W.shown + dir;
  if (target < 0 || target >= W.rounds.length) return;
  stop();
  const R = S.rounds[W.rounds[target].r];
  if (R.t < S.view[0] || R.t > S.view[1]) { $('view').value = 'all'; setView(); }
  S.t = R.t;
  render();
}
$('round-prev').addEventListener('click', () => stepRound(-1));
$('round-next').addEventListener('click', () => stepRound(1));

// What the unit events say, measured: sizes, times, and how little the start-of-unit core sample
// predicts (threads move between cores during a unit).
function unitSummary(units) {
  const ms = (xs) => (median(xs) / 1000).toFixed(1);
  const dur = units.map((u) => u.t1 - u.t0), mb = units.reduce((s, u) => s + u.arg, 0) / 1e6;
  let text = `${units.length} units, ${mb.toFixed(0)} MB; median ${ms(dur)} ms per unit.`;
  const P = units.filter((u) => !u.e).map((u) => u.t1 - u.t0), E = units.filter((u) => u.e).map((u) => u.t1 - u.t0);
  if (S.coreLabels && P.length && E.length) {
    text += ` ${Math.round(100 * E.length / units.length)}% of units started on an E-core, yet they took about as`
      + ` long (median ${ms(E)} ms) as those that started on a P-core (${ms(P)} ms): a unit is long enough for`
      + ' macOS to move its thread between core types, so the start sample says little about where the work ran.';
  }
  return text;
}

// Phase 1 grid: one cell per work unit, in corpus order.
const CELL = 12, CELL_GAP = 2;
function layoutUnits() {
  const canvas = $('units'), w = canvas.parentElement.clientWidth;
  const n = W.units.length;
  const cols = Math.max(1, Math.floor((w + CELL_GAP) / (CELL + CELL_GAP)));
  const rows = n ? Math.ceil(n / cols) : 0;
  const h = rows ? rows * (CELL + CELL_GAP) : 0;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  W.cols = cols;
  $('units-note').textContent = W.note;
  drawUnits();
}
function drawUnits() {
  const canvas = $('units');
  if (!canvas.height) return;
  const ctx = canvas.getContext('2d'), C = S.colors, dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
  const t = S.t;
  W.units.forEach((u, i) => {
    const x = (i % W.cols) * (CELL + CELL_GAP), y = Math.floor(i / W.cols) * (CELL + CELL_GAP);
    if (t >= u.t1) {
      ctx.fillStyle = C.groups[0];
      ctx.beginPath(); ctx.roundRect(x, y, CELL, CELL, 2); ctx.fill();
    } else if (t >= u.t0) {
      ctx.fillStyle = C.surface;
      ctx.beginPath(); ctx.roundRect(x + 0.75, y + 0.75, CELL - 1.5, CELL - 1.5, 2); ctx.fill();
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.lineWidth = 1;
    } else {
      ctx.fillStyle = C.grid;
      ctx.beginPath(); ctx.roundRect(x, y, CELL, CELL, 2); ctx.fill();
    }
  });
  renderUnitsPerThread();
}
// Units finished per thread by now: dynamic scheduling hands more units to faster cores.
function renderUnitsPerThread() {
  const host = $('units-per-thread');
  if (!W.units.length) { host.replaceChildren(); return; }
  const n = S.lanes.length, done = new Array(n).fill(0), all = new Array(n).fill(0);
  for (const u of W.units) {
    all[u.tid] += 1;
    if (S.t >= u.t1) done[u.tid] += 1;
  }
  const max = Math.max(...all, 1);
  if (!host.childElementCount || host.childElementCount !== n) {
    host.replaceChildren();
    for (let tid = 0; tid < n; ++tid) {
      const row = el('div', { class: 'pt-row' });
      row.append(el('span', { class: 'pt-label' }), el('span', { class: 'pt-bar' }), el('span', { class: 'pt-n tabular' }));
      host.append(row);
    }
  }
  [...host.children].forEach((row, tid) => {
    row.children[0].textContent = `t${tid}`;
    row.children[1].style.width = `${(100 * done[tid] / max).toFixed(1)}%`;
    row.children[2].textContent = `${done[tid]} / ${all[tid]}`;
  });
}
$('units').addEventListener('mousemove', (ev) => {
  const tip = $('units-tip');
  const r = ev.target.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const col = Math.floor(x / (CELL + CELL_GAP)), row = Math.floor(y / (CELL + CELL_GAP));
  const u = col < W.cols ? W.units[row * W.cols + col] : null;
  if (!u) { tip.hidden = true; return; }
  const core = S.coreLabels ? ` (${u.e ? 'E' : 'P'}-core)` : '';
  tip.replaceChildren(el('b', {}, `Work unit ${u.unit}`), ` · ${(u.arg / 1e6).toFixed(2)} MB`, el('br'),
    `thread ${u.tid} · cpu ${u.cpu}${core}`, el('br'), `${fmtTime(u.t1 - u.t0)}, from ${fmtTime(u.t0)}`);
  tip.hidden = false;
  tip.style.left = `${Math.min(x + 14, ev.target.clientWidth - tip.offsetWidth - 4)}px`;
  tip.style.top = `${y + 14}px`;
});
$('units').addEventListener('mouseleave', () => { $('units-tip').hidden = true; });

// ---------- the race (B9) ----------
// Two or three recorded runs replayed side by side on one clock. Lane A is the baseline.
const RACE = { lanes: [], runs: [], t: 0, end: 0, playing: false, last: 0, ids: [] };
const SLOTS = ['race-a', 'race-b', 'race-c'];

function runLabel(run) {
  const tm = run.timings.timings;
  const mt = run.merge_threads ? `, merges on ${run.merge_threads_used}` : '';
  return `${run.merge}, ${run.threads} thread${run.threads > 1 ? 's' : ''}${mt}` +
    (run.full_rounds ? ', every round batched' : '') + ` · ${tm.total.toFixed(2)} s`;
}
const sameCase = (a, b) => a && b && a.input === b.input && a.vocab_size === b.vocab_size;

async function refreshRuns() {
  RACE.runs = (await getJSON('/api/runs')).runs;
  fillSlot('race-a', RACE.runs, false);
  fillPartners();
}
function fillSlot(id, runs, optional) {
  const sel = $(id), keep = sel.value;
  sel.replaceChildren();
  if (optional) sel.append(el('option', { value: '' }, 'none'));
  for (const r of runs) {
    sel.append(el('option', { value: r.id }, `${runLabel(r)} · ${r.input} v${fmt(r.vocab_size)} · ${r.id}`));
  }
  if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;
}
// Lanes B and C only offer runs on lane A's corpus and vocabulary: a race needs the same work.
function fillPartners() {
  const a = RACE.runs.find((r) => r.id === $('race-a').value);
  const partners = RACE.runs.filter((r) => sameCase(a, r) && r.id !== (a && a.id));
  fillSlot('race-b', partners, false);
  fillSlot('race-c', partners, true);
}
$('race-a').addEventListener('change', fillPartners);

// Pick a default pair: the newest run and the newest incr-serial run on the same case.
function pickDefaultLanes() {
  const newest = RACE.runs[0];
  if (!newest) return false;
  const serial = RACE.runs.find((r) => r.merge === 'incr-serial' && r.threads === 1 && sameCase(r, newest));
  if (!serial || serial.id === newest.id) return false;
  $('race-a').value = serial.id;
  fillPartners();
  $('race-b').value = newest.id;
  return true;
}

async function loadRace() {
  stopRace();
  const ids = SLOTS.map((s) => $(s).value).filter(Boolean);
  if (ids.length < 2) { setStatus('race-status', 'Pick at least lanes A and B.', true); return; }
  setStatus('race-status', 'Loading traces…');
  try {
    RACE.lanes = await Promise.all(ids.map(async (id, i) => {
      const run = RACE.runs.find((r) => r.id === id);
      const [trace, mergesTxt] = await Promise.all([
        getJSON(`/api/runs/${id}/trace.json`), getText(`/api/runs/${id}/merges.txt`)]);
      const d = parseTrace(trace, run);
      const eng = ENGINES.find((e) => e.merge === run.merge) || ENGINES[0];
      return { slot: 'ABC'[i], run, d, total: mergesTxt.split('\n').slice(1).filter((l) => l.length).length,
        color: eng.color, traced3: d.lanes.some((l) => l.ev.some((e) => e.g >= 2)),
        traced1: d.lanes.some((l) => l.ev.some((e) => KINDS[e.k] === 'PRETOK_UNIT')) };
    }));
  } catch (e) {
    setStatus('race-status', e.message, true);
    return;
  }
  RACE.ids = ids;
  RACE.end = Math.max(...RACE.lanes.map((L) => L.d.end));
  buildRaceLanes();
  RACE.t = 0;
  for (const id of ['race-play', 'race-restart']) $(id).disabled = false;
  setStatus('race-status', `${RACE.lanes.length} lanes loaded: ${RACE.lanes[0].run.input}, vocab ${fmt(RACE.lanes[0].run.vocab_size)}.`);
  // Stills: #end for the finish, #at=0.3 for 30% of the way through the slowest lane.
  const at = [...HASH].find((h) => h.startsWith('at='));
  if (HASH.has('end')) RACE.t = RACE.end;
  else if (at) RACE.t = Math.min(1, Math.max(0, +at.slice(3) || 0)) * RACE.end;
  renderRace();
}

// One row per lane: label, progress bar, counters and the core map (P row, E row).
function buildRaceLanes() {
  const host = $('race-lanes');
  host.replaceChildren();
  const cpus = Math.max(...RACE.lanes.map((L) => L.d.cpus));
  const eCores = RACE.lanes[0].d.eCores;
  for (const L of RACE.lanes) {
    const box = el('div', { class: 'race-lane' });
    const head = el('div', { class: 'race-head' });
    head.append(el('b', { class: 'slot' }, L.slot), el('span', {}, runLabel(L.run)), el('span', { class: 'race-flag' }));
    const bar = el('div', { class: 'bar' });
    const fill = el('div', { class: 'bar-fill' });
    fill.style.background = `var(${L.color})`;
    bar.append(fill);
    const stats = el('div', { class: 'race-stats' });
    const merges = el('span', { class: 'tabular' }), phase = el('span'), clock = el('span', { class: 'tabular' });
    stats.append(merges, phase, clock);
    const cores = el('div', { class: 'cores', title: 'cores sampled busy at this instant' });
    const tiles = [];
    const row = (label, list, small) => {
      const r = el('div', { class: 'core-row' });
      r.append(el('span', { class: 'core-row-label' }, label));
      for (const c of list) {
        const t = el('span', { class: `core${small ? ' e' : ''}`, title: `cpu ${c}` });
        tiles[c] = t;
        r.append(t);
      }
      cores.append(r);
    };
    const all = Array.from({ length: cpus }, (_, c) => c);
    if (eCores >= 0) {
      row('P', all.filter((c) => c >= eCores), false);
      row('E', all.filter((c) => c < eCores), true);
    } else {
      row('cpu', all, false);
    }
    const note = el('p', { class: 'muted small race-note' });
    box.append(head, bar, stats, cores, note);
    host.append(box);
    L.ui = { fill, merges, phase, clock, tiles, flag: head.lastChild, note };
  }
}

// The latest event on one thread that was still running within [t - w, t], or null. With w = 0
// this is the event covering t. A frame of the replay covers a sliver of trace time and traced
// events have untraced gaps between them, so the core map looks back over a small window
// (RACE_WINDOW of the race) rather than flickering with each gap.
function recentEvent(lane, t, w) {
  const ev = lane.ev;
  let lo = 0, hi = ev.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ev[m].t0 <= t) lo = m + 1; else hi = m; }
  for (let j = lo - 1, n = 0; j >= 0 && n < 64; --j, ++n) {
    if (ev[j].t1 >= t - w) return ev[j];
  }
  return null;
}
// Where a thread was last seen before t (for the dashed "working, untraced" tile).
function lastCpuBefore(lane, t) {
  const ev = lane.ev;
  let lo = 0, hi = ev.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ev[m].t0 <= t) lo = m + 1; else hi = m; }
  return lo ? ev[lo - 1].cpu : undefined;
}
const RACE_WINDOW = 0.005;  // the core map's look-back, as a share of the race clock

function renderRace() {
  if (!RACE.lanes.length) return;
  const t = RACE.t;
  $('race-clock').textContent = `t = ${fmtTime(t)} of ${fmtTime(RACE.end)}`;
  const A = RACE.lanes[0];
  let allDone = true;
  for (const L of RACE.lanes) {
    const { d, ui } = L;
    const finished = t >= d.end - 1;
    allDone = allDone && finished;
    const phase = t < d.p1End ? 1 : t < d.p3[0] ? 2 : finished ? 4 : 3;
    // Merges landed: incr-batch rounds are traced; other engines spread evenly over the measured phase 3.
    let done = 0;
    if (d.rounds.length) {
      const landed = landedIn(d.rounds, t);
      done = landed ? d.rounds[landed - 1].start + d.rounds[landed - 1].k : 0;
    } else if (phase >= 3) {
      done = Math.min(L.total, Math.floor(L.total * (t - d.p3[0]) / Math.max(1, d.p3[1] - d.p3[0])));
    }
    if (finished) done = L.total;
    ui.fill.style.width = `${(100 * done / L.total).toFixed(2)}%`;
    ui.merges.textContent = `${fmt(done)} / ${fmt(L.total)} merges`;
    ui.phase.textContent = ['', 'phase 1: pre-tokenize', 'phase 2: word table + pair counts',
      'phase 3: merges', 'done'][phase];
    ui.clock.textContent = finished ? `${(d.end / 1e6).toFixed(2)} s` : fmtTime(t);
    // Core map: one tile per sampled cpu of a thread's current event, coloured by the work kind.
    for (const tile of ui.tiles) if (tile) { tile.style.background = ''; tile.classList.remove('untraced'); }
    let lit = 0, untraced = false;
    if (!finished) {
      const w = RACE.end * RACE_WINDOW;
      for (const lane of d.lanes) {
        const e = recentEvent(lane, t, w);
        if (!e || e.cpu === undefined || !ui.tiles[e.cpu]) continue;
        ui.tiles[e.cpu].style.background = `var(${GROUP_VARS[e.g]})`;
        ++lit;
      }
      // Nothing traced lately but the run is in a measured phase: one thread is working (the
      // serial pre-tokenizer and the serial merge engines are timed, not traced per event).
      if (!lit && phase >= 1 && phase <= 3) {
        const cpu = lastCpuBefore(d.lanes[0], t);
        const tile = ui.tiles[cpu] || ui.tiles.find((x) => x);
        if (tile) tile.classList.add('untraced');
        untraced = true;
      }
    }
    ui.note.textContent = untraced ? 'dashed tile: 1 thread working, timed but not traced per event' : '';
    // Flag: the first lane home, and every lane's time against lane A.
    if (finished) {
      const first = RACE.lanes.every((M) => M.d.end >= d.end);
      const vsA = L === A ? 'baseline' : A.d.end / d.end >= 1
        ? `${(A.d.end / d.end).toFixed(2)}× faster than A` : `${(d.end / A.d.end).toFixed(2)}× slower than A`;
      ui.flag.textContent = `${first ? '🏁 first · ' : ''}${vsA}`;
      ui.flag.classList.toggle('win', first);
    } else {
      ui.flag.textContent = '';
      ui.flag.classList.remove('win');
    }
  }
  // Exactness: every lane wrote the same merges.txt.
  const b = $('race-badge');
  const shas = new Set(RACE.lanes.map((L) => L.run.sha256));
  const state = !allDone ? 'pending' : shas.size === 1 ? 'ok' : 'bad';
  b.className = `badge ${state}`;
  b.textContent = { pending: '… same merges.txt? checked at the finish',
    ok: `✓ same merges.txt · sha256 ${[...shas][0].slice(0, 12)}…`, bad: '✗ merges.txt differ' }[state];
}
function landedIn(rounds, t) {
  let lo = 0, hi = rounds.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (rounds[m].t <= t) lo = m + 1; else hi = m; }
  return lo;
}

function raceSpeed() {
  const v = $('race-speed').value;
  return v === 'auto' ? RACE.end / 1e6 / 20 : +v;
}
function playRace() {
  if (!RACE.lanes.length) return;
  if (RACE.t >= RACE.end) RACE.t = 0;
  RACE.playing = true;
  RACE.last = performance.now();
  $('race-play').textContent = 'Pause';
  requestAnimationFrame(raceFrame);
}
function stopRace() {
  RACE.playing = false;
  $('race-play').textContent = 'Play';
}
function raceFrame(ts) {
  if (!RACE.playing) return;
  const dt = Math.min(0.1, (ts - RACE.last) / 1000);
  RACE.last = ts;
  RACE.t = Math.min(RACE.end, RACE.t + dt * raceSpeed() * 1e6);
  renderRace();
  if (RACE.t >= RACE.end) { stopRace(); return; }
  requestAnimationFrame(raceFrame);
}
$('race-play').addEventListener('click', () => (RACE.playing ? stopRace() : playRace()));
$('race-restart').addEventListener('click', () => { stopRace(); RACE.t = 0; renderRace(); });
$('race-load').addEventListener('click', loadRace);

// Train the race set with the Train form's corpus, vocab and threads: A = incr-serial on one
// thread (the project's speedup baseline), B = incr-batch on all threads with merges on the
// P-cores, C (optional) = the same without --merge-threads auto.
$('race-train').addEventListener('click', async () => {
  const input = $('input').value, vocab_size = +$('vocab').value, threads = +$('threads').value;
  const set = [{ merge: 'incr-serial', threads: 1 }, { merge: 'incr-batch', threads, merge_threads: 'auto' }];
  if ($('race-third').checked) set.push({ merge: 'incr-batch', threads });
  $('race-train').disabled = true;
  const ids = [];
  try {
    for (const [i, s] of set.entries()) {
      setStatus('race-status', `Training lane ${'ABC'[i]}: ${s.merge}, ${s.threads} thread${s.threads > 1 ? 's' : ''}…`);
      const run = await getJSON('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input, vocab_size, ...s }) });
      ids.push(run.id);
    }
    await refreshRuns();
    $('race-a').value = ids[0];
    fillPartners();
    $('race-b').value = ids[1];
    $('race-c').value = ids[2] || '';
    await loadRace();
  } catch (e) {
    setStatus('race-status', e.message, true);
  } finally {
    $('race-train').disabled = false;
  }
});

async function initRace() {
  try {
    await refreshRuns();
    if (pickDefaultLanes()) await loadRace();
    else setStatus('race-status', 'Train a race set, or pick two finished runs on the same corpus and vocab.');
  } catch (e) {
    setStatus('race-status', `Server error: ${e.message}`, true);
  }
  if (HASH.has('race')) $('race-card').scrollIntoView();
}

// ---------- theme + resize ----------
$('theme').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  readColors();
  draw();
  drawUnits();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); draw(); drawUnits(); });
let resizeTimer = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (S.lanes.length) { layoutLanes(); draw(); buildSpan(); layoutUnits(); S.landed = -1; render(); }
    if (BENCH.rows.length) drawBench();
  }, 150);
});

init();
