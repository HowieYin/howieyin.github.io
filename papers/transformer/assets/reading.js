import { attention, kvBytes, blockFlops } from "./model.mjs";

const el = id => document.getElementById(id);
const number = id => Number(el(id).value);
const mib = value => (value / 1024 ** 2).toFixed(2);
let selectedQuery = 0;

function updateAttention() {
  const count = number("token-count"), dim = number("head-dim"), gain = number("logit-gain");
  const causal = el("causal-mask").checked, scaled = el("scaled-attention").checked;
  selectedQuery = Math.min(selectedQuery, count - 1);
  const Q = Array.from({ length: count }, (_, i) =>
    Array.from({ length: dim }, (_, d) => (d % count === i ? gain : 0)));
  const V = Array.from({ length: count }, (_, i) => i % 2 ? [0, 20] : [10, 0]);
  const result = attention(Q, Q, V, { causal, scaled });
  el("gain-output").value = gain.toFixed(1);
  const grid = el("attention-grid");
  grid.style.setProperty("--tokens", count);
  grid.replaceChildren();
  const corner = document.createElement("span");
  corner.textContent = "Q ↓ / K →";
  grid.append(corner);
  for (let j = 0; j < count; j++) {
    const label = document.createElement("span");
    label.textContent = `k${j}`;
    grid.append(label);
  }
  for (let i = 0; i < count; i++) {
    const row = document.createElement("button");
    row.type = "button";
    row.textContent = `q${i}`;
    row.className = "query-button";
    row.setAttribute("aria-pressed", i === selectedQuery);
    row.setAttribute("aria-label", `读取 Query ${i} 的输出`);
    row.addEventListener("click", () => { selectedQuery = i; updateAttention(); });
    grid.append(row);
    result.weights[i].forEach((weight, j) => {
      const cell = document.createElement("span");
      const masked = causal && j > i;
      cell.textContent = masked ? "×" : weight.toFixed(3);
      cell.className = masked ? "weight masked" : "weight";
      cell.style.backgroundColor = masked ? "#101722" : `rgba(107,164,234,${0.07 + weight * 0.75})`;
      cell.setAttribute("aria-label", `q${i} → k${j}: ${masked ? "masked, zero" : weight.toFixed(6)}`);
      grid.append(cell);
    });
  }
  const weights = result.weights[selectedQuery];
  const entropy = -weights.reduce((sum, p) => sum + (p ? p * Math.log(p) : 0), 0);
  el("attention-readout").textContent =
    `q${selectedQuery} · weights = [${weights.map(v => v.toFixed(4)).join(", ")}]\n` +
    `output = [${result.output[selectedQuery].map(v => v.toFixed(4)).join(", ")}] · row sum = ${weights.reduce((a, b) => a + b, 0).toFixed(6)}\n` +
    `entropy = ${entropy.toFixed(4)} nats · divisor = ${scaled ? Math.sqrt(dim).toFixed(4) : "1"} · ${causal ? "causal j≤i" : "bidirectional"}`;
  window.attentionState = { count, dim, gain, causal, scaled, selectedQuery, ...result };
}
["token-count", "head-dim", "logit-gain", "causal-mask", "scaled-attention"].forEach(id =>
  el(id).addEventListener("input", updateAttention));
el("reset-attention").addEventListener("click", () => {
  el("token-count").value = 2; el("head-dim").value = 2; el("logit-gain").value = 1;
  el("causal-mask").checked = false; el("scaled-attention").checked = true;
  selectedQuery = 0; updateAttention();
});

function updateCache() {
  const tokens = number("cache-tokens"), layers = number("decoder-layers"), heads = number("kv-heads");
  const bytes = number("cache-bytes"), batch = number("cache-batch"), source = number("source-tokens");
  const parameters = { layers, batch, tokens, heads, dim: 64, bytes };
  const self = kvBytes(parameters), cross = kvBytes({ ...parameters, tokens: source });
  // Base query heads stay eight when the hypothetical KV sharing changes.
  const matrix = batch * 8 * tokens ** 2 * bytes;
  const cost = blockFlops(tokens, 512, 2048), total = cost.projection + cost.mixing + cost.ffn;
  ["cache-tokens", "decoder-layers", "source-tokens", "cache-batch"].forEach(id =>
    el(`${id}-output`).value = String(number(id)));
  el("cache-readout").textContent =
    `Self KV = ${mib(self)} MiB · Cross KV = ${mib(cross)} MiB · Total KV = ${mib(self + cross)} MiB\n` +
    `Materialized A, one self-attention layer = ${mib(matrix)} MiB (8 query heads)\n` +
    `Encoder-style block, B=1, n=${tokens}: ${(total / 1e9).toFixed(3)} GFLOPs`;
  const rows = [["Q/K/V/O projections", cost.projection, "#91baf1"], ["QKᵀ + AV", cost.mixing, "#8ed0bf"], ["FFN", cost.ffn, "#d2b68b"]];
  const bars = el("flop-bars");
  bars.replaceChildren(...rows.map(([name, value, color]) => {
    const row = document.createElement("div");
    row.className = "flop-row";
    const label = document.createElement("span");
    label.textContent = `${name} · ${(value / 1e9).toFixed(3)}G · ${(value / total * 100).toFixed(1)}%`;
    const track = document.createElement("div");
    track.className = "bar-track";
    const bar = document.createElement("div");
    bar.style.width = `${value / total * 100}%`; bar.style.backgroundColor = color;
    track.append(bar); row.append(label, track); return row;
  }));
  window.cacheState = { ...parameters, source, self, cross, matrix, cost, total };
}
["cache-tokens", "decoder-layers", "source-tokens", "kv-heads", "cache-bytes", "cache-batch"].forEach(id =>
  el(id).addEventListener("input", updateCache));

let playing = false, timer;
function stop() {
  playing = false; clearInterval(timer); el("play-decode").textContent = "Play growth";
}
function step() {
  const input = el("cache-tokens");
  const next = Math.min(Number(input.max), Number(input.value) + 256);
  input.value = next; updateCache();
  if (next === Number(input.max)) stop();
}
el("step-decode").addEventListener("click", step);
el("play-decode").addEventListener("click", () => {
  if (playing) return stop();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return step();
  playing = true; el("play-decode").textContent = "Pause growth"; timer = setInterval(step, 700);
});
el("reset-cache").addEventListener("click", () => {
  stop();
  for (const [id, value] of Object.entries({ "cache-tokens": 4096, "decoder-layers": 6, "source-tokens": 512, "kv-heads": 8, "cache-bytes": 2, "cache-batch": 1 })) el(id).value = value;
  updateCache();
});
new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }).observe(el("cache-widget"));
document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", stop);

const modal = el("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
  const original = button.querySelector("img");
  el("zoom-image").src = original.src; el("zoom-image").alt = original.alt;
  el("full-figure").href = original.src; modal.showModal();
}));
el("close-dialog").addEventListener("click", () => modal.close());
modal.addEventListener("click", event => { if (event.target === modal) modal.close(); });
if (matchMedia("(max-width:1000px)").matches) el("contents").open = false;
const headings = [...document.querySelectorAll("article h2")];
let scrollFrame;
function updateContents() {
  const current = headings.filter(heading => heading.getBoundingClientRect().top <= 100).at(-1) || headings[0];
  document.querySelectorAll(".toc a").forEach(link =>
    link.classList.toggle("active", link.hash === `#${current.id}`));
}
addEventListener("scroll", () => {
  cancelAnimationFrame(scrollFrame);
  scrollFrame = requestAnimationFrame(updateContents);
}, { passive: true });
updateContents();
document.querySelectorAll("pre").forEach(pre => {
  const button = document.createElement("button");
  button.className = "copy-code"; button.textContent = "Copy"; button.type = "button";
  button.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(pre.querySelector("code").textContent); button.textContent = "Copied"; }
    catch { button.textContent = "Select to copy"; }
  });
  pre.prepend(button);
});
updateAttention(); updateCache();
