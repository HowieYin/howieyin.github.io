"use strict";
const $ = id => document.getElementById(id);
const ns = "http://www.w3.org/2000/svg";
const svgNode = (tag, attrs, text = "") => {
  const node = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  node.textContent = text;
  return node;
};
function updatePattern() {
  const n = Number($("pattern-n").value), radius = Number($("pattern-r").value);
  const mode = $("pattern-mode").value, d = mode === "dilated" ? Number($("pattern-d").value) : 1;
  const g = mode === "global" ? Number($("pattern-g").value) : 0;
  $("pattern-source").max = n - 1;
  const source = Math.min(Number($("pattern-source").value), n - 1);
  $("pattern-source").value = source;
  const layers = Number($("pattern-layers").value);
  const globals = Array.from({length: g}, (_, i) => i);
  const matrix = mode === "full" ? Array.from({length: n}, () => Array(n).fill(true)) : LongformerMath.pattern(n, radius, d, globals, false);
  const reach = LongformerMath.reachable(matrix, source, layers);
  const svg = $("attention-svg"), step = 480 / n;
  svg.replaceChildren();
  svg.append(svgNode("text", {x: 290, y: 18, fill: "#c4d4ed", "text-anchor": "middle", "font-size": 13}, "Key / Value position →"));
  svg.append(svgNode("text", {x: 18, y: 300, fill: "#c4d4ed", transform: "rotate(-90 18 300)", "text-anchor": "middle", "font-size": 13}, "Query position →"));
  matrix.forEach((row, q) => row.forEach((allowed, k) => {
    if (allowed) svg.append(svgNode("rect", {x: 50 + k * step, y: 35 + q * step, width: Math.max(.5, step - .5), height: Math.max(.5, step - .5), fill: globals.includes(q) || globals.includes(k) ? "#e9bf78" : "#789dcb"}));
  }));
  svg.append(svgNode("rect", {x: 49, y: 34 + source * step, width: 482, height: step + 1, fill: "none", stroke: "#eff6ff", "stroke-width": 2}));
  svg.append(svgNode("text", {x: 50, y: 543, fill: "#c4d4ed", "font-size": 13}, `Source ${source} → reachable after ${layers} layer(s)`));
  for (let k = 0; k < n; k++) svg.append(svgNode("rect", {x: 50 + k * step, y: 558, width: step - .5, height: 17, fill: k === source ? "#e9bf78" : reach.includes(k) ? "#8fc7b7" : "#293447"}));
  [["pattern-n", n], ["pattern-r", radius], ["pattern-d", d], ["pattern-g", g], ["pattern-source", source], ["pattern-layers", layers]].forEach(([id, value]) => $(id + "-out").value = value);
  const edges = matrix.flat().filter(Boolean).length;
  $("pattern-readout").textContent = `n=${n} · w=2r=${2 * radius} · d=${d} · g=${g} · connections=${edges}/${n * n} (${(edges / n ** 2 * 100).toFixed(2)}%) · reached=${reach.length}/${n} · positions=[${reach.join(", ")}]`;
  window.patternState = {n, radius, d, g, source, layers, edges, reach, mode};
}
function updateCost() {
  const read = id => Number($(id).value);
  const n = read("cost-n"), w = read("cost-w"), g = read("cost-g"), m = read("cost-m");
  const D = read("cost-D"), layers = read("cost-layers"), heads = read("cost-heads"), bytes = read("cost-bytes");
  const c = LongformerMath.costs(n, w, g, m, D, layers, heads, bytes);
  const mib = value => (value / 1048576).toFixed(2) + " MiB";
  $("cost-readout").textContent = `Encoder single-head edges: ${c.sparseEntries.toLocaleString()} / dense ${c.denseEntries.toLocaleString()} (${(100 * c.sparseEntries / c.denseEntries).toFixed(2)}%)\nEncoder single-layer score storage ×${heads} heads: ${mib(c.sparseScoreBytes)} / dense ${mib(c.denseScoreBytes)}\nDecoder teacher-forced score: self ${c.decoderEntries.toLocaleString()} · cross ${c.crossEntries.toLocaleString()} (single head/layer)\nAll ${layers} decoder layers KV (batch=1): cross ${mib(c.crossKV)} · self ${mib(c.selfKV)} · total ${mib(c.crossKV + c.selfKV)}`;
  window.costState = {n, w, g, m, D, layers, heads, bytes, ...c};
}
document.querySelectorAll("#attention-widget input, #attention-widget select").forEach(el => el.addEventListener("input", updatePattern));
document.querySelectorAll("#cost-widget input, #cost-widget select").forEach(el => el.addEventListener("input", updateCost));
$("relay-preset").addEventListener("click", () => {
  $("pattern-mode").value = "global";
  $("pattern-n").value = 16;
  $("pattern-r").value = 1;
  $("pattern-g").value = 1;
  $("pattern-source").value = 15;
  $("pattern-layers").value = 0;
  updatePattern();
});
$("relay-step").addEventListener("click", () => {
  $("pattern-layers").value = (Number($("pattern-layers").value) + 1) % 9;
  updatePattern();
});
const dialog = $("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
  const source = button.querySelector("img");
  $("zoom-image").src = button.dataset.zoom;
  $("zoom-image").alt = source.alt;
  dialog.showModal();
}));
$("close-dialog").addEventListener("click", () => dialog.close());
const compactLayout = matchMedia("(max-width:1000px)");
$("contents").open = !compactLayout.matches;
compactLayout.addEventListener("change", event => { $("contents").open = !event.matches; });
const headings = [...document.querySelectorAll("article h2")];
let scrollPending = false;
const updateContents = () => {
  const current = headings.filter(h => h.getBoundingClientRect().top <= 160).at(-1) || headings[0];
  document.querySelectorAll(".toc a").forEach(a => a.classList.toggle("active", a.hash === "#" + current.id));
  scrollPending = false;
};
addEventListener("scroll", () => {
  if (!scrollPending) { scrollPending = true; requestAnimationFrame(updateContents); }
}, {passive:true});
updateContents();
updatePattern();
updateCost();
