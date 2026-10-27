import { cacheModel, sharedAttention } from "./model.mjs";

const element = id => document.getElementById(id);
let steps = 0;
function bytesLabel(value) {
  if (!value) return "0 B";
  const labels = ["B", "KiB", "MiB", "GiB", "TiB"];
  const power = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** power).toFixed(2)} ${labels[power]}`;
}
function inputNumber(id, lower, upper) {
  const value = Number(element(id).value);
  return Math.min(upper, Math.max(lower, Number.isFinite(value) ? Math.round(value) : lower));
}
function updateCache() {
  const parameters = {
    heads: 32, width: 128, kvHeads: Number(element("kv-heads").value),
    batch: inputNumber("batch-size", 1, 128), layers: inputNumber("layer-count", 1, 128),
    length: inputNumber("context-length", 0, 262144),
    bytes: Number(element("storage-bytes").value), window: Number(element("window-length").value), step: steps,
  };
  const model = cacheModel(parameters);
  const label = parameters.kvHeads === 1 ? "MQA" : parameters.kvHeads === 32 ? "MHA" : `GQA-${parameters.kvHeads}`;
  element("selected-label").textContent = label;
  element("mha-size").textContent = bytesLabel(model.mhaCapacity);
  element("selected-size").textContent = bytesLabel(model.capacity);
  element("mha-bar").style.width = model.activeLength ? "100%" : "0%";
  element("selected-bar").style.width = model.activeLength ? `${100 / model.ratio}%` : "0%";
  element("cache-readout").textContent =
    `${model.length} tokens (${model.activeLength} visible) · capacity / one-pass reads: ${bytesLabel(model.capacity)} · ${model.ratio}× smaller than MHA · QK/PV: ${(model.flops / 1e9).toFixed(3)} GFLOPs · ${model.intensity.toFixed(2)} FLOPs/byte · added tokens: ${steps}`;
  window.mqaState = { ...parameters, ...model, label };
}
for (const id of ["kv-heads", "batch-size", "layer-count", "context-length", "storage-bytes", "window-length"]) {
  element(id).addEventListener("input", () => { steps = 0; updateCache(); });
}
element("step-token").addEventListener("click", () => { steps += 1; updateCache(); });
element("reset-cache").addEventListener("click", () => {
  const defaults = { "kv-heads": "8", "batch-size": "1", "layer-count": "32", "context-length": "4096", "storage-bytes": "2", "window-length": "0" };
  for (const [id, value] of Object.entries(defaults)) element(id).value = value;
  steps = 0;
  updateCache();
});

function updateQueries() {
  const strength = Number(element("query-strength").value);
  const state = sharedAttention(strength);
  element("query-strength-value").textContent = strength.toFixed(1);
  element("query-rows").replaceChildren();
  for (let j = 0; j < 3; j += 1) {
    const row = document.createElement("tr");
    for (const text of [String(j), `(${state.keys[j].join(",")})`]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    for (let head = 0; head < 2; head += 1) {
      const cell = document.createElement("td");
      const fill = document.createElement("span");
      fill.className = `weight-fill head-${head}`;
      fill.style.width = `${state.weights[head][j] * 100}%`;
      cell.className = "weight-cell";
      cell.append(fill, document.createTextNode(state.weights[head][j].toFixed(3)));
      row.append(cell);
    }
    element("query-rows").append(row);
  }
  element("query-readout").textContent = state.outputs.map((o, i) =>
    `Q${i + 1}=(${state.queries[i].join(",")}) → o${i + 1}=(${o.map(v => v.toFixed(3)).join(",")})`).join(" · ");
  window.mqaQueryState = state;
}
element("query-strength").addEventListener("input", updateQueries);
updateCache();
updateQueries();

const modal = element("figure-modal");
let zoomTrigger = null;
for (const button of document.querySelectorAll("[data-zoom]")) {
  button.addEventListener("click", () => {
    zoomTrigger = button;
    const source = button.querySelector("img");
    modal.querySelector("img").src = source.src;
    modal.querySelector("img").alt = source.alt;
    modal.querySelector("a").href = source.src;
    modal.showModal();
  });
}
modal.querySelector("button").addEventListener("click", () => modal.close());
modal.addEventListener("close", () => zoomTrigger?.focus());
modal.addEventListener("click", event => { if (event.target === modal) modal.close(); });
for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(element(button.dataset.copy).textContent);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select code to copy";
    }
  });
}
