const select = identifier => document.getElementById(identifier);
const namespace = "http://www.w3.org/2000/svg";
const vectorRotate = (vector, angle) => [vector[0] * Math.cos(angle) - vector[1] * Math.sin(angle), vector[0] * Math.sin(angle) + vector[1] * Math.cos(angle)];
const vectorDot = (first, second) => first[0] * second[0] + first[1] * second[1];
const svgElement = (name, attributes, text = "") => {
  const element = document.createElementNS(namespace, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  element.textContent = text;
  return element;
};
const drawPlane = (svg, center, vectors, title) => {
  svg.append(svgElement("circle", {cx: center, cy: 150, r: 100, fill: "none", stroke: "#41506a"}));
  svg.append(svgElement("path", {d: `M${center - 110} 150h220M${center} 40v220`, fill: "none", stroke: "#334259"}));
  svg.append(svgElement("text", {x: center, y: 24, fill: "#bdcde3", "text-anchor": "middle", "font-size": 13}, title));
  vectors.forEach(({vector, color, dashed}) => {
    const horizontal = center + vector[0] * 95;
    const vertical = 150 - vector[1] * 95;
    svg.append(svgElement("line", {x1: center, y1: 150, x2: horizontal, y2: vertical, stroke: color, "stroke-width": dashed ? 1.5 : 3, "stroke-dasharray": dashed ? "4 4" : ""}));
    svg.append(svgElement("circle", {cx: horizontal, cy: vertical, r: dashed ? 2 : 5, fill: color}));
  });
};
let playing = false;
let lastShift = 0;
let animationFrame;
const stopAnimation = () => {
  playing = false;
  cancelAnimationFrame(animationFrame);
  select("animate-shift").textContent = "Play shift";
};
const updateRotation = () => {
  const query = Number(select("query-position").value);
  const key = Number(select("key-position").value);
  const offset = Number(select("shared-offset").value);
  const frequency = Number(select("frequency").value);
  const rotatedQuery = vectorRotate([1, 0], (query + offset) * frequency);
  const rotatedKey = vectorRotate([0, 1], (key + offset) * frequency);
  const direct = vectorDot(rotatedQuery, rotatedKey);
  const relative = vectorDot([1, 0], vectorRotate([0, 1], (key - query) * frequency));
  select("query-output").value = String(query);
  select("key-output").value = String(key);
  select("offset-output").value = String(offset);
  select("frequency-output").value = frequency.toFixed(6);
  const svg = select("rotation-svg");
  svg.replaceChildren();
  drawPlane(svg, 150, [{vector: [1, 0], color: "#90b8f4", dashed: true}, {vector: [0, 1], color: "#83cdbd", dashed: true}, {vector: rotatedQuery, color: "#90b8f4"}, {vector: rotatedKey, color: "#83cdbd"}], `Absolute: m=${query + offset}, n=${key + offset}`);
  drawPlane(svg, 450, [{vector: [1, 0], color: "#90b8f4"}, {vector: vectorRotate([0, 1], (key - query) * frequency), color: "#83cdbd"}], `Relative: Δ=n−m=${key - query}`);
  select("rotation-readout").textContent = `Q′=[${rotatedQuery.map(value => value.toFixed(6)).join(", ")}] · K′=[${rotatedKey.map(value => value.toFixed(6)).join(", ")}]\nDirect Q′·K′=${direct.toFixed(9)} · Relative qᵀRΔk=${relative.toFixed(9)} · Error=${Math.abs(direct - relative).toExponential(2)} · norms=${Math.hypot(...rotatedQuery).toFixed(6)}, ${Math.hypot(...rotatedKey).toFixed(6)}`;
  window.ropeState = {query, key, offset, frequency, direct, relative, error: Math.abs(direct - relative)};
};
["query-position", "key-position", "shared-offset", "frequency"].forEach(identifier => select(identifier).addEventListener("input", updateRotation));
select("shift-step").addEventListener("click", () => {
  select("shared-offset").value = (Number(select("shared-offset").value) + 1) % 21;
  updateRotation();
});
select("animate-shift").addEventListener("click", () => {
  if (playing) return stopAnimation();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return select("shift-step").click();
  playing = true;
  lastShift = 0;
  select("animate-shift").textContent = "Pause shift";
  const frame = timestamp => {
    if (!playing) return;
    if (timestamp - lastShift > 600) {
      select("shift-step").click();
      lastShift = timestamp;
    }
    animationFrame = requestAnimationFrame(frame);
  };
  animationFrame = requestAnimationFrame(frame);
});
select("reset-rotation").addEventListener("click", () => {
  stopAnimation();
  select("query-position").value = 2;
  select("key-position").value = 5;
  select("shared-offset").value = 0;
  select("frequency").value = 0.5235987756;
  updateRotation();
});
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) stopAnimation();
}).observe(select("rotation-svg"));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopAnimation();
});
const updateSpectrum = () => {
  const dimension = Number(select("rotary-dimension").value);
  const base = Number(select("rotary-base").value);
  const distance = Number(select("distance").value);
  const frequencies = Array.from({length: dimension / 2}, (_, index) => base ** (-2 * index / dimension));
  const average = value => frequencies.reduce((sum, frequency) => sum + Math.cos(value * frequency), 0) / frequencies.length;
  const svg = select("spectrum-svg");
  svg.replaceChildren();
  svg.append(svgElement("path", {d: "M45 25v205h525M45 125h525", fill: "none", stroke: "#53627a"}));
  [0, 64, 128, 192, 256].forEach(value => svg.append(svgElement("text", {x: 45 + value / 256 * 525, y: 254, fill: "#bdcde3", "text-anchor": "middle", "font-size": 12}, value)));
  [1, 0, -1].forEach(value => svg.append(svgElement("text", {x: 30, y: 128 - value * 100, fill: "#bdcde3", "text-anchor": "end", "font-size": 12}, value)));
  [[average, "#90b8f4"], [Math.cos, "#83cdbd"]].forEach(([calculate, color]) => {
    const points = Array.from({length: 513}, (_, index) => `${45 + index / 512 * 525},${125 - calculate(index / 2) * 100}`).join(" ");
    svg.append(svgElement("polyline", {points, fill: "none", stroke: color, "stroke-width": 1.6}));
  });
  const horizontal = 45 + distance / 256 * 525;
  svg.append(svgElement("path", {d: `M${horizontal} 25v205`, fill: "none", stroke: "#e9bf78", "stroke-dasharray": "4 4"}));
  select("distance-output").value = String(distance);
  select("spectrum-readout").textContent = `d=${dimension} · base=${base} · Δ=${distance} · Average cosine=${average(distance).toFixed(6)} · Single-frequency cos(Δ)=${Math.cos(distance).toFixed(6)} · Not an attention probability`;
  select("frequency-table").replaceChildren(...frequencies.map((frequency, index) => {
    const row = document.createElement("tr");
    [index, frequency.toPrecision(6), (2 * Math.PI / frequency).toFixed(2), Math.cos(distance * frequency).toFixed(6)].forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    return row;
  }));
  window.spectrumState = {dimension, base, distance, average: average(distance), single: Math.cos(distance)};
};
["rotary-dimension", "rotary-base", "distance"].forEach(identifier => select(identifier).addEventListener("input", updateSpectrum));
const dialog = select("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
  select("zoom-image").src = button.dataset.zoom;
  const image = button.querySelector("img");
  select("zoom-image").width = image.width > 0 ? image.naturalWidth : Number(image.getAttribute("width"));
  dialog.showModal();
}));
select("close-dialog").addEventListener("click", () => dialog.close());
if (matchMedia("(max-width:1000px)").matches) select("contents").open = false;
const headings = document.querySelectorAll("article h2");
new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll(".toc a").forEach(link => link.classList.toggle("active", link.hash === `#${entry.target.id}`));
    }
  });
}, {rootMargin: "0px 0px -65% 0px"}).observe(headings[0]);
headings.forEach(heading => {
  if (heading !== headings[0]) new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) document.querySelectorAll(".toc a").forEach(link => link.classList.toggle("active", link.hash === `#${heading.id}`));
  }, {rootMargin: "0px 0px -65% 0px"}).observe(heading);
});
updateRotation();
updateSpectrum();
