const grid = document.querySelector("#patch-grid");
const controlIds = ["grid-height", "grid-width", "frame-count", "frame-index", "direction", "start-position"];
let selectedPatch = { row: 1, column: 1 };

function coordinates(frame, row, column, height, width, start) {
  const base = start + frame * (height + width - 1);
  return [
    base + column + row,
    base + column - row + height - 1,
    base - column - row + height + width - 2,
    base - column + row + width - 1,
  ];
}

function updateGrid() {
  const height = Number(document.querySelector("#grid-height").value);
  const width = Number(document.querySelector("#grid-width").value);
  const frames = Number(document.querySelector("#frame-count").value);
  const frameInput = document.querySelector("#frame-index");
  frameInput.max = String(frames - 1);
  frameInput.value = String(Math.min(Number(frameInput.value), frames - 1));
  const frame = Number(frameInput.value);
  const start = Number(document.querySelector("#start-position").value);
  const direction = Number(document.querySelector("#direction").value);
  const stride = height + width - 1;
  selectedPatch.row = Math.min(selectedPatch.row, height - 1);
  selectedPatch.column = Math.min(selectedPatch.column, width - 1);
  grid.style.gridTemplateColumns = `repeat(${width}, minmax(0, 1fr))`;
  grid.replaceChildren();
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const values = coordinates(frame, row, column, height, width, start);
      const value = values[direction];
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = String(value);
      button.setAttribute("aria-label", `Patch w=${column}, h=${row}, direction ${direction + 1}, position ${value}`);
      button.setAttribute("aria-pressed", String(row === selectedPatch.row && column === selectedPatch.column));
      button.classList.toggle("selected", row === selectedPatch.row && column === selectedPatch.column);
      const shade = stride === 1 ? 0.5 : (value - start - frame * stride) / (stride - 1);
      button.style.backgroundColor = `hsl(216 40% ${20 + shade * 28}%)`;
      button.addEventListener("click", () => {
        selectedPatch = { row, column };
        updateGrid();
        const targetIndex = row * width + column;
        grid.children[targetIndex].focus({ preventScroll: true });
      });
      grid.append(button);
    }
  }
  const values = coordinates(frame, selectedPatch.row, selectedPatch.column, height, width, start);
  document.querySelector("#patch-values").textContent = `w=${selectedPatch.column}, h=${selectedPatch.row}, t=${frame}: [${values.join(", ")}]`;
  document.querySelector("#span-values").textContent = `frame: ${start + frame * stride}–${start + (frame + 1) * stride - 1}; next text: ${start + frames * stride}`;
  document.querySelector("#token-values").textContent = `${frames * height * width} tokens; ${frames * stride} position IDs; span/token=${(stride / (height * width)).toFixed(3)}`;
  document.querySelector("#center-values").textContent = `center=${start + frame * stride + (height + width - 2) / 2}; all 4 directions equal`;
  document.querySelector("#frame-label").textContent = `Frame ${frame}`;
  document.querySelector("#grid-legend").textContent = `Direction ${direction + 1} · darker = smaller position · lighter = larger position · not attention`;
  window.vropeState = { height, width, frames, frame, start, direction, stride, values, selectedPatch, tokens: frames * height * width, nextText: start + frames * stride };
}

controlIds.forEach((identifier) => {
  const control = document.getElementById(identifier);
  control.addEventListener("input", updateGrid);
});

function updatePhase() {
  const frequency = Number(document.querySelector("#phase-frequency").value);
  const distance = Number(document.querySelector("#phase-distance").value);
  const plot = document.querySelector("#phase-path");
  const marker = document.querySelector("#phase-marker");
  const width = 640;
  const height = 160;
  const maximum = 40;
  const points = [];
  for (let step = 0; step <= 320; step += 1) {
    const delta = step / 320 * maximum;
    points.push(`${step === 0 ? "M" : "L"}${(20 + delta / maximum * (width - 40)).toFixed(2)},${(height / 2 - Math.cos(delta * frequency) * 55).toFixed(2)}`);
  }
  plot.setAttribute("d", points.join(" "));
  marker.setAttribute("cx", String(20 + distance / maximum * (width - 40)));
  marker.setAttribute("cy", String(height / 2 - Math.cos(distance * frequency) * 55));
  document.querySelector("#phase-readout").textContent = `θ=${frequency.toFixed(2)}; Δp=${distance}; cos(Δpθ)=${Math.cos(distance * frequency).toFixed(4)}; period=${(2 * Math.PI / frequency).toFixed(2)}`;
}

["phase-frequency", "phase-distance"].forEach((identifier) => document.getElementById(identifier).addEventListener("input", updatePhase));
updateGrid();
updatePhase();

const modal = document.querySelector("#figure-modal");
let lastFigureButton = null;
document.querySelectorAll("[data-figure-src]").forEach((button) => {
  button.addEventListener("click", () => {
    lastFigureButton = button;
    const image = modal.querySelector("img");
    image.src = button.dataset.figureSrc;
    image.alt = button.querySelector("img").alt;
    modal.querySelector("p").textContent = image.alt;
    modal.querySelector("a").href = image.src;
    modal.showModal();
  });
});
modal.querySelector("button").addEventListener("click", () => modal.close());
modal.addEventListener("close", () => lastFigureButton?.focus());
modal.addEventListener("click", (event) => {
  if (event.target === modal) modal.close();
});

const toc = document.querySelector("#contents");
const mobileLayout = matchMedia("(max-width: 960px)");
function updateToc() { toc.open = !mobileLayout.matches; }
mobileLayout.addEventListener("change", updateToc);
updateToc();
const sectionLinks = new Map([...document.querySelectorAll(".toc a")].map((link) => [link.hash.slice(1), link]));
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      sectionLinks.forEach((link) => {
        link.classList.remove("active");
        link.removeAttribute("aria-current");
      });
      const activeLink = sectionLinks.get(entry.target.id);
      if (activeLink) {
        activeLink.classList.add("active");
        activeLink.setAttribute("aria-current", "location");
      }
    }
  }
}, { rootMargin: "-5% 0px -70% 0px" });
document.querySelectorAll(".article h2[id]").forEach((heading) => observer.observe(heading));
window.vropeCoordinates = coordinates;
