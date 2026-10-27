"use strict";
const $ = id => document.getElementById(id);
const svgNode = (tag, attrs, text = "") => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  node.textContent = text;
  return node;
};
function updateGate() {
  const state = GLUMath.scalar($("gate-kind").value, Number($("gate-a").value), Number($("gate-u").value));
  $("gate-a-out").value = state.a.toFixed(2);
  $("gate-u-out").value = state.up.toFixed(2);
  const svg = $("gate-svg"), scale = Math.max(4, Math.abs(state.up) * 4);
  const px = a => 50 + (a + 4) * 60, py = y => 170 - y / scale * 120;
  svg.replaceChildren();
  svg.append(svgNode("path", {d:"M50 50V290H530M50 170H530M290 50V290",stroke:"#3e5069",fill:"none"}));
  const curve = multiplier => Array.from({length:161}, (_,i) => {
    const a = -4 + i / 20;
    return `${i ? "L" : "M"}${px(a)} ${py(GLUMath.activation(state.kind, a) * multiplier)}`;
  }).join(" ");
  svg.append(svgNode("path", {d:curve(1),stroke:"#90b8f4","stroke-width":2,fill:"none"}));
  svg.append(svgNode("path", {d:curve(state.up),stroke:"#d9b779","stroke-width":2,fill:"none"}));
  svg.append(svgNode("circle", {cx:px(state.a),cy:py(state.product),r:5,fill:"#d9b779"}));
  for (const [x,y,text,color] of [
    [50,22,`${state.kind} · blue f(a) / gold f(a) × u`,"#bdcce2"],
    [50,315,"−4","#bdcce2"],[290,315,"0","#bdcce2"],[510,315,"a=4","#bdcce2"],
    [54,45,`y=+${scale}`,"#91a4bf"],[54,286,`y=−${scale}`,"#91a4bf"]
  ]) svg.append(svgNode("text", {x,y,fill:color,"font-size":12}, text));
  $("gate-readout").textContent = `a=${state.a.toFixed(2)} · u=${state.up.toFixed(2)} → f(a)=${state.gate.toFixed(5)} → product=${state.product.toFixed(5)}\n这里 a 与 u 是不同投影的标量。输出还要经过 W₂；曲线不代表模型质量或概率。`;
  window.gateState = state;
}
function updateBudget() {
  const num = id => Number($(id).value);
  const c = GLUMath.budget(num("budget-d"), num("budget-h0"), num("budget-m"), num("budget-bytes"), $("budget-mode").value, num("budget-align"));
  const count = x => x.toLocaleString("en-US"), size = x => x >= 1048576 ? `${(x / 1048576).toFixed(3)} MiB` : `${(x / 1024).toFixed(3)} KiB`;
  $("budget-readout").textContent = `单个 FFN · M=${c.M} · D=${c.D} · H₀=${c.H0} → H=${c.H} (理想 ${c.ideal.toFixed(3)})\n权重: ordinary ${count(c.baseParams)} / gated ${count(c.gateParams)} (${c.ratio.toFixed(4)}×)\n权重容量: ${size(c.baseWeightBytes)} / ${size(c.gateWeightBytes)}\n主要 GEMM FLOPs: ${count(c.baseFlops)} / ${count(c.gateFlops)}\n隐层驻留: ordinary ${size(c.baseHidden)}; gate+up ${size(c.gatePair)}; 再保留 product ${size(c.gatePairAndProduct)}\n逻辑隐层读写: ordinary unfused ${size(c.baseTraffic)} / producer-fused ${size(c.baseFusedTraffic)}\ngated fused-product ${size(c.gateFusedProductTraffic)} / separate-SiLU ${size(c.gateSeparateTraffic)}\n${c.H !== c.ideal ? "整数或硬件倍数 rounding 后，预算不再严格相同。" : "理想 2/3 宽度在本组维度中可精确实现。"} 不含 Attention、KV、input/output、workspace 或实际 cache；不预测延迟。`;
  const bars = $("budget-bars");
  bars.replaceChildren();
  const largest = Math.max(c.baseWeightBytes, c.gateWeightBytes, 1);
  for (const [index, label, value, color] of [
    [0,"Ordinary weights",c.baseWeightBytes,"#90b8f4"],
    [1,"Gated weights",c.gateWeightBytes,"#d9b779"]
  ]) {
    const y = 28 + index * 57;
    bars.append(svgNode("text", {x:0,y,fill:"#bdcce2","font-size":12}, `${label} · ${size(value)}`));
    bars.append(svgNode("rect", {x:0,y:y+10,width:540 * value / largest,height:14,rx:2,fill:color}));
  }
  window.budgetState = c;
}
document.querySelectorAll("#gate-widget input, #gate-widget select").forEach(el => el.addEventListener("input", updateGate));
document.querySelectorAll("#budget-widget select").forEach(el => el.addEventListener("input", updateBudget));
$("gate-negative").addEventListener("click", () => {
  $("gate-kind").value = "SwiGLU"; $("gate-a").value = -1; $("gate-u").value = 2; updateGate();
});
$("gate-flip").addEventListener("click", () => { $("gate-u").value = -Number($("gate-u").value); updateGate(); });
$("budget-reset").addEventListener("click", () => {
  for (const [id,value] of [["d",768],["h0",3072],["m",1],["bytes",2],["mode","matched"],["align",0]]) $("budget-"+id).value = value;
  updateBudget();
});
const dialog = $("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(button => button.addEventListener("click", () => {
  $("zoom-image").src = button.dataset.zoom;
  $("zoom-image").alt = button.querySelector("img").alt;
  dialog.showModal();
}));
$("close-dialog").addEventListener("click", () => dialog.close());
const compact = matchMedia("(max-width:1000px)");
$("contents").open = !compact.matches;
compact.addEventListener("change", event => { $("contents").open = !event.matches; });
const headings = [...document.querySelectorAll("article h2")];
let scrollPending = false;
function updateContents() {
  const current = headings.filter(h => h.getBoundingClientRect().top <= 160).at(-1) || headings[0];
  document.querySelectorAll(".toc a").forEach(a => a.classList.toggle("active", a.hash === "#" + current.id));
  scrollPending = false;
}
document.querySelectorAll(".toc a").forEach(a => a.addEventListener("click", event => {
  const heading = document.querySelector(a.hash);
  if (!heading) return;
  event.preventDefault();
  if (compact.matches) $("contents").open = false;
  history.pushState(null, "", a.hash);
  heading.scrollIntoView({block:"start", behavior:"instant"});
  updateContents();
}));
addEventListener("scroll", () => {
  if (!scrollPending) { scrollPending = true; requestAnimationFrame(updateContents); }
}, {passive:true});
updateGate(); updateBudget(); updateContents();
