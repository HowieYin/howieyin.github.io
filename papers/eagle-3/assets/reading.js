(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const listen = (ids, update) => ids.forEach(id => $(id).addEventListener("input", update));
  const num = (id) => Number($(id).value);
  const fmt = (x, digits = 2) => x.toFixed(digits);

  function updateMask() {
    const round = num("mask-round"), anchor = num("mask-anchor");
    const box = $("mask-keys"); box.replaceChildren();
    for (let block = 0; block <= round; block++) {
      const label = document.createElement("div"); label.className = "mask-group";
      label.textContent = block === 0 ? "Original teacher keys" : `Generated keys · round ${block}`;
      box.append(label);
      for (let j = 0; j < 4; j++) {
        const allowed = block === 0 ? j <= anchor : j === anchor;
        const cell = document.createElement("div");
        cell.className = `mask-key${block ? " generated" : ""}${allowed ? " allowed" : ""}`;
        cell.textContent = block ? `R${block}·${j}` : `T${j}`;
        cell.setAttribute("aria-label", `${cell.textContent}: ${allowed ? "visible" : "blocked"}`);
        box.append(cell);
      }
    }
    $("mask-status").textContent = `Query: simulated round ${round + 1}, anchor ${anchor} · ${anchor + 1} teacher keys + ${round} same-anchor generated keys. All visible scores share one softmax.`;
  }
  listen(["mask-round", "mask-anchor"], updateMask); updateMask();
  $("mask-reset").addEventListener("click", () => {
    $("mask-round").value = "1"; $("mask-anchor").value = "2"; updateMask();
  });

  function updateResults() {
    const model = $("result-model").value, temp = num("result-temp"), task = num("result-task");
    const pair = window.EAGLE_RESULTS.find(row => row.model === model && row.temperature === temp);
    const e2 = pair.e2[task], e3 = pair.e3[task];
    $("result-e2").textContent = `${fmt(e2[0])}×`; $("result-e3").textContent = `${fmt(e3[0])}×`;
    $("result-bar-e2").style.width = `${e2[0] / 7 * 100}%`;
    $("result-bar-e3").style.width = `${e3[0] / 7 * 100}%`;
    $("result-status").textContent = `Table 1 · ${model} · T=${temp} · ${$("result-task").selectedOptions[0].text}: τ ${fmt(e2[1])} → ${fmt(e3[1])}; speed ratio change ${fmt((e3[0] / e2[0] - 1) * 100)}%; elapsed-time reduction ${fmt((1 - e2[0] / e3[0]) * 100)}%.`;
  }
  listen(["result-model", "result-temp", "result-task"], updateResults); updateResults();
  $("result-reset").addEventListener("click", () => {
    $("result-model").value = "L31 8B"; $("result-temp").value = "0"; $("result-task").value = "5"; updateResults();
  });

  function updateCost() {
    const depth = num("cost-depth"), alpha = num("cost-alpha"), decay = num("cost-decay");
    const ar = num("cost-ar"), step = num("cost-draft"), verify = num("cost-verify"), misc = num("cost-misc");
    $("depth-value").textContent = depth; $("alpha-value").textContent = `${fmt(alpha * 100, 0)}%`;
    $("decay-value").textContent = fmt(decay);
    if (["cost-ar","cost-draft","cost-verify","cost-misc"].some(id => !$(id).value || !$(id).checkValidity())) {
      $("cost-status").textContent = "Enter valid cost values within the stated ranges to calculate speedup.";
      return;
    }
    let survival = 1, accepted = 0;
    const rows = $("cost-rows"); rows.replaceChildren();
    for (let i = 1; i <= depth; i++) {
      const conditional = alpha * decay ** (i - 1);
      survival *= conditional; accepted += survival;
      const row = document.createElement("tr");
      [String(i), `${fmt(conditional * 100)}%`, `${fmt(survival * 100)}%`].forEach(text => {
        const cell = document.createElement("td"); cell.textContent = text; row.append(cell);
      });
      rows.append(row);
    }
    const cycle = depth * step + verify + misc, tau = 1 + accepted, speed = tau * ar / cycle;
    $("cost-status").textContent = `Expected accepted drafts: ${fmt(accepted, 4)} · progress τ: ${fmt(tau, 4)} tokens/cycle · draft ${fmt(depth * step)} + verify ${fmt(verify)} + misc ${fmt(misc)} = ${fmt(cycle)} ms/cycle · analytical speedup ${fmt(speed, 4)}×.`;
  }
  listen(["cost-depth", "cost-alpha", "cost-decay", "cost-ar", "cost-draft", "cost-verify", "cost-misc"], updateCost); updateCost();
  $("cost-reset").addEventListener("click", () => {
    const defaults = {"cost-depth":6,"cost-alpha":.8,"cost-decay":1,"cost-ar":10,"cost-draft":.5,"cost-verify":12,"cost-misc":1};
    Object.entries(defaults).forEach(([id,value]) => { $(id).value = value; }); updateCost();
  });

  const dialog = $("image-modal");
  let opener;
  document.querySelectorAll(".zoom-link").forEach(link => link.addEventListener("click", event => {
    event.preventDefault(); opener = link;
    $("zoom-image").src = link.href; $("zoom-image").alt = link.querySelector("img").alt;
    $("zoom-caption").textContent = link.dataset.zoom;
    $("zoom-original").href = link.href;
    dialog.showModal(); $("zoom-close").focus();
  }));
  const closeZoom = () => { dialog.close(); opener?.focus(); };
  $("zoom-close").addEventListener("click", closeZoom);
  dialog.addEventListener("cancel", event => { event.preventDefault(); closeZoom(); });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && dialog.open) { event.preventDefault(); closeZoom(); }
  });
  dialog.addEventListener("click", event => { if (event.target === dialog) closeZoom(); });
  dialog.addEventListener("close", () => opener?.focus());
})();
