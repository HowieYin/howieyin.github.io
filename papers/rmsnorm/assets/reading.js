(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const fmt = x => Number.isFinite(x) ? x.toFixed(4) : "undefined";
  function updateNorm() {
    const bases = { balanced: [-3,-1,1,3], prefix: [1,1,100,100], zero: [0,0,0,0] };
    const scale = +$("scale").value, shift = +$("shift").value;
    const a = bases[$("preset").value].map(x => scale * x + shift);
    const p = +$("ratio").value, eps = +$("epsilon").value;
    const ln = RMSModel.norm(a,"ln",1,eps), rms = RMSModel.norm(a,"rms",1,eps), part = RMSModel.norm(a,"rms",p,eps);
    $("scale-value").textContent = scale; $("shift-value").textContent = shift;
    $("norm-cells").innerHTML = a.map((v,i)=>`<tr><td>${i+1}</td><td>${fmt(v)}</td><td>${fmt(ln.out[i])}</td><td>${fmt(rms.out[i])}</td><td>${fmt(part.out[i])}</td></tr>`).join("");
    const groups = [["LN",ln],["RMS",rms],["pRMS",part]];
    const max = Math.max(1,...groups.flatMap(([,g])=>g.out.filter(Number.isFinite).map(Math.abs)));
    $("vector-chart").innerHTML = groups.map(([name,g],j)=>`<div class="vector-row"><span>${name}</span><div class="vector-items">${g.out.map((v,i)=>`<div class="component"><span class="component-label">a${i+1}</span><div class="component-axis"><i style="--length:${Number.isFinite(v)?Math.abs(v)/max*48:0}%;--left:${v<0?50-Math.abs(v)/max*48:50}%;--tone:${j}"></i></div><span>${fmt(v)}</span></div>`).join("")}</div></div>`).join("");
    const warning = [...ln.out,...rms.out,...part.out].some(v=>!Number.isFinite(v)) ? " · Zero denominator: paper algebra undefined; choose epsilon > 0" : "";
    $("norm-readout").textContent = `mean=${fmt(rms.mean)} · full RMS=${fmt(rms.rms)} · LN den=${fmt(ln.den)} · RMS den=${fmt(rms.den)} · prefix k=${part.k} · prefix den=${fmt(part.den)}${warning}`;
  }
  function updateCost() {
    const M=+$("rows").value,D=+$("hidden").value,s=+$("dtype").value,f=+$("share").value/100,k=+$("speed").value;
    $("share-value").textContent = `${Math.round(f*100)}%`; $("speed-value").textContent = `${k}×`;
    if(!Number.isInteger(M)||!Number.isInteger(D)||M<1||D<1||M>1e6||D>131072) { $("cost-readout").textContent="Enter integer rows 1–1,000,000 and hidden size 1–131,072.";return; }
    const c = RMSModel.costs(M,D,s,f,k), time = 100/c.speedup;
    const bytes = c.bytes<1048576?`${(c.bytes/1024).toFixed(2)} KiB`:`${(c.bytes/1048576).toFixed(2)} MiB`;
    $("cost-readout").textContent = `IO lower bound: ${bytes} · statistic cache: ${c.statisticBytes.toLocaleString()} bytes · model speedup: ${c.speedup.toFixed(4)}× · infinite-k limit: ${f===1?"∞":(1/(1-f)).toFixed(4)}×`;
    $("amdahl-bar").style.width = `${time}%`; $("new-time").textContent = `${time.toFixed(2)}%`;
  }
  for(const id of ["preset","scale","shift","ratio","epsilon"]) $(id).addEventListener("input",updateNorm);
  for(const id of ["rows","hidden","dtype","share","speed"]) $(id).addEventListener("input",updateCost);
  $("norm-reset").addEventListener("click",()=>{ $("preset").value="balanced";$("scale").value=1;$("shift").value=0;$("ratio").value=".5";$("epsilon").value=0;updateNorm(); });
  const dialog=$("figure-modal");
  document.querySelectorAll(".zoom-link").forEach(a=>a.addEventListener("click",e=>{
    if(typeof dialog.showModal!=="function")return;
    e.preventDefault(); $("zoom-image").src=a.href; $("zoom-image").alt=a.querySelector("img").alt; $("zoom-title").textContent=a.dataset.zoom;
    $("zoom-download").href=a.href; dialog.showModal();
  }));
  $("zoom-close").addEventListener("click",()=>dialog.close());
  dialog.addEventListener("click",e=>{if(e.target===dialog)dialog.close();});
  updateNorm(); updateCost();
})();
