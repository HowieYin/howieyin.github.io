"use strict";
const $ = id => document.getElementById(id);
const ns = "http://www.w3.org/2000/svg";
function node(tag, attrs, text = "") {
  const e = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k,v));
  e.textContent = text;
  return e;
}
function base(svg, xlabel, ylabel) {
  svg.replaceChildren();
  svg.append(node("path", {d:"M60 28V240H560",stroke:"#71869f",fill:"none"}));
  svg.append(node("text",{x:310,y:284,fill:"#ccd8e8","text-anchor":"middle","font-size":13},xlabel));
  svg.append(node("text",{x:22,y:130,fill:"#ccd8e8","text-anchor":"middle","font-size":12,transform:"rotate(-90 22 130)"},ylabel));
}
function updateBatch() {
  const r = 10 ** Number($("batch-r").value), p = ScalingMath.batch(r);
  $("batch-r-out").value = r.toFixed(3) + " × Bcrit";
  const svg = $("batch-svg");base(svg,"S / Smin（log scale）","E / Emin（log scale）");
  const xy = q => [60+Math.log10(q.steps)*230,240-Math.log10(q.examples)*100];
  const pts = Array.from({length:101},(_,i)=>xy(ScalingMath.batch(10**(-2+i*.04))).join(","));
  svg.append(node("polyline",{points:pts,fill:"none",stroke:"#8fafdf","stroke-width":2}));
  const [x,y] = xy(p);
  svg.append(node("circle",{cx:x,cy:y,r:6,fill:"#e9bf78"}));
  for (const k of [1,2,10,100]) {
    svg.append(node("text",{x:60+Math.log10(k)*230,y:258,fill:"#9eacc1","font-size":11,"text-anchor":"middle"},String(k)));
    svg.append(node("text",{x:49,y:243-Math.log10(k)*100,fill:"#9eacc1","font-size":11,"text-anchor":"end"},String(k)));
  }
  $("batch-readout").textContent=`B/Bcrit = ${r.toFixed(3)}\nS/Smin = ${p.steps.toFixed(4)} · E/Emin = C/Cmin = ${p.examples.toFixed(4)}\nCmin / actual C = ${(100*p.minComputeFraction).toFixed(2)}%`;
  window.batchState={r,...p};
}
function updateAllocation() {
  const decades=Number($("budget").value),policy=$("policy").value;
  const p=ScalingMath.allocation(decades,policy);
  $("budget-out").value=`10^${decades.toFixed(1)} × base compute`;
  const svg=$("allocation-svg");base(svg,"增加资源的 log10 倍率","资源");
  const items=[["Parameters N",Math.log10(p.n),"#8fafdf"],["Processed tokens E",Math.log10(p.tokens),"#8fc7b7"]];
  items.forEach(([label,log,color],i)=>{
    const y=65+i*83;
    svg.append(node("text",{x:65,y:y-10,fill:"#ccd8e8","font-size":13},label));
    svg.append(node("rect",{x:65,y,width:Math.max(1,log/9*470),height:24,rx:3,fill:color}));
    svg.append(node("text",{x:65,y:y+45,fill:"#9eacc1","font-size":12},`10^${log.toFixed(4)}`));
  });
  const mult=n=> n.toLocaleString("en-US",{maximumFractionDigits:4});
  $("allocation-readout").textContent=`策略：${policy}\nN ×${mult(p.n)} · E ×${mult(p.tokens)} · N×E = ${(p.n*p.tokens).toExponential(4)}\n`+
    (p.b===null?"此模式不推算 B、S 或统一 loss 比例。":`B ×${mult(p.b)} · Smin ×${mult(p.s)} · loss ×${p.loss.toFixed(5)}\npN=${p.exponents.n.toFixed(6)}, pB=${p.exponents.b.toFixed(6)}, pS=${p.exponents.s.toFixed(6)}`);
  window.allocationState={decades,policy,...p};
}
function updateSuboptimal() {
  const r=Number($("model-r").value),p=ScalingMath.suboptimal(r);
  $("model-r-out").value=r.toFixed(2)+" × Neff";
  const svg=$("suboptimal-svg");base(svg,"N / Neff","ratio（cap 4）");
  const xy=(r,v)=>[60+(r-.1)/3.9*500,240-Math.min(4,v)*50];
  for(const [key,color] of [["compute","#e9bf78"],["steps","#8fafdf"]]){
    const pts=Array.from({length:196},(_,i)=>{
      const x=.1+i*.02,q=ScalingMath.suboptimal(x);return q.valid?xy(x,q[key]).join(","):null;
    }).filter(Boolean).join(" ");
    svg.append(node("polyline",{points:pts,fill:"none",stroke:color,"stroke-width":2}));
    if(p.valid){const [x,y]=xy(r,p[key]);svg.append(node("circle",{cx:x,cy:y,r:5,fill:color}));}
  }
  svg.append(node("text",{x:70,y:25,fill:"#e9bf78","font-size":12},"Compute"));
  svg.append(node("text",{x:200,y:25,fill:"#8fafdf","font-size":12},"Steps"));
  for(const x of [.1,1,2.2,4])svg.append(node("text",{x:xy(x,0)[0],y:258,fill:"#9eacc1","font-size":11,"text-anchor":"middle"},String(x)));
  for(const y of [0,1,2,4])svg.append(node("text",{x:49,y:xy(1,y)[1]+4,fill:"#9eacc1","font-size":11,"text-anchor":"end"},String(y)));
  $("suboptimal-readout").textContent=p.valid?`固定目标 loss · N/Neff=${r.toFixed(2)}\nCompute ratio=${p.compute.toFixed(5)} · Steps ratio=${p.steps.toFixed(5)}\n这是解析预测；少 updates 不直接等于同幅度 wall-clock 加速。`:"模型容量地板高于目标，解析式无有限训练预算解。";
  window.suboptimalState={r,...p};
}
document.querySelectorAll("#batch-widget input").forEach(e=>e.addEventListener("input",updateBatch));
document.querySelectorAll("#allocation-widget input,#allocation-widget select").forEach(e=>e.addEventListener("input",updateAllocation));
document.querySelectorAll("#suboptimal-widget input").forEach(e=>e.addEventListener("input",updateSuboptimal));
$("batch-reset").addEventListener("click",()=>{$("batch-r").value=0;updateBatch();});
$("model-reset").addEventListener("click",()=>{$("model-r").value=2.2;updateSuboptimal();});
const dialog=$("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(b=>b.addEventListener("click",()=>{
  $("zoom-image").src=b.dataset.zoom;$("zoom-image").alt=b.querySelector("img").alt;dialog.showModal();
}));
$("close-dialog").addEventListener("click",()=>dialog.close());
const compact=matchMedia("(max-width:1000px)");
$("contents").open=!compact.matches;
compact.addEventListener("change",e=>{$("contents").open=!e.matches;});
const headings=[...document.querySelectorAll("article h2")];
let pending=false;
function contents(){
  const h=headings.filter(h=>h.getBoundingClientRect().top<=160).at(-1)||headings[0];
  document.querySelectorAll(".toc a").forEach(a=>a.classList.toggle("active",a.hash==="#"+h.id));pending=false;
}
addEventListener("scroll",()=>{if(!pending){pending=true;requestAnimationFrame(contents);}},{passive:true});
updateBatch();updateAllocation();updateSuboptimal();contents();
