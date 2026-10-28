"use strict";
const $=id=>document.getElementById(id),M=EagleMath;
const ns="http://www.w3.org/2000/svg";
function node(tag,attrs,text=""){
  const n=document.createElementNS(ns,tag);
  Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));n.textContent=text;return n;
}
function text(svg,x,y,t,color="#c5d4e9",size=14){
  svg.append(node("text",{x,y,fill:color,"font-size":size},t));
}
function rect(svg,x,y,width,height,fill){
  svg.append(node("rect",{x,y,width,height,fill,rx:3}));
}
const val=id=>Number($(id).value),out=(id,v)=>$(id+"-out").value=v;
const parents=[-1,0,0,1,1,2,3],tokens=["I","am","always","ready","excited","learning","today"];
function updateTree(){
  const selected=val("tree-node"),prefix=val("tree-prefix"),path=M.ancestors(parents,selected),depths=M.depths(parents);
  const s=$("tree-svg");s.replaceChildren();
  const xy=[[310,32],[190,105],[440,105],[110,180],[265,180],[470,180],[110,245]];
  text(s,18,22,"Target tree: ancestors + self only","#9eb6d5",13);
  xy.forEach(([x,y],i)=>{
    if(i) s.append(node("path",{d:`M${xy[parents[i]][0]} ${xy[parents[i]][1]+15}V${y-30}H${x}V${y-17}`,stroke:path.includes(i)?"#8fc8b8":"#34465e","stroke-width":2,fill:"none"}));
  });
  xy.forEach(([x,y],i)=>{
    const fill=i===selected?"#e6bf7e":path.includes(i)?"#8fc8b8":"#25364d";
    s.append(node("circle",{cx:x,cy:y,r:19,fill}));
    text(s,x-4,y+5,String(i),path.includes(i)?"#101722":"#cad7e9",13);
    text(s,x+26,y+4,tokens[i],i===selected?"#e6bf7e":"#c5d4e9",13);
  });
  text(s,18,294,`Query ${selected}: visible prefix[0:${prefix}) + nodes`, "#c5d4e9",13);
  for(let i=0;i<7;i++){
    rect(s,20+i*84,308,74,27,path.includes(i)?"#8fc8b8":"#25364d");
    text(s,44+i*84,327,String(i),path.includes(i)?"#101722":"#c5d4e9",13);
  }
  out("tree-prefix",prefix);
  $("tree-widget-readout").textContent=`query=${selected} (${tokens[selected]}) · ancestry=[${path.join(",")}]\ndepth=${depths[selected]} · target position=${prefix+depths[selected]}\nf_${prefix+depths[selected]} + embedding(t_${prefix+depths[selected]+1}) → predicted next feature → head → next-next proposal`;
  window.treeState={selected,prefix,path,depth:depths[selected],position:prefix+depths[selected]};
}
function updateMass(){
  const preset=$("mass-preset").value;
  let p=[.6,.3,.1],q=[val("mass-q0"),(1-val("mass-q0"))*val("mass-q1"),(1-val("mass-q0"))*(1-val("mass-q1"))];
  if(preset==="equal")q=[...p];
  if(preset==="disjoint"){p=[1,0,0];q=[0,.5,.5]}
  $("mass-q0").disabled=preset!=="toy";$("mass-q1").disabled=preset!=="toy";
  const r=M.exactMass(p,q),s=$("mass-svg");s.replaceChildren();
  const labels=["target p","proposal q","accepted mass","fallback mass","final output"],rows=[p,q,r.accepted,r.residual.map(v=>v*r.rejectedTotal),r.output],colors=["#e4bb7b","#7b9fce","#8dc7b6"];
  rows.forEach((row,j)=>{
    const y=36+j*50;text(s,14,y+16,labels[j],"#c5d4e9",13);
    let x=150;row.forEach((v,i)=>{rect(s,x,y,450*v,25,colors[i]);if(v>.075)text(s,x+6,y+17,`${"ABC"[i]} ${v.toFixed(2)}`,"#0e1827",12);x+=450*v});
  });
  out("mass-q0",q[0].toFixed(2));out("mass-q1",val("mass-q1").toFixed(3));
  $("mass-widget-readout").textContent=`accepted=${r.acceptedTotal.toFixed(4)} · rejected=${r.rejectedTotal.toFixed(4)}\nresidual=[${r.residual.map(v=>v.toFixed(4))}] · output=[${r.output.map(v=>v.toFixed(4))}]\nΣoutput=${r.output.reduce((a,b)=>a+b,0).toFixed(4)} · max |output−p|=${Math.max(...r.output.map((v,i)=>Math.abs(v-p[i]))).toExponential(2)}`;
  window.massState={p,q,...r};
}
function updateCost(){
  const measured=M.measured($("cost-mode").value);
  const c=M.cost({tau:val("cost-tau"),verify:val("cost-verify"),draft:val("cost-draft"),steps:val("cost-steps"),other:val("cost-other")});
  const k=M.capacity({batch:1,context:val("kv-context"),nodes:val("kv-nodes"),layers:32,heads:val("kv-heads"),dim:128,bytes:2,hidden:4096});
  const s=$("cost-svg");s.replaceChildren();
  text(s,15,25,"Measured Table 4 row · same gpt-fast baseline","#e4bb7b");
  rect(s,150,45,420*measured.baseline/160.4,25,"#526984");text(s,15,63,`baseline ${measured.baseline}`, "#c5d4e9",13);
  rect(s,150,83,420*measured.eagle/160.4,25,"#8fc8b8");text(s,15,102,`EAGLE ${measured.eagle}`,"#c5d4e9",13);
  text(s,15,142,`Analytical cost: verify + ${val("cost-steps")}×draft + other = ${c.cycle.toFixed(2)} AR`, "#9eb6d5",13);
  text(s,15,171,`τ=${val("cost-tau").toFixed(1)} → hypothetical speed ${c.speed.toFixed(2)}×`, "#c5d4e9",16);
  text(s,15,207,`Ideal KV: prefix ${(k.prefix/2**30).toFixed(3)} GiB; tree ${(k.tree/2**20).toFixed(2)} MiB`, "#e4bb7b",14);
  ["cost-tau","cost-verify","cost-draft","cost-steps","cost-other","kv-nodes"].forEach(id=>out(id,val(id)));
  $("cost-widget-readout").textContent=`Author TPS: ${measured.baseline} → ${measured.eagle}; incremental ${measured.ratio.toFixed(3)}×\nDerived mean interval: ${measured.baseTpot.toFixed(3)} → ${measured.tpot.toFixed(3)} ms (not measured TPOT)\nHypothetical break-even: τ > ${c.threshold.toFixed(2)}; modeled speed=${c.speed.toFixed(3)}×\nPrefix KV ${(k.prefix/1e9).toFixed(3)} GB / ${(k.prefix/2**30).toFixed(3)} GiB; temporary tree ${(k.tree/2**20).toFixed(3)} MiB\nTemporary features ${(k.features/1024).toFixed(2)} KiB; draft KV / logits / workspace excluded.`;
  window.costState={measured,...c,...k};
}
for(const [id,fn] of [["tree-widget",updateTree],["mass-widget",updateMass],["cost-widget",updateCost]])
  document.querySelectorAll(`#${id} input,#${id} select`).forEach(el=>el.addEventListener("input",fn));
$("tree-step").addEventListener("click",()=>{$("tree-node").value=(val("tree-node")+1)%7;updateTree()});
$("tree-reset").addEventListener("click",()=>{$("tree-node").value=0;$("tree-prefix").value=8;updateTree()});
const dialog=$("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(btn=>btn.addEventListener("click",()=>{
  $("zoom-image").src=btn.dataset.zoom;$("zoom-image").alt=btn.querySelector("img").alt;dialog.showModal();
}));
$("close-dialog").addEventListener("click",()=>dialog.close());
const compact=matchMedia("(max-width:1000px)");
$("contents").open=!compact.matches;
compact.addEventListener("change",e=>{$("contents").open=!e.matches});
document.querySelectorAll(".toc a").forEach(a=>a.addEventListener("click",()=>{if(compact.matches)$("contents").open=false}));
const headings=[...document.querySelectorAll("article h2")];let pending=false;
function active(){
  const current=headings.filter(h=>h.getBoundingClientRect().top<160).at(-1)||headings[0];
  document.querySelectorAll(".toc a").forEach(a=>a.classList.toggle("active",a.hash==="#"+current.id));pending=false;
}
addEventListener("scroll",()=>{if(!pending){pending=true;requestAnimationFrame(active)}},{passive:true});
updateTree();updateMass();updateCost();active();
