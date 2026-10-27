"use strict";
const $=id=>document.getElementById(id);
const ns="http://www.w3.org/2000/svg";
const svgNode=(tag,attrs,text="")=>{
  const e=document.createElementNS(ns,tag);
  Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));
  e.textContent=text;return e;
};
const f=n=>n.toFixed(4);
function updateScale(){
  const s=+$("scale-s").value,x=+$("scale-x").value,dominant=+$("scale-max").value;
  const z=AWQMath.scaleToy(s,x,dominant),svg=$("scale-svg");
  svg.replaceChildren();
  svg.append(svgNode("text",{x:20,y:22,fill:"#bdcde5","font-size":13},"各通道对输出误差的贡献（带符号）"));
  const magnitude=Math.max(.02,...z.contribution.map(Math.abs)),axis=300,unit=210/magnitude;
  svg.append(svgNode("line",{x1:axis,y1:38,x2:axis,y2:223,stroke:"#8b9eb7"}));
  z.contribution.forEach((error,i)=>{
    const y=49+i*44,width=Math.abs(error)*unit;
    svg.append(svgNode("text",{x:20,y:y+17,fill:i===0?"#e2bc7c":"#b5c6df","font-size":12},`j=${i} · x=${z.X[i]}`));
    svg.append(svgNode("rect",{x:error<0?axis-width:axis,y,width:Math.max(width,.8),height:24,fill:i===0?"#e2bc7c":"#7f9fc9",rx:3}));
    svg.append(svgNode("text",{x:555,y:y+17,fill:"#d5e2f4","text-anchor":"end","font-size":12},f(error)));
  });
  svg.append(svgNode("text",{x:300,y:249,fill:"#b5c6df","text-anchor":"middle","font-size":12},"← 负贡献             0             正贡献 →"));
  $("scale-readout").textContent=`s=${s} · Δ′=${f(z.delta)} · 重要通道原坐标步长 Δ′/s=${f(z.delta/s)}\n参考输出=${f(z.reference)} · 量化输出=${f(z.quantized)} · |总误差|=${f(z.error)}\n重要通道误差上界=${f(z.salientBound)} · 全通道三角上界=${f(z.totalBound)}\n重建 W=[${z.reconstructed.map(f).join(", ")}]`;
  [["scale-s",s],["scale-x",x],["scale-max",dominant]].forEach(([id,v])=>$(id+"-out").value=v);
  window.scaleState={s,x,dominant,...z};
}
let stage=0;
function updatePacking(){
  const offset=+$("pack-offset").value;
  const weights=Array.from({length:32},(_,i)=>(i+offset)%16),bytes=AWQMath.pack(weights),unpacked=AWQMath.unpack(bytes);
  const svg=$("packing-svg");svg.replaceChildren();
  const text=(x,y,t,color="#c9d8ed",size=12)=>svg.append(svgNode("text",{x,y,fill:color,"font-size":size},t));
  text(15,20,["① 原始 32 个 unsigned INT4 codes","② 离线：低 wᵢ / 高 wᵢ₊₁₆ 合入 byte","③ 运行时：AND 与 SHIFT 还原两组"][stage]);
  for(let row=0;row<2;row++){
    text(15,48+row*64,row===0?"w₀ … w₁₅":"w₁₆ … w₃₁",row===0?"#e2bc7c":"#91b5e6");
    for(let i=0;i<16;i++){
      const x=15+i*34,y=58+row*64;
      svg.append(svgNode("rect",{x,y,width:31,height:28,rx:3,fill:row===0?"#3c3324":"#21324b",stroke:row===0?"#8e744c":"#577da9"}));
      text(x+10,y+19,String(weights[i+row*16]),row===0?"#efd19c":"#bdd8fc",11);
    }
  }
  text(15,207,stage===0?"下一步：每个 byte 配对相隔 16 的两项":"Packed bytes：high | low（十六进制）");
  for(let i=0;i<16;i++){
    const x=15+i*34;
    svg.append(svgNode("rect",{x,y:218,width:31,height:31,rx:3,fill:stage>0?"#18342f":"#18202c",stroke:"#48625d",opacity:stage>0?1:.4}));
    text(x+5,238,bytes[i].toString(16).padStart(2,"0"),stage>0?"#b4dfcd":"#748299",11);
  }
  text(15,283,stage===2?"low=Pw & 0x0f；high=(Pw >> 4) & 0x0f → 顺序完全还原":"点击“下一步”观察离线 packing 与运行时 unpack",stage===2?"#b4dfcd":"#9bacbf");
  $("packing-readout").textContent=`阶段 ${stage+1}/3 · offset=${offset} · 32 codes → 16 bytes → 32 codes\nroundtrip=${unpacked.every((w,i)=>w===weights[i])?"PASS":"FAIL"} · 仅解包，不含 convert、scale、zero 与 FMA`;
  $("pack-offset-out").value=offset;
  window.packState={stage,offset,weights,bytes,unpacked};
}
function updateCapacity(){
  const parameters=+$("memory-params").value*1e9,group=+$("memory-group").value,format=$("memory-format").value;
  const L=+$("memory-layers").value,T=+$("memory-context").value,H=+$("memory-heads").value,B=+$("memory-batch").value;
  const weights=AWQMath.weightBytes(parameters,group,format),fp16=2*parameters,kv=AWQMath.kvBytes(L,B,T,H,128,2);
  const gib=n=>(n/2**30).toFixed(3)+" GiB",bpw=weights*8/parameters;
  $("capacity-readout").textContent=`权重：${gib(weights)} · ${(weights/1e9).toFixed(5)} GB · ${bpw.toFixed(5)} bit/weight\nFP16 权重：${gib(fp16)} · 权重容量比=${(fp16/weights).toFixed(3)}×\nFP16 KV：${gib(kv)} · L=${L}, B=${B}, T=${T}, H_kv=${H}, d_h=128\n权重 + 逻辑 KV：${gib(weights+kv)}（不含 vision / 未量化层 / activation / workspace / padding / runtime）`;
  const svg=$("capacity-svg");svg.replaceChildren();
  const max=Math.max(fp16,weights+kv),u=480/max;
  [["FP16 weights",fp16,0],["INT4 weights + KV",weights,kv]].forEach(([label,w,k],i)=>{
    const y=40+i*77;
    svg.append(svgNode("text",{x:15,y:y-13,fill:"#c9d8ed","font-size":12},label));
    svg.append(svgNode("rect",{x:15,y,width:w*u,height:28,fill:"#7f9fc9",rx:3}));
    if(k)svg.append(svgNode("rect",{x:15+w*u,y,width:k*u,height:28,fill:"#e2bc7c",rx:3}));
    svg.append(svgNode("text",{x:515,y:y+19,fill:"#c9d8ed","font-size":12},gib(w+k)));
  });
  svg.append(svgNode("text",{x:15,y:190,fill:"#9fb1c9","font-size":12},"蓝：权重容量    黄：FP16 KV；只比较容量，不估计速度"));
  window.capacityState={parameters,group,format,L,T,H,B,weights,fp16,kv,bpw};
}
document.querySelectorAll("#scale-widget input").forEach(e=>e.addEventListener("input",updateScale));
document.querySelectorAll("#packing-widget input").forEach(e=>e.addEventListener("input",updatePacking));
document.querySelectorAll("#capacity-widget select").forEach(e=>e.addEventListener("input",updateCapacity));
$("pack-step").addEventListener("click",()=>{stage=(stage+1)%3;updatePacking()});
$("pack-reset").addEventListener("click",()=>{stage=0;$("pack-offset").value=0;updatePacking()});
const dialog=$("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(button=>button.addEventListener("click",()=>{
  $("zoom-image").src=button.dataset.zoom;$("zoom-image").alt=button.querySelector("img").alt;dialog.showModal();
}));
$("close-dialog").addEventListener("click",()=>dialog.close());
const compact=matchMedia("(max-width:1000px)");$("contents").open=!compact.matches;
compact.addEventListener("change",e=>{$("contents").open=!e.matches});
const headings=[...document.querySelectorAll("article h2")];
let pending=false;
function updateTOC(){
  const current=headings.filter(h=>h.getBoundingClientRect().top<=160).at(-1)||headings[0];
  document.querySelectorAll(".toc a").forEach(a=>a.classList.toggle("active",a.hash==="#"+current.id));pending=false;
}
addEventListener("scroll",()=>{if(!pending){pending=true;requestAnimationFrame(updateTOC)}},{passive:true});
updateScale();updatePacking();updateCapacity();updateTOC();
