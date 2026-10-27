"use strict";
const $=id=>document.getElementById(id);
const NS="http://www.w3.org/2000/svg";
function node(tag,attrs,text=""){
  const e=document.createElementNS(NS,tag);
  for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
  e.textContent=text;return e;
}
function updateToy(){
  const t=GPTQMath.toy(Number($("toy-rho").value));
  const step=Number($("toy-step").value);
  $("toy-rho-out").value=t.rho.toFixed(2);
  $("toy-step-out").value=["原始权重","固定第一项、补偿第二项","再量化第二项"][step];
  const current=step===0?[t.w,t.w]:step===1?[0,t.secondBefore]:[0,t.q2];
  const svg=$("toy-svg");svg.replaceChildren();
  svg.append(node("text",{x:25,y:25},"固定格点：−2, −1, 0, 1 · W=[0.49, 0.49]"));
  const rows=[["RTN",[0,0],"#9aa9be"],["补偿路径",current,"#a6c4ef"]];
  rows.forEach(([label,values,color],r)=>{
    const y=60+r*105;
    svg.append(node("text",{x:25,y:y+20},label));
    values.forEach((v,c)=>{
      const x=170+c*180;
      svg.append(node("rect",{x,y,width:145,height:60,rx:6,fill:"#19283d",stroke:color}));
      svg.append(node("text",{x:x+72,y:y+37,"text-anchor":"middle",fill:color},`w${c+1} = ${v.toFixed(4)}`));
    });
  });
  svg.append(node("path",{d:"M243 235v20h180v-20",fill:"none",stroke:"#dec594","stroke-width":2}));
  svg.append(node("text",{x:333,y:282,"text-anchor":"middle",fill:"#dec594"},`相关性 ρ=${t.rho.toFixed(2)} · 补偿 +${t.firstCorrection.toFixed(4)}`));
  const loss=step===0?0:step===1?t.firstLoss:t.gptqLoss;
  $("toy-readout").textContent=`Step ${step}/2 · 当前补偿路径误差 ΔCΔᵀ = ${loss.toFixed(5)}\n最终 RTN loss = ${t.rtnLoss.toFixed(5)} · 最终 GPTQ loss = ${t.gptqLoss.toFixed(5)}\n最终 Q=[0,${t.q2}]；边界 ρ≈0.02041 时第二项跨过 0.5 格点。`;
  window.toyState={...t,step,current,currentLoss:loss};
}
function updateMemory(){
  const val=id=>Number($(id).value);
  const c={params:val("mem-params"),bits:val("mem-bits"),group:val("mem-group"),
    fpFraction:val("mem-fp")/100,scaleBits:16,zeroBits:val("mem-bits"),
    layers:96,kvHeads:val("mem-heads"),headDim:128,batch:val("mem-batch"),
    length:val("mem-length"),kvBits:16};
  const m=GPTQMath.memory(c),gib=b=>(b/2**30).toFixed(2);
  const svg=$("memory-svg");svg.replaceChildren();
  const max=Math.max(m.fpWeightBytes,m.totalBytes);
  const width=b=>Math.max(1,480*b/max);
  svg.append(node("text",{x:20,y:25},"相同纵轴单位：GiB · 权重与 FP16 KV 分开计数"));
  svg.append(node("rect",{x:20,y:50,width:width(m.fpWeightBytes),height:30,rx:4,fill:"#7f8ea3"}));
  svg.append(node("text",{x:20,y:105},`FP16 weights：${gib(m.fpWeightBytes)} GiB`));
  svg.append(node("rect",{x:20,y:135,width:width(m.weightBytes),height:30,fill:"#a0bbe1"}));
  svg.append(node("rect",{x:20+width(m.weightBytes),y:135,width:width(m.kvBytes),height:30,fill:"#e2bf83"}));
  svg.append(node("text",{x:20,y:192},`压缩 weights：${gib(m.weightBytes)} · KV：${gib(m.kvBytes)} GiB`));
  $("memory-readout").textContent=`有效位宽 = ${m.effectiveBits.toFixed(4)} bit / 量化权重\n权重 = ${(m.weightBytes/1e9).toFixed(2)} GB / ${gib(m.weightBytes)} GiB\nFP16 KV = ${gib(m.kvBytes)} GiB；两项合计 = ${gib(m.totalBytes)} GiB\n理想权重读取量压缩比 = ${(m.fpWeightBytes/m.weightBytes).toFixed(2)}×（不是 latency 加速）`;
  window.memoryState={...c,...m};
}
document.querySelectorAll("#toy-widget input").forEach(e=>e.addEventListener("input",updateToy));
document.querySelectorAll("#memory-widget input,#memory-widget select").forEach(e=>e.addEventListener("input",updateMemory));
$("toy-next").addEventListener("click",()=>{$("toy-step").value=(Number($("toy-step").value)+1)%3;updateToy();});
$("toy-reset").addEventListener("click",()=>{$("toy-step").value=0;$("toy-rho").value=.95;updateToy();});
const dialog=$("figure-dialog");
document.querySelectorAll("[data-zoom]").forEach(b=>b.addEventListener("click",()=>{
  $("zoom-image").src=b.dataset.zoom;$("zoom-image").alt=b.querySelector("img").alt;dialog.showModal();
}));
$("close-dialog").addEventListener("click",()=>dialog.close());
document.querySelectorAll(".table-scroll").forEach(e=>e.addEventListener("keydown",event=>{
  if(event.key==="ArrowRight"||event.key==="ArrowLeft"){
    event.preventDefault();e.scrollLeft+=event.key==="ArrowRight"?80:-80;
  }
}));
const mobile=matchMedia("(max-width:1000px)");
$("contents").open=!mobile.matches;
mobile.addEventListener("change",e=>{$("contents").open=!e.matches;});
const headings=[...document.querySelectorAll("article h2")];
let pending=false;
function contents(){
  const current=headings.filter(h=>h.getBoundingClientRect().top<=160).at(-1)||headings[0];
  document.querySelectorAll(".toc a").forEach(a=>a.classList.toggle("active",a.hash==="#"+current.id));
  pending=false;
}
addEventListener("scroll",()=>{if(!pending){pending=true;requestAnimationFrame(contents);}},{passive:true});
contents();updateToy();updateMemory();
