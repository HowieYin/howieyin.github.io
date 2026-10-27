import {normalize, normalizeTensor, cost} from "./math.mjs";
const $=id=>document.getElementById(id);
const fmt=(x,d=5)=>Number(x).toFixed(d);
const svgText=(x,y,text,color="#cad8eb",size=13)=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}">${text}</text>`;
const vector=x=>"["+x.map(v=>fmt(v,4)).join(", ")+"]";
function updateNorm(){
  const mode=$("norm-mode").value,scale=+$("norm-scale").value,shift=+$("norm-shift").value;
  const eps=+$("norm-eps").value,g=+$("norm-gain").value,b=+$("norm-bias").value;
  for(const name of ["scale","shift","gain","bias"])$("norm-"+name+"-out").textContent=$("norm-"+name).value;
  const a=[1,2,3,4], input=a.map((x,i)=>x*scale+shift+(i===0&&$("feature-shift").checked?2:0));
  const ref=normalize(a,mode,eps,g,b),out=normalize(input,mode,eps,g,b);
  if(out.y.some(x=>!Number.isFinite(x))){
    $("norm-svg").innerHTML=svgText(20,70,"ε=0 · denominator=0","#e6bd88",18)+svgText(20,110,"归一化未定义：请选择 ε > 0 或改变输入。","#d4deed",16);
    $("norm-readout").textContent=`input ${vector(input)}\nmean=${fmt(out.mean)} · variance=${fmt(out.variance)}\nε=0 且分母为 0：原数学定义未定义。请选择 ε > 0 或改变输入。`;
    window.normState={mode,scale,shift,eps,g,b,input,defined:false,...out};
    return;
  }
  const max=Math.max(2,...ref.y.map(Math.abs),...out.y.map(Math.abs));
  let svg=svgText(15,20,"Output y = gain × normalized + bias")+"<path d=\"M285 35V235\" stroke=\"#7890b1\"/>";
  for(let i=0;i<4;i++){
    const y=50+i*45;
    svg+=svgText(15,y+14,`d=${i}`)+svgText(55,y+14,fmt(input[i],2));
    for(const [v,color,dy] of [[ref.y[i],"#718cae",0],[out.y[i],"#9ec7ff",16]]){
      const length=Math.abs(v)/max*150,x=v>=0?285:285-length;
      svg+=`<rect x="${x}" y="${y+dy}" width="${length}" height="12" rx="2" fill="${color}"/>`;
    }
    svg+=svgText(455,y+14,fmt(out.y[i],4));
  }
  svg+=svgText(15,259,"灰蓝：原向量输出   浅蓝：变换后输出","#9daec6",12);
  $("norm-svg").innerHTML=svg;
  const delta=Math.max(...out.y.map((x,i)=>Math.abs(x-ref.y[i])));
  $("norm-readout").textContent=`input ${vector(input)}\nmean=${fmt(out.mean)} · variance=${fmt(out.variance)} · denominator=${fmt(1/out.rstd)}\ny=${vector(out.y)}\nmax |Δy| vs original=${fmt(delta)} · ε=${eps}`;
  window.normState={mode,scale,shift,eps,g,b,input,delta,defined:true,...out};
}
function updateAxis(){
  const k=+$("axis-other").value,mode=$("axis-mode").value;
  $("axis-other-out").textContent=k+"×";
  const tensor=[[[1,2,3,4],[2,3,5,7],[4,8,9,12]],[[5,7,9,11],[6,8,10,13],[7,9,11,15]].map(row=>row.map(x=>x*k))];
  const y=normalizeTensor(tensor,mode);
  let svg=svgText(108,20,"d=0")+svgText(204,20,"d=1")+svgText(300,20,"d=2")+svgText(396,20,"d=3");
  for(let b=0;b<2;b++)for(let t=0;t<3;t++){
    const row=b*3+t,top=38+row*49;
    svg+=svgText(10,top+23,`b${b} / t${t}`,b===0?"#d1e0f4":"#aab9ce",12);
    for(let d=0;d<4;d++){
      const selected=mode==="ln"?b===0&&t===0:mode==="bn"?t===0&&d===0:b===0&&d===0;
      const x=90+d*96;
      svg+=`<rect x="${x}" y="${top}" width="84" height="38" rx="4" fill="${selected?"#254b74":"#152233"}" stroke="${selected?"#9fc7fa":"#3b4d64"}"/>`;
      svg+=svgText(x+13,top+25,String(tensor[b][t][d]),selected?"#e8f2ff":"#b4c1d4",13);
    }
  }
  $("axis-svg").innerHTML=svg;
  const messages={ln:"浅蓝格：b0/t0 的 4 个隐藏单元共享统计。改 b1 不改变这条输出。",bn:"浅蓝格：同 t0/d0 跨 B=2 统计。toy BN 无 running stats；不是 RNN BN 的完整实现。",time:"浅蓝格：同 b0/d0 跨 T=3 统计。t0 输出依赖未来 t1/t2，此处演示 causal leakage。"};
  $("axis-readout").textContent=messages[mode]+"\nb0/t0 standardized output = "+vector(y[0][0]);
  window.axisState={mode,k,y};
}
function updateCost(){
  const c=cost(+$("cost-b").value,+$("cost-t").value,+$("cost-d").value,+$("cost-s").value,+$("cost-layers").value);
  const mib=x=>fmt(x/2**20,4)+" MiB",kib=x=>fmt(x/1024,2)+" KiB";
  $("cost-readout").textContent=`一处 LN · ideal fused activation IO: ${mib(c.fused)}\n五 kernel 物化模型: ${mib(c.unfused)} · ratio 4× (not latency)\n训练 mean+rstd: ${kib(c.savedStats)} · γ/β FP32: ${kib(c.params)}\n${c.layers} 处 LN · 单 decode step 理想主激活: ${mib(c.decode)}\n无历史 KV；s=${c.s} bytes，B=${c.B}, T=${c.T}, D=${c.D}`;
  window.costState=c;
}
for(const el of document.querySelectorAll("#norm-widget input,#norm-widget select"))el.addEventListener("input",updateNorm);
for(const el of document.querySelectorAll("#axis-widget input,#axis-widget select"))el.addEventListener("input",updateAxis);
for(const el of document.querySelectorAll("#cost-widget select"))el.addEventListener("input",updateCost);
const preset=(scale,shift,eps)=>{
  $("norm-scale").value=scale;$("norm-shift").value=shift;$("norm-eps").value=eps;
  $("norm-mode").value="ln";$("norm-gain").value=1;$("norm-bias").value=0;$("feature-shift").checked=false;updateNorm();
};
$("norm-reset").addEventListener("click",()=>preset(1,0,"0.00001"));
$("norm-negative").addEventListener("click",()=>preset(-1,0,"0"));
$("norm-small").addEventListener("click",()=>preset(.05,0,"1"));
updateNorm();updateAxis();updateCost();
const dialog=$("figure-dialog"),zoom=$("zoom-image");
for(const btn of document.querySelectorAll("[data-zoom]"))btn.addEventListener("click",()=>{zoom.src=btn.dataset.zoom;zoom.alt=btn.getAttribute("aria-label");dialog.showModal()});
$("close-dialog").addEventListener("click",()=>dialog.close());
dialog.addEventListener("click",e=>{if(e.target===dialog)dialog.close()});
const headings=[...document.querySelectorAll("article h2")];
let scrollQueued=false;
const updateContents=()=>{
  const current=headings.filter(h=>h.getBoundingClientRect().top<=180).at(-1)||headings[0];
  for(const a of document.querySelectorAll(".toc a"))a.classList.toggle("active",a.hash==="#"+current.id);
  scrollQueued=false;
};
addEventListener("scroll",()=>{
  if(!scrollQueued){scrollQueued=true;requestAnimationFrame(updateContents)}
},{passive:true});
updateContents();
const narrow=matchMedia("(max-width:999px)");
const contentsMode=()=>$("contents").open=!narrow.matches;
contentsMode();narrow.addEventListener("change",contentsMode);
