(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const nodes=[{name:"R",parent:-1,depth:0},{name:"A1",parent:0,depth:1},{name:"A2",parent:0,depth:1},
    ...[1,2].flatMap((parent)=>[1,2,3].map(i=>({name:`B${i}(A${parent})`,parent,depth:2})))];
  const paths=[[0,1,3],[0,1,4],[0,1,5],[0,2,6],[0,2,7],[0,2,8]];
  let query=0,phase=0;
  function ancestors(index){const set=new Set();while(index>=0){set.add(index);index=nodes[index].parent;}return set;}
  function treeUpdate(){
    const allowed=ancestors(query),path=paths[+$("tree-path").value],accepted=path.slice(0,+$("tree-depth").value+1);
    $("tree-nodes").innerHTML=nodes.map((n,i)=>`<button class="tree-node ${allowed.has(i)?"ancestor":""} ${i===query?"queried":""} ${phase===2&&accepted.includes(i)?"committed":""}" style="--depth:${n.depth}" data-node="${i}" aria-pressed="${query===i}" type="button">${n.name}<small>depth ${n.depth} · pos S+${n.depth}</small></button>`).join("");
    $("tree-mask").innerHTML=`<caption>Rows: query · columns: candidate keys · old prefix always visible</caption><thead><tr><th>Q / K</th>${nodes.map(n=>`<th>${n.name}</th>`).join("")}</tr></thead><tbody>${nodes.map((n,i)=>`<tr class="${i===query?"mask-query":""}"><th>${n.name}</th>${nodes.map((_,j)=>`<td class="${ancestors(i).has(j)?"visible-key":""}">${ancestors(i).has(j)?"1":"·"}</td>`).join("")}</tr>`).join("")}</tbody>`;
    const states=["Proposed","Verified","Committed"];
    $("tree-readout").textContent=`${states[phase]} · q=9 · 6 leaves · query ${nodes[query].name} sees old prefix + ${[...allowed].reverse().map(i=>nodes[i].name).join(" → ")} · accepted prefix ${accepted.map(i=>nodes[i].name).join(" → ")}${phase===2?` · logical length S+${accepted.length}`:" · not yet committed"}`;
    $("tree-cache").innerHTML='<span class="prefix-cell">Old KV · S</span>'+ (phase===2?accepted:nodes.map((_,i)=>i)).map(i=>`<span class="${phase===2?"kept-cell":"temp-cell"}">${nodes[i].name}</span>`).join("");
    $("tree-step").textContent=["Next: verify","Next: commit","Replay"][phase];
  }
  $("tree-nodes").addEventListener("click",e=>{const b=e.target.closest("[data-node]");if(!b)return;query=+b.dataset.node;treeUpdate();$("tree-nodes").querySelector(`[data-node="${query}"]`).focus();});
  $("tree-step").addEventListener("click",()=>{phase=(phase+1)%3;treeUpdate();});
  $("tree-reset").addEventListener("click",()=>{phase=0;query=0;$("tree-path").value="0";$("tree-depth").value="2";treeUpdate();});
  ["tree-path","tree-depth"].forEach(id=>$(id).addEventListener("input",()=>{phase=0;treeUpdate();}));
  function acceptance(){
    const t=+$("temperature").value,eps=+$("epsilon").value,delta=+$("delta").value;
    const weights=[2,1,0,-1].map(x=>Math.exp((x-2)/t)),sum=weights.reduce((a,b)=>a+b,0),p=weights.map(w=>w/sum);
    const H=-p.reduce((v,x)=>v+x*Math.log(x),0),threshold=Math.min(eps,delta*Math.exp(-H)),mass=p.reduce((v,x)=>v+(x>threshold?x:0),0),match=p.reduce((v,x)=>v+x*x,0);
    $("temperature-value").textContent=t.toFixed(1);$("epsilon-value").textContent=eps.toFixed(2);$("delta-value").textContent=delta.toFixed(2);
    $("acceptance-bars").innerHTML=p.map((x,i)=>`<div class="prob-row"><span>token ${i+1}</span><span class="bar-track"><i class="bar-fill ${x>threshold?"selected":""}" style="width:${x*100}%"></i></span><span>${(x*100).toFixed(2)}% · ${x>threshold?"allowed":"reject"}</span></div>`).join("");
    $("acceptance-readout").textContent=`p=q_D · strict rejection acceptance 100% · independent matching Σp² ${(100*match).toFixed(2)}% · H=${H.toFixed(4)} · typical threshold=${threshold.toFixed(4)} · allowed probability mass ${(mass*100).toFixed(2)}%`;
  }
  ["temperature","epsilon","delta"].forEach(id=>$(id).addEventListener("input",acceptance));
  function costRows(){
    const n=$("cost-table").value,d=window.MedusaCostData[n],letter=n==="6"?"S":"B";
    const current=+$("cost-row").value;
    $("cost-row").innerHTML=d.rows.map(r=>`<option value="${r.row}">${letter}=${r.row}</option>`).join("");
    $("cost-row").value=d.rows.some(r=>r.row===current)?String(current):(n==="6"?"1024":"1");
    cost();
  }
  function cost(){
    const n=$("cost-table").value,d=window.MedusaCostData[n],row=d.rows.find(r=>r.row===+$("cost-row").value),q=+$("cost-q").value,[flops,oi]=row.values[d.q.indexOf(q)],[base,baseOI]=row.values[0],ratio=flops/base,time=q/ratio;
    $("cost-readout").textContent=`Table ${n} · ${n==="6"?"S":"B"}=${row.row} · q=${q} · ${flops} TFLOP/s · ${oi} FLOP/byte · vs q=1: ${ratio.toFixed(2)}× FLOP/s · ${(oi/baseOI).toFixed(2)}× intensity · approximate cycle time ${time.toFixed(2)}×`;
    $("cost-compare").innerHTML=`<div><strong>${ratio.toFixed(2)}×</strong><span>compute throughput</span></div><div><strong>~${q}×</strong><span>matrix work</span></div><div><strong>~${time.toFixed(2)}×</strong><span>operator time / q=1</span></div>`;
  }
  $("cost-table").addEventListener("input",costRows);
  ["cost-row","cost-q"].forEach(id=>$(id).addEventListener("input",cost));
  const dialog=$("figure-modal");let trigger;
  document.querySelectorAll(".zoom-link").forEach(a=>a.addEventListener("click",e=>{
    if(typeof dialog.showModal!=="function")return;e.preventDefault();trigger=a;
    $("zoom-image").src=a.href;$("zoom-image").alt=a.querySelector("img").alt;
    $("zoom-title").textContent=a.dataset.zoom;$("zoom-download").href=a.href;
    dialog.showModal();dialog.querySelector(".zoom-scroll").scrollTo(0,0);
  }));
  $("zoom-close").addEventListener("click",()=>dialog.close());
  dialog.addEventListener("click",e=>{if(e.target===dialog)dialog.close();});
  dialog.addEventListener("close",()=>trigger?.focus({preventScroll:true}));
  document.querySelectorAll(".copy-code").forEach(b=>b.addEventListener("click",async()=>{
    const pre=b.parentElement.nextElementSibling;
    try{await navigator.clipboard.writeText(pre.textContent);b.textContent="Copied";}
    catch{const r=document.createRange();r.selectNodeContents(pre);const s=getSelection();s.removeAllRanges();s.addRange(r);b.textContent="Selected — copy";}
  }));
  treeUpdate();acceptance();costRows();
})();
