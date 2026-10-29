(() => {
  'use strict';
  const byId=id=>document.getElementById(id);
  const theme=byId('theme-toggle');
  try {document.documentElement.classList.toggle('light',localStorage.getItem('paper-reading-theme')==='light');}catch{}
  const themeLabel=()=>theme.textContent=document.documentElement.classList.contains('light')?'Dark mode':'Light mode';
  themeLabel();
  theme.addEventListener('click',()=>{
    const light=document.documentElement.classList.toggle('light');
    try {localStorage.setItem('paper-reading-theme',light?'light':'dark');}catch{}
    themeLabel();
  });
  const contents=byId('contents');
  if(matchMedia('(max-width:800px)').matches)contents.open=false;
  const tocLinks=[...document.querySelectorAll('.toc a')];
  const observer=new IntersectionObserver(entries=>{
    for(const e of entries)if(e.isIntersecting)tocLinks.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));
  },{rootMargin:'-10% 0px -72% 0px'});
  document.querySelectorAll('.reading h2[id]').forEach(h=>observer.observe(h));
  tocLinks.forEach(a=>a.addEventListener('click',()=>{if(matchMedia('(max-width:800px)').matches)contents.open=false;}));
  const modal=byId('figure-modal');let focus;
  document.querySelectorAll('.reading figure img').forEach(img=>{
    img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','Zoom: '+img.alt);
    const open=()=>{
      focus=img;byId('modal-image').src=img.src;byId('modal-image').alt=img.alt;
      byId('modal-caption').textContent=img.closest('figure').querySelector('figcaption')?.textContent||img.alt;
      byId('modal-source').href=img.src;modal.showModal();
    };
    img.addEventListener('click',open);
    img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});
  });
  byId('modal-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
  modal.addEventListener('close',()=>focus?.focus({preventScroll:true}));
  document.querySelectorAll('.code-frame,.equation').forEach(frame=>{
    const button=frame.querySelector('.copy-button');if(!button)return;
    button.addEventListener('click',async()=>{
      try {await navigator.clipboard.writeText(frame.querySelector('code').textContent);button.textContent='Copied';}
      catch {button.textContent='Select text';}
      setTimeout(()=>button.textContent='Copy',1800);
    });
  });
  let step=0,timer=null;
  const stop=()=>{clearInterval(timer);timer=null;byId('grpo-play').textContent='Play';byId('grpo-play').setAttribute('aria-pressed','false');};
  function updateGRPO(){
    const n=Number(byId('success-count').value),eps=Number(byId('clip-epsilon').value),ratio=Number(byId('policy-ratio').value);
    const mean=n/4,std=Math.sqrt(mean*(1-mean)),clipped=Math.max(1-eps,Math.min(1+eps,ratio));
    byId('ratio-value').textContent=ratio.toFixed(1);
    const grid=byId('answer-grid');grid.replaceChildren();
    for(let i=0;i<4;i++){
      const r=i<n?1:0,a=std>0?(r-mean)/std:NaN,obj=std>0?Math.min(ratio*a,clipped*a):NaN;
      const card=document.createElement('div');card.className='answer-cell'+(r?' correct':'');
      const lines=[`Answer ${i+1}`,step>=1?`reward ${r}`:'sampled',step>=2?(std>0?`A ${a.toFixed(3)}`:'A undefined'):'',step>=3?(std>0?`obj ${obj.toFixed(3)}`:'no defined update'):''];
      lines.forEach(t=>{if(t){const p=document.createElement('span');p.textContent=t;card.append(p);}});grid.append(card);
    }
    document.querySelectorAll('#grpo-explorer .step-track li').forEach((x,i)=>x.classList.toggle('current',i===step));
    byId('grpo-result').textContent=std>0?`Mean ${mean.toFixed(2)} · population std ${std.toFixed(3)}\nClip interval [${(1-eps).toFixed(1)}, ${(1+eps).toFixed(1)}] · ratio after clip ${clipped.toFixed(1)}\n${['Old policy samples four outputs.','Verifier assigns binary outcomes.','Same-question baseline creates signed advantages.','Positive and negative advantages enter min differently.'][step]}`:'Zero std: written advantage is undefined.\nStabilization/filtering choices are not disclosed by this equation.';
  }
  ['success-count','clip-epsilon','policy-ratio'].forEach(id=>byId(id).addEventListener('input',updateGRPO));
  byId('grpo-step').addEventListener('click',()=>{stop();step=(step+1)%4;updateGRPO();});
  byId('grpo-replay').addEventListener('click',()=>{stop();step=0;updateGRPO();});
  byId('grpo-play').addEventListener('click',()=>{
    if(timer){stop();return;}
    byId('grpo-play').textContent='Pause';byId('grpo-play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{step++;if(step>3){step=3;stop();}updateGRPO();},1100);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  updateGRPO();
  const metrics={AIME:[77.9,59,74,78.1,79.8],GPQA:[75.8,66.1,70.7,71.2,71.5],CNMO:[88.1,58,73.9,77.3,78.8],IFEval:[46.6,71.7,72,78.1,83.3],Alpaca:[24.7,50.1,55.8,62.1,87.6],Arena:[53.6,77,73.2,75.6,92.3],LCB:[50,57.5,63.5,64.6,65.9],Aider:[12.2,6.7,25.6,44.8,53.3]};
  function bars(target,names,values,format){
    target.replaceChildren(...values.map((v,i)=>{
      const row=document.createElement('div'),text=document.createElement('span'),bar=document.createElement('i');
      text.textContent=names[i]+' · '+format(v);bar.style.width=Math.max(0,Math.min(100,v))+'%';bar.setAttribute('aria-hidden','true');
      row.append(text,bar);return row;
    }));
  }
  function updateStage(){
    const v=metrics[byId('stage-metric').value],names=['Zero','Dev1','Dev2','Dev3','R1'];
    bars(byId('stage-bars'),names,v,x=>x.toFixed(1)+'%');
    byId('stage-result').textContent=v.slice(1).map((x,i)=>`${names[i]} → ${names[i+1]}: ${x-v[i]>=0?'+':''}${(x-v[i]).toFixed(1)} pp`).join('\n');
  }
  byId('stage-metric').addEventListener('input',updateStage);updateStage();
  function updateCost(){
    const ctx=Number(byId('cache-context').value),batch=Number(byId('cache-batch').value),tpot=Number(byId('assumed-tpot').value);
    const raw=Number(byId('thinking-tokens').value),tokens=Math.max(0,Math.min(131072,Number.isFinite(raw)?raw:0));
    const per=61*(512+64)*2,kv=ctx*per/2**30,sec=tokens*tpot/1000;
    byId('context-value').textContent=ctx+' tokens';byId('tpot-value').textContent=tpot+' ms/token';
    bars(byId('cost-bars'),['KV vs 128 GiB analytical scale','Thinking time vs 4,000 s scale'],[kv*batch/128*100,sec/4000*100],x=>x.toFixed(2)+'%');
    byId('cost-result').textContent=`BF16 latent cache: ${(per/1024).toFixed(3)} KiB/token\nPer request: ${kv.toFixed(3)} GiB · ${batch} requests: ${(kv*batch).toFixed(3)} GiB\nThinking only: ${sec.toFixed(2)} s (${tokens} × ${tpot} ms)\nConditional arithmetic; not measured latency or total memory.`;
  }
  ['cache-context','cache-batch','thinking-tokens','assumed-tpot'].forEach(id=>byId(id).addEventListener('input',updateCost));
  updateCost();
})();
