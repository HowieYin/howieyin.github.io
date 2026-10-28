(function(){
  'use strict';
  const $=id=>document.getElementById(id),m=window.DFlashModel;
  let step=3,timer=null;
  const words=['speculative','decoding','is','fast'];
  function stop(){clearInterval(timer);timer=null;$('prefix-play').textContent='Play';}
  function drawPrefix(){
    const miss=+$('mismatch').value;
    $('draft-tokens').replaceChildren();$('target-tokens').replaceChildren();
    for(let i=0;i<4;i++){
      for(const [id,word] of [['draft-tokens',words[i]],['target-tokens',i===miss?'verified':words[i]]]){
        const b=document.createElement('b');
        b.textContent=step===0?'—':id==='target-tokens'&&step<2?'?':word;
        if(step>=2)b.className=i<miss?'accepted':i===miss?'rejected':'discarded';
        $(id).append(b);
      }
    }
    [...$('step-track').children].forEach((e,i)=>e.classList.toggle('current',i===step));
    const states=[
      'Context: target has supplied the clean anchor and verified hidden features.',
      'Parallel draft: four masked positions are predicted in one forward pass.',
      `Verify: target proposals agree for the first ${miss} positions; ${miss===4?'all four accepted.':'the first mismatch truncates the prefix.'}`,
      `Commit: accepted proposals = ${miss}; τ = ${miss+1}. ${miss===4?'Keep the four proposals and the target bonus.':'Keep the prefix and target correction; discard every later proposal.'}`
    ];
    $('prefix-reading').textContent=states[step];
  }
  $('mismatch').addEventListener('change',()=>{stop();step=3;drawPrefix();});
  $('prefix-step').addEventListener('click',()=>{stop();step=(step+1)%4;drawPrefix();});
  $('prefix-replay').addEventListener('click',()=>{stop();step=0;drawPrefix();});
  $('prefix-play').addEventListener('click',()=>{
    if(timer){stop();return;}
    if(step===3)step=0;
    drawPrefix();
    if(matchMedia('(prefers-reduced-motion: reduce)').matches){
      step=3;drawPrefix();return;
    }
    $('prefix-play').textContent='Pause';
    timer=setInterval(()=>{step++;drawPrefix();if(step===3)stop();},1100);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  function survival(){
    const a=+$('accept-prob').value;
    $('accept-prob-value').textContent=a.toFixed(2);
    $('survival-reading').textContent=`E[τ] = ${m.expectedLength(Array(4).fill(a)).toFixed(4)} for 4 conditional probabilities of ${a.toFixed(2)}.`;
  }
  $('accept-prob').addEventListener('input',survival);survival();drawPrefix();
  const data={
    q8math:{baseline:[230,861,1666,3133,5694],draft:[1175,3884,7485,12268,16076],rounded:[5.1,4.5,4.5,3.9,2.8],tau:8.01},
    q8code:{baseline:[229,868,1649,3253,5462],draft:[955,3092,6010,9919,13116],rounded:[4.2,3.6,3.6,3,2.4],tau:6.50},
    coder:{baseline:[229,686,1068,1681,2713],draft:[802,2078,3442,5429,8314],rounded:[3.5,3,3.2,3.2,3.1],tau:8.09}
  };
  function serving(){
    const d=data[$('serving-task').value],i=+$('serving-concurrency').value;
    const base=d.baseline[i],draft=d.draft[i],ratio=draft/base;
    $('serving-bars').replaceChildren();
    for(const [name,value] of [['AR baseline',base],['DFlash',draft]]){
      const row=document.createElement('div'),label=document.createElement('span'),bar=document.createElement('i');
      label.textContent=`${name} · ${value.toLocaleString('en-US')} tok/s`;
      bar.style.width=(100*value/draft)+'%';row.append(label,bar);$('serving-bars').append(row);
    }
    $('serving-reading').textContent=`${base} → ${draft} tok/s; recalculated speedup ${ratio.toFixed(3)}×; +${((ratio-1)*100).toFixed(1)}%.\nPaper rounded: ${d.rounded[i].toFixed(1)}×; average τ ${d.tau.toFixed(2)} (not a per-concurrency measurement).`;
  }
  for(const id of ['serving-task','serving-concurrency'])$(id).addEventListener('change',serving);serving();
  const mib=x=>(x/2**20).toFixed(2)+' MiB';
  function memory(){
    const x=m.memory({B:+$('memory-batch').value,S:+$('memory-length').value,D:2048,F:5,L:+$('memory-layers').value,G:+$('memory-groups').value,d:128,b:2});
    $('memory-reading').textContent=`Projection weights: ${mib(x.projection)}.\nInput / output activation: ${mib(x.input)} / ${mib(x.output)}.\nPersistent draft KV: ${mib(x.kv)}. Decode B16 projection input + output, per request: 384 KiB.`;
  }
  for(const id of ['memory-length','memory-batch','memory-layers','memory-groups'])$(id).addEventListener('change',memory);memory();
  function cost(){
    const target=+$('target-ms').value,draft=+$('draft-ms').value,verify=+$('verify-ms').value,tau=+$('cost-tau').value;
    $('cost-tau-value').textContent=tau.toFixed(1);
    if(![target,draft,verify,tau].every(x=>Number.isFinite(x)&&x>0)){
      $('cost-reading').textContent='Enter positive latency values to evaluate Equation (1).';return;
    }
    const ratio=m.speedup(target,draft,verify,tau);
    $('cost-reading').textContent=`Assumed cycle ${(draft+verify).toFixed(2)} ms / τ ${tau.toFixed(1)} = ${((draft+verify)/tau).toFixed(2)} ms/token; speedup ${ratio.toFixed(2)}×.${ratio<1?' Slower than the assumed AR baseline.':''}`;
  }
  for(const id of ['target-ms','draft-ms','verify-ms','cost-tau'])$(id).addEventListener('input',cost);cost();
  const theme=$('theme-toggle');
  const setTheme=light=>{document.documentElement.classList.toggle('light',light);theme.textContent=light?'Dark mode':'Light mode';theme.setAttribute('aria-pressed',String(light));};
  theme.addEventListener('click',()=>setTheme(!document.documentElement.classList.contains('light')));
  const contents=$('contents'),mq=matchMedia('(max-width:800px)');
  const adapt=()=>contents.open=!mq.matches;adapt();mq.addEventListener('change',adapt);
  const links=[...document.querySelectorAll('.toc a')];
  links.forEach(a=>a.addEventListener('click',()=>{if(mq.matches)contents.open=false;}));
  const headings=links.map(a=>document.querySelector(a.getAttribute('href')));
  const active=()=>{
    let current=headings[0];for(const h of headings)if(h.getBoundingClientRect().top<130)current=h;
    links.forEach(a=>{const on=a.hash==='#'+current.id;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
  };
  let queued=false;addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(()=>{active();queued=false;});}},{passive:true});active();
  const modal=$('image-modal'),image=$('zoom-image');let opener;
  document.querySelectorAll('.zoom-link').forEach(a=>a.addEventListener('click',e=>{
    if(e.metaKey||e.ctrlKey)return;
    e.preventDefault();opener=a;image.src=a.href;image.alt=a.querySelector('img').alt;
    $('zoom-caption').textContent=a.closest('figure').querySelector('figcaption').textContent;
    $('zoom-original').href=a.href;modal.showModal();
  }));
  $('zoom-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
  modal.addEventListener('close',()=>opener?.focus());
  document.querySelectorAll('.copy-button').forEach(button=>button.addEventListener('click',async()=>{
    const pre=button.closest('.code-frame').querySelector('pre');
    try{await navigator.clipboard.writeText(pre.textContent);button.textContent='Copied';}
    catch{const range=document.createRange();range.selectNodeContents(pre);const sel=getSelection();sel.removeAllRanges();sel.addRange(range);button.textContent='Selected';}
    setTimeout(()=>button.textContent='Copy',1800);
  }));
})();
