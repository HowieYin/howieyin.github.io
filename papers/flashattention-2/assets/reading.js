(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const theme = byId('theme-toggle');
  let savedTheme = null;
  try { savedTheme = localStorage.getItem('paper-reading-theme'); } catch {}
  document.documentElement.classList.toggle('light', savedTheme === 'light');
  const themeLabel = () => { theme.textContent = document.documentElement.classList.contains('light') ? 'Dark mode' : 'Light mode'; };
  themeLabel();
  theme.addEventListener('click', () => {
    const light = document.documentElement.classList.toggle('light');
    try { localStorage.setItem('paper-reading-theme', light ? 'light' : 'dark'); } catch {}
    themeLabel();
  });
  const contents = byId('contents');
  if (matchMedia('(max-width:800px)').matches) contents.open = false;
  const tocLinks = [...document.querySelectorAll('.toc a')];
  const observed = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      tocLinks.forEach(a => a.classList.toggle('active', a.hash === '#' + entry.target.id));
    }
  }, {rootMargin: '-10% 0px -72% 0px'});
  document.querySelectorAll('.reading h2[id]').forEach(h => observed.observe(h));
  tocLinks.forEach(a => a.addEventListener('click', () => {
    if (matchMedia('(max-width:800px)').matches) contents.open = false;
  }));
  const modal = byId('figure-modal');
  let previousFocus = null;
  document.querySelectorAll('.reading figure img').forEach(img => {
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', 'Zoom: ' + img.alt);
    const open = () => {
      previousFocus = img;
      byId('modal-image').src = img.src;
      byId('modal-image').alt = img.alt;
      byId('modal-caption').textContent = img.closest('figure').querySelector('figcaption')?.textContent || img.alt;
      byId('modal-source').href = img.src;
      modal.showModal();
    };
    img.addEventListener('click', open);
    img.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
    });
  });
  byId('modal-close').addEventListener('click', () => modal.close());
  modal.addEventListener('click', e => { if (e.target === modal) modal.close(); });
  modal.addEventListener('close', () => previousFocus?.focus({preventScroll:true}));
  document.querySelectorAll('.code-frame,.equation').forEach(frame => {
    const button = frame.querySelector('.copy-button');
    if (!button) return;
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(frame.querySelector('code').textContent);
        button.textContent = 'Copied';
      } catch { button.textContent = 'Select text'; }
      setTimeout(() => {button.textContent = 'Copy';}, 1800);
    });
  });
  const training = [
    {name:'GPT3-1.3B / 2k', values:[142,189,196]},
    {name:'GPT3-1.3B / 8k', values:[72,170,220]},
    {name:'GPT3-2.7B / 2k', values:[149,189,205]},
    {name:'GPT3-2.7B / 8k', values:[80,175,225]}
  ];
  function bars(target, labels, values, suffix, denominator) {
    target.replaceChildren(...values.map((v,i)=>{
      const row=document.createElement('div'), text=document.createElement('span'), bar=document.createElement('i');
      text.textContent=labels[i]+' · '+v.toFixed(3)+suffix;
      bar.style.width=Math.min(100,Math.max(0,v/denominator*100))+'%';
      bar.setAttribute('aria-hidden','true');row.append(text,bar);return row;
    }));
  }
  let step=0;
  function softmax() {
    const mode=byId('softmax-case').value;
    const scores=mode==='same'?[4,3,0,1]:mode==='large'?[1000,1001,1004,1003]:[0,1,4,3];
    const values=[1,2,3,4];
    let m=-Infinity,l=0,U=0,alpha=0,oldM=-Infinity;
    for(let b=0;b<Math.min(step,2);b++){
      const start=b*2,newM=Math.max(m,scores[start],scores[start+1]);
      oldM=m;alpha=Math.exp(m-newM);
      const e=scores.slice(start,start+2).map(s=>Math.exp(s-newM));
      l=alpha*l+e[0]+e[1];U=alpha*U+e[0]*values[start]+e[1]*values[start+1];m=newM;
    }
    const fullM=Math.max(...scores), fullE=scores.map(s=>Math.exp(s-fullM));
    const reference=fullE.reduce((a,e,i)=>a+e*values[i],0)/fullE.reduce((a,e)=>a+e,0);
    [...byId('softmax-track').children].forEach((x,i)=>x.classList.toggle('current',i===step));
    byId('softmax-next').disabled=step===3;
    const text=step===0?'Initial: m=−∞, l=0, U=0':`After KV block ${Math.min(step,2)}:\nm=${m.toFixed(3)}, previous m=${oldM===-Infinity?'−∞':oldM.toFixed(3)}\nalpha=exp(old m−new m)=${alpha.toFixed(6)}\nl=${l.toFixed(6)}, U=${U.toFixed(6)}`;
    byId('softmax-result').textContent=text+(step===3?`\nNormalize once: O=U/l=${(U/l).toFixed(9)}\nDense stable reference=${reference.toFixed(9)}; absolute error=${Math.abs(U/l-reference).toExponential(2)}`:'\nValues=[1,2,3,4]; scores=['+scores.join(',')+']');
    bars(byId('softmax-bars'),['l: denominator','U: numerator'],[l,U],'',Math.max(1,U,l));
  }
  byId('softmax-next').addEventListener('click',()=>{step=Math.min(3,step+1);softmax();});
  byId('softmax-reset').addEventListener('click',()=>{step=0;softmax();});
  byId('softmax-case').addEventListener('input',()=>{step=0;softmax();});
  softmax();
  function parallel() {
    const B=+byId('parallel-batch').value,H=+byId('parallel-heads').value,N=+byId('parallel-seq').value,tile=+byId('parallel-tile').value;
    const causal=byId('parallel-mask').value==='causal', slicedQ=byId('parallel-warp').value==='q';
    const T=Math.ceil(N/tile), old=B*H, tasks=old*T, total=T*T, active=causal?T*(T+1)/2:total, masked=causal?T:0, skip=total-active;
    bars(byId('task-bars'),['FA1: batch × heads','FA2: batch × heads × row blocks'],[old,tasks],' CTA',tasks);
    const shown=Math.min(16,T), grid=byId('tile-grid');grid.style.gridTemplateColumns=`repeat(${shown},1fr)`;
    const cells=[];
    for(let i=0;i<shown;i++)for(let j=0;j<shown;j++){
      const e=document.createElement('span');e.className=causal?(j>i?'skip':j===i?'mask':'full'):'full';cells.push(e);
    }
    grid.replaceChildren(...cells);
    const logicalBytes=(slicedQ?1:4)*tile*128*4;
    byId('parallel-result').textContent=`B=${B}, Hq=${H}, N=${N}, Br=Bc=${tile}, T=${T}\nFA1 ${old} tasks; FA2 ${tasks} tasks\nTask supply / 108 SMs: ${(old/108).toFixed(2)} → ${(tasks/108).toFixed(2)} (not occupancy)\nPer head: ${active} computed tiles = ${active-masked} full + ${masked} diagonal masked; ${skip} skipped\n${slicedQ?'Sliced-Q: distinct output rows, no output partial-sum reduction.':'Sliced-K: shared output rows, partial sums must be reduced.'}\nLogical FP32 output accumulator volume (d=128,W=4): ${(logicalBytes/1024).toFixed(0)} KiB, not allocation.`;
  }
  ['parallel-batch','parallel-heads','parallel-seq','parallel-tile','parallel-mask','parallel-warp'].forEach(id=>byId(id).addEventListener('input',parallel));
  parallel();
  function updateTraining(){
    const item=training[+byId('training-model').value], [base,fa1,fa2]=item.values;
    bars(byId('training-bars'),['Without FlashAttention','FlashAttention','FlashAttention-2'],item.values,' TFLOPs/s/GPU',312);
    byId('training-result').textContent=`${item.name}\nFA2/FA1=${(fa2/fa1).toFixed(4)}×; +${((fa2/fa1-1)*100).toFixed(2)}% throughput\nFixed-work time change: −${((1-fa1/fa2)*100).toFixed(2)}%\nFA2 / without FA=${(fa2/base).toFixed(4)}×\nFA2 MFU under paper definition=${(fa2/312*100).toFixed(2)}%; causal attention term not halved.`;
  }
  byId('training-model').addEventListener('input',updateTraining);updateTraining();
})();
