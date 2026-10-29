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
  function updateRanking() {
    const k=Number(byId('candidate-count').value),gap=Number(byId('reward-gap').value),bias=Number(byId('reward-bias').value);
    const scores=Array.from({length:k},(_,i)=>(k-1-i)*gap+bias);
    byId('ranking-candidates').replaceChildren(...scores.map((r,i)=>{
      const d=document.createElement('div');d.textContent=String.fromCharCode(65+i)+' · reward '+r.toFixed(2);return d;
    }));
    const pairs=[];let total=0;
    for(let w=0;w<k;w++)for(let l=w+1;l<k;l++){
      const delta=scores[w]-scores[l],p=1/(1+Math.exp(-delta)),loss=Math.log1p(Math.exp(-delta));
      total+=loss;pairs.push(`${String.fromCharCode(65+w)}≻${String.fromCharCode(65+l)} · Δ=${delta.toFixed(2)} · P=${p.toFixed(3)} · loss=${loss.toFixed(3)}`);
    }
    byId('ranking-pairs').textContent=pairs.join('\n');byId('ranking-pairs').style.whiteSpace='pre-line';
    byId('ranking-result').textContent=`K=${k} · pairs=${pairs.length}\nNaive per-pair forwards: ${2*pairs.length} completion evaluations\nGrouped forwards: ${k} completion evaluations (${(2*pairs.length/k).toFixed(1)}× arithmetic reduction)\nPair mean loss: ${(total/pairs.length).toFixed(6)}\nDouble-averaged counterexample: ${(total/pairs.length**2).toFixed(6)}\nCommon bias does not change differences or loss. Strict labels are assumed even when scores tie.`;
  }
  ['candidate-count','reward-gap','reward-bias'].forEach(id=>byId(id).addEventListener('input',updateRanking));updateRanking();
  function updateKL() {
    const p=Number(byId('policy-prob').value),a=byId('sample-action').value,n=Number(byId('response-length').value);
    const q=a==='A'?p:1-p,logratio=Math.log(q/.5),expected=p*Math.log(p/.5)+(1-p)*Math.log((1-p)/.5);
    byId('kl-token-line').replaceChildren(...Array.from({length:n},(_,i)=>{
      const s=document.createElement('span');s.className=logratio<0?'negative':'';s.textContent=`t${i+1}: ${a}\n${logratio.toFixed(4)} nats`;s.style.whiteSpace='pre-line';return s;
    }));
    byId('kl-result').textContent=`Reference P(A)=0.5 · policy P(A)=${p}\nSample sequence log-ratio: ${(n*logratio).toFixed(6)} nats\nSample KL penalty (β=0.02): ${(-.02*n*logratio).toFixed(6)}\nExact expected forward KL: ${(n*expected).toFixed(6)} nats ≥ 0\nSample can be negative; expectation is nonnegative. Sequence sum, not token mean.`;
  }
  ['policy-prob','sample-action','response-length'].forEach(id=>byId(id).addEventListener('input',updateKL));updateKL();
  const states=[
    ['Actor πold generates response; save old logp + mask.','No parameter updates; reference/RM fixed; critic awaits trajectory.'],
    ['Score the same tokens with reference, RM and critic.','Reference→token logp; RM→terminal score; critic→values.'],
    ['Token KL + terminal RM form rewards; compute masked returns / advantages.','Detach rewards / advantages from actor gradient.'],
    ['Actor πnew computes new/old ratio and clipped loss.','Old rollout logp stays fixed; reference is NOT the PPO ratio denominator.'],
    ['Accumulate PPO + γ·ptx actor gradients, then update actor; update critic separately.','RM/reference frozen; pretrain ratio8 and gamma27.8 have different roles.'],
    ['Updated actor becomes next rollout snapshot.','Old policy changes each round; frozen reference remains initial SFT variant.']
  ];
  let step=0;
  function showStep(){
    [...byId('ppo-track').children].forEach((e,i)=>e.classList.toggle('current',i===step));
    byId('ppo-result').textContent=`Step ${step+1}/6\n${states[step][0]}\n${states[step][1]}`;
    byId('model-lifecycle').replaceChildren(...['Actor: '+(step===4?'update':step===5?'new snapshot':'current / old snapshot'),'Reference: frozen initialization','RM 6B: frozen scalar scorer','Critic 6B: '+(step===4?'value update':'values / baseline')].map((text,i)=>{
      const p=document.createElement('p');p.textContent=text;if(i===1||i===2)p.className='frozen';return p;
    }));
    byId('ppo-back').disabled=step===0;byId('ppo-next').disabled=step===5;
  }
  byId('ppo-next').addEventListener('click',()=>{step=Math.min(5,step+1);showStep();});
  byId('ppo-back').addEventListener('click',()=>{step=Math.max(0,step-1);showStep();});
  byId('ppo-reset').addEventListener('click',()=>{step=0;showStep();});showStep();

})();
