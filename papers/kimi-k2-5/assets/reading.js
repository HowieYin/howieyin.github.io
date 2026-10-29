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
      byId('modal-image').src = img.currentSrc || img.src;
      byId('modal-image').alt = img.alt;
      byId('modal-caption').textContent = img.closest('figure').querySelector('figcaption')?.textContent || img.alt;
      byId('modal-source').href = img.currentSrc || img.src;
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
  const number = (id, min, max) => {
    const raw = Number(byId(id).value);
    return Math.round(Math.max(min, Math.min(max, Number.isFinite(raw) ? raw : min)));
  };
  function bars(id, labels, values, scale=100) {
    byId(id).replaceChildren(...values.map((v,i)=>{
      const row=document.createElement('div'), label=document.createElement('span'), bar=document.createElement('i');
      label.textContent=labels[i]+' · '+v.toFixed(1);
      bar.style.width=Math.max(0,Math.min(100,v/scale*100))+'%';
      bar.setAttribute('aria-hidden','true');row.append(label,bar);return row;
    }));
  }
  function vision() {
    const frames=number('frames',1,8192),r=Number(byId('resolution').value),g=Number(byId('temporal').value),contexts=number('contexts',1,100);
    const patches=(r/14)**2, chunks=Math.ceil(frames/g),tokens=chunks*patches/4,kv=tokens*61*(512+64)*2;
    const percent=tokens/262144*100;
    byId('context-bar').style.width=Math.min(100,percent)+'%';
    byId('vision-result').textContent=`${frames} frames · ${patches.toLocaleString()} patches/frame\n${chunks} chunks × ${patches/4} tokens/chunk = ${tokens.toLocaleString()} visual tokens\n${percent.toFixed(2)}% of 256K context${tokens>262144?' · EXCEEDS context limit':''}\nMLA BF16 latent KV: ${(kv/2**30).toFixed(6)} GiB / context\n${contexts} independent contexts: ${(contexts*kv/2**30).toFixed(3)} GiB\n${(262144-tokens).toLocaleString()} tokens remaining before markers/text`;
  }
  ['frames','resolution','temporal','contexts'].forEach(id=>byId(id).addEventListener('input',vision));vision();
  function critical() {
    const main=number('main-steps',0,100),n=number('branches',1,100),s=number('sub-steps',1,100),tail=number('tail-steps',1,200),workers=number('workers',1,100);
    const work=(n-1)*s+tail,slowest=Math.max(n>1?s:0,tail),total=main+work,path=main+slowest,lower=main+Math.max(slowest,Math.ceil(work/workers));
    bars('work-bars',['Total serial work','Unlimited-worker CriticalSteps','Finite-worker lower bound'],[total,path,lower],total);
    byId('critical-result').textContent=`main ${main} + branch work ${work} = ${total} steps total\nCriticalSteps: ${path}; ideal ratio: ${(total/path).toFixed(2)}×\n${workers} workers: completion lower bound ≥ ${lower} steps\nThese are equal-step teaching units, not seconds or measured speedup.`;
  }
  ['main-steps','branches','sub-steps','tail-steps','workers'].forEach(id=>byId(id).addEventListener('input',critical));critical();
  function results() {
    const key=byId('benchmark').value,managed=byId('baseline').value==='managed';
    const values={browse:[60.6,78.4],wide:[72.7,79],house:[41.6,58.3]}[key];
    const managedOption=byId('baseline').querySelector('[value=managed]');
    managedOption.disabled=key!=='browse';
    if (key!=='browse' && managed) byId('baseline').value='raw';
    const base=key==='browse' && byId('baseline').value==='managed'?74.9:values[0],swarm=values[1],delta=swarm-base;
    bars('result-bars',['Baseline','Agent Swarm'],[base,swarm]);
    byId('result-result').textContent=`${base.toFixed(1)} → ${swarm.toFixed(1)}\nAbsolute: +${delta.toFixed(1)} percentage points\nRelative: +${(delta/base*100).toFixed(3)}%\n${key==='wide'?'Metric: Item-F1; not complete-problem accuracy.':key==='house'?'Internal benchmark; not independently reproducible from this report.':'BrowseComp benchmark score; no matched-cost or uncertainty estimate.'}`;
  }
  ['benchmark','baseline'].forEach(id=>byId(id).addEventListener('change',results));results();
  let tocPending=false;
  const updateToc=()=>{
    tocPending=false;
    const headings=[...document.querySelectorAll('.reading h2[id]')];
    const current=headings.filter(h=>h.getBoundingClientRect().top<=150).at(-1);
    if(current)tocLinks.forEach(a=>a.classList.toggle('active',a.hash==='#'+current.id));
  };
  addEventListener('scroll',()=>{if(!tocPending){tocPending=true;requestAnimationFrame(updateToc);}}, {passive:true});

})();
