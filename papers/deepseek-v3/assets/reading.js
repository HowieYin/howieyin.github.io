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
  const calc=DeepSeekModels;
  const number=(id,min,max)=>Math.min(max,Math.max(min,Number(byId(id).value)||min));
  function bars(target,labels,values,suffix){
    const max=Math.max(...values,1);
    target.replaceChildren(...values.map((v,i)=>{
      const row=document.createElement('div'),text=document.createElement('span'),bar=document.createElement('i');
      text.textContent=labels[i]+' · '+v.toFixed(3)+suffix;
      bar.style.width=(v/max*100)+'%';bar.setAttribute('aria-hidden','true');row.append(text,bar);return row;
    }));
  }
  function mla(){
    const tokens=number('mla-tokens',1,1048576),batch=number('mla-batch',1,256),bytes=+byId('mla-bytes').value,ranks=+byId('mla-ranks').value;
    const x=calc.cache({tokens,batch,bytes,ranks});
    bars(byId('mla-bars'),['MLA · logical one copy','Materialized same-head K/V','MLA · replicated rank aggregate'],[x.gib,x.materialized/2**30,x.replicated/2**30],' GiB');
    byId('mla-result').textContent=`S=${tokens}, B=${batch}, layers=61, bytes=${bytes}\nMLA: ${x.logical.toLocaleString('en-US')} bytes = ${x.gib.toFixed(6)} GiB\nSame-head materialized / MLA = ${x.ratio.toFixed(4)}×\nReplicated across ${ranks} ranks: ${(x.replicated/2**30).toFixed(6)} GiB aggregate\nEach logical token/layer: 512 latent + 64 RoPE K. No cached Q.`;
    const a=calc.absorbExample();
    byId('mla-algebra').textContent=`Materialized content scores: [${a.naive.map(x=>x.toFixed(5)).join(', ')}]\nAbsorbed scores: [${a.absorbed.map(x=>x.toFixed(5)).join(', ')}]\nMaximum score error: ${a.scoreError.toExponential(2)}\nMaximum output error: ${a.outputError.toExponential(2)}`;
  }
  ['mla-tokens','mla-batch','mla-bytes','mla-ranks'].forEach(id=>byId(id).addEventListener('input',mla));mla();
  function moe(){
    const scores=[.9,.8,.7,.2],bias=[0,0,0,+byId('moe-bias').value],x=calc.route(scores,bias,2);
    byId('expert-grid').replaceChildren(...scores.map((s,i)=>{
      const e=document.createElement('div'),at=x.indices.indexOf(i);
      e.className='expert-box'+(at>=0?' selected':'');
      e.textContent=`Expert ${i+1}\nscore=${s.toFixed(2)} · bias=${bias[i].toFixed(2)}\nselection=${(s+bias[i]).toFixed(2)}\n${at>=0?'SELECTED · weight='+x.weights[at].toFixed(6):'Not selected · weight=0'}`;
      e.style.whiteSpace='pre-line';return e;
    }));
    byId('moe-result').textContent=`Expert 4 bias=${bias[3].toFixed(2)}\nSelected experts: ${x.indices.map(i=>i+1).join(', ')}\nOriginal-score weights: ${x.weights.map(x=>x.toFixed(6)).join(', ')}\nSum=${x.weights.reduce((a,b)=>a+b,0).toFixed(6)} · bias is excluded from weights.`;
  }
  byId('moe-bias').addEventListener('input',moe);
  byId('moe-reset').addEventListener('click',()=>{byId('moe-bias').value=0;moe();});moe();
  function mtp(){
    const a=+byId('mtp-accept').value,c=+byId('mtp-cost').value,tpot=number('mtp-tpot',.1,1000),length=+byId('mtp-length').value,x=calc.speculative(a,c);
    const baseline=length*tpot/1000,modified=baseline/x.speed;
    bars(byId('mtp-bars'),['Normal decode · selected length','One-proposal analytical decode'],[baseline,modified],' s');
    byId('mtp-result').textContent=`a=${a.toFixed(2)}, c=${c.toFixed(2)}\nExpected outputs/round=${x.outputs.toFixed(2)}\nSpeedup=(1+a)/(1+c)=${x.speed.toFixed(6)}×\n${c===a?'Break-even':x.breakEven?'Faster in this toy model':'Slower in this toy model'} · threshold c < a\nTPOT=${tpot} ms; length=${length} tokens\nDecode only: ${baseline.toFixed(3)} s → ${modified.toFixed(3)} s\nTable 9 math length growth: 769 → 1510 = +96.36% (separate experiment).`;
  }
  ['mtp-accept','mtp-cost','mtp-tpot','mtp-length'].forEach(id=>byId(id).addEventListener('input',mtp));mtp();

})();