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
const configurations = {
  '7': {n:6.7e9,d:4096,h:32,l:32,m:11008},
  '13': {n:13e9,d:5120,h:40,l:40,m:13824},
  '33': {n:32.5e9,d:6656,h:52,l:60,m:17920},
  '65': {n:65.2e9,d:8192,h:64,l:80,m:22016}
};
function memory(c,b,t,weightBits,kvBits,heads) {
  const weights=c.n*weightBits/8;
  const perToken=2*c.l*heads*128*kvBits/8;
  const kv=b*t*perToken;
  return {weights,perToken,kv,total:weights+kv};
}
function breakEven(extra,saving,served) {
  return {tokens:saving>0?extra/saving*1e6:Infinity,net:served/1e6*saving-extra};
}
function parameterCount(c) {
  return 2*32000*c.d+c.l*(4*c.d*c.d+3*c.d*c.m+2*c.d)+c.d;
}
// The builder prepends the arithmetic module to this classic-script closure.
function bounded(id,min,max) {
  const n=Number(byId(id).value);
  return Number.isFinite(n)?Math.max(min,Math.min(max,n)):min;
}
function bars(id,labels,values) {
  byId(id).replaceChildren(...values.map((v,i)=>{
    const row=document.createElement('div'),label=document.createElement('span'),bar=document.createElement('i');
    label.textContent=labels[i];bar.style.width=Math.max(0,Math.min(100,v))+'%';
    row.append(label,bar);return row;
  }));
}
function updateMemory() {
  const c=configurations[byId('model-size').value],b=Math.trunc(bounded('batch-size',1,128));
  const t=bounded('context-length',128,2048),w=Number(byId('weight-bits').value),s=Number(byId('kv-bits').value);
  const native=byId('kv-heads').value==='native',heads=native?c.h:Number(byId('kv-heads').value);
  const m=memory(c,b,t,w,s,heads);
  byId('context-value').textContent=t+' tokens';
  bars('memory-bars',['裸权重 · '+(m.weights/2**30).toFixed(3)+' GiB','KV · '+(m.kv/2**30).toFixed(5)+' GiB'],[m.weights/m.total*100,m.kv/m.total*100]);
  byId('memory-result').textContent=`Weights: ${(m.weights/1e9).toFixed(3)} GB / ${(m.weights/2**30).toFixed(3)} GiB\nKV: ${(m.kv/2**30).toFixed(5)} GiB (${(m.perToken/2**20).toFixed(7)} MiB/token/sequence)\nSubtotal: ${(m.total/2**30).toFixed(3)} GiB · B=${b}, T=${t}, L=${c.l}, Hkv=${heads}\n${native?'Original LLaMA v1 MHA head count.':'Hypothetical shared KV heads; no v1 checkpoint quality claim.'}\n${s===16?'FP16/BF16 KV assumption.':'Hypothetical 8-bit KV; quantization metadata excluded.'} Weight precision and KV precision are independent.`;
}
function updateCost() {
  const extra=bounded('extra-cost',0,1e12),saving=bounded('token-saving',-1e4,1e4),served=bounded('served-tokens',0,1e15);
  const c=breakEven(extra,saving,served),recovered=served/1e6*saving;
  const fraction=extra>0?recovered/extra*100:0;
  bars('cost-bars',['额外成本 · '+extra.toLocaleString(),'累计回收 · '+recovered.toLocaleString()+(extra>0?' ('+fraction.toFixed(1)+'%)':'')],[100,fraction]);
  byId('cost-result').textContent=`Break-even: ${Number.isFinite(c.tokens)?c.tokens.toLocaleString()+' generated tokens':'No finite break-even; per-token saving must be positive.'}\nNet saving after ${served.toLocaleString()} tokens: ${c.net.toLocaleString()} currency units\nAnalytical scenario · equal quality and common billing unit assumed.`;
}
['model-size','weight-bits','kv-bits','kv-heads','context-length','batch-size'].forEach(id=>byId(id).addEventListener('input',updateMemory));
['extra-cost','token-saving','served-tokens'].forEach(id=>byId(id).addEventListener('input',updateCost));
updateMemory();updateCost();

})();