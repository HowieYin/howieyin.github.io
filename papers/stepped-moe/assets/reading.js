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
  const table9 = {16:[.722,.298,.653,.513],32:[.706,.284,.654,.508],64:[.695,.217,.650,.489]};
  const labels = ['Core-En-0s','Core-En-1s','MMLU-5s','GSM-8k'];
  function bars(target, names, values, format) {
    target.replaceChildren(...values.map((value, i) => {
      const row = document.createElement('div');
      const text = document.createElement('span');
      text.textContent = names[i] + ' · ' + format(value);
      const bar = document.createElement('i');
      bar.style.width = Math.max(0,Math.min(100,value)) + '%';
      bar.setAttribute('aria-hidden','true');
      row.append(text,bar);
      return row;
    }));
  }
  function updateSegment() {
    const segment = Number(byId('segment-size').value);
    const length = Number(byId('output-length').value);
    byId('output-length-value').textContent = length + ' tokens';
    bars(byId('quality-bars'), labels, table9[segment].map(v=>v*100), v=>v.toFixed(1)+'%');
    const groups = Math.ceil(length/segment);
    byId('segment-result').textContent = `${length} tokens / S=${segment}\n${groups} mask groups incl. initial group; ${Math.max(0,groups-1)} later refreshes\nSource: original Table 9. Refresh count: arithmetic, not a latency measurement.`;
  }
  ['segment-size','output-length'].forEach(id=>byId(id).addEventListener('input',updateSegment));
  updateSegment();
  function updateMemory() {
    const budget = Number(byId('active-budget').value);
    const bits = Number(byId('weight-bits').value);
    const length = Number(byId('context-length').value);
    const heads = Number(byId('kv-heads').value);
    const weights = budget*1e9*bits/8;
    const totalWeights = 12e9*bits/8;
    const kvPerToken = 2*56*heads*128*2;
    const kv = length*kvPerToken;
    const combined = weights+kv;
    byId('context-value').textContent = length + ' tokens';
    bars(byId('memory-bars'),['Active weights','KV cache'],[weights/combined*100,kv/combined*100],v=>v.toFixed(1)+'% of subtotal');
    byId('memory-result').textContent = `Active weights: ${(weights/1e9).toFixed(3)} GB / ${(weights/2**30).toFixed(3)} GiB\nBF16 KV: ${(kv/2**30).toFixed(3)} GiB (${(kvPerToken/1024).toFixed(0)} KiB/token)\nSubtotal lower bound: ${(combined/2**30).toFixed(3)} GiB\nFull 12B weights on storage: ${(totalWeights/1e9).toFixed(1)} GB\n${heads===12?'Paper literal 12 KV heads.':'Hypothetical head count; not a configuration reported for Stepped MoE.'}`;
  }
  ['active-budget','weight-bits','context-length','kv-heads'].forEach(id=>byId(id).addEventListener('input',updateMemory));
  updateMemory();
})();
