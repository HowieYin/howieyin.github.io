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
  const modalImage = byId('modal-image');
  const viewport = document.createElement('div');
  viewport.className = 'modal-viewport';
  viewport.tabIndex = 0;
  viewport.setAttribute('aria-label', 'Image detail; scroll to explore when zoomed');
  modalImage.before(viewport);
  viewport.append(modalImage);
  const tools = document.createElement('div');
  tools.className = 'modal-tools';
  const controls = document.createElement('div');
  controls.className = 'modal-zoom-controls';
  modal.append(tools);
  tools.append(byId('modal-source'), controls);
  function zoomButton(id, label, title, action) {
    const button = document.createElement('button');
    button.id = id;
    button.type = 'button';
    button.textContent = label;
    button.setAttribute('aria-label', title);
    button.title = title;
    button.addEventListener('click', action);
    controls.append(button);
    return button;
  }
  let zoom = 1;
  const zoomOut = zoomButton('modal-zoom-out', '−', 'Zoom out', () => setZoom(zoom / 2));
  zoomButton('modal-fit', 'Fit', 'Fit entire image to window', () => setZoom(1));
  const zoomIn = zoomButton('modal-zoom-in', '+', 'Zoom in; scroll to explore details', () => setZoom(zoom * 2));
  function applyZoom() {
    viewport.classList.toggle('is-zoomed', zoom > 1);
    if (zoom === 1) {
      modalImage.style.removeProperty('width');
      modalImage.style.removeProperty('height');
    } else if (modalImage.naturalWidth && modalImage.naturalHeight) {
      const ratio = modalImage.naturalWidth / modalImage.naturalHeight;
      const fitHeight = Math.min(viewport.clientHeight, viewport.clientWidth / ratio);
      modalImage.style.width = `${fitHeight * ratio * zoom}px`;
      modalImage.style.height = `${fitHeight * zoom}px`;
    }
    zoomOut.disabled = zoom === 1;
    zoomIn.disabled = zoom === 8;
  }
  function setZoom(value) {
    zoom = Math.max(1, Math.min(8, value));
    applyZoom();
    viewport.scrollTo(0, 0);
  }
  modalImage.addEventListener('load', applyZoom);
  new ResizeObserver(() => { if (modal.open) applyZoom(); }).observe(viewport);
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
      setZoom(1);
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

  function number(id,min,max) {
    const v=Number(byId(id).value);
    return Math.max(min,Math.min(max,Number.isFinite(v)?v:min));
  }
  function bars(id,labels,values,suffix) {
    const max=Math.max(1,...values);
    byId(id).replaceChildren(...values.map((v,i)=>{
      const r=document.createElement('div'),t=document.createElement('span'),b=document.createElement('i');
      t.textContent=labels[i]+' · '+v.toFixed(3)+suffix;
      b.style.width=(v/max*100)+'%';r.append(t,b);return r;
    }));
  }
  function clip(){
    const s=number('clip-score',20,1000),tau=number('clip-threshold',30,200),g=s>tau?tau/s:1;
    const scheme=byId('clip-scheme').value;
    const qc=scheme==='mla'?Math.sqrt(g):1,kc=qc,qr=scheme==='none'?1:g;
    const content=s*.75*qc*kc,rotary=s*.25*qr,total=content+rotary;
    byId('clip-score-label').textContent=s;
    byId('clip-threshold-label').textContent=tau;
    bars('clip-bars',['Original score','After fixed-input scaling'],[s,total],' logit');
    byId('clip-result').textContent=`S=${s}; τ=${tau}; γ=${g.toFixed(6)}\nqC=${qc.toFixed(6)}; kC=${kc.toFixed(6)}; qR=${qr.toFixed(6)}; shared kR=1\nContent=${content.toFixed(3)}; rotary=${rotary.toFixed(3)}; total=${total.toFixed(3)}\n${total>tau+.00001?'Above threshold: content was not scaled.':g===1?'Below threshold: no intervention.':'Fixed-input identity holds: total=γS. Not a future-input guarantee.'}`;
  }
  ['clip-score','clip-threshold','clip-scheme'].forEach(id=>byId(id).addEventListener('input',clip));clip();
  function memory(){
    const N=number('memory-seq',4096,131072),B=number('memory-batch',1,128),bytes=number('memory-precision',1,2),H=number('memory-heads',64,128);
    const latent=B*N*61*576*bytes/2**30,expanded=B*N*61*H*320*bytes/2**30;
    const fullBF=1.043e12*2/2**30,fullFP=1.043e12/2**30,active=32.6e9*2/2**30;
    bars('memory-bars',['Optimized latent cache','Expanded per-head K/V cache'],[latent,expanded],' GiB');
    byId('memory-result').textContent=`B=${B}; N=${N}; H=${H}; bytes=${bytes}\nLatent cache=${latent.toFixed(6)} GiB; expanded=${expanded.toFixed(6)} GiB\nExpanded / latent=${(expanded/latent).toFixed(4)}×\nAll weights: BF16=${fullBF.toFixed(2)} GiB; FP8≈${fullFP.toFixed(2)} GiB\nActive weights / token (BF16)≈${active.toFixed(2)} GiB; does not determine residency.\nCache results omit scales, pages, replication and workspace.`;
  }
  ['memory-seq','memory-batch','memory-precision','memory-heads'].forEach(id=>byId(id).addEventListener('input',memory));memory();
  function budget(){
    const rollout=number('budget-rollout',1,3600),update=number('budget-update',1,3600),sw=number('budget-switch',0,600),limit=number('budget-limit',4096,16384),attempt=number('budget-output',1024,24576),total=rollout+update+sw;
    const elems=[rollout,update,sw].map((t,i)=>{const e=document.createElement('span');e.style.width=(t/total*100)+'%';e.title=['Rollout','Update','Switch'][i]+' '+t+'s';return e;});
    byId('budget-bars').replaceChildren(...elems);byId('budget-output-label').textContent=attempt;
    byId('budget-result').textContent=`Iteration=${total}s; useful rollout+update share=${((rollout+update)/total*100).toFixed(3)}%\nSwitch share=${(sw/total*100).toFixed(3)}%; ideal zero-switch speedup=${(total/(rollout+update)).toFixed(4)}×\nAttempt=${attempt}; allowed=${limit}; emitted=${Math.min(attempt,limit)} tokens\n${attempt>limit?'TRUNCATED: paper assigns a penalty; exact value undisclosed. Incomplete tool calls require explicit handling.':'Within budget; completion is still not guaranteed.'}\nTimes are illustrative inputs, not measured K2 performance.`;
  }
  ['budget-rollout','budget-update','budget-switch','budget-limit','budget-output'].forEach(id=>byId(id).addEventListener('input',budget));budget();
})();
