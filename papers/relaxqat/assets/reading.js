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
  const headings = [...document.querySelectorAll('.reading h2[id]')];
  let contentsQueued = false;
  const updateContents = () => {
    contentsQueued = false;
    let current = headings[0];
    for (const heading of headings) {
      if (heading.getBoundingClientRect().top <= window.innerHeight * 0.2) current = heading;
      else break;
    }
    tocLinks.forEach(a => a.classList.toggle('active', a.hash === '#' + current?.id));
  };
  const queueContents = () => {
    if (!contentsQueued) { contentsQueued = true; requestAnimationFrame(updateContents); }
  };
  window.addEventListener('scroll', queueContents, {passive:true});
  window.addEventListener('resize', queueContents);
  updateContents();
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
        await navigator.clipboard.writeText((frame.querySelector('.algorithm-copy code') || frame.querySelector('code')).textContent);
        button.textContent = 'Copied';
      } catch { button.textContent = 'Select text'; }
      setTimeout(() => {button.textContent = 'Copy';}, 1800);
    });
  });
  const M=globalThis.RelaxModel;
  const ns='http://www.w3.org/2000/svg';
  function el(name,attrs,text) {
    const node=document.createElementNS(ns,name);
    for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,String(value));
    if (text!==undefined) node.textContent=text;
    return node;
  }
  function line(svg,x1,y1,x2,y2,color='#50657a') {svg.append(el('line',{x1,y1,x2,y2,stroke:color,'stroke-width':1}));}
  function label(svg,x,y,text){svg.append(el('text',{x,y},text));}
  function schedule() {
    const t=Number(byId('schedule-step').value),ms=Number(byId('schedule-ms').value),n=Number(byId('schedule-n').value);
    const svg=byId('schedule-chart');svg.replaceChildren();
    const x=v=>48+v/10000*600,y=v=>185-(v-2)*65;
    for(const b of [2,3,4]){line(svg,48,y(b),648,y(b));label(svg,12,y(b)+4,b+'b');}
    for(const step of [0,2000,4000,6000,8000,10000])label(svg,x(step)-12,215,String(step));
    let d='';
    for(let step=0;step<=10000;step+=20)d+=(step===0?'M':'L')+x(step)+','+y(M.scheduleAt(step,ms,n))+' ';
    svg.append(el('path',{d,fill:'none',stroke:'#74a9cb','stroke-width':3}));
    const b=M.scheduleAt(t,ms,n);
    line(svg,x(t),35,x(t),190,'#d4b280');
    svg.append(el('circle',{cx:x(t),cy:y(b),r:5,fill:'#d4b280'}));
    byId('schedule-result').textContent=`step=${t} · effective b=${b.toFixed(4)} · k=${(2**(b-1)).toFixed(4)}\nhalf-cycle=${ms} steps · ${n} full cycle(s) · δ/α=${(1/2**(b-1)).toFixed(5)}\nHigher b narrows internal rounding cells; it does not specify physical packed bits.`;
  }
  ['schedule-step','schedule-ms','schedule-n'].forEach(id=>byId(id).addEventListener('input',schedule));schedule();
  function grid() {
    const kind=byId('grid-kind').value;
    byId('grid-bits').disabled=kind!=='two';
    const b=kind==='two'?Number(byId('grid-bits').value):(kind==='literal'?1.58:Math.log2(3));
    const svg=byId('grid-chart');svg.replaceChildren();
    const x=v=>48+(v+1.5)/3*600,y=v=>220-(v+1.2)/2.4*190;
    for(const v of [-1,0,1]){line(svg,48,y(v),648,y(v));label(svg,6,y(v)+4,v.toFixed(1));}
    for(const v of [-1.5,-1,0,1,1.5])label(svg,x(v)-12,247,v.toFixed(1));
    label(svg,240,272,'latent θ · α=1');label(svg,49,18,'QDQ w');
    const methods=kind==='two'?['pareto','relax']:['ternary'];
    for(const method of methods){
      let d='';
      for(let i=0;i<=1200;i++){const theta=-1.5+i/400;d+=(i===0?'M':'L')+x(theta)+','+y(M.quantize(theta,1,b,method))+' ';}
      svg.append(el('path',{d,fill:'none',stroke:method==='pareto'?'#d49a8d':'#77b8a2','stroke-width':2}));
    }
    const k=2**(b-1),end=M.quantize(3,1,b,kind==='two'?'relax':'ternary');
    byId('grid-result').textContent=`b=${b.toFixed(7)} · k=${k.toFixed(7)} · δ/α=${(1/k).toFixed(7)}\nRelax endpoint: ±${end.toFixed(7)}α\n${kind==='two'?'Red: ParetoQ SEQ · Green: RelaxSEQ. At b=2: α×{−0.75,−0.25,0.25,0.75}.':kind==='literal'?'Literal 1.58 produces fractional clipped codes; the common amplitude can be absorbed into an export scale.':'Exact log₂3 gives codes {−1,0,1}, QDQ=2α/3 × code.'}`;
  }
  ['grid-bits','grid-kind'].forEach(id=>byId(id).addEventListener('input',grid));grid();
  function memory() {
    const value=id=>Number(byId(id).value);
    const v=M.storage(value('memory-params'),value('memory-bits'),value('memory-width'),value('memory-tokens'),value('memory-layers'),value('memory-heads'),128);
    const bars=byId('memory-bars');bars.replaceChildren();
    const total=v.weights+v.scales+v.kv;
    for(const [name,num] of [['Packed weights',v.weights],['FP16 channel scales',v.scales],['Standard BF16 KV',v.kv]]){
      const row=document.createElement('div'),txt=document.createElement('span'),bar=document.createElement('i');
      txt.textContent=name+' · '+(num/2**30).toFixed(4)+' GiB';
      bar.style.width=Math.max(.4,num/total*100)+'%';bar.setAttribute('aria-hidden','true');
      row.append(txt,bar);bars.append(row);
    }
    byId('memory-result').textContent=`Weights: ${(v.weights/1e9).toFixed(4)} GB / ${(v.weights/2**30).toFixed(4)} GiB\nScales: ${(v.scales/2**20).toFixed(3)} MiB\nKV (${value('memory-tokens')} tokens): ${(v.kv/2**30).toFixed(4)} GiB\nSubtotal: ${(total/2**30).toFixed(4)} GiB\nAll values are conditional arithmetic, not measured runtime memory. Weight bits do not change KV bytes.`;
  }
  ['memory-params','memory-bits','memory-width','memory-tokens','memory-layers','memory-heads'].forEach(id=>byId(id).addEventListener('input',memory));memory();

})();
