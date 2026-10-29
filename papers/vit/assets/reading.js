(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let saved;
  try { saved = localStorage.getItem('paper-reading-theme'); } catch {}
  document.documentElement.classList.toggle('light', saved === 'light');
  const theme = $('theme-toggle');
  function themeLabel() { theme.textContent = document.documentElement.classList.contains('light') ? 'Dark mode' : 'Light mode'; }
  themeLabel();
  theme.addEventListener('click', () => {
    const light = document.documentElement.classList.toggle('light');
    try { localStorage.setItem('paper-reading-theme', light ? 'light' : 'dark'); } catch {}
    themeLabel();
  });
  if (matchMedia('(max-width:800px)').matches) $('contents').open = false;
  const links = [...document.querySelectorAll('.toc a')];
  const observer = new IntersectionObserver(entries => {
    for (const e of entries) if (e.isIntersecting) links.forEach(a => a.classList.toggle('active', a.hash === '#' + e.target.id));
  }, { rootMargin: '-10% 0px -72% 0px' });
  document.querySelectorAll('.reading h2').forEach(h => observer.observe(h));
  links.forEach(a => a.addEventListener('click', () => {
    if (matchMedia('(max-width:800px)').matches) $('contents').open = false;
  }));
  let priorFocus;
  const modal = $('figure-modal');
  document.querySelectorAll('figure img').forEach(img => {
    img.tabIndex = 0;
    img.setAttribute('role','button');
    img.setAttribute('aria-label','Zoom: '+img.alt);
    const open = () => {
      priorFocus=img;
      $('modal-image').src=img.src;
      $('modal-image').alt=img.alt;
      $('modal-caption').textContent=img.closest('figure').querySelector('figcaption').textContent;
      $('modal-source').href=img.src;
      modal.showModal();
    };
    img.addEventListener('click',open);
    img.addEventListener('keydown',e=>{
      if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}
    });
  });
  $('modal-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
  modal.addEventListener('close',()=>priorFocus?.focus({preventScroll:true}));

  let selected=0;
  function patch() {
    const p=Number($('patch-size').value),g=224/p,n=g*g;
    selected=Math.min(selected,n-1);
    $('patch-grid').style.gridTemplateColumns=`repeat(${g},1fr)`;
    const cells=Array.from({length:n},(_,i)=>{
      const cell=document.createElement('span');
      cell.className=i===selected?'focus':'';
      cell.textContent=g<=7?i+1:'';
      return cell;
    });
    $('patch-grid').replaceChildren(...cells);
    $('patch-result').textContent=`224 / ${p} = ${g} patches per side\nN = ${n} image tokens; S = ${n+1} incl. CLS\nEach patch: ${p*p*3} RGB scalars → D channels\nHighlighted: row ${Math.floor(selected/g)+1}, col ${selected%g+1}`;
  }
  $('patch-size').addEventListener('input',patch);
  $('patch-grid').addEventListener('click',e=>{
    const i=[...$('patch-grid').children].indexOf(e.target);
    if(i>=0){selected=i;patch();}
  });
  patch();

  const source=Array.from({length:14},(_,y)=>Array.from({length:14},(_,x)=>(x+y)/26));
  function bilinear(x,y) {
    const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(13,x0+1),y1=Math.min(13,y0+1);
    const a=x-x0,b=y-y0;
    return (1-b)*((1-a)*source[y0][x0]+a*source[y0][x1])+b*((1-a)*source[y1][x0]+a*source[y1][x1]);
  }
  function draw(canvas,g,interpolated) {
    const ctx=canvas.getContext('2d'),size=canvas.width/g;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    for(let y=0;y<g;y++)for(let x=0;x<g;x++){
      const v=interpolated?bilinear(x*13/(g-1),y*13/(g-1)):source[y][x];
      ctx.fillStyle=`rgb(${Math.round(35+130*v)},${Math.round(65+130*v)},${Math.round(85+130*v)})`;
      ctx.fillRect(x*size,y*size,size,size);
      ctx.strokeStyle='#a8bdd133';ctx.strokeRect(x*size,y*size,size,size);
    }
  }
  function position() {
    const g=Number($('target-grid').value);
    draw($('pos-old'),14,false);draw($('pos-new'),g,true);
    $('position-result').textContent=`Spatial position table: [196,D] → [${g*g},D]\nCLS position: [1,D] → [1,D] unchanged\nFinal table: [${g*g+1},D]\nSynthetic scalar channel; bilinear, endpoint-aligned grid illustration.`;
  }
  $('target-grid').addEventListener('input',position);position();

  const config={B:[12,768,12,86e6],L:[24,1024,16,307e6],H:[32,1280,16,632e6]};
  function cost() {
    const [l,d,k,params]=config[$('cost-model').value],r=Number($('cost-resolution').value),p=Number($('cost-patch').value),b=Number($('cost-batch').value);
    if(r%p){$('cost-bars').replaceChildren();$('cost-result').textContent='Invalid grid: image side must be divisible by patch side. No padding is assumed.';return;}
    const n=(r/p)**2,s=n+1,linear=24*s*d*d*l*b,attn=4*s*s*d*l*b,embed=2*n*p*p*3*d*b;
    $('cost-bars').replaceChildren(...[['Linear layers',linear],['Attention QK + AV',attn]].map(([name,v])=>{
      const row=document.createElement('div'),label=document.createElement('span'),bar=document.createElement('i');
      label.textContent=`${name}: ${(v/1e9).toFixed(2)} GFLOPs · ${(100*v/(linear+attn)).toFixed(1)}%`;
      bar.style.width=100*v/(linear+attn)+'%';row.append(label,bar);return row;
    }));
    $('cost-result').textContent=`Grid ${r/p} × ${r/p}; N=${n}; S=${s} incl. CLS\nEncoder major matmuls: ${((linear+attn)/1e9).toFixed(2)} GFLOPs / batch\nPatch projection: ${(embed/1e9).toFixed(2)} GFLOPs / batch\nTotal major matmuls: ${((linear+attn+embed)/1e9).toFixed(2)} GFLOPs / batch\nOne attention map / layer / batch (2 bytes): ${(b*k*s*s*2/2**20).toFixed(3)} MiB\nRounded model weights (2 bytes): ${(params*2/2**20).toFixed(1)} MiB\nNot latency or total peak memory. One MAC = 2 FLOPs.`;
  }
  ['cost-model','cost-resolution','cost-patch','cost-batch'].forEach(id=>$(id).addEventListener('input',cost));
  cost();
})();
