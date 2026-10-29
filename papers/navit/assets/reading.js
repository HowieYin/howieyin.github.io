function pack(lengths,capacity){
  if(!Number.isInteger(capacity)||capacity<1||lengths.some(n=>!Number.isInteger(n)||n<1||n>capacity))throw new RangeError('Every image must fit a sequence');
  const bins=[];
  for(const n of lengths){let bin=bins.find(b=>b.reduce((a,x)=>a+x,0)+n<=capacity);if(!bin){bin=[];bins.push(bin);}bin.push(n);}
  return bins;
}
function mask(ids){return ids.map(i=>ids.map(j=>Number(i>=0&&i===j)));}
function coordinates(row,col,rows,cols){return {x:col,y:row,rx:col/cols,ry:row/rows};}
function cost(s,d,l,heads,bytes=2){
  const projection=8*s*d*d,attention=4*s*s*d,mlp=16*s*d*d;
  return {projection,attention,mlp,total:(projection+attention+mlp)*l,mapMiB:heads*s*s*bytes/1048576};
}
function schedule(n,mu,tau){return .2+.6/(1+Math.exp(-(n-mu)/tau));}

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


  document.querySelectorAll('.copy-button').forEach(button=>button.addEventListener('click',async()=>{
    try{await navigator.clipboard.writeText(button.closest('.code-frame').querySelector('code').textContent);button.textContent='Copied';}
    catch{button.textContent='Select code to copy';}
  }));
})();

(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const colours=['#a4c8e5','#a9cdbb','#d9b697','#b9b0d8','#93bec5','#c7bf90'];
  function packing(){
    const cap=+$('pack-capacity').value;
    const lengths=$('pack-lengths').value.trim().split(/[\s,]+/).map(Number);
    let bins;
    try{bins=pack(lengths,cap);}catch{
      $('pack-result').textContent=`每张图片需为 1–${cap} 的整数 tokens；以逗号或空格分隔。`;
      $('pack-bins').replaceChildren();$('mask-grid').replaceChildren();return;
    }
    const rows=[];
    let image=0;
    for(const bin of bins){
      const row=document.createElement('div');row.className='packed-row';
      for(const length of bin){
        const span=document.createElement('span');
        span.style.width=100*length/cap+'%';span.style.background=colours[image%6];
        span.textContent=String.fromCharCode(65+image)+' · '+length;
        row.append(span);image++;
      }
      const spare=cap-bin.reduce((a,b)=>a+b,0);
      if(spare){const span=document.createElement('span');span.className='padding';span.style.width=100*spare/cap+'%';span.textContent='pad '+spare;row.append(span);}
      rows.push(row);
    }
    $('pack-bins').replaceChildren(...rows);
    const used=lengths.reduce((a,b)=>a+b,0),total=cap*bins.length;
    $('pack-result').textContent=`First-fit：${lengths.length} 张图 → ${bins.length} 个长度 ${cap} 的容器\n有效 tokens ${used} / ${total}；padding ${(100*(total-used)/total).toFixed(2)}%\n这里只演示有限样本，不复现论文 <2% padding 统计。`;
    // A,A,A,B,B,C,pad,pad: fixed miniature reveals the mask without a huge matrix.
    const ids=[0,0,0,1,1,2,-1,-1],allowed=mask(ids);
    const tokens=['A','A','A','B','B','C','pad','pad'];
    $('mask-grid').style.gridTemplateColumns='repeat(9,minmax(0,1fr))';
    const cells=[];
    for(let y=-1;y<8;y++)for(let x=-1;x<8;x++){
      const c=document.createElement('span');
      if(y===-1||x===-1){c.textContent=y===-1?(x<0?'Q/K':tokens[x]):tokens[y];c.className='matrix-label';}
      else{c.className=allowed[y][x]?'allowed':'blocked';c.textContent=allowed[y][x]?'1':'·';c.title=`query ${y}, key ${x}: ${allowed[y][x]?'same image':'masked'}`;}
      cells.push(c);
    }
    $('mask-grid').replaceChildren(...cells);
  }
  ['pack-lengths','pack-capacity'].forEach(id=>$(id).addEventListener('input',packing));packing();

  function costs(){
    const s=+$('cost-sequence').value,d=+$('cost-width').value,l=+$('cost-depth').value,h=+$('cost-heads').value;
    if(d%h){$('cost-bars').replaceChildren();$('cost-result').textContent='Invalid heads：hidden D 必须能被 heads 整除，请选择 16 heads。';return;}
    const n=+$('cost-image').value,m=Math.floor(s/n),rem=s-m*n;
    const c=cost(s,d,l,h),semantic=4*d*l*(m*n*n+rem*rem);
    const bars=[['QKV + output',c.projection*l],['QKᵀ + AV · dense',c.attention*l],['MLP · ratio 4',c.mlp*l]];
    $('cost-bars').replaceChildren(...bars.map(([name,v])=>{
      const row=document.createElement('div'),label=document.createElement('span'),bar=document.createElement('i');
      label.textContent=`${name}: ${(v/1e9).toFixed(3)} GFLOPs · ${(100*v/c.total).toFixed(1)}%`;
      bar.style.width=100*v/c.total+'%';row.append(label,bar);return row;
    }));
    $('cost-result').textContent=`S=${s}, D=${d}, L=${l}, heads=${h}\nEncoder major matmuls：${(c.total/1e9).toFixed(3)} GFLOPs / container\n一层一份显式 attention map · 2 bytes：${c.mapMiB.toFixed(3)} MiB\n同图 attention 下界示意：${(semantic/1e9).toFixed(3)} GFLOPs（${m}×${n} tokens${rem?` + ${rem}`:''}）\n稠密 attention / 块内 attention：${(c.attention*l/semantic).toFixed(2)}×\n下界把余量当作一张独立有效图；不含 padding。不是 kernel latency 或实际物理峰值。`;
  }
  ['cost-sequence','cost-width','cost-depth','cost-heads','cost-image'].forEach(id=>$(id).addEventListener('input',costs));costs();

  function dropping(){
    const mode=$('drop-mode').value,mu=+$('drop-mid').value,tau=(mode==='increasing'?1:-1)*+$('drop-speed').value;
    const canvas=$('drop-chart'),w=canvas.clientWidth,h=Math.max(220,Math.round(w*.438));
    canvas.width=w*2;canvas.height=h*2;
    const ctx=canvas.getContext('2d');ctx.scale(2,2);
    ctx.clearRect(0,0,w,h);ctx.fillStyle='#0c1420';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='#4b6279';ctx.lineWidth=1;
    for(const r of [.2,.5,.8]){
      const y=h-40-r*(h-60);ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(w-15,y);ctx.stroke();
      ctx.fillStyle='#b4c7da';ctx.font='12px monospace';ctx.fillText(r.toFixed(1),12,y+4);
    }
    ctx.strokeStyle='#a8c9e7';ctx.lineWidth=3;ctx.beginPath();
    for(let i=0;i<=200;i++){
      const n=i/200,rate=mode==='fixed'?.5:schedule(n,mu,tau);
      const x=48+n*(w-65),y=h-40-rate*(h-60);i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    }
    ctx.stroke();ctx.fillStyle='#b4c7da';ctx.fillText('0',46,h-10);ctx.fillText('images seen / N',w-145,h-10);
    const begin=mode==='fixed'?.5:schedule(0,mu,tau),end=mode==='fixed'?.5:schedule(1,mu,tau);
    $('drop-result').textContent=`ρmin=0.2, ρmax=0.8；μ/N=${mu.toFixed(2)}, τ/N=${mode==='fixed'?'—':tau.toFixed(2)}\n起点 drop ${(begin*100).toFixed(1)}% → 终点 ${(end*100).toFixed(1)}%\n每图原有 196 tokens：预期保留 ${(196*(1-begin)).toFixed(1)} → ${(196*(1-end)).toFixed(1)}\n横轴是累计 images seen / N；N 为参照总图片数，不是训练 steps。\n曲线来自原始 Equation (1)；未模拟准确率或固定 wall-time 下的图片数。`;
  }
  ['drop-mode','drop-mid','drop-speed'].forEach(id=>$(id).addEventListener('input',dropping));
  window.addEventListener('resize',dropping);dropping();
})();
