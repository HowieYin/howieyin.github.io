(() => {
  'use strict';
  const $=id=>document.getElementById(id), M=window.MLAModel;
  const theme=$('theme-toggle');let saved=null;
  try{saved=localStorage.getItem('paper-reading-theme');}catch{}
  document.documentElement.classList.toggle('light',saved==='light');
  const label=()=>theme.textContent=document.documentElement.classList.contains('light')?'Dark mode':'Light mode';
  label();theme.addEventListener('click',()=>{
    const light=document.documentElement.classList.toggle('light');
    try{localStorage.setItem('paper-reading-theme',light?'light':'dark');}catch{}label();
  });
  if(matchMedia('(max-width:800px)').matches)$('contents').open=false;
  const toc=[...document.querySelectorAll('.toc a')];
  const observer=new IntersectionObserver(es=>{for(const e of es)if(e.isIntersecting)toc.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));},{rootMargin:'-10% 0px -72% 0px'});
  document.querySelectorAll('.reading h2[id]').forEach(h=>observer.observe(h));
  toc.forEach(a=>a.addEventListener('click',()=>{if(matchMedia('(max-width:800px)').matches)$('contents').open=false;}));
  const modal=$('figure-modal');let focus=null;
  document.querySelectorAll('.reading figure img').forEach(img=>{
    img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','Zoom: '+img.alt);
    const open=()=>{focus=img;$('modal-image').src=img.src;$('modal-image').alt=img.alt;$('modal-caption').textContent=img.closest('figure').querySelector('figcaption')?.textContent||img.alt;$('modal-source').href=img.src;modal.showModal();};
    img.addEventListener('click',open);img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});
  });
  $('modal-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
  modal.addEventListener('close',()=>focus?.focus({preventScroll:true}));
  document.querySelectorAll('.copy-button').forEach(button=>button.addEventListener('click',async()=>{
    try{await navigator.clipboard.writeText(button.closest('.code-frame').querySelector('code').textContent);button.textContent='Copied';}catch{button.textContent='Select text';}
  }));
  function bars(target,labels,values,max){
    target.replaceChildren(...values.map((v,i)=>{
      const row=document.createElement('div'),text=document.createElement('span'),bar=document.createElement('i');
      text.textContent=labels[i]+' · '+v.toFixed(4)+' GiB';bar.style.width=Math.min(100,v/max*100)+'%';row.append(text,bar);return row;
    }));
  }
  function cache(){
    const S=+$('cache-seq').value,B=+$('cache-batch').value,bits=+$('cache-bits').value,budget=+$('cache-budget').value;
    const kinds=['mla','67b','expanded'],v=kinds.map(k=>M.cacheBytes(k,S,B,bits)/2**30);
    bars($('cache-bars'),['V2 compressed latent','DeepSeek-67B GQA8','V2 expanded eager'],v,v[2]);
    const cap=Math.floor(budget*2**30/M.cacheBytes('mla',S,1,bits));
    $('cache-result').textContent=`S=${S}, batch=${B}, nominal bits=${bits}\nV2 MLA=${v[0].toFixed(4)} GiB; 67B GQA8=${v[1].toFixed(4)} GiB; expanded V2=${v[2].toFixed(4)} GiB\nSame-precision MLA reduction vs 67B: 82.2368%\nKV-only budget ${budget} GiB → MLA maximum batch ${cap}\nHeadline reconstruction: V2 6-bit vs 67B BF16 = 93.3388% nominal byte reduction.`;
  }
  ['cache-seq','cache-batch','cache-bits','cache-budget'].forEach(id=>$(id).addEventListener('input',cache));cache();
  let step=0;
  function absorb(){
    const second=$('absorb-query').value==='b';
    const t={c:[[1,2],[-1,1],[2,-1]],kr:[.3,-.1,.5],q:second?[-.2,1]:[.6,-.4],qr:second?-.5:.8,uk:[[1,.5],[-.2,.7]],uv:[[.5,-.3],[.8,.2]]};
    const a=M.absorbedAttention(t),e=M.expandedAttention(t),fmt=x=>'['+x.map(v=>v.toFixed(6)).join(', ')+']';
    [...$('absorb-track').children].forEach((li,i)=>li.classList.toggle('current',i===step));
    $('absorb-next').disabled=step===3;
    $('latent-grid').replaceChildren(...t.c.map((c,i)=>{const p=document.createElement('p');p.textContent=`t${i}: c=${fmt(c)}, kR=${t.kr[i]}`;return p;}));
    const lines=['Persistent cache: c1,c2,c3 + rotated kR. UK and UV are model weights.'];
    if(step>=1)lines.push('Effective query UKᵀqC='+fmt(a.effective));
    if(step>=2)lines.push('Scores='+fmt(a.scores)+'\nSoftmax weights='+fmt(a.weights)+'\nLatent aggregate z='+fmt(a.z));
    if(step>=3)lines.push('Absorbed output='+fmt(a.out)+'\nExpanded reference='+fmt(e.out)+'\nMaximum absolute error='+Math.max(...a.out.map((v,i)=>Math.abs(v-e.out[i]))).toExponential(2));
    $('absorb-result').textContent=lines.join('\n');
  }
  $('absorb-next').addEventListener('click',()=>{step=Math.min(3,step+1);absorb();});
  $('absorb-reset').addEventListener('click',()=>{step=0;absorb();});
  $('absorb-query').addEventListener('input',()=>{step=0;absorb();});absorb();
  function routing(){
    const D=8,N=160,K=6,limit=+$('route-limit').value,spread=$('route-case').value==='spread';
    const raw=Array.from({length:N},(_,i)=>1+((i*7)%19)/20);
    for(let i=0;i<6;i++)raw[spread?i*20:i]=10-i*.5;
    const sum=raw.reduce((a,b)=>a+b,0),scores=raw.map(v=>v/sum);
    const r=M.route(scores,D,K,limit),unrestricted=M.route(scores,D,K,D),fan=new Set(r.selected.map(i=>Math.floor(i/20))).size;
    const grid=$('expert-grid');grid.replaceChildren(...Array.from({length:D},(_,g)=>{
      const group=document.createElement('div'),title=document.createElement('p'),cells=document.createElement('div');
      group.className='expert-group'+(r.allowed.includes(g)?' allowed':'');title.textContent='Device '+g;cells.className='expert-cells';
      for(let i=g*20;i<(g+1)*20;i++){const cell=document.createElement('span');cell.textContent=i;cell.title=`Expert ${i}: affinity ${scores[i].toFixed(5)}`;if(r.selected.includes(i))cell.className='selected';cells.append(cell);}
      group.append(title,cells);return group;
    }));
    const retained=r.selected.filter(i=>unrestricted.selected.includes(i)).length;
    $('route-result').textContent=`Allowed device groups: ${r.allowed.join(', ')}\nSelected experts: ${r.selected.join(', ')}\nActual device fan-out=${fan} ≤ ${limit}; routed expert assignments=${K}\nOverlap with unrestricted top-6: ${retained}/6\nShared experts=2; total active experts=8. Affinities are synthetic, no latency measured.`;
  }
  ['route-limit','route-case'].forEach(id=>$(id).addEventListener('input',routing));routing();
})();
