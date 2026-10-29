(() => {
  'use strict';
  const byId=id=>document.getElementById(id), api=globalThis.LlamaMath;
  const theme=byId('theme-toggle');
  try{document.documentElement.classList.toggle('light',localStorage.getItem('paper-reading-theme')==='light');}catch{}
  const label=()=>theme.textContent=document.documentElement.classList.contains('light')?'Dark mode':'Light mode';
  theme.addEventListener('click',()=>{
    const light=document.documentElement.classList.toggle('light');
    try{localStorage.setItem('paper-reading-theme',light?'light':'dark');}catch{}
    label();
  });label();
  const contents=byId('contents'), toc=[...document.querySelectorAll('.toc a')];
  if(matchMedia('(max-width:800px)').matches)contents.open=false;
  const observer=new IntersectionObserver(entries=>{
    entries.forEach(e=>{if(e.isIntersecting)toc.forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));});
  },{rootMargin:'-10% 0px -72% 0px'});
  document.querySelectorAll('.reading h2[id]').forEach(h=>observer.observe(h));
  toc.forEach(a=>a.addEventListener('click',()=>{if(matchMedia('(max-width:800px)').matches)contents.open=false;}));
  const modal=byId('figure-modal');let focus=null;
  document.querySelectorAll('.reading figure img').forEach(img=>{
    img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','Zoom: '+img.alt);
    const open=()=>{focus=img;byId('modal-image').src=img.src;byId('modal-image').alt=img.alt;
      byId('modal-caption').textContent=img.closest('figure').querySelector('figcaption')?.textContent||img.alt;
      byId('modal-source').href=img.src;modal.showModal();};
    img.addEventListener('click',open);img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}});
  });
  byId('modal-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
  modal.addEventListener('close',()=>focus?.focus({preventScroll:true}));
  document.querySelectorAll('.copy-button').forEach(b=>b.addEventListener('click',async()=>{
    try{await navigator.clipboard.writeText(b.closest('.code-frame').querySelector('code').textContent);b.textContent='Copied';}
    catch{b.textContent='Select text';}
  }));
  function margin(){
    const delta=+byId('margin-delta').value,m=+byId('margin-value').value,r=api.ranking(delta,m);
    byId('margin-result').textContent=`Δ=${delta.toFixed(1)}, m=${m.toFixed(3)}\nloss=${r.loss.toFixed(6)}\n∂loss/∂Δ=${r.gradient.toFixed(6)}\nGradient descent increases Δ; absolute scores remain uncalibrated.`;
    const x=v=>40+(v+4)*50,y=v=>195-v/8*175;
    let path='';
    for(let i=0;i<=100;i++){const d=-4+i/10;path+=(i?'L':'M')+x(d).toFixed(2)+' '+y(api.ranking(d,m).loss).toFixed(2);}
    byId('margin-curve').setAttribute('d',path);
    byId('margin-dot').setAttribute('cx',x(delta));byId('margin-dot').setAttribute('cy',y(r.loss));
  }
  ['margin-delta','margin-value'].forEach(id=>byId(id).addEventListener('input',margin));margin();
  let step=0;
  function gatt(){
    const segments=step===0?['U1 + instruction','A1','U2 + instruction','A2','U3 + instruction','A3']:
      ['U1 + instruction','A1','U2','A2','U3','A3'];
    byId('gatt-tokens').replaceChildren(...segments.map((text,i)=>{
      const token=document.createElement('div'),name=document.createElement('b'),tag=document.createElement('span');
      name.textContent=text;tag.textContent=step===2?(i===5?'loss=1 · target':'loss=0 · context'):(step===0?'teacher generation':'student causal context');
      token.className=step===2?(i===5?'loss-on':'loss-off'):'';token.append(name,tag);return token;
    }));
    [...byId('gatt-track').children].forEach((e,i)=>e.classList.toggle('current',i===step));
    byId('gatt-next').disabled=step===2;
    byId('gatt-result').textContent=[
      'Teacher: instruction injected at each user turn. Generate instruction-consistent assistant answers.',
      'Student: instruction retained only in U1; prior generated assistant messages remain in the prefix.',
      'Loss mask: U1/A1/U2/A2/U3=0; A3=1.\nCausal attention still sees all earlier turns. Loss mask ≠ attention mask.'
    ][step];
  }
  byId('gatt-next').addEventListener('click',()=>{step=Math.min(2,step+1);gatt();});
  byId('gatt-reset').addEventListener('click',()=>{step=0;gatt();});gatt();
  function kv(){
    const c=+byId('kv-context').value,b=+byId('kv-batch').value,h=+byId('kv-heads').value,
      bytes=+byId('kv-bytes').value,tp=+byId('kv-tp').value,r=api.kvCapacity(c,b,h,bytes,tp),
      mha=api.kvCapacity(c,b,64,bytes,tp);
    const labels=['Logical compact KV','Per rank KV','All ranks with replication'],
      values=[r.logicalGiB,r.perRankGiB,r.physicalGiB];
    byId('kv-bars').replaceChildren(...values.map((v,i)=>{
      const row=document.createElement('div'),t=document.createElement('span'),bar=document.createElement('i');
      t.textContent=labels[i]+' · '+v.toFixed(5)+' GiB';bar.style.width=v/Math.max(mha.physicalGiB,r.physicalGiB)*100+'%';
      bar.setAttribute('aria-hidden','true');row.append(t,bar);return row;
    }));
    byId('kv-result').textContent=`B=${b}, C=${c}, Hkv=${h}, element=${bytes} bytes, TP=${tp}\nLogical compact KV: ${r.logicalGiB.toFixed(5)} GiB\nPer rank: ${r.perRankGiB.toFixed(5)} GiB (${r.localHeads} KV head/rank)\nAll ranks: ${r.physicalGiB.toFixed(5)} GiB; replication=${r.replication}×\nMHA counterfactual: ${mha.logicalGiB.toFixed(5)} GiB logical\n${c>4096?'Beyond checkpoint context: capacity extrapolation only.':'Within original checkpoint context capacity.'}`;
  }
  ['kv-context','kv-batch','kv-heads','kv-bytes','kv-tp'].forEach(id=>byId(id).addEventListener('input',kv));kv();
})();
