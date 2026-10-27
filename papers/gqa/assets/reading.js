(function(){
  'use strict';
  const $=id=>document.getElementById(id), model=window.GQAModel;
  function grouping(){
    const G=+$('group-count').value,H=32,grid=$('head-grid');
    grid.replaceChildren();
    for(let h=0;h<H;h++){
      const g=model.groupForHead(h,H,G),cell=document.createElement('div');
      cell.className=`query-cell group-${g%2?'odd':'even'}`;
      cell.textContent=`Q${h}`;
      const sub=document.createElement('span');sub.textContent=`→ KV${g}`;cell.append(sub);grid.append(cell);
    }
    $('group-reading').textContent=`H=32，G=${G}，每个 KV 服务 ${H/G} 个 Query；逻辑 cache 为 MHA 的 ${(100*G/H).toFixed(2)}%。${G===1?' MQA 端点。':G===H?' MHA 端点。':''}`;
  }
  const gib=x=>(x/2**30).toFixed(3)+' GiB';
  function cache(){
    const S=+$('cache-length').value,B=+$('cache-batch').value,G=+$('cache-groups').value,p=+$('cache-precision').value,P=+$('cache-tp').value;
    const spec={layers:36,batch:B,length:S,groups:G,dim:128,bytes:p};
    const logical=model.cacheBytes(spec),base=model.cacheBytes({...spec,groups:32}),physical=model.cacheBytes({...spec,groups:model.physicalGroups(G,P)});
    $('length-value').textContent=S.toLocaleString('en-US');
    const bars=$('cache-bars');bars.replaceChildren();
    for(const [name,value] of [['Logical GQA',logical],['MHA reference',base],['Physical global',physical]]){
      const row=document.createElement('div');row.className='cache-row';
      const label=document.createElement('span');label.textContent=name;
      const track=document.createElement('div');track.className='cache-track';track.setAttribute('aria-label',`${name}: ${gib(value)}`);
      const fill=document.createElement('div');fill.className='cache-fill';fill.style.width=(100*value/base)+'%';track.append(fill);row.append(label,track);bars.append(row);
    }
    $('cache-reading').textContent=`逻辑 KV ${gib(logical)}；假想 MHA ${gib(base)}。\nTP=${P}：全球物理 KV ${gib(physical)}；每 rank ${gib(physical/P)}。\n逻辑减少 ${(100*(1-G/32)).toFixed(2)}%；物理 replication factor ${(physical/logical).toFixed(2)}×。${p<2?' 低位格式仅计 payload。':''}`;
  }
  const data=window.GQAData;
  function alpha(){
    const i=+$('alpha-choice').value,f=data.figure5,g=f.GQA[i],m=f.MQA[i],sign=x=>(x>=0?'+':'')+x.toFixed(3);
    $('alpha-reading').textContent=`α=${100*f.alpha[i]}%：GQA ${g.toFixed(3)}，MQA ${m.toFixed(3)}。\n相对 MHA ${f.MHA.toFixed(3)}：${sign(g-f.MHA)} / ${sign(m-f.MHA)}。`;
  }
  function latency(){
    const i=+$('latency-choice').value,f=data.figure6,t=f.time_s[i],delta=100*(t/f.time_s[0]-1);
    $('latency-reading').textContent=`G=${f.group_counts[i]}：${t.toFixed(3)} s；相对 G=1，时间 ${delta>=0?'+':''}${delta.toFixed(2)}%。${i===1?' 此点略低于 G=1；无误差条，不判断统计显著性。':''}`;
  }
  $('group-count').addEventListener('change',grouping);
  for(const id of ['cache-length','cache-batch','cache-groups','cache-precision','cache-tp']) $(id).addEventListener('input',cache);
  $('alpha-choice').addEventListener('change',alpha);$('latency-choice').addEventListener('change',latency);
  grouping();cache();alpha();latency();
  const contents=$('contents'),mq=matchMedia('(max-width:800px)');
  const adapt=()=>contents.open=!mq.matches;adapt();mq.addEventListener('change',adapt);
  const links=[...document.querySelectorAll('.toc a')];
  links.forEach(a=>a.addEventListener('click',()=>{if(mq.matches) contents.open=false;}));
  const headings=links.map(a=>document.querySelector(a.getAttribute('href')));
  const active=()=>{
    let current=headings[0];
    for(const h of headings) if(h.getBoundingClientRect().top<130)current=h;
    links.forEach(a=>{const on=a.getAttribute('href')==='#'+current.id;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
  };
  let queued=false;addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(()=>{active();queued=false;});}},{passive:true});active();
  const modal=$('image-modal'),image=$('zoom-image');let opener;
  document.querySelectorAll('.zoom-link').forEach(a=>a.addEventListener('click',e=>{
    if(e.metaKey||e.ctrlKey)return;
    e.preventDefault();opener=a;image.src=a.href;image.alt=a.querySelector('img').alt;
    $('zoom-caption').textContent=a.closest('figure').querySelector('figcaption').textContent;
    $('zoom-original').href=a.href;modal.showModal();
  }));
  $('zoom-close').addEventListener('click',()=>modal.close());
  modal.addEventListener('click',e=>{if(e.target===modal)modal.close();});
  modal.addEventListener('close',()=>{if(opener)opener.focus();});
  document.querySelectorAll('.copy-button').forEach(button=>button.addEventListener('click',async()=>{
    const text=button.closest('.code-frame').querySelector('pre').textContent;
    try{await navigator.clipboard.writeText(text);button.textContent='Copied';}
    catch{const range=document.createRange();range.selectNodeContents(button.closest('.code-frame').querySelector('pre'));const selection=getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent='Selected';}
    setTimeout(()=>button.textContent='Copy',1800);
  }));
})();
