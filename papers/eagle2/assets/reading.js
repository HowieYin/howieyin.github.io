(() => {
  'use strict';
  const $=id=>document.getElementById(id),M=window.TreeMath;
  const input=[
    {id:'root',label:'It',parent:null,c:1,x:390,y:35},
    {id:'is',label:'is',parent:'root',c:.6,x:230,y:140},
    {id:'has',label:'has',parent:'root',c:.2,x:570,y:140},
    {id:'a',label:'a',parent:'is',c:.8,x:130,y:250},
    {id:'the',label:'the',parent:'is',c:.1,x:300,y:250},
    {id:'to',label:'to',parent:'has',c:.7,x:490,y:250},
    {id:'a2',label:'a₂',parent:'has',c:.1,x:670,y:250},
    {id:'good',label:'good',parent:'a',c:.7,x:70,y:360},
    {id:'nice',label:'nice',parent:'a',c:.1,x:200,y:360},
    {id:'be',label:'be',parent:'to',c:.6,x:440,y:360},
    {id:'do',label:'do',parent:'to',c:.2,x:570,y:360}
  ];
  function tree(){
    const is=+$('confidence-is').value,has=+$('confidence-has').value,m=+$('tree-budget').value;
    $('confidence-is-out').textContent=is.toFixed(2);$('confidence-has-out').textContent=has.toFixed(2);$('tree-budget-out').textContent=m;
    const nodes=M.scoreTree(input.map(n=>({...n,c:n.id==='is'?is:n.id==='has'?has:n.c})));
    const selected=M.selectTree(nodes,m),keep=new Set(selected.map(n=>n.id)),mask=M.maskFor(selected);
    const previous=$('tree-query').value;
    $('tree-query').innerHTML=selected.map(n=>`<option value="${n.id}">${n.label}</option>`).join('');
    $('tree-query').value=keep.has(previous)?previous:selected[selected.length-1].id;
    const query=selected.findIndex(n=>n.id===$('tree-query').value);
    let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 430" role="img" aria-label="路径置信度树；蓝色为所选节点">';
    for(const n of nodes){
      if(n.parent){const p=nodes.find(p=>p.id===n.parent);
        svg+=`<path d="M${p.x} ${p.y+26}L${n.x} ${n.y-26}" stroke="#6e819d" fill="none"/><text class="edge-label" x="${(p.x+n.x)/2+6}" y="${(p.y+n.y)/2}">${n.c.toFixed(2)}</text>`;}
    }
    for(const n of nodes)svg+=`<g class="tree-node ${keep.has(n.id)?'kept':'removed'}"><rect x="${n.x-49}" y="${n.y-25}" width="98" height="53" rx="6"/><text x="${n.x}" y="${n.y-2}" text-anchor="middle">${n.label}</text><text x="${n.x}" y="${n.y+17}" text-anchor="middle" class="node-value">${n.value.toFixed(3)}</text></g>`;
    $('tree-drawing').innerHTML=svg+'</svg>';
    $('tree-mask').style.setProperty('--nodes',m);
    $('tree-mask').innerHTML='<span>Q / K</span>'+selected.map(n=>`<span>${n.label}</span>`).join('')+
      selected.map((n,i)=>`<button class="query-button" data-query="${n.id}" aria-pressed="${i===query}">${n.label}</button>`+
        mask[i].map((v,j)=>`<span class="mask-cell ${v?'visible':'hidden'} ${i===query?'current':''}" aria-label="${n.label} ${v?'可见':'不可见'} ${selected[j].label}">${v?'●':'·'}</span>`).join('')).join('');
    $('tree-mask').querySelectorAll('[data-query]').forEach(b=>b.addEventListener('click',()=>{$('tree-query').value=b.dataset.query;tree();}));
    const sees=selected.filter((n,j)=>mask[query][j]).map(n=>n.label).join(' / ');
    $('tree-readout').textContent=`保留 ${m} / 11（含 root） · 拓扑展平：${selected.map(n=>n.label).join(' ')}\nQuery ${selected[query].label} 可见：共享已验证 prefix + ${sees}\nPosition id = prefix length + depth；当前相对深度 ${selected[query].depth}。`;
  }
  for(const id of ['confidence-is','confidence-has','tree-budget'])$(id).addEventListener('input',tree);
  $('tree-query').addEventListener('change',tree);
  $('tree-reset').addEventListener('click',()=>{$('confidence-is').value=.6;$('confidence-has').value=.2;$('tree-budget').value=8;tree();});
  function cost(){
    const obj={};for(const key of ['layers','tokens','heads','dim','bytes'])obj[key]=+$('cost-'+key).value;obj.batch=1;
    const bytes=M.cost(obj);
    $('cost-readout').textContent=`新候选临时 target KV：${(bytes/1e6).toFixed(2)} MB / ${(bytes/2**20).toFixed(2)} MiB\n2 × ${obj.layers} layers × ${obj.tokens} nodes × ${obj.heads} KV heads × ${obj.dim} head dim × ${obj.bytes} bytes\n仅新增逻辑 KV；不含 prefix、draft、logits、mask、workspace 或 allocator 峰值。`;
  }
  for(const key of ['layers','tokens','heads','dim','bytes'])$('cost-'+key).addEventListener('change',cost);
  tree();cost();
  const dialog=$('figure-dialog');
  document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
    $('zoom-image').src=button.dataset.zoom;$('zoom-image').alt=button.querySelector('img').alt;
    $('zoom-source').href=button.dataset.zoom;dialog.showModal();
  }));
  $('close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
  const narrow=matchMedia('(max-width:1000px)');
  if(narrow.matches)$('contents').open=false;
  narrow.addEventListener('change',e=>{$('contents').open=!e.matches;});
  const links=[...document.querySelectorAll('.toc a')];
  const headings=[...document.querySelectorAll('.article h2')];
  function activeSection(){
    const current=[...headings].reverse().find(h=>h.getBoundingClientRect().top<=130)||headings[0];
    links.forEach(a=>a.classList.toggle('active',a.hash==='#'+current.id));
  }
  addEventListener('scroll',activeSection,{passive:true});activeSection();
  links.forEach(a=>a.addEventListener('click',()=>{if(narrow.matches)$('contents').open=false;}));
  document.querySelectorAll('.article pre').forEach(pre=>{
    const button=document.createElement('button');button.className='copy-code';button.textContent='Copy';
    button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(pre.querySelector('code').textContent);button.textContent='Copied';}catch{button.textContent='Select code';}});
    pre.append(button);
  });
})();
