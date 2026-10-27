(() => {
  'use strict';
  const $=id=>document.getElementById(id), M=window.MoEMath;
  let seed=1;
  const clean=[2,1,.5,0,-.5,-1];
  function noises() {
    let state=seed;
    const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return (state+.5)/4294967296;};
    return clean.map(()=>Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random()));
  }
  function router() {
    const k=+$('router-k').value, amplitude=+$('router-noise').value;
    $('router-k-out').textContent=k; $('router-noise-out').textContent=amplitude.toFixed(2);
    const noisy=clean.map((v,i)=>v+noises()[i]*amplitude), g=M.gate(noisy,k);
    $('router-rows').innerHTML=clean.map((v,i)=>`<div class="expert-row ${g[i]>0?'selected':''}"><span>E${i}</span><span>${v.toFixed(2)} → ${noisy[i].toFixed(2)}</span><span class="gate-track"><span style="width:${g[i]*100}%"></span></span><span>${g[i].toFixed(4)}</span></div>`).join('');
    $('router-readout').textContent=`seed=${seed} · ${k} / 6 专家激活 · gate sum=${g.reduce((a,b)=>a+b,0).toFixed(4)}\n${k===1?'Top-1 重新归一化后权重恒为 1；选中集合不变时，任务损失的 gate-logit 导数为 0。':'选中集合内的 softmax 可以传播任务梯度；集合切换仍然不连续。'}`;
  }
  $('router-k').addEventListener('input',router);
  $('router-noise').addEventListener('input',router);
  $('router-resample').addEventListener('click',()=>{seed++;router();});
  $('router-reset').addEventListener('click',()=>{seed=1;$('router-k').value=2;$('router-noise').value=0;router();});
  function costs() {
    const obj={};for(const key of ['tokens','experts','k','d','h','bytes']) obj[key]=+$('cost-'+key).value;
    const c=M.cost(obj), num=v=>v>0&&v<.01?v.toPrecision(3):v.toLocaleString('en-US',{maximumFractionDigits:2});
    $('cost-readout').textContent=`仅专家矩阵权重：${num(c.expertParams/1e9)} B 总参数\n每 token：${num(c.activeParams/1e6)} M 激活参数；${num(c.mac/1e6)} M MAC\n静态专家权重：${num(c.weightBytes/1e9)} GB（${num(c.weightBytes/2**30)} GiB）\n均衡时每专家平均：${num(c.perExpert)} token assignments / batch\n前向输入+输出全远端上界：${num(c.traffic/1e9)} GB / batch\nGate、主干、bias、optimizer、activation、metadata、backward 均未计入。`;
  }
  for(const key of ['tokens','experts','k','d','h','bytes']) $('cost-'+key).addEventListener('change',costs);
  $('cost-largest').addEventListener('click',()=>{$('cost-experts').value=131072;costs();});
  $('cost-best').addEventListener('click',()=>{$('cost-experts').value=65536;costs();});
  router();costs();
  const dialog=$('figure-dialog');
  document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
    $('zoom-image').src=button.dataset.zoom;
    $('zoom-image').alt=button.querySelector('img').alt;
    $('zoom-source').href=button.dataset.zoom;
    dialog.showModal();
  }));
  $('close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog) dialog.close();});
  const narrow=matchMedia('(max-width:1000px)');
  if(narrow.matches) $('contents').open=false;
  narrow.addEventListener('change',event=>{$('contents').open=!event.matches;});
  const links=[...document.querySelectorAll('.toc a')];
  const observer=new IntersectionObserver(entries=>{
    const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
    if(visible[0]) links.forEach(a=>a.classList.toggle('active',a.hash==='#'+visible[0].target.id));
  },{rootMargin:'0px 0px -65% 0px'});
  document.querySelectorAll('.article h2').forEach(el=>observer.observe(el));
  links.forEach(a=>a.addEventListener('click',()=>{if(matchMedia('(max-width:1000px)').matches) $('contents').open=false;}));
  document.querySelectorAll('.article pre').forEach(pre=>{
    const button=document.createElement('button');button.className='copy-code';button.textContent='Copy';
    button.addEventListener('click',async()=>{
      try {await navigator.clipboard.writeText(pre.querySelector('code')?.textContent||pre.textContent.replace('Copy',''));button.textContent='Copied';}
      catch {button.textContent='Select code';}
    }); pre.append(button);
  });
})();
