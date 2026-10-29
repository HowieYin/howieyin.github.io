(() => {
  'use strict';
  const el = id => document.getElementById(id);
  let saved = null;
  try { saved = localStorage.getItem('paper-reading-theme'); } catch {}
  document.documentElement.classList.toggle('light', saved === 'light');
  const themeLabel = () => { el('theme-toggle').textContent = document.documentElement.classList.contains('light') ? 'Dark mode' : 'Light mode'; };
  themeLabel();
  el('theme-toggle').addEventListener('click', () => {
    const light = document.documentElement.classList.toggle('light');
    try { localStorage.setItem('paper-reading-theme', light ? 'light' : 'dark'); } catch {}
    themeLabel();
  });
  const contents = el('contents'), links = [...document.querySelectorAll('.toc a')];
  if (matchMedia('(max-width:800px)').matches) contents.open = false;
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) if (entry.isIntersecting) {
      links.forEach(a => a.classList.toggle('active', a.hash === '#' + entry.target.id));
    }
  }, {rootMargin:'-10% 0px -72% 0px'});
  document.querySelectorAll('.reading h2[id]').forEach(h => observer.observe(h));
  links.forEach(a => a.addEventListener('click', () => {
    if (matchMedia('(max-width:800px)').matches) contents.open = false;
  }));
  const modal = el('figure-modal');
  let previousFocus;
  document.querySelectorAll('.reading figure img').forEach(img => {
    img.tabIndex = 0; img.setAttribute('role','button'); img.setAttribute('aria-label','Zoom: '+img.alt);
    const open = () => {
      previousFocus = img;
      el('modal-image').src = img.src; el('modal-image').alt = img.alt;
      el('modal-caption').textContent = img.closest('figure').querySelector('figcaption').textContent;
      el('modal-source').href = img.src; modal.showModal();
    };
    img.addEventListener('click',open);
    img.addEventListener('keydown',event => { if(['Enter',' '].includes(event.key)) { event.preventDefault(); open(); } });
  });
  el('modal-close').addEventListener('click',() => modal.close());
  modal.addEventListener('click',e => { if(e.target===modal)modal.close(); });
  modal.addEventListener('close',() => previousFocus?.focus({preventScroll:true}));
  document.querySelectorAll('.code-frame,.equation').forEach(frame => {
    const button = frame.querySelector('.copy-button'); if(!button)return;
    button.addEventListener('click',async () => {
      try { await navigator.clipboard.writeText(frame.querySelector('code').textContent); button.textContent='Copied'; }
      catch { button.textContent='Select text'; }
      setTimeout(() => { button.textContent='Copy'; },1800);
    });
  });
  const charts=[];
  const chartTheme=()=>document.documentElement.classList.contains('light');
  function makeChart(id){const c=echarts.init(el(id),null,{renderer:'svg'});charts.push(c);return c;}
  const dsaChart=makeChart('dsa-chart'),qualityChart=makeChart('quality-chart');
  function chartOption(names,series,max,unit){
    const light=chartTheme(),ink=light?'#283440':'#dce1e8',muted=light?'#617181':'#93a0b2';
    return {animation:false,aria:{enabled:true},color:['#82a8d4','#98bcae'],
      textStyle:{fontFamily:'IBM Plex Mono, monospace',color:ink},
      tooltip:{trigger:'axis',valueFormatter:v=>v.toFixed(3)+unit},
      legend:{top:0,textStyle:{color:ink}},grid:{left:58,right:20,top:48,bottom:48,containLabel:true},
      xAxis:{type:'category',data:names,axisLabel:{color:muted,fontSize:12},axisLine:{lineStyle:{color:muted}}},
      yAxis:{type:'value',max,name:unit,axisLabel:{color:muted},splitLine:{lineStyle:{color:light?'#d5dce4':'#29303c'}}},
      series:series.map(s=>({...s,type:'bar',barMaxWidth:58,label:{show:true,position:'top',color:ink,formatter:p=>p.value.toFixed(2)}}))};
  }
  function number(id,min,max){const v=Number(el(id).value);return Math.max(min,Math.min(max,Number.isFinite(v)?v:min));}
  function dsaUI(){
    const L=Math.trunc(number('dsa-length',1,202752)),s=Number(el('dsa-precision').value),k=Math.min(2048,L);
    const kv=78*576*2*L,index=78*128*s*L;
    dsaChart.setOption(chartOption(['缓存 payload'],[{name:'MLA KV BF16',data:[kv/2**30]},{name:'Indexer keys',data:[index/2**30]}],undefined,'GiB'),true);
    el('dsa-result').textContent=`L=${L.toLocaleString('en-US')} · selected=${k.toLocaleString('en-US')}\nMLA KV=${(kv/2**30).toFixed(5)} GiB\nIndexer keys=${(index/2**30).toFixed(5)} GiB\nCombined=${((kv+index)/2**30).toFixed(5)} GiB\nFinal-query core candidates: dense=${L}, sparse=${k}; candidate ratio=${(L/k).toFixed(3)}×\nIndexer still scores all ${L} history positions. Candidate ratio is NOT model speedup.`;
  }
  function asyncUI(){
    const r=number('async-ratio',.1,3),lag=Math.trunc(number('async-lag',0,10)),fresh=lag<=4,inside=r>.8&&r<1.28,weight=fresh&&inside?r:0;
    el('lag-state').textContent=`Lag ${lag}: ${fresh?'保留 trajectory':'丢弃 trajectory'}`;
    el('ratio-state').textContent=`Ratio ${r.toFixed(2)}: ${inside?'区间内':'边界 / 区间外'}`;
    el('loss-state').textContent=`示例权重 f=${weight.toFixed(2)}`;
    el('async-result').textContent=`Toy strict interval: 0.80 < r < 1.28\nToy trajectory keep: lag ≤ 4\nToken weight after both filters=${weight.toFixed(2)}\n${!fresh?'整个轨迹先因版本滞后被丢弃。':!inside?'当前 token 不贡献梯度；未被 clip 到边界。':'当前 token 按 ratio 加权；仍须满足 model / padding mask。'}`;
  }
  function qualityUI(){
    const vue=el('quality-stack').value==='vue',spread=el('quality-pattern').value==='spread';
    const csr=vue?[77.1,74.3]:[71,70.7],isr=vue?[32.7,46.9]:[34.6,39.7];
    qualityChart.setOption(chartOption(['CSR (%)','ISR (%)'],[{name:'GLM-5',data:[csr[0],isr[0]]},{name:'Opus 4.5',data:[csr[1],isr[1]]}],100,'%'),true);
    el('quality-grid').replaceChildren(...Array.from({length:5},(_,i)=>{
      const row=document.createElement('div'),name=document.createElement('span');name.textContent=`Toy ${i+1}`;row.append(name);
      for(let j=0;j<5;j++){const pass=spread?j!==i:i!==0,b=document.createElement('b');b.textContent=pass?'✓':'×';b.className=pass?'check-pass':'check-fail';b.setAttribute('aria-label',`实例${i+1}检查${j+1}: ${pass?'通过':'失败'}`);row.append(b);}
      return row;
    }));
    el('quality-result').textContent=`PAPER · ${vue?'Vue':'React'} · Table 8\nGLM-5 CSR=${csr[0]}%, ISR=${isr[0]}%\nOpus 4.5 CSR=${csr[1]}%, ISR=${isr[1]}%\nSEPARATE TOY · 5 instances × 5 checks\nCSR=20/25=80%; ISR=${spread?'0/5=0%':'4/5=80%'}\nSame total passed checks can imply very different complete-instance reliability.`;
  }
  ['dsa-length','dsa-precision'].forEach(id=>el(id).addEventListener('input',dsaUI));
  ['async-ratio','async-lag'].forEach(id=>el(id).addEventListener('input',asyncUI));
  ['quality-stack','quality-pattern'].forEach(id=>el(id).addEventListener('input',qualityUI));
  el('theme-toggle').addEventListener('click',()=>{dsaUI();qualityUI();});
  window.addEventListener('resize',()=>charts.forEach(c=>c.resize()));
  dsaUI();asyncUI();qualityUI();

})();
