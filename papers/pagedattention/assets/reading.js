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
  const states = [
    {name:'Prefill 7',a:[7,1],b:[],data:{7:[1,2,3,4],1:[5,6,7]},note:'已有 7 个 KV；表 [7,1]，尾块已用 3/4。首个输出 token 已采样，但还没有自己的 KV。'},
    {name:'Append 8',a:[7,1],b:[],data:{7:[1,2,3,4],1:[5,6,7,8]},note:'写入第 8 个 KV，原尾块恰好填满；没有新分配。'},
    {name:'Append 9',a:[7,1,3],b:[],data:{7:[1,2,3,4],1:[5,6,7,8],3:[9]},note:'第 9 个 KV 需要新物理块 3；映射 [7,1,3]。'},
    {name:'Fork',a:[7,1,3],b:[7,1,3],data:{7:[1,2,3,4],1:[5,6,7,8],3:[9]},note:'Fork 只复制表；块 7、1、3 的 refcount 都变为 2。'},
    {name:'A: COW',a:[7,1,5],b:[7,1,3],data:{7:[1,2,3,4],1:[5,6,7,8],3:[9],5:[9,'A10']},note:'A 写入前发现块 3 共享且未满：复制 3→5，再写 A10。块 3 ref=1；块 5 ref=1；共享满块 7/1 不复制。'},
    {name:'B: append',a:[7,1,5],b:[7,1,3],data:{7:[1,2,3,4],1:[5,6,7,8],3:[9,'B10'],5:[9,'A10']},note:'B 独占块 3，可直接写 B10；没有第二次 COW。A/B 总计 4 个物理块、16 槽、12 个实际 KV。'},
    {name:'Free A',a:[],b:[7,1,3],data:{7:[1,2,3,4],1:[5,6,7,8],3:[9,'B10'],5:[]},note:'设备完成后释放 A：块 5 ref=0 回收；块 7/1 仍被 B 引用，不能回收。'},
    {name:'Free B',a:[],b:[],data:{7:[],1:[],3:[],5:[]},note:'设备完成后释放 B：所有引用归零，可复用。这是无持久 APC 的生命周期示例。'}
  ];
  let step = 0;
  function blockUI() {
    const state = states[step];
    el('block-track').replaceChildren(...states.map((s,i) => {
      const li=document.createElement('li');li.textContent=s.name;li.classList.toggle('current',i===step);
      if(i===step)li.setAttribute('aria-current','step');return li;
    }));
    el('logical-tables').replaceChildren(...[['A',state.a],['B',state.b]].map(([name,table]) => {
      const p=document.createElement('p');p.textContent=`Sequence ${name}: ${table.length?'['+table.join(', ')+']':'—'}`;return p;
    }));
    const refs = {};
    [...state.a,...state.b].forEach(id => {refs[id]=(refs[id]||0)+1;});
    el('physical-pool').replaceChildren(...Object.entries(state.data).map(([id,tokens]) => {
      const frame=document.createElement('div');frame.className='physical-block';
      if(!refs[id])frame.classList.add('free-block');
      const title=document.createElement('span');title.textContent=`Physical ${id} · ref=${refs[id]||0}`;
      const slots=document.createElement('div');slots.className='kv-slots';
      for(let i=0;i<4;i++){const cell=document.createElement('b');cell.textContent=tokens[i]??'·';cell.classList.toggle('empty',tokens[i]===undefined);slots.append(cell);}
      frame.append(title,slots);return frame;
    }));
    el('block-result').textContent=`Step ${step+1}/8\n${state.note}`;
    el('block-prev').disabled=step===0;el('block-next').disabled=step===states.length-1;
  }
  el('block-prev').addEventListener('click',() => {step=Math.max(0,step-1);blockUI();});
  el('block-next').addEventListener('click',() => {step=Math.min(states.length-1,step+1);blockUI();});
  el('block-reset').addEventListener('click',() => {step=0;blockUI();});
  blockUI();
  function integer(id,min,max) {
    const value=Number(el(id).value);
    return Math.max(min,Math.min(max,Number.isFinite(value)?Math.trunc(value):min));
  }
  function bars(id,names,values) {
    el(id).replaceChildren(...values.map((v,i) => {
      const row=document.createElement('div'), text=document.createElement('span'),bar=document.createElement('i');
      text.textContent=names[i]+' · '+v.toFixed(2)+'%';
      bar.style.width=Math.max(0,Math.min(100,v))+'%';bar.setAttribute('aria-hidden','true');
      row.append(text,bar);return row;
    }));
  }
  function memoryUI() {
    const layers=integer('kv-layers',1,256),heads=Number(el('kv-heads').value),dim=integer('head-dim',16,512),bytes=Number(el('kv-bytes').value);
    const b=Number(el('block-size').value),p=integer('prompt-length',0,32768),suffix=integer('suffix-length',1,32768),n=Number(el('branch-count').value);
    const t=p+suffix,per=2*layers*heads*dim*bytes,padded=Math.ceil(t/b)*b;
    const shared=el('sharing').value==='yes'?Math.floor(p/b)*b:0;
    const slots=n*padded-(n-1)*shared,base=n*padded,slack=n*(padded-t),memory=slots*per;
    const literal=layers===40&&heads===40&&dim===128&&bytes===2;
    el('branch-value').textContent=String(n);
    bars('memory-bars',['有效 KV 槽占比','尾部空槽占比'],[(slots-slack)/slots*100,slack/slots*100]);
    el('memory-result').textContent=`${per.toLocaleString('en-US')} bytes/token = ${(per/1024).toFixed(3)} KiB/token\nT=${t}; per-branch padded=${padded} slots; N=${n}\nShared full-prefix=${shared} tokens; unique physical=${slots/b} blocks / ${slots} slots\nTail slack=${slack} slots (${(slack/slots*100).toFixed(3)}%)\nKV payload=${(memory/2**30).toFixed(5)} GiB / ${(memory/1e9).toFixed(5)} GB\nSharing saves=${((base-slots)*per/2**30).toFixed(5)} GiB vs no-sharing\n${literal?'Paper background: OPT-13B FP16 KV.':'Hypothetical configuration; not a paper-reported accuracy/performance result.'}`;
  }
  ['kv-layers','kv-heads','head-dim','kv-bytes','block-size','prompt-length','suffix-length','branch-count','sharing'].forEach(id => el(id).addEventListener('input',memoryUI));
  memoryUI();
  function prefixUI() {
    const diff=Number(el('prefix-diff').value),length=integer('query-length',1,16),same=el('prefix-salt').value==='same';
    const common=Math.min(12,length,diff?diff-1:12),hit=same?Math.floor(common/4)*4:0;
    el('prefix-blocks').replaceChildren(...[0,1,2,3].map(j => {
      const row=document.createElement('div');row.className='prefix-block';
      const reused=(j+1)*4<=hit;
      row.classList.toggle('hit-block',reused);
      const start=j*4+1,end=Math.min((j+1)*4,length);
      row.textContent=start>length?`Block ${j}: outside query`:`Block ${j}: tokens ${start}–${end} · ${reused?'HIT':'not reused'}`;
      return row;
    }));
    el('prefix-result').textContent=`Contiguous common tokens=${common}; salt=${same?'same':'different'}\nReusable full-block KV=${hit} tokens / ${hit/4} blocks\nQuery remainder requiring KV computation=${length-hit} tokens\n${diff&&same?'A differing causal prefix invalidates all later block keys, even when later token text matches.':same?'Cache has only 12 tokens; partial blocks are excluded in this documented mode.':'Different isolation salt prevents reuse across the entire prefix.'}`;
  }
  ['prefix-diff','query-length','prefix-salt'].forEach(id => el(id).addEventListener('input',prefixUI));
  prefixUI();
})();
