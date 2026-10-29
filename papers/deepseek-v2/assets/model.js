/* Analytical models for the three explainers. No measured latency is inferred. */
(function(root){
  'use strict';
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const mv=(m,v)=>m.map(row=>dot(row,v));
  const transpose=m=>m[0].map((_,i)=>m.map(row=>row[i]));
  const softmax=x=>{const m=Math.max(...x),e=x.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
  const cacheBytes=(kind,sequence,batch,bits)=>{
    const elements={mla:60*(512+64),'67b':95*2*8*128,expanded:60*128*(192+128),mha:60*2*128*128};
    return elements[kind]*sequence*batch*bits/8;
  };
  function route(scores,D,K,M){
    const groupSize=scores.length/D;
    const groups=Array.from({length:D},(_,g)=>({g,max:Math.max(...scores.slice(g*groupSize,(g+1)*groupSize))}));
    groups.sort((a,b)=>b.max-a.max||a.g-b.g);
    const allowed=groups.slice(0,M).map(x=>x.g);
    const selected=[...scores.keys()].filter(i=>allowed.includes(Math.floor(i/groupSize))).sort((i,j)=>scores[j]-scores[i]||i-j).slice(0,K);
    return {allowed,selected};
  }
  function expandedAttention(t){
    const scores=t.c.map((c,i)=>(dot(t.q,mv(t.uk,c))+t.qr*t.kr[i])/Math.sqrt(t.q.length+1));
    const weights=softmax(scores),v=t.c.map(c=>mv(t.uv,c));
    const out=v[0].map((_,i)=>v.reduce((s,row,j)=>s+weights[j]*row[i],0));
    return {scores,weights,out};
  }
  function absorbedAttention(t){
    const effective=mv(transpose(t.uk),t.q);
    const scores=t.c.map((c,i)=>(dot(effective,c)+t.qr*t.kr[i])/Math.sqrt(t.q.length+1));
    const weights=softmax(scores),z=t.c[0].map((_,i)=>t.c.reduce((s,row,j)=>s+weights[j]*row[i],0));
    return {scores,weights,effective,z,out:mv(t.uv,z)};
  }
  const model={cacheBytes,route,expandedAttention,absorbedAttention};
  if(typeof module!=='undefined'&&module.exports)module.exports=model;
  else root.MLAModel=model;
})(typeof globalThis!=='undefined'?globalThis:this);
