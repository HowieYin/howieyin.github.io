(function (root) {
  'use strict';
  function gate(logits, k) {
    if (!Number.isInteger(k) || k < 1 || k > logits.length || !logits.every(Number.isFinite)) throw Error('Invalid top-k');
    const selected = logits.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v || a.i-b.i).slice(0,k);
    const max=selected[0].v, sum=selected.reduce((a,b)=>a+Math.exp(b.v-max),0);
    const result=logits.map(()=>0); selected.forEach(({v,i})=>result[i]=Math.exp(v-max)/sum);
    return result;
  }
  function cv(values) {
    const mean=values.reduce((a,b)=>a+b,0)/values.length;
    return mean ? Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/values.length)/mean : 0;
  }
  function statistics(gates) {
    const importance=gates[0].map((_,i)=>gates.reduce((s,row)=>s+row[i],0));
    const counts=gates[0].map((_,i)=>gates.filter(row=>row[i]>0).length);
    return {importance,counts,cvImportance:cv(importance),cvCount:cv(counts)};
  }
  function excludingThresholds(noisy,k) {
    return noisy.map((_,i)=>noisy.filter((_,j)=>i!==j).sort((a,b)=>b-a)[k-1]);
  }
  function batchwise(scores,k) {
    const b=scores.length,n=scores[0].length,m=k*b/n;
    if (!Number.isInteger(m)) throw Error('Demo requires integer expert quota');
    const mask=scores.map(row=>row.map(()=>0));
    for(let i=0;i<n;i++) scores.map((row,t)=>({v:row[i],t})).sort((a,b)=>b.v-a.v||a.t-b.t).slice(0,m).forEach(({t})=>mask[t][i]=1);
    const coverage=mask.map(row=>row.reduce((a,b)=>a+b,0));
    const gates=scores.map((row,t)=>{
      const sum=row.reduce((s,v,i)=>s+v*mask[t][i],0);
      return sum ? row.map((v,i)=>v*mask[t][i]/sum) : null;
    });
    return {gates,coverage,counts:scores[0].map((_,i)=>mask.reduce((s,row)=>s+row[i],0))};
  }
  function cost({tokens,experts,k,d,h,bytes}) {
    if (![tokens,experts,k,d,h,bytes].every(v=>Number.isFinite(v)&&v>0) || k>experts) throw Error('Invalid capacity configuration');
    const one=2*d*h;
    return {oneExpert:one,expertParams:experts*one,activeParams:k*one,mac:k*one,weightBytes:experts*one*bytes,perExpert:tokens*k/experts,traffic:tokens*k*2*d*bytes};
  }
  const api={gate,statistics,excludingThresholds,batchwise,cost};
  if(typeof module==='object'&&module.exports) module.exports=api;
  else root.MoEMath=api;
})(typeof globalThis!=='undefined'?globalThis:this);
