(function(root) {
  'use strict';
  function kvCapacity(context,batch,heads,bytes,tp) {
    if(![1,2,4,8,16,32,64].includes(tp)||!Number.isFinite(context)||context<=0||batch<=0||heads<=0||
      (heads>=tp?heads%tp:tp%heads)!==0) throw new RangeError('Incompatible KV/TP shape');
    const localHeads=Math.max(1,heads/tp);
    const perHead=2*80*batch*context*128*bytes/(2**30);
    return {logicalGiB:perHead*heads,perRankGiB:perHead*localHeads,
      physicalGiB:perHead*localHeads*tp,localHeads,replication:localHeads*tp/heads};
  }
  function ranking(delta,margin) {
    const z=margin-delta;
    const loss=Math.max(z,0)+Math.log1p(Math.exp(-Math.abs(z)));
    const gradient=z>=0?-1/(1+Math.exp(-z)):-Math.exp(z)/(1+Math.exp(z));
    return {loss,gradient};
  }
  function rewardBranch(tag,safety,helpful) {
    const useSafety=tag||safety<0.15;
    return {name:useSafety?'Safety':'Helpful',score:useSafety?safety:helpful};
  }
  const api={kvCapacity,ranking,rewardBranch};
  if(typeof module!=='undefined')module.exports=api;
  else root.LlamaMath=api;
})(typeof globalThis!=='undefined'?globalThis:this);
