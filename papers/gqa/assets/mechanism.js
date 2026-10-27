/* Agent-created analytical model. No latency prediction. */
(function(root){
  'use strict';
  function groupForHead(h,H,G){
    if(!Number.isInteger(H)||!Number.isInteger(G)||G<1||H%G||h<0||h>=H) throw new RangeError('Equal groups require H divisible by G');
    return Math.floor(h/(H/G));
  }
  function cacheBytes({layers,batch,length,groups,dim,bytes}){
    const values=[layers,batch,length,groups,dim,bytes];
    if(values.some(x=>!Number.isFinite(x)||x<0)) throw new RangeError('Invalid cache dimensions');
    return 2*layers*batch*length*groups*dim*bytes;
  }
  function physicalGroups(G,P){
    if(!Number.isInteger(G)||!Number.isInteger(P)||G<1||P<1||(G>=P?G%P:P%G)) throw new RangeError('This TP model requires exact head sharding or replication');
    return Math.max(G,P);
  }
  function causalAllowed(q,k,prefix){return k<=prefix+q;}
  function poolHeads(heads,G){
    const H=heads.length;
    groupForHead(0,H,G);
    return Array.from({length:G},(_,g)=>heads[0].map((_,j)=>heads.slice(g*H/G,(g+1)*H/G).reduce((a,x)=>a+x[j],0)/(H/G)));
  }
  const api={groupForHead,cacheBytes,physicalGroups,causalAllowed,poolHeads};
  if(typeof module!=='undefined') module.exports=api;
  else root.GQAModel=api;
})(typeof window!=='undefined'?window:globalThis);
