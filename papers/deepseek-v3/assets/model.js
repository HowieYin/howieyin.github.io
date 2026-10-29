(function(root){
  'use strict';
  function cache({tokens=131072,batch=1,layers=61,bytes=2,ranks=4}={}){
    const logical=tokens*batch*layers*576*bytes;
    return {logical,gib:logical/2**30,replicated:logical*ranks,materialized:tokens*batch*layers*40960*bytes,ratio:40960/576};
  }
  function route(scores,bias,k){
    const indices=scores.map((s,i)=>i).sort((a,b)=>(scores[b]+bias[b])-(scores[a]+bias[a])||a-b).slice(0,k);
    const total=indices.reduce((s,i)=>s+scores[i],0);
    return {indices,weights:indices.map(i=>scores[i]/total)};
  }
  function speculative(a,c){return {speed:(1+a)/(1+c),outputs:1+a,breakEven:c<a};}
  function absorbExample(){
    const q=[.3,-.2],K=[[1,.5],[-.3,2]],V=[[.7,-.1],[.4,.8]],latents=[[1,2],[-.5,1],[.2,-.4]];
    const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),mv=(m,v)=>m.map(r=>dot(r,v));
    const qLat=[q[0]*K[0][0]+q[1]*K[1][0],q[0]*K[0][1]+q[1]*K[1][1]];
    const naive=latents.map(c=>dot(q,mv(K,c))),absorbed=latents.map(c=>dot(qLat,c));
    const exps=naive.map(x=>Math.exp(x-Math.max(...naive))),sum=exps.reduce((a,b)=>a+b),p=exps.map(x=>x/sum);
    const materialized=[0,1].map(d=>latents.reduce((s,c,j)=>s+p[j]*mv(V,c)[d],0));
    const z=[0,1].map(d=>latents.reduce((s,c,j)=>s+p[j]*c[d],0)),output=mv(V,z);
    return {naive,absorbed,materialized,output,scoreError:Math.max(...naive.map((v,i)=>Math.abs(v-absorbed[i]))),outputError:Math.max(...output.map((v,i)=>Math.abs(v-materialized[i])))};
  }
  const api={cache,route,speculative,absorbExample};
  if(typeof module==='object')module.exports=api;else root.DeepSeekModels=api;
})(typeof globalThis==='object'?globalThis:this);
