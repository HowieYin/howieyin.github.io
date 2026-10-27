(function(root){
  "use strict";
  const sum=a=>a.reduce((x,y)=>x+y,0);
  const normalize=a=>{const s=sum(a);return s>1e-14?a.map(v=>v/s):a.map(()=>0)};
  function exactMass(p,q){
    const accepted=p.map((v,i)=>Math.min(v,q[i]));
    const residual=normalize(p.map((v,i)=>Math.max(0,v-q[i])));
    const acceptedTotal=sum(accepted),rejectedTotal=Math.max(0,1-acceptedTotal);
    return {accepted,residual,acceptedTotal,rejectedTotal,output:accepted.map((v,i)=>v+rejectedTotal*residual[i])};
  }
  function conditional(q,removed){
    const a=q.map((v,i)=>removed.includes(i)?0:v);
    if(sum(a)<=1e-14)throw Error("exhausted support");
    return normalize(a);
  }
  function ancestors(parents,node){
    const path=[],seen=new Set();
    while(node>=0){
      if(seen.has(node))throw Error("cycle in tree");
      if(node>=parents.length)throw Error("invalid parent");
      seen.add(node);path.push(node);node=parents[node];
    }
    return path.reverse();
  }
  const depths=p=>p.map((_,i)=>ancestors(p,i).length-1);
  function capacity({batch,context,nodes,layers,heads,dim,bytes,hidden}){
    const unit=2*batch*layers*heads*dim*bytes;
    return {prefix:unit*context,tree:unit*nodes,features:batch*nodes*hidden*bytes};
  }
  function measured(mode){
    const [baseline,eagle]=mode==="int4"?[106.9,160.4]:[55.1,100.2];
    return {baseline,eagle,ratio:eagle/baseline,tpot:1000/eagle,baseTpot:1000/baseline};
  }
  function cost({tau,verify,draft,steps,other}){
    const cycle=verify+draft*steps+other;
    return {cycle,speed:tau/cycle,threshold:cycle};
  }
  const api={exactMass,conditional,ancestors,depths,capacity,measured,cost};
  if(typeof module!=="undefined")module.exports=api;
  else root.EagleMath=api;
})(typeof globalThis!=="undefined"?globalThis:this);
