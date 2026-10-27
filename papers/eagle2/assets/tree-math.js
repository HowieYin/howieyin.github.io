(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.TreeMath=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  function scoreTree(input){
    const lookup=new Map();
    return input.map((n,index)=>{
      if(lookup.has(n.id))throw Error('Duplicate node');
      if(!Number.isFinite(n.c)||n.c<0||n.c>1)throw Error('Confidence outside [0,1]');
      const parent=n.parent==null?null:lookup.get(n.parent);
      if(n.parent!=null&&!parent)throw Error('Parent must precede child');
      const logValue=parent?parent.logValue+Math.log(n.c):0;
      const result={...n,index,depth:parent?parent.depth+1:0,logValue,value:Math.exp(logValue)};
      lookup.set(n.id,result);return result;
    });
  }
  function selectTree(nodes,m){
    if(!Number.isInteger(m)||m<1||m>nodes.length)throw Error('Invalid budget');
    const ranked=[...nodes].sort((a,b)=>{
      if(a.logValue!==b.logValue)return a.logValue>b.logValue?-1:1;
      return a.depth-b.depth||a.index-b.index;
    });
    return ranked.slice(0,m).sort((a,b)=>a.index-b.index);
  }
  function maskFor(nodes){
    const indices=new Map(nodes.map((n,i)=>[n.id,i]));
    return nodes.map((node,i)=>{
      const row=Array(nodes.length).fill(0);let current=node;
      while(current){const j=indices.get(current.id);if(j==null)throw Error('Missing ancestor');row[j]=1;
        current=current.parent==null?null:nodes[indices.get(current.parent)];
        if(node.parent!=null&&j===i&&!current)throw Error('Missing parent');
      }return row;
    });
  }
  const cost=({layers,batch,tokens,heads,dim,bytes})=>2*layers*batch*tokens*heads*dim*bytes;
  return {scoreTree,selectTree,maskFor,cost};
});
