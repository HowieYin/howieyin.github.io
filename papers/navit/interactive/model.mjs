export function pack(lengths,capacity){
  if(!Number.isInteger(capacity)||capacity<1||lengths.some(n=>!Number.isInteger(n)||n<1||n>capacity))throw new RangeError('Every image must fit a sequence');
  const bins=[];
  for(const n of lengths){let bin=bins.find(b=>b.reduce((a,x)=>a+x,0)+n<=capacity);if(!bin){bin=[];bins.push(bin);}bin.push(n);}
  return bins;
}
export function mask(ids){return ids.map(i=>ids.map(j=>Number(i>=0&&i===j)));}
export function coordinates(row,col,rows,cols){return {x:col,y:row,rx:col/cols,ry:row/rows};}
export function cost(s,d,l,heads,bytes=2){
  const projection=8*s*d*d,attention=4*s*s*d,mlp=16*s*d*d;
  return {projection,attention,mlp,total:(projection+attention+mlp)*l,mapMiB:heads*s*s*bytes/1048576};
}
export function schedule(n,mu,tau){return .2+.6/(1+Math.exp(-(n-mu)/tau));}
