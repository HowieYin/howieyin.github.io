export function normalize(a, mode="ln", epsilon=1e-5, gain=1, bias=0) {
  const mean=a.reduce((s,x)=>s+x,0)/a.length;
  const variance=a.reduce((s,x)=>s+(x-mean)**2,0)/a.length;
  const moment=mode==="ln"?variance:a.reduce((s,x)=>s+x*x,0)/a.length;
  const rstd=1/Math.sqrt(moment+epsilon);
  const z=a.map(x=>(x-(mode==="ln"?mean:0))*rstd);
  return {mean,variance,moment,rstd,z,y:z.map(x=>gain*x+bias)};
}
export function normalizeRows(rows, ...args) {
  return rows.map(row=>normalize(row,...args));
}
export function normalizeTensor(a, mode="ln", eps=1e-5) {
  return a.map((batch,b)=>batch.map((row,t)=>row.map((x,d)=>{
    const group=mode==="ln"?row:mode==="bn"?a.map(batch=>batch[t][d]):batch.map(row=>row[d]);
    const stat=normalize(group,"ln",eps);
    return (x-stat.mean)*stat.rstd;
  })));
}
export function cost(B,T,D,s,layers) {
  const N=B*T*D;
  return {B,T,D,s,layers,fused:2*N*s,unfused:8*N*s,savedStats:2*B*T*4,
          params:2*D*4,decode:layers*2*B*D*s};
}
