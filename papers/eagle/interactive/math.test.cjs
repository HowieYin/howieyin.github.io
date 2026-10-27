const {test}=require("node:test");
const assert=require("node:assert/strict");
const M=require("./math.cjs");
const close=(x,y)=>assert(Math.abs(x-y)<1e-10,`${x} != ${y}`);
test("accept + rejection residual reconstruct target distribution",()=>{
  const p=[.6,.3,.1],q=[.2,.5,.3],r=M.exactMass(p,q);
  close(r.acceptedTotal,.6);close(r.rejectedTotal,.4);
  r.output.forEach((v,i)=>close(v,p[i]));
  assert.deepEqual(r.residual,[1,0,0]);
});
test("identical distributions have no fallback and no NaN",()=>{
  const r=M.exactMass([.2,.3,.5],[.2,.3,.5]);
  close(r.acceptedTotal,1);assert.deepEqual(r.residual,[0,0,0]);
  assert(r.output.every(Number.isFinite));
});
test("disjoint support entirely falls back",()=>{
  const r=M.exactMass([1,0,0],[0,.5,.5]);
  close(r.rejectedTotal,1);assert.deepEqual(r.output,[1,0,0]);
});
test("without replacement renormalizes and exhausts support",()=>{
  assert.deepEqual(M.conditional([.2,.5,.3],[1]),[.4,0,.6]);
  assert.throws(()=>M.conditional([.2,.5,.3],[0,1,2]),/exhausted/);
});
test("sequential conditional proposals preserve target",()=>{
  const p=[.6,.3,.1],q=[.2,.5,.3];
  const output=[0,0,0];
  // Enumerate all first candidates, accept or reject, then second candidates.
  for(let x=0;x<3;x++){
    const first=M.exactMass(p,q);
    output[x]+=first.accepted[x];
    const reject=q[x]-first.accepted[x];
    const next=M.exactMass(first.residual,M.conditional(q,[x]));
    next.output.forEach((v,i)=>output[i]+=reject*v);
  }
  output.forEach((v,i)=>close(v,p[i]));
});
test("branch visibility is self and ancestors, never siblings",()=>{
  const parents=[-1,0,0,1,1,2,3];
  assert.deepEqual(M.ancestors(parents,6),[0,1,3,6]);
  assert.deepEqual(M.depths(parents),[0,1,1,2,2,2,3]);
  assert.throws(()=>M.ancestors([-1,2,1],1),/cycle/);
});
test("KV byte accounting distinguishes prefix and tree",()=>{
  const r=M.capacity({batch:1,context:4096,nodes:26,layers:32,heads:32,dim:128,bytes:2,hidden:4096});
  assert.equal(r.prefix,2147483648);assert.equal(r.tree,13631488);
  assert.equal(r.features,212992);
});
test("speed ratios and break-even use total cycle cost",()=>{
  close(M.measured("fp16").ratio,100.2/55.1);
  close(M.measured("int4").tpot,1000/160.4);
  close(M.cost({tau:4,verify:1.2,draft:.08,steps:5,other:.1}).speed,4/1.7);
  close(M.cost({tau:1.7,verify:1.2,draft:.08,steps:5,other:.1}).speed,1);
});
