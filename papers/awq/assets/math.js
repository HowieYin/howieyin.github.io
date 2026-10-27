(function (root) {
  "use strict";
  function weightBytes(parameters, group, format) {
    const metadata = {official:32,packed:20,symmetric:16}[format];
    if (!metadata || parameters < 0 || group <= 0) throw new Error("Invalid storage configuration");
    return parameters * (4 + metadata / group) / 8;
  }
  function kvBytes(layers,batch,tokens,heads,headDim,bytes) {
    return 2*layers*batch*tokens*heads*headDim*bytes;
  }
  function pack(weights) {
    if(weights.length!==32 || weights.some(w=>!Number.isInteger(w)||w<0||w>15)) throw new Error("Expected 32 unsigned nibbles");
    return Array.from({length:16},(_,i)=>weights[i]|(weights[i+16]<<4));
  }
  function unpack(bytes) {
    return [...bytes.map(b=>b&15),...bytes.map(b=>(b>>4)&15)];
  }
  function roundEven(value) {
    const floor=Math.floor(value),fraction=value-floor;
    return fraction===.5?(floor%2===0?floor:floor+1):Math.round(value);
  }
  function scaleToy(s,x,dominant) {
    const W=[.13,.7,1.2,-dominant],X=[x,.2,.1,.1],S=[s,1,1,1];
    const scaled=W.map((w,i)=>w*S[i]),delta=Math.max(...scaled.map(Math.abs))/8;
    const reconstructed=scaled.map((w,i)=>roundEven(w/delta)*delta/S[i]);
    const contribution=reconstructed.map((w,i)=>(w-W[i])*X[i]);
    const reference=W.reduce((a,w,i)=>a+w*X[i],0);
    const quantized=reconstructed.reduce((a,w,i)=>a+w*X[i],0);
    return {W,X,S,scaled,delta,reconstructed,contribution,reference,quantized,
      error:Math.abs(quantized-reference),salientBound:.5*delta*x/s,
      totalBound:.5*delta*X.reduce((a,v,i)=>a+Math.abs(v)/S[i],0)};
  }
  const api={weightBytes,kvBytes,pack,unpack,scaleToy};
  if(typeof module!=="undefined" && module.exports) module.exports=api;
  else root.AWQMath=api;
})(typeof globalThis!=="undefined"?globalThis:this);
