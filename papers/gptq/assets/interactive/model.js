(function(root) {
  "use strict";
  function toy(rho) {
    if (!(rho >= 0 && rho < 1)) throw new RangeError("rho must be in [0,1)");
    const w=.49, firstCorrection=rho*w, secondBefore=w+firstCorrection;
    const q2=Math.max(-2,Math.min(1,Math.round(secondBefore)));
    const loss=(a,b)=>a*a+b*b+2*rho*a*b;
    return {rho,w,firstCorrection,secondBefore,q1:0,q2,
      rtnLoss:loss(-w,-w),gptqLoss:loss(-w,q2-w),
      firstLoss:loss(-w,firstCorrection)};
  }
  function memory(c) {
    const effectiveBits=c.bits+(c.group>0?(c.scaleBits+c.zeroBits)/c.group:0);
    const weightBytes=c.params*1e9*((1-c.fpFraction)*effectiveBits/8+c.fpFraction*2);
    const kvBytes=2*c.layers*c.kvHeads*c.headDim*c.batch*c.length*c.kvBits/8;
    return {effectiveBits,weightBytes,kvBytes,totalBytes:weightBytes+kvBytes,
      fpWeightBytes:c.params*1e9*2};
  }
  const api={toy,memory};
  if(typeof module!=="undefined"&&module.exports) module.exports=api;
  else root.GPTQMath=api;
})(typeof globalThis!=="undefined"?globalThis:this);
