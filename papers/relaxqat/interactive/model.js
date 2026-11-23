(function(root) {
  'use strict';
  const finite = (...v) => { if (!v.every(Number.isFinite)) throw new RangeError('Finite numbers required'); };
  function bitsAt(omega, start, end) {
    finite(omega,start,end);
    return start-(start-end)*Math.sin(Math.PI/2*Math.max(0,Math.min(1,omega)));
  }
  // Continuous-time explanatory reconstruction of the textual schedule.
  // Original Algorithms 1/2 discretize samples and contain a cycle-boundary ambiguity.
  function scheduleAt(t, ms, cycles, start=4, end=2) {
    finite(t,ms,cycles,start,end);
    if (ms<=0 || cycles<0) throw new RangeError('Invalid schedule');
    if (t<ms) return bitsAt(t/ms,start,end);
    for (let i=0;i<cycles;i++) {
      const offset=t-(3000+i*2600);
      if (offset>=0 && offset<=ms) return bitsAt(1-offset/ms,start,end);
      if (offset>ms && offset<=2*ms) return bitsAt((offset-ms)/ms,start,end);
    }
    return end;
  }
  function roundEven(x) {
    const lower=Math.floor(x), fraction=x-lower;
    return fraction===0.5 ? (lower%2===0 ? lower : lower+1) : Math.round(x);
  }
  const clip = (x,lo,hi) => Math.max(lo,Math.min(hi,x));
  function quantize(theta,alpha,b,kind) {
    finite(theta,alpha,b);
    if (alpha<=0 || b<=1) throw new RangeError('Positive scale and bits > 1 required');
    const k=2**(b-1), delta=alpha/k;
    if (kind==='pareto') return delta*(roundEven(clip(theta/alpha,-1,1)*k-0.5)+0.5);
    if (kind==='ternary') return delta*clip(roundEven(theta/delta),-k+0.5,k-0.5);
    return delta*(clip(roundEven(theta/delta-0.5),-k,k-1)+0.5);
  }
  function storage(billions,bits,inputWidth,tokens,layers,heads,headDim) {
    finite(billions,bits,inputWidth,tokens,layers,heads,headDim);
    if (billions<=0 || inputWidth<=0 || bits<=0 || tokens<0 || layers<0 || heads<0 || headDim<0) throw new RangeError('Invalid storage assumptions');
    return {weights:billions*1e9*bits/8,scales:billions*1e9/inputWidth*2,kv:2*tokens*layers*heads*headDim*2};
  }
  const model={bitsAt,scheduleAt,roundEven,quantize,storage};
  if (typeof module!=='undefined' && module.exports) module.exports=model;
  else root.RelaxModel=model;
})(typeof globalThis!=='undefined' ? globalThis : this);
