export const configurations = {
  '7': {n:6.7e9,d:4096,h:32,l:32,m:11008},
  '13': {n:13e9,d:5120,h:40,l:40,m:13824},
  '33': {n:32.5e9,d:6656,h:52,l:60,m:17920},
  '65': {n:65.2e9,d:8192,h:64,l:80,m:22016}
};
export function memory(c,b,t,weightBits,kvBits,heads) {
  const weights=c.n*weightBits/8;
  const perToken=2*c.l*heads*128*kvBits/8;
  const kv=b*t*perToken;
  return {weights,perToken,kv,total:weights+kv};
}
export function breakEven(extra,saving,served) {
  return {tokens:saving>0?extra/saving*1e6:Infinity,net:served/1e6*saving-extra};
}
export function parameterCount(c) {
  return 2*32000*c.d+c.l*(4*c.d*c.d+3*c.d*c.m+2*c.d)+c.d;
}
