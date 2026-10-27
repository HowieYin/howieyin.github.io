"""Independent numerical check: grouped contraction vs expanded-KV reference."""
import json
from pathlib import Path
import numpy as np

rng=np.random.default_rng(230513245)
results=[]
def softmax(x):
    y=np.exp(x-x.max(axis=-1,keepdims=True))
    return y/y.sum(axis=-1,keepdims=True)
for H,G in [(8,8),(8,1),(8,2),(8,4)]:
    B,Tq,Tk,d=2,2,5,4
    q=rng.normal(size=(B,Tq,H,d))
    k=rng.normal(size=(B,Tk,G,d))
    v=rng.normal(size=(B,Tk,G,d))
    allowed=np.arange(Tk)[None,:] <= (3+np.arange(Tq))[:,None]
    qg=q.reshape(B,Tq,G,H//G,d)
    scores=np.einsum('btgrd,bsgd->bgrts',qg,k)/np.sqrt(d)
    probs=softmax(np.where(allowed,scores,-np.inf))
    native=np.einsum('bgrts,bsgd->btgrd',probs,v).reshape(B,Tq,H,d)
    kr=np.repeat(k,H//G,axis=2);vr=np.repeat(v,H//G,axis=2)
    reference=np.zeros_like(native)
    for b in range(B):
        for h in range(H):
            p=softmax(np.where(allowed,q[b,:,h]@kr[b,:,h].T/np.sqrt(d),-np.inf))
            reference[b,:,h]=p@vr[b,:,h]
    error=float(abs(reference-native).max())
    assert error<1e-12
    results.append({'H':H,'G':G,'max_abs_error':error})
D,H,G,d=6,8,2,4
x=rng.normal(size=(3,D));w=rng.normal(size=(D,H,d))
pooled=w.reshape(D,G,H//G,d).mean(axis=2)
before=np.einsum('ti,ihj->thj',x,w)
after=np.einsum('ti,igj->tgj',x,pooled)
pool_error=float(abs(before.reshape(3,G,H//G,d).mean(axis=2)-after).max())
assert pool_error<1e-12
# A non-symmetric case demonstrates pooling does not commute with softmax.
old=softmax(np.array([[1.,4.],[3.,1.]])).mean(axis=0)
new=softmax(np.array([[2.,2.5]]))[0]
assert not np.allclose(old,new)
report={'status':'passed','grouped_vs_repeat_reference':results,'linear_pooling_max_abs_error':pool_error,'softmax_non_equivalence':{'mean_softmax':old.tolist(),'softmax_mean':new.tolist()},'scope':'Float64 toy mechanism; not checkpoint uptraining or hardware performance reproduction'}
path=Path(__file__).with_name('numerical-results.json')
path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
