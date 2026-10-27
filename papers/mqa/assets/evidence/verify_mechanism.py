import json
import math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
rng = np.random.default_rng(191102150)
b, n, d, h, k = 2, 7, 16, 4, 4
x = rng.normal(size=(b, n, d))
pq = rng.normal(size=(h,d,k)) / math.sqrt(d)
pk = rng.normal(size=(d,k)) / math.sqrt(d)
pv = rng.normal(size=(d,k)) / math.sqrt(d)
po = rng.normal(size=(h,d,k)) / math.sqrt(h*k)
q = np.einsum("bnd,hdk->bhnk",x,pq)
keys = x @ pk
values = x @ pv
logits = np.einsum("bhnk,bmk->bhnm",q,keys)
logits += np.where(np.triu(np.ones((n,n)),1), -np.inf,0)[None,None]
def softmax(z):
    z = np.exp(z-z.max(axis=-1,keepdims=True))
    return z/z.sum(axis=-1,keepdims=True)
weights = softmax(logits)
full = np.einsum("bhnv,hdv->bnd",np.einsum("bhnm,bmv->bhnv",weights,values),po)
# Independent explicit-head reference shares projection weights, not an arbitrary MHA checkpoint.
reference = np.zeros_like(full)
for batch in range(b):
    for position in range(n):
        for head in range(h):
            qi = x[batch,position] @ pq[head]
            ki = x[batch,:position+1] @ pk
            vi = x[batch,:position+1] @ pv
            wi = softmax(qi @ ki.T)
            reference[batch,position] += (wi @ vi) @ po[head].T
assert np.allclose(full, reference, atol=1e-12)
incremental=[]
cache_k=np.empty((b,0,k))
cache_v=np.empty((b,0,k))
for t in range(n):
    qt=np.einsum("bd,hdk->bhk",x[:,t],pq)
    cache_k=np.concatenate([cache_k,(x[:,t]@pk)[:,None,:]],axis=1)
    cache_v=np.concatenate([cache_v,(x[:,t]@pv)[:,None,:]],axis=1)
    wt=softmax(np.einsum("bhk,bmk->bhm",qt,cache_k))
    ot=np.einsum("bhm,bmv->bhv",wt,cache_v)
    incremental.append(np.einsum("bhv,hdv->bd",ot,po))
incremental=np.stack(incremental,axis=1)
assert np.allclose(full,incremental,atol=1e-12)
saved_mt=18*2*1024*128*7
added_mt=12*2*1024*(5440-4096)
saved_lm=6*2*1024*128*7
added_lm=6*2*1024*(9088-8192)
assert saved_mt==added_mt==33030144
assert saved_lm==added_lm==11010048
results={"scope":"CPU algebra and reported arithmetic only; no training, BLEU/PPL or TPU/GPU speed replication", "causal_reference_max_error":float(np.max(np.abs(full-reference))),"incremental_max_error":float(np.max(np.abs(full-incremental))),"cache_shape":list(cache_k.shape),"mt_parameter_reallocation":saved_mt,"lm_parameter_reallocation":saved_lm,"greedy_decoder_speedup_rounded":46/3.8,"greedy_step_speedup_raw":47/3.9,"greedy_end_to_end_rounded":(1.7+46)/(1.5+3.8),"beam4_decoder_speedup":203/32,"translation_ppl_relative_increase":math.exp(1.439-1.424)-1,"lm_ppl_relative_increase":30.2/29.9-1}
(root/"mechanism-checks.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
