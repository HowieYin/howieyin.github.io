"""CPU teaching experiment: identities, counterexamples, and raw KV capacity.
Not a trained-model evaluation or a GPU speed benchmark. Run with Python + NumPy.
"""
from pathlib import Path
import json, platform, numpy as np

def softmax(x):
    e=np.exp(x-x.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)

def dense(q,k,v,mask=None):
    s=q@k.T/np.sqrt(k.shape[-1])
    if mask is not None: s=np.where(mask,s,-np.inf)
    return softmax(s)@v

def tiled(q,k,v,block):
    """FP64 online softmax, no N-by-N score matrix is retained."""
    m=np.full((len(q),1),-np.inf)
    l=np.zeros((len(q),1)); u=np.zeros((len(q),v.shape[-1]))
    for start in range(0,len(k),block):
        s=q@k[start:start+block].T/np.sqrt(k.shape[-1])
        new=np.maximum(m,s.max(axis=-1,keepdims=True))
        a=np.exp(m-new); p=np.exp(s-new)
        u=a*u+p@v[start:start+block]
        l=a*l+p.sum(axis=-1,keepdims=True); m=new
    return u/l

def main():
    rng=np.random.default_rng(20261116)
    q,k,v=[rng.normal(size=(257,32)).astype(np.float64) for _ in range(3)]
    ref=dense(q,k,v)
    block_errors={str(b):float(np.max(np.abs(tiled(q,k,v,b)-ref))) for b in [1,7,32,128,257]}
    mask=np.tril(np.ones((257,257),dtype=bool))
    causal=dense(q,k,v,mask)
    cache=np.concatenate([dense(q[t:t+1],k[:t+1],v[:t+1]) for t in range(257)])
    cache_error=float(np.max(np.abs(cache-causal)))
    # A distant matching token is removed by a local-only mask.
    qr=np.array([[1.,0.]])
    kr=np.array([[8.,0.],[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
    vr=np.array([[1.,0.],[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
    full=dense(qr,kr,vr)
    local=dense(qr,kr[-2:],vr[-2:])
    # Low-dimensional positive linear features preserve only aggregate moments.
    # Same S and z, different softmax query outputs: this is a finite-state collision.
    keys=np.array([0.,1.,2.])
    phi=np.stack([np.ones(3),keys],axis=1)
    va=np.array([1.,-2.,1.]); vb=np.zeros(3)
    sa=phi.T@va; sb=phi.T@vb; z=phi.sum(axis=0)
    phiq=np.array([1.,1.])
    lin_a=float(phiq@sa/(phiq@z)); lin_b=float(phiq@sb/(phiq@z))
    weights=softmax(keys.reshape(1,-1))[0]
    sf_a=float(weights@va); sf_b=float(weights@vb)
    # GQA agrees with MHA only when those MHA heads already share exactly the K/V.
    hq,hkv=8,2
    qh=rng.normal(size=(hq,1,16)); kh=rng.normal(size=(hkv,19,16)); vh=rng.normal(size=(hkv,19,16))
    out_gqa=np.stack([dense(qh[h],kh[h//4],vh[h//4]) for h in range(hq)])
    km=np.repeat(kh,4,axis=0); vm=np.repeat(vh,4,axis=0)
    out_mha=np.stack([dense(qh[h],km[h],vm[h]) for h in range(hq)])
    # Educational common geometry; NOT actual DS-V2's layer count or full-model RAM.
    layers,batch,length,dh,bytes_per_value=32,1,32768,128,2
    sizes={name:int(2*layers*batch*length*heads*dh*bytes_per_value)
           for name,heads in [("MHA_32",32),("GQA_8",8),("MQA_1",1)]}
    sizes["MLA_latent512_rope64"]=int(layers*batch*length*(512+64)*bytes_per_value)
    results={
      "identity":"actually executed deterministic CPU teaching experiment",
      "environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},
      "seed":20261116,"dtype":"float64","geometry":{"tokens":257,"head_dimension":32},
      "online_softmax_max_abs_errors":block_errors,
      "causal_prefix_cache_max_abs_error":cache_error,
      "distant_token_mask_counterexample":{"dense":full.tolist(),"last2_only":local.tolist(),"max_abs_error":float(np.max(np.abs(full-local)))},
      "linear_state_collision":{"phi_keys":phi.tolist(),"values_a":va.tolist(),"values_b":vb.tolist(),"S_a":sa.tolist(),"S_b":sb.tolist(),"z":z.tolist(),"linear_output_a":lin_a,"linear_output_b":lin_b,"softmax_output_a":sf_a,"softmax_output_b":sf_b},
      "shared_head_identity_max_abs_error":float(np.max(np.abs(out_mha-out_gqa))),
      "raw_cache_bytes":sizes,"raw_cache_GiB":{k:v/(1024**3) for k,v in sizes.items()},
      "scope":"No trained checkpoint, corpus task score, wall-clock GPU speed, FP16 validation, or full physical-memory measurement."
    }
    assert max(block_errors.values())<1e-12 and cache_error<1e-12
    assert np.allclose(sa,sb) and lin_a==lin_b and abs(sf_a-sf_b)>0.1
    assert results["distant_token_mask_counterexample"]["max_abs_error"]>0.9
    assert results["shared_head_identity_max_abs_error"]==0
    (Path(__file__).parent/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
