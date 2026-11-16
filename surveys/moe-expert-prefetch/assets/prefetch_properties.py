"""CPU teaching experiment: actual MoE numerical execution, no GPU benchmark.

Python 3 + NumPy. Seeded random weights, 128 independent inputs, 3 MoE layers,
64 experts/layer, top-2, hidden dim 16, FFN dim 32. Deliberately inject wrong
prefetch sets. Reference routing is recomputed from current inputs at each layer.
"""
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
import json
import platform
import numpy as np

ROOT=Path(__file__).resolve().parent
SEED=20261116
E,K,D,F,L,N=64,2,16,32,3,128
rng=np.random.default_rng(SEED)
gates=rng.normal(0,.2,(L,D,E))
up=rng.normal(0,.15,(L,E,D,F))
down=rng.normal(0,.15,(L,E,F,D))
inputs=rng.normal(0,1,(N,D))

def route(h,l):
    scores=h@gates[l]
    chosen=np.argsort(-scores,kind="stable")[:K]
    s=scores[chosen]; w=np.exp(s-s.max()); w/=w.sum()
    return chosen,w

def expert(h,l,e,u=up,d=down):
    z=h@u[l,e]
    z=z/(1+np.exp(-z))
    return z@d[l,e]

def reference():
    h=inputs.copy(); trace=[]
    for l in range(L):
        ids=[]; rows=[]
        for x in h:
            chosen,w=route(x,l); ids.append(chosen)
            rows.append(x+sum(wi*expert(x,l,int(e)) for e,wi in zip(chosen,w)))
        trace.append(np.array(ids)); h=np.array(rows)
    return h,np.array(trace)

def make_predictions(trace,wrong_fraction):
    pred=trace.copy()
    wrong_count=round(L*N*wrong_fraction)
    for ix in rng.permutation(L*N)[:wrong_count]:
        l,t=divmod(int(ix),N)
        alternatives=np.setdiff1d(np.arange(E),trace[l,t])
        pred[l,t]=rng.choice(alternatives,K,replace=False)
    return pred

def execute(pred,approximate=False):
    h=inputs.copy(); cache=[OrderedDict() for _ in range(L)]
    traffic={"prefetch_copies":0,"demand_copies":0,"demand_hits":0}
    def fetch(l,e,kind):
        if e in cache[l]:
            cache[l].move_to_end(e)
            if kind=="demand":traffic["demand_hits"]+=1
            return cache[l][e]
        # Actual byte copies of weight tensors into an eight-slot CPU cache.
        value=(up[l,e].copy(),down[l,e].copy())
        traffic[f"{kind}_copies"]+=1
        if len(cache[l])==8:cache[l].popitem(last=False)
        cache[l][e]=value
        return value
    for l in range(L):
        rows=[]
        for t,x in enumerate(h):
            for e in pred[l,t]:fetch(l,int(e),"prefetch")
            actual,w=route(x,l)
            if approximate:
                actual=pred[l,t]
                scores=(x@gates[l])[actual]; w=np.exp(scores-scores.max());w/=w.sum()
            outputs=[]
            for e,wi in zip(actual,w):
                # Demand fallback fetches every actually selected expert.
                u,d=fetch(l,int(e),"demand")
                z=x@u; z=z/(1+np.exp(-z))
                outputs.append(wi*(z@d))
            rows.append(x+sum(outputs))
        h=np.array(rows)
    return h,traffic

def classification(actual,pred):
    tp=sum(len(set(a)&set(p)) for a,p in zip(actual.reshape(-1,K),pred.reshape(-1,K)))
    positive=L*N*K; fp=L*N*K-tp; fn=positive-tp; tn=L*N*E-tp-fp-fn
    precision=tp/(tp+fp);recall=tp/(tp+fn)
    return {"TP":tp,"FP":fp,"TN":tn,"FN":fn,
            "Accuracy":(tp+tn)/(L*N*E),"Precision":precision,"Recall":recall,
            "F1":2*precision*recall/(precision+recall) if precision+recall else 0,
            "exact_set_match":float(np.mean(np.all(np.sort(actual,axis=-1)==np.sort(pred,axis=-1),axis=-1)))}

def feasible(tasks):
    return all(i+1<=t["deadline"] for i,t in enumerate(sorted(tasks,key=lambda t:t["deadline"])))

def scheduling():
    jobs=[{"id":"A","weight":100,"deadline":3},{"id":"B","weight":99,"deadline":3},
          {"id":"C","weight":90,"deadline":1},{"id":"D","weight":89,"deadline":2}]
    selected=[]
    for task in sorted(jobs,key=lambda t:-t["weight"]):
        if feasible(selected+[task]):selected.append(task)
    schedule=sorted(selected,key=lambda t:t["deadline"])
    best=max(sum(t["weight"] for t in subset) for n in range(5)
             for subset in combinations(jobs,n) if feasible(subset))
    greedy=sum(t["weight"] for t in selected)
    fifo=sum(t["weight"] for i,t in enumerate(jobs) if i+1<=t["deadline"])
    assert greedy==best==289 and fifo==199
    # Model async completion with slot generations, independent of real CUDA.
    slot={"tag":2,"state":"Fetch"}
    def complete(tag):
        if tag==slot["tag"]:slot["state"]="Used"
    complete(1);assert slot["state"]=="Fetch"
    complete(2);assert slot["state"]=="Used"
    return {"unit_job_assumptions":"all releases 0, equal one-unit duration, one channel",
            "deadline_order":[t["id"] for t in schedule],"matroid_weight":greedy,
            "exhaustive_optimum":best,"FIFO_timely_weight":fifo,
            "inflight_copy_wait_teaching_units":{"whole_9_arrival_at_1":8,"chunk_3_arrival_at_1":2},
            "stale_generation_completion":"ignored; current completion accepted"}

target,trace=reference()
results=[]
for fraction in [0,.25,.5,.75,1]:
    pred=make_predictions(trace,fraction)
    exact,traffic=execute(pred)
    approx,_=execute(pred,True)
    error=float(np.max(np.abs(exact-target)))
    rel=float(np.linalg.norm(approx-target)/np.linalg.norm(target))
    assert error==0
    assert (rel==0) if fraction==0 else (rel>0)
    results.append({"wrong_set_fraction":fraction,"metrics":classification(trace,pred),
                    "fallback_max_abs_error":error,"approximate_relative_L2":rel,
                    "cache_weight_traffic":traffic})

correct=np.arange(100)<80
timely=np.arange(100)>=20
joint=float(np.mean(correct&timely))
product=float(np.mean(correct)*np.mean(timely))
assert joint==.6 and np.isclose(product,.64)
payload={
 "identity":"actual CPU numerical toy and logical scheduling tests; synthetic weights and inputs",
 "seed":SEED,"environment":{"python":platform.python_version(),"numpy":np.__version__,
                             "platform":platform.platform()},
 "configuration":{"layers":L,"experts":E,"top_k":K,"inputs":N,"hidden":D,"ffn":F,
                  "dtype":"float64","cache_slots_per_layer":8,"autoregressive":False},
 "results":results,
 "always_negative":{"TP":0,"FP":0,"TN":L*N*(E-K),"FN":L*N*K,
                    "Accuracy":1-K/E,"Precision":None,"Recall":0,"F1":0},
 "batch_union_counterexample":{"actual":[[0,1],[2,3]],"predicted":[[2,3],[0,1]],
                               "token_recall":0,"union_recall":1},
 "joint_counterexample":{"correct":.8,"timely":.8,"joint":joint,"marginal_product":product},
 "scheduling":scheduling(),
 "limits":["No learned predictor, pretrained checkpoint, token perplexity or task accuracy.",
           "No real DMA, GPU, SSD or timing speedup measurement.",
           "Float64 same operation order; not a mixed-kernel bitwise-equivalence proof."],
}
(ROOT/"results.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(payload,ensure_ascii=False,indent=2))
