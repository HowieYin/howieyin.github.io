"""Real handwritten digits: recurrent representations, state recovery and timing.

This is a small CPU diagnostic, not a reproduction of any paper benchmark.
Run with an interpreter containing torch, numpy and scikit-learn.
"""
from pathlib import Path
import copy,hashlib,json,platform,time
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parent
ROOT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

class Classifier(nn.Module):
    def __init__(self,kind,input_size,hidden=48):
        super().__init__()
        self.rnn=getattr(nn,kind)(input_size,hidden,batch_first=True)
        self.head=nn.Linear(hidden,10)
    def forward(self,x,state=None):
        y,state=self.rnn(x,state)
        return self.head(y[:,-1]),state

def clone(state):
    return tuple(s.detach().clone() for s in state) if isinstance(state,tuple) else state.detach().clone()

def metrics(target,pred):
    cm=confusion_matrix(target,pred,labels=range(10))
    per=[]
    for c in range(10):
        tp=int(cm[c,c]);fn=int(cm[c].sum()-tp);fp=int(cm[:,c].sum()-tp)
        tn=int(cm.sum()-tp-fn-fp)
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        per.append({"class":c,"TP":tp,"FP":fp,"TN":tn,"FN":fn,
          "precision":precision,"recall":recall,"f1":f1,"support":tp+fn})
    return {"n":len(target),"correct":int(np.trace(cm)),"accuracy":float(np.trace(cm)/cm.sum()),
      "macro_precision":float(np.mean([r["precision"] for r in per])),
      "macro_recall":float(np.mean([r["recall"] for r in per])),
      "macro_f1":float(np.mean([r["f1"] for r in per])),"confusion_matrix":cm.tolist(),"per_class":per}

def bench(fn,repeats=60):
    for _ in range(8):fn()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-start)/1e6)
    return {"median_ms":float(np.median(values)),"p10_ms":float(np.percentile(values,10)),
      "p90_ms":float(np.percentile(values,90)),"repeats":repeats,"warmups":8}

def stream(model,x,chunk,state=None,reset=False,drop_c=False):
    for offset in range(0,x.shape[1],chunk):
        if reset:state=None
        if drop_c and isinstance(state,tuple):state=(state[0],torch.zeros_like(state[1]))
        logits,state=model(x[:,offset:offset+chunk],state)
    return logits,state

def diagnose(model,x,y):
    with torch.no_grad():
        full,state=model(x)
        base_pred=full.argmax(-1).numpy();rows=[]
        for size in [1,3,5,7]:
            out,rest=stream(model,x,size)
            rows.append({"chunk":size,"max_logit_error":float((out-full).abs().max()),
              "prediction_changes":int((out.argmax(-1).numpy()!=base_pred).sum())})
        cut=x.shape[1]//2
        _,prefix=model(x[:,:cut])
        torch.save(clone(prefix),ROOT/"state-restore.pt")
        restored=torch.load(ROOT/"state-restore.pt",weights_only=True)
        suffix,_=model(x[:,cut:],restored)
        reset,_=stream(model,x,3,reset=True)
        contamination=clone(state)
        # A previous request occupies the same batch slots. The next request is
        # the reverse ordering of held-out images; only the state is mismatched.
        carried,_=model(x.flip(0),contamination)
        clean_next,_=model(x.flip(0))
        transpose,_=model(x.flip(1))
        failures={
          "reset_each_chunk":metrics(y,reset.argmax(-1).numpy()),
          "cross_request_state":metrics(y[::-1].copy(),carried.argmax(-1).numpy()),
          "cross_request_clean":metrics(y[::-1].copy(),clean_next.argmax(-1).numpy()),
          "reverse_sequence":metrics(y,transpose.argmax(-1).numpy())}
        if isinstance(prefix,tuple):
            wrong,_=stream(model,x,3,drop_c=True)
            failures["drop_lstm_cell_state"]=metrics(y,wrong.argmax(-1).numpy())
        # Quantize only the boundary state; weights and arithmetic remain FP32.
        def q8(v):
            scale=v.abs().max()/127
            return v if float(scale)==0 else (v/scale).round().clamp(-127,127)*scale
        quant=tuple(q8(v) for v in prefix) if isinstance(prefix,tuple) else q8(prefix)
        qout,_=model(x[:,cut:],quant)
        byte_count=sum(v.numel()*v.element_size() for v in state) if isinstance(state,tuple) else state.numel()*state.element_size()
        single=x[:1]
        full_time=bench(lambda:model(single))
        stepped_time=bench(lambda:stream(model,single,1))
        # The benchmark returns logits only at the last step; head is called
        # at every chunk in streaming mode and that difference is stated.
    probe=x[:4].detach().clone().requires_grad_(True)
    objective=model(probe)[0].square().mean()
    full_grad=torch.autograd.grad(objective,probe)[0]
    probe2=x[:4].detach().clone().requires_grad_(True)
    _,pstate=model(probe2[:,:cut]);pstate=clone(pstate)
    objective2=model(probe2[:,cut:],pstate)[0].square().mean()
    detached_grad=torch.autograd.grad(objective2,probe2)[0]
    return {"chunks":rows,"state_restore_max_logit_error":float((suffix-full).abs().max()),
      "state_restore_prediction_changes":int((suffix.argmax(-1).numpy()!=base_pred).sum()),
      "state_boundary_int8_max_logit_error":float((qout-full).abs().max()),
      "state_boundary_int8_prediction_changes":int((qout.argmax(-1).numpy()!=base_pred).sum()),
      "state_bytes_batch360":byte_count,"state_bytes_per_image":byte_count//len(x),
      "failure_metrics":failures,"full_sequence_timing":full_time,"stepwise_timing":stepped_time,
      "detach_probe":{"n":4,"cut":cut,"full_forward_loss":float(objective.detach()),
        "detached_forward_loss":float(objective2.detach()),
        "full_prefix_grad_l1":float(full_grad[:,:cut].abs().sum()),
        "detached_prefix_grad_l1":float(detached_grad[:,:cut].abs().sum()),
        "suffix_gradient_max_error":float((full_grad[:,cut:]-detached_grad[:,cut:]).abs().max())}}

data=load_digits()
images=np.asarray(data.images,dtype=np.float32)/16.
target=np.asarray(data.target,dtype=np.int64)
indices=np.arange(len(target))
train,rest=train_test_split(indices,test_size=.4,random_state=20261123,stratify=target)
valid,test=train_test_split(rest,test_size=.5,random_state=20261124,stratify=target[rest])
np.savez_compressed(ROOT/"dataset-and-splits.npz",images=images,target=target,train=train,valid=valid,test=test)
config={"dataset":"sklearn load_digits; real UCI handwritten digits","dataset_n":len(target),
 "input_range":[0.,1.],"train_n":len(train),"validation_n":len(valid),"test_n":len(test),
 "split_random_states":[20261123,20261124],"seeds":[7,19,41],"epochs":35,"batch":64,
 "hidden":48,"layers":1,"optimizer":"Adam","learning_rate":.003,"gradient_clip_norm":1.,
 "checkpoint_selection":"lowest held-out validation cross entropy; test untouched until selection",
 "precision":"FP32","device":"CPU","threads":1,"torch":torch.__version__,"numpy":np.__version__,
 "platform":platform.platform(),"machine":platform.machine(),"timing_boundary":"rnn forward plus linear head, input already normalized, excludes model load and disk IO",
 "parameter_budget":"same hidden width, different parameters; not a matched-parameter benchmark"}
results={"config":config,"models":[]}
for layout in ["rows","pixels"]:
    allx=torch.from_numpy(images if layout=="rows" else images.reshape(-1,64,1))
    ally=torch.from_numpy(target)
    for kind in ["RNN","GRU","LSTM"]:
        for seed in config["seeds"]:
            torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed)
            model=Classifier(kind,allx.shape[-1]);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
            best=float("inf");beststate=None;trajectory=[];start=time.perf_counter()
            for epoch in range(config["epochs"]):
                model.train();order=torch.randperm(len(train),generator=rng).numpy();total=0.
                for begin in range(0,len(train),64):
                    ix=train[order[begin:begin+64]]
                    optimizer.zero_grad();logits,_=model(allx[ix])
                    loss=nn.functional.cross_entropy(logits,ally[ix]);loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    total+=float(loss.detach())*len(ix)
                model.eval()
                with torch.no_grad():
                    vl=model(allx[valid])[0];v=float(nn.functional.cross_entropy(vl,ally[valid]))
                trajectory.append({"epoch":epoch+1,"train_ce":total/len(train),"valid_ce":v})
                if v<best:best=v;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch+1
            model.load_state_dict(beststate);model.eval()
            key=f"{layout}-{kind.lower()}-seed{seed}"
            torch.save({"kind":kind,"input_size":allx.shape[-1],"hidden":48,"state_dict":beststate},ROOT/(key+".pt"))
            with torch.no_grad():pred=model(allx[test])[0].argmax(-1).numpy()
            entry={"key":key,"layout":layout,"kind":kind,"seed":seed,
              "parameters":sum(p.numel() for p in model.parameters()),"best_epoch":bestepoch,
              "best_valid_ce":best,"training_seconds":time.perf_counter()-start,"trajectory":trajectory,
              "test_metrics":metrics(target[test],pred),"test_predictions":pred.tolist(),
              "diagnostic":diagnose(model,allx[test],target[test])}
            results["models"].append(entry)
            (ROOT/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(key,"acc",entry["test_metrics"]["accuracy"],"elapsed",entry["training_seconds"],flush=True)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in {".npz",".pt",".json",".py"}}
(ROOT/"sha256.json").write_text(json.dumps(manifest,indent=2))
print("Completed all real-data models and fault diagnostics",flush=True)
