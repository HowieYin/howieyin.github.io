"""Actual WikiText/Pythia memory diagnostic; neither Engram pretraining nor a serving benchmark."""
from pathlib import Path
import copy,hashlib,importlib.util,json,math,platform,time
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
R=Path("/tmp/engram-survey-work")
B=Path("/tmp/llm-ptq-survey-work")
torch.set_num_threads(4)
torch.manual_seed(20261119)
device="mps" if torch.backends.mps.is_available() else "cpu"
spec=importlib.util.spec_from_file_location("official",R/"code/engram_demo_v1.py")
official=importlib.util.module_from_spec(spec)
import sys
LR=float(sys.argv[1]) if len(sys.argv)>1 else .001
TAG="-"+sys.argv[2] if len(sys.argv)>2 else ""
sys.modules["official"]=official
spec.loader.exec_module(official)
tok=AutoTokenizer.from_pretrained(B/"model",local_files_only=True)
def token_stream(split):
    return tok("\n\n".join(pq.read_table(B/f"{split}.parquet").column("text").to_pylist()),return_tensors="pt").input_ids[0]
train_stream=token_stream("train")
test_stream=token_stream("test")
train=train_stream[:8192].reshape(-1,128)
test=test_stream[:4096].reshape(-1,256)
mapping=official.NgramHashMapping([1021,1021],3,128,4,[1],str(B/"model"),tok.eos_token_id,17)
def hashes(ids): return torch.from_numpy(mapping.hash(ids.numpy())[1])
train_hash=[hashes(x[None]) for x in train]
test_hash=[hashes(x[None]) for x in test]
model=AutoModelForCausalLM.from_pretrained(B/"model",local_files_only=True,attn_implementation="eager",dtype=torch.float32).eval().requires_grad_(False).to(device)
hidden=model.config.hidden_size
state={"ids":None,"hash":None,"memory":None,"capture":None}
def add(m,args):
    x=args[0]
    if state["capture"] is not None: state["capture"].append(x.detach().cpu())
    if state["memory"] is not None: return (x+state["memory"](x,state["hash"]),*args[1:])
class Memory(nn.Module):
    def __init__(self,mode):
        super().__init__()
        self.mode=mode
        sizes=[s for order in mapping.vocab_size_across_layers[1] for s in order]
        self.register_buffer("offsets",torch.tensor([0]+list(np.cumsum(sizes)[:-1]),dtype=torch.long))
        self.table=nn.Embedding(sum(sizes),32)
        nn.init.normal_(self.table.weight,0,.02)
        self.value=nn.Linear(256,hidden,bias=False)
        nn.init.zeros_(self.value.weight)
        if mode=="layer-gated":
            self.key=nn.Linear(256,hidden,bias=False)
            self.normx=nn.RMSNorm(hidden)
            self.normk=nn.RMSNorm(hidden)
            self.conv=official.ShortConv(hidden,4,3,hc_mult=1)
    def components(self,x,h):
        e=self.table(h.to(x.device)+self.offsets).flatten(-2)
        v=self.value(e)
        if self.mode=="input": return v
        score=(self.normx(x)*self.normk(self.key(e))).sum(-1)/math.sqrt(hidden)
        score=score.abs().clamp_min(1e-6).sqrt()*score.sign()
        return score.sigmoid().unsqueeze(-1)*v
    def forward(self,x,h):
        u=self.components(x,h)
        return u if self.mode=="input" else u+self.conv(u.unsqueeze(2)).squeeze(2)
@torch.inference_mode()
def evaluate(windows=test,hs=test_hash):
    losses=[];first=None
    for i,(x,h) in enumerate(zip(windows,hs)):
        state["ids"]=x[None]
        state["hash"]=h
        y=model(x[None].to(device),use_cache=False).logits[0,:-1].float()
        losses.append(float(F.cross_entropy(y,x[1:].to(device),reduction="sum").cpu()))
        if i==0: first=y[:32].cpu()
    count=sum(x.numel()-1 for x in windows)
    nll=sum(losses)/count
    return {"nll":nll,"ppl":math.exp(nll),"targets":count},first
result={
"identity":"actual 70M frozen pretrained backbone; input lookup and single-layer gated memory are teaching-scale adapters, not full OE/Engram",
"model":"EleutherAI/pythia-70m-deduped","model_revision":json.loads((B/"model-meta.json").read_text())["sha"],
"dataset":"WikiText2 raw-v1","dataset_revision":json.loads((B/"dataset-meta.json").read_text())["sha"],
"protocol":{"train_tokens":8192,"train_window":128,"test_tokens":4096,"test_window":256,"join":"two newlines, original rows",
 "steps":120,"batch":1,"lr":LR,"optimizer":"AdamW default betas/eps, weight_decay=0","seeds":[20261119,20261120,20261121],
 "backbone":"frozen, eval mode, eager attention, float32","memory":"n=2/3; 4 heads/order; 32 values/head; prime buckets from1021; zero value projection; hash seed17",
 "input_injection":"before embedding-layer normalization, at gpt_neox.emb_dropout",
 "layer_injection":"before block index1 (second block); one branch; official signed-sqrt gate; kernel4/dilation3 causal conv",
 "fairness":"same data order per seed and step budget, unequal trainable parameters and backward compute; no hyperparameter search or universal method ranking",
 "history":"each train/test window starts a fresh request; EOS pad id for hash; no inter-window carry"},
"environment":{"platform":platform.platform(),"torch":torch.__version__,"device":device,"python":platform.python_version()},
"token_hashes":{k:hashlib.sha256(x.numpy().tobytes()).hexdigest() for k,x in [("train",train),("test",test)]},
"runs":[]}
base,base_logits=evaluate()
result["baseline"]=base
print("BASELINE",base,flush=True)
# Trace actual suffix keys separately from update frequency.
trace_ids=train_stream[:65536][None]
trace_hash=hashes(trace_ids)[0].numpy()
compressed=mapping.compressed_tokenizer(trace_ids.numpy())[0]
trace=[]
for n in [2,3]:
    keys=[tuple(int(t) for t in compressed[i-n+1:i+1]) for i in range(n-1,len(compressed))]
    unique=sorted(set(keys))
    key_ids=np.asarray(unique,dtype=np.int64)
    h=mapping._get_ngram_hashes(key_ids,1)[:,-1,(n-2)*4:(n-1)*4]
    bucket_counts=[len(set(h[:,j])) for j in range(4)]
    joint=len(set(map(tuple,h.tolist())))
    from collections import Counter
    events=Counter(keys)
    seen={}
    affected_events=0
    for key,address in zip(unique,h[:,0]):
        seen.setdefault(int(address),[]).append(key)
    colliding=set(k for bucket in seen.values() if len(bucket)>1 for k in bucket)
    affected_events=sum(events[k] for k in colliding)
    trace.append({"order":n,"events":len(keys),"unique_canonical_keys":len(unique),"occupied_rows_per_head":bucket_counts,
      "distinct_joint_addresses":joint,"joint_key_merges":len(unique)-joint,
      "head0_colliding_unique_keys":len(colliding),"head0_events_in_colliding_buckets":affected_events,
      "bucket_sizes":mapping.vocab_size_across_layers[1][n-2]})
result["hash_trace"]={"tokens":trace_ids.numel(),"canonical_vocab":len(mapping.compressed_tokenizer),
 "original_vocab":len(tok),"trace_sha256":hashlib.sha256(trace_ids.numpy().tobytes()).hexdigest(),"orders":trace,
 "warning":"canonical normalization aliases excluded before counting hash collisions; sharing one head differs from sharing the whole concatenation"}
for seed in result["protocol"]["seeds"]:
    for mode in ["input","layer-gated"]:
        torch.manual_seed(seed)
        mem=Memory(mode).to(device)
        state["memory"]=mem
        module=model.gpt_neox.emb_dropout if mode=="input" else model.gpt_neox.layers[1]
        handle=module.register_forward_pre_hook(add)
        opt=torch.optim.AdamW(mem.parameters(),lr=LR,weight_decay=0)
        rng=np.random.default_rng(seed)
        order=np.concatenate([rng.permutation(len(train)) for _ in range(2)])[:120]
        start=time.perf_counter()
        history=[]
        for step,index in enumerate(order):
            state["hash"]=train_hash[index]
            x=train[index][None].to(device)
            opt.zero_grad(set_to_none=True)
            loss=model(x,labels=x,use_cache=False).loss
            loss.backward()
            opt.step()
            history.append(float(loss.detach().cpu()))
            if (step+1)%30==0: print("TRAIN",seed,mode,step+1,history[-1],flush=True)
        if device=="mps": torch.mps.synchronize()
        seconds=time.perf_counter()-start
        score,ref=evaluate()
        row={"seed":seed,"mode":mode,"trainable_parameters":sum(p.numel() for p in mem.parameters()),
             "training_wall_seconds":seconds,"loss_history":history,**score}
        if mode=="layer-gated" and seed==20261119:
            torch.save(mem.cpu().state_dict(),R/f"trained-memory{TAG}.pt")
            # Exact stream equivalence uses real layer activations and CPU float64 to isolate the memory branch.
            mem.to(device)
            state["capture"]=[]
            evaluate(test[:1],test_hash[:1])
            xfull=state["capture"][0].double()
            state["capture"]=None
            cpu=copy.deepcopy(mem).cpu().double()
            full_ids=test[:1]
            full_h=test_hash[0]
            with torch.inference_mode():
                whole=cpu(xfull,full_h)
                def streaming(lose_tokens=False,lose_conv=False):
                    outs=[];id_history=full_ids[:,:0];u_history=xfull[:,:0];position=0
                    for width in [37,29,1,53,7,129]:
                        ids=full_ids[:,position:position+width]
                        ix=torch.cat([id_history,ids],dim=1)
                        h=hashes(ix)[:,-width:]
                        u=cpu.components(xfull[:,position:position+width],h)
                        ux=torch.cat([u_history,u],dim=1)
                        y=u+cpu.conv(ux.unsqueeze(2)).squeeze(2)[:,-width:]
                        outs.append(y)
                        id_history=ix[:,-2:] if not lose_tokens else ix[:,:0]
                        u_history=ux[:,-9:] if not lose_conv else ux[:,:0]
                        position+=width
                    return torch.cat(outs,dim=1)
                good=streaming()
                result["streaming"]={"dtype":"CPU float64, real block1 input activations",
                 "chunk_sizes":[37,29,1,53,7,129],"token_history":2,"convolution_history":9,
                 "correct_max_abs":float((good-whole).abs().max()),
                 "dropped_token_history_max_abs":float((streaming(True)-whole).abs().max()),
                 "dropped_convolution_history_max_abs":float((streaming(False,True)-whole).abs().max()),
                 "restored_max_abs":float((streaming()-whole).abs().max())}
            original_forward=mem.forward
            def fault_forward(x,h):
                parts=[]
                for position in range(0,x.shape[1],13):
                    hx=hashes(state["ids"][:,position:position+13])
                    parts.append(original_forward(x[:,position:position+13],hx))
                return torch.cat(parts,dim=1)
            mem.forward=fault_forward
            bad,bad_logits=evaluate()
            mem.forward=original_forward
            fixed,fixed_logits=evaluate()
            row["full_model_state_fault"]={
                "fault":"reset token and convolution history at each13token chunk inside actual model forward",
                "fault_nll":bad["nll"],"fault_ppl":bad["ppl"],
                "fault_sample_logits_max_abs":float((bad_logits-ref).abs().max()),
                "restored_nll":fixed["nll"],"restored_sample_logits_max_abs":float((fixed_logits-ref).abs().max())}
            # Quantize only trained table then restore full function.
            w=mem.table.weight.detach().clone()
            qrows=[]
            for bits in [8,4,2]:
                qmax=2**(bits-1)-1
                scale=w.abs().amax(-1,keepdim=True).clamp_min(1e-12)/qmax
                q=(w/scale).round().clamp(-qmax,qmax)
                with torch.no_grad():mem.table.weight.copy_(q*scale)
                val,_=evaluate()
                qrows.append({"bits":bits,**val,"relative_weight_l2":float(((q*scale-w).norm()/w.norm()).cpu()),
                  "theoretical_payload_bytes":w.numel()*bits/8,
                  "row_scale_bytes_fp16":w.shape[0]*2,
                  "actual_compute":"codes dequantized to FP32 before gather; no measured packed kernel speed or read-byte saving"})
            with torch.no_grad():mem.table.weight.copy_(w)
            restored,rl=evaluate()
            row["quantized_tables"]=qrows
            row["restoration"]={"nll":restored["nll"],"logits_max_abs":float((rl-ref).abs().max())}
        result["runs"].append(row)
        print("RESULT",{k:v for k,v in row.items() if k not in {"loss_history","quantized_tables"}},flush=True)
        (R/f"experiment-results{TAG}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        handle.remove()
        state["memory"]=None
        del opt,mem
        if device=="mps":torch.mps.empty_cache()
final,fl=evaluate()
result["base_restoration"]={"nll":final["nll"],"logits_max_abs":float((fl-base_logits).abs().max())}
(R/f"experiment-results{TAG}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
print("COMPLETE",result["streaming"],result["base_restoration"],flush=True)
