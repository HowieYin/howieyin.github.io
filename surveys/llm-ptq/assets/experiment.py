"""Real pretrained Pythia diagnostic; fake quantization, not low-bit kernels."""
from pathlib import Path
import argparse, json, hashlib, math, platform, time
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir",type=Path,default=Path("inputs"),
                    help="Directory containing frozen model, splits and metadata")
B=parser.parse_args().data_dir
torch.manual_seed(20261119); torch.set_num_threads(4)
device="mps" if torch.backends.mps.is_available() else "cpu"
tokenizer=AutoTokenizer.from_pretrained(B/"model",local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(B/"model",local_files_only=True,
    attn_implementation="eager",dtype=torch.float32).eval().requires_grad_(False).to(device)
def corpus(split):
    return "\n\n".join(pq.read_table(B/f"{split}.parquet").column("text").to_pylist())
def tokens(split,count):
    return tokenizer(corpus(split),return_tensors="pt").input_ids[0,:count].reshape(-1,256)
train=tokens("train",2048);test=tokens("test",4096)
layers={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith("gpt_neox.layers.")}
original={n:m.weight.detach().cpu().clone() for n,m in layers.items()}
energy={n:torch.zeros(m.in_features,dtype=torch.float64) for n,m in layers.items()}
samples={n:0 for n in layers};example_inputs={}
def hook(name):
    def collect(m,args):
        x=args[0].detach().reshape(-1,m.in_features).cpu().double()
        if name not in example_inputs:example_inputs[name]=x[:32].float()
        energy[name]+=x.square().sum(0);samples[name]+=x.shape[0]
    return collect
handles=[m.register_forward_pre_hook(hook(n)) for n,m in layers.items()]
with torch.inference_mode():
    for x in train:model(x[None].to(device),use_cache=False)
for h in handles:h.remove()
energy={n:(v/samples[n]).float() for n,v in energy.items()}

def qdq(w,bits,granularity,weighted=None):
    qmax=2**(bits-1)-1
    if granularity=="tensor":z=w.reshape(1,-1)
    elif granularity=="row":z=w
    else:z=w.reshape(w.shape[0],-1,128).reshape(-1,128)
    peak=z.abs().amax(-1,keepdim=True).clamp(min=1e-12)
    def candidate(ratio):
        scale=peak*ratio/qmax
        return torch.clamp(torch.round(z/scale),-qmax,qmax)*scale
    best=candidate(1.0);ratios=torch.ones(z.shape[0],1)
    if weighted is not None:
        importance=weighted.expand(w.shape[0],-1).reshape_as(z)
        score=((best-z).square()*importance).sum(-1,keepdim=True)
        for ratio in [.99,.97,.95,.9,.85,.8]:
            out=candidate(ratio)
            other=((out-z).square()*importance).sum(-1,keepdim=True)
            improve=other<score
            best=torch.where(improve,out,best);score=torch.minimum(score,other)
            ratios=torch.where(improve,torch.full_like(ratios,ratio),ratios)
    return best.reshape_as(w),float((ratios<1).float().mean())

@torch.inference_mode()
def evaluate():
    losses=[];logits=[]
    for i,x in enumerate(test):
        y=model(x[None].to(device),use_cache=False).logits[0,:-1].float()
        losses.append(float(F.cross_entropy(y,x[1:].to(device),reduction="sum").cpu()))
        if i==0:logits=y[:32].cpu()
    nll=sum(losses)/(len(test)*255)
    return {"nll_nats":nll,"perplexity":math.exp(nll),"scored_tokens":len(test)*255},logits

result={"identity":"actual pretrained-model diagnostic; all quantized matrices dequantized back to float32",
 "model":"EleutherAI/pythia-70m-deduped","model_revision":json.loads((B/"model-meta.json").read_text())["sha"],
 "parameters":sum(p.numel() for p in model.parameters()),"linear_layers":len(layers),
 "quantized_parameters":sum(w.numel() for w in original.values()),
 "excluded":"embedding, output head, normalization, biases, activations and KV cache",
 "environment":{"python":platform.python_version(),"torch":torch.__version__,
  "transformers":transformers.__version__,"numpy":np.__version__,"pyarrow":__import__("pyarrow").__version__,
  "platform":platform.platform(),"device":device,"compute_dtype":"float32","seed":20261119},
 "protocol":{"dataset":"Salesforce/wikitext / wikitext-2-raw-v1",
  "dataset_revision":json.loads((B/"dataset-meta.json").read_text())["sha"],
  "calibration":"first 2048 train tokens, eight nonoverlapping length256 windows",
  "evaluation":"first 4096 test tokens, sixteen nonoverlapping length256 windows, independent window prefixes and reset positions; no extra BOS inserted; 4080 targets",
  "joining":"all original rows joined by two newlines; no cleanup, no chat template",
  "token_hashes":{s:hashlib.sha256(t.numpy().tobytes()).hexdigest() for s,t in [("train",train),("test",test)]},
  "rounding":"nearest, ties-to-even, symmetric narrow-range (-7..7 or -127..127), no zero point",
  "calibration_objective":"diagonal activation-energy weighted weight MSE, independently per output-row/input group; not GPTQ/AWQ reproduction",
  "clipping_candidates":[1,.99,.97,.95,.9,.85,.8]},
 "runs":[]}
baseline,ref=evaluate();result["runs"].append({"mode":"FP32",**baseline})
print(result["runs"][-1],flush=True)
for name,bits,gran,cal in [("W8 per-tensor",8,"tensor",False),("W4 per-tensor",4,"tensor",False),
 ("W4 per-output-row",4,"row",False),("W4 group128 RTN",4,"group",False),
 ("W4 group128 calibrated clip",4,"group",True)]:
    clipped=[];errsum=0.;normsum=0.;proxy_err=0.;proxy_norm=0.;layer_stats=[]
    for n,m in layers.items():
        w=original[n];out,fraction=qdq(w,bits,gran,energy[n] if cal else None)
        m.weight.copy_(out.to(device))
        clipped.append(fraction);errsum+=float((out-w).square().sum());normsum+=float(w.square().sum())
        proxy_err+=float(((out-w).square()*energy[n]).sum())
        proxy_norm+=float((w.square()*energy[n]).sum())
        layer_stats.append({"layer":n,"shape":list(w.shape),"max_abs":float(w.abs().max()),
            "weight_relative_l2":float((out-w).norm()/w.norm()),
            "zero_fraction":float((out==0).float().mean())})
    score,current=evaluate()
    row={"mode":name,**score,"weight_relative_l2":math.sqrt(errsum/normsum),
         "mean_clipped_group_fraction":float(np.mean(clipped)),
         "activation_energy_proxy_relative_l2":math.sqrt(proxy_err/proxy_norm),
         "per_layer":layer_stats,
         "sample_logits_relative_l2":float((current-ref).norm()/ref.norm())}
    result["runs"].append(row);print({k:v for k,v in row.items() if k!="per_layer"},flush=True)
    (B/"experiment-results.json").write_text(json.dumps(result,indent=2))
with torch.no_grad():
    for n,m in layers.items():m.weight.copy_(original[n].to(device))
restored,restore_logits=evaluate()
result["restore_check"]={"nll_nats":restored["nll_nats"],"logits_max_abs":float((ref-restore_logits).abs().max())}
# Two-window CPU check isolates backend differences from weight quantization.
model.to("cpu")
cpu_checks=[]
for mode in ["FP32","W8 per-tensor"]:
    if mode!="FP32":
        with torch.no_grad():
            for n,m in layers.items():m.weight.copy_(qdq(original[n],8,"tensor")[0])
    with torch.inference_mode():
        loss=sum(float(F.cross_entropy(model(x[None],use_cache=False).logits[0,:-1].float(),x[1:],reduction="sum")) for x in test[:2])
    cpu_checks.append({"mode":mode,"nll_nats":loss/510,"scored_tokens":510})
result["cpu_spot_check"]=cpu_checks
# Export-format fault injection uses real weight values, no synthetic data.
w=next(iter(original.values()))[:32,:128].clone()
qmax=7;s=w.abs().amax(1,keepdim=True).clamp(min=1e-12)/qmax
codes=torch.clamp(torch.round(w/s),-7,7).to(torch.int8)
unsigned=(codes+8).numpy().astype(np.uint8)
packed=unsigned[:,0::2]|(unsigned[:,1::2]<<4)
unpacked=np.empty_like(unsigned);unpacked[:,0::2]=packed&15;unpacked[:,1::2]=packed>>4
decoded=torch.from_numpy(unpacked.astype(np.int16)-8).float()*s
bad=torch.from_numpy(unpacked.astype(np.int16)-7).float()*s
transposed=torch.from_numpy(unpacked.T.copy().reshape(unpacked.shape).astype(np.int16)-8).float()*s
def relative(x,y):return float((x-y).norm()/y.norm())
result["format_faults"]={"real_matrix_shape":list(w.shape),"payload_bytes":int(packed.nbytes),
 "float32_bytes":w.numel()*4,"round_trip_code_mismatches":int(np.sum(unsigned!=unpacked)),
 "correct_decode_max_abs":float((decoded-codes.float()*s).abs().max()),
 "wrong_zero_offset_relative_l2":relative(bad,decoded),
 "wrong_layout_relative_l2":relative(transposed,decoded),
 "meaning":"offset8 is a storage encoding for symmetric signed codes; using offset7 or reinterpreting a transpose corrupts the same real quantized values"}
x=next(iter(example_inputs.values()))[:,:128];y=x@decoded.T
result["format_faults"]["correct_output_max_abs"]=float((x@decoded.T-y).abs().max())
result["format_faults"]["wrong_zero_offset_output_relative_l2"]=relative(x@bad.T,y)
result["format_faults"]["wrong_layout_output_relative_l2"]=relative(x@transposed.T,y)
(B/"experiment-results.json").write_text(json.dumps(result,indent=2))
print(result["restore_check"],result["format_faults"],flush=True)
