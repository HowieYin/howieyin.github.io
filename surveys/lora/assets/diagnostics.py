"""Adapter algebra, quantization order and retained-tensor diagnostics on real inputs.

This is not a LoRAHub, QLoRA or LoRA-FA paper reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, required=True)
parser.add_argument("--adapter-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.data_dir/"model", local_files_only=True)
text = "\n\n".join(pq.read_table(args.data_dir/"test.parquet").column("text").to_pylist())
test = tok(text, return_tensors="pt").input_ids[0,:4096].reshape(-1,256)
def base(dtype=torch.float32, where=device):
    return AutoModelForCausalLM.from_pretrained(args.data_dir/"model", local_files_only=True,
        attn_implementation="eager", dtype=dtype).to(where).eval().requires_grad_(False)
def artifact(seed, rank):
    path=args.adapter_dir/f"adapter-{seed}-lora-{rank}.pt"
    return torch.load(path,map_location="cpu",weights_only=True), hashlib.sha256(path.read_bytes()).hexdigest()
def diff(a,b):
    a,b=a.detach().cpu(),b.detach().cpu()
    return {"max_abs":float((a-b).abs().max()),"mean_abs":float((a-b).abs().mean()),
            "rmse":float((a-b).square().mean().sqrt())}
def quant(w):
    """Teaching symmetric rowwise W4 RTN with 15 used levels, not NF4."""
    scale=w.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7
    return torch.round(w/scale).clamp(-7,7)*scale

class Adapter(nn.Module):
    def __init__(self,layer,factors,fixed_a=False,quantize_base=False):
        super().__init__()
        self.base=layer.requires_grad_(False)
        self.A=nn.Parameter(factors["A"].to(layer.weight.device,layer.weight.dtype),
                            requires_grad=not fixed_a)
        self.B=nn.Parameter(factors["B"].to(layer.weight.device,layer.weight.dtype))
        if quantize_base:
            with torch.no_grad():
                self.base.weight.copy_(quant(self.base.weight))
    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x,self.A),self.B)

model=base()
name="gpt_neox.layers.0.attention.query_key_value"
captured=[]
handle=model.get_submodule(name).register_forward_pre_hook(lambda module,inputs: captured.append(inputs[0].detach().cpu()))
with torch.no_grad():
    model(test[0:1,:64].to(device),use_cache=False)
handle.remove()
x=captured[0].reshape(-1,model.get_submodule(name).in_features).double()
w=model.get_submodule(name).weight.detach().cpu().double()
payload1,hash1=artifact(20261119,4)
payload2,hash2=artifact(20261120,4)
f1,f2=payload1["adapters"][name],payload2["adapters"][name]
A1,B1=f1["A"].double(),f1["B"].double()
A2,B2=f2["A"].double(),f2["B"].double()
dense=.5*(B1@A1)+.5*(B2@A2)
factor=(.5*(B1+B2))@(.5*(A1+A2))
joined=torch.cat((.5*B1,.5*B2),dim=1)@torch.cat((A1,A2),dim=0)
report={
 "identity":"Actual trained adapters and frozen WikiText2 activations; local algebra diagnostic",
 "model_revision":payload1["metadata"]["model_revision"],
 "inputs":{"seed19_rank4_sha256":hash1,"seed20_rank4_sha256":hash2,
           "test_token_sha256":hashlib.sha256(test.numpy().tobytes()).hexdigest()},
 "composition":{"layer":name,"weights":[.5,.5],"each_rank":4,"concatenated_rank_bound":8,
   "factor_average_vs_delta_average":diff(factor,dense),
   "concatenation_vs_delta_average":diff(joined,dense),
   "factor_average_activation_output_error":diff(F.linear(x,factor),F.linear(x,dense)),
   "concatenation_activation_output_error":diff(F.linear(x,joined),F.linear(x,dense)),
   "limit":"same-task different-seed adapters; algebra does not establish multi-task quality"},
 "quantization_order_layer":{
   "format":"per-output-row symmetric W4 RTN, qmax7; 15 used levels; CPU float64 GEMM",
   "weights_side_vs_requantized":diff(quant(w)+B1@A1,quant(w+B1@A1)),
   "outputs_side_vs_requantized":diff(F.linear(x,quant(w))+F.linear(x,B1@A1),F.linear(x,quant(w+B1@A1))),
   "limit":"no NF4, double quantization, low-bit kernel, quantized training or QA-LoRA group restriction"},
}
assert report["composition"]["concatenation_vs_delta_average"]["max_abs"]<1e-12
assert report["composition"]["factor_average_vs_delta_average"]["max_abs"]>1e-8
del model

@torch.no_grad()
def score(model):
    total=0.
    for t in test:
        logits=model(t[None].to(device),use_cache=False).logits[0,:-1].float()
        total+=float(F.cross_entropy(logits,t[1:].to(device),reduction="sum").cpu())
    nll=total/4080
    return {"nll_nats":nll,"ppl":float(torch.exp(torch.tensor(nll,dtype=torch.float64))),"targets":4080}
report["quantization_order_model"]=[]
for mode in ["float_adapter","W4_base_plus_float_adapter","merge_then_W4"]:
    model=base()
    for n,factors in payload1["adapters"].items():
        original=model.get_submodule(n)
        parent,child=n.rsplit(".",1)
        if mode=="merge_then_W4":
            original.weight.copy_(quant(original.weight+factors["B"].to(device)@factors["A"].to(device)))
        else:
            setattr(model.get_submodule(parent),child,Adapter(original,factors,
                quantize_base=mode=="W4_base_plus_float_adapter"))
    report["quantization_order_model"].append({"mode":mode,**score(model)})
    del model
    if device=="mps":
        torch.mps.empty_cache()

# Exactly one Linear, counting unique retained non-parameter storage, not device peak.
report["retained_tensors"]=[]
for rank in [4,16,64]:
    payload,_=artifact(20261119,rank)
    for fixed in [False,True]:
        layer=nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.float32)
        with torch.no_grad():
            layer.weight.copy_(w.float())
        adapted=Adapter(layer,payload["adapters"][name],fixed_a=fixed)
        parameter_storages={p.untyped_storage().data_ptr() for p in adapted.parameters()}
        saved=[]
        def pack(tensor):
            saved.append({"shape":list(tensor.shape),"storage":tensor.untyped_storage().data_ptr(),
                          "bytes":tensor.untyped_storage().nbytes(),"parameter":tensor.untyped_storage().data_ptr() in parameter_storages})
            return tensor
        value=x.float().clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            adapted(value).square().mean().backward()
        unique={entry["storage"]:entry["bytes"] for entry in saved if not entry["parameter"]}
        report["retained_tensors"].append({"rank":rank,"fixed_A":fixed,
            "non_parameter_unique_retained_storage_bytes":sum(unique.values()),"saved":saved})
report["retained_tensor_boundary"]="CPU float32, single1536x512 Linear,64 actual token activations; x needs gradient; includes loss/output buffers; not whole-model peak, optimizer memory, CUDA memory or specialized LoRA-FA optimizer"

# Recheck the originally failing merge in CPU float64 on the full actual model.
model=base(torch.float64,"cpu")
for n,factors in payload1["adapters"].items():
    parent,child=n.rsplit(".",1)
    setattr(model.get_submodule(parent),child,Adapter(model.get_submodule(n),factors))
with torch.no_grad():
    before=model(test[0:1,:64],use_cache=False).logits
    for n in payload1["adapters"]:
        module=model.get_submodule(n)
        module.base.weight.add_(module.B@module.A)
        parent,child=n.rsplit(".",1)
        setattr(model.get_submodule(parent),child,module.base)
    after=model(test[0:1,:64],use_cache=False).logits
report["cpu_float64_merge"]={**diff(before,after),
    "argmax_mismatches":int((before.argmax(-1)!=after.argmax(-1)).sum()),
    "argmax_positions":64}
assert report["cpu_float64_merge"]["max_abs"]<1e-9
(args.output_dir/"diagnostics.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!="retained_tensors"},indent=2))
