"""Separate recurrent work, output heads and state-copy costs on actual inputs."""
from pathlib import Path
import json,platform,subprocess,time
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
data=np.load(R/"dataset-and-splits.npz")
images=torch.from_numpy(data["images"][data["test"][:1]])
def measure(fn):
    for _ in range(12):fn()
    values=[]
    for _ in range(100):
        start=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-start)/1e6)
    return {"median_ms":float(np.median(values)),"p10_ms":float(np.percentile(values,10)),
      "p90_ms":float(np.percentile(values,90)),"samples_ms":values,"warmup":12,"repeats":100}
records=[]
with torch.no_grad():
    for layout in ["rows","pixels"]:
        x=images if layout=="rows" else images.reshape(1,64,1)
        for kind in ["RNN","GRU","LSTM"]:
            c=torch.load(R/f"{layout}-{kind.lower()}-seed7.pt",weights_only=True)
            cell=getattr(nn,kind)(c["input_size"],48,batch_first=True)
            head=nn.Linear(48,10)
            cell.load_state_dict({k[4:]:v for k,v in c["state_dict"].items() if k.startswith("rnn.")})
            head.load_state_dict({k[5:]:v for k,v in c["state_dict"].items() if k.startswith("head.")})
            cell.eval();head.eval()
            y,state=cell(x)
            def steps(with_head):
                s=None
                for t in range(x.shape[1]):
                    z,s=cell(x[:,t:t+1],s)
                    if with_head:head(z[:,-1])
                return z,s
            def copy_state():
                return tuple(v.clone() for v in state) if isinstance(state,tuple) else state.clone()
            def cached_step():
                return cell(x[:,:1],state)
            records.append({"layout":layout,"kind":kind,"seed":7,"batch":1,"length":x.shape[1],
              "recurrent_full":measure(lambda:cell(x)),"recurrent_step_calls":measure(lambda:steps(False)),
              "head_once":measure(lambda:head(y[:,-1])),"recurrent_and_head_each_step":measure(lambda:steps(True)),
              "cached_one_step":measure(cached_step),"clone_all_state":measure(copy_state),
              "full_vs_step_output_max_error":float((steps(False)[0][:,-1]-y[:,-1]).abs().max())})
hardware={"cpu":subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip(),
  "physical_memory_bytes":int(subprocess.check_output(["sysctl","-n","hw.memsize"],text=True)),
  "logical_cpu":int(subprocess.check_output(["sysctl","-n","hw.ncpu"],text=True)),
  "os":platform.platform(),"torch":torch.__version__,"precision":"FP32","threads":1,
  "device":"CPU","scope":"warm forward, actual held-out handwritten image already normalized; excludes model load, preprocessing and disk IO; no GPU/NPU measurements"}
(R/"runtime-results.json").write_text(json.dumps({"hardware":hardware,"measurements":records},indent=2))
print(hardware)
for row in records:print(row["layout"],row["kind"],{k:round(v["median_ms"],6) for k,v in row.items() if isinstance(v,dict)})
