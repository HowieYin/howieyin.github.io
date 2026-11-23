"""Real pretrained CNN layout diagnostic; no labelled accuracy evaluation."""
from pathlib import Path
import copy,hashlib,json,platform,statistics,time,subprocess
import torch,torchvision
from torchvision.models import resnet18,ResNet18_Weights,mobilenet_v2,MobileNet_V2_Weights
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"evidence/experiment";OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
DATA=Path("/tmp/image-consistency-survey-work/data/images")
paths=[DATA/f"{s}.jpg" for s in ["live_subject/animal/00","live_subject/animal/01",
 "live_subject/human/00","live_subject/human/01","object/00","object/01","style/00","style/01"]]
assert all(p.is_file() for p in paths)
rec={"date":"2026-11-23","scope":"8 selected real DreamBench++ images; layout/logit/runtime diagnostic only; no ground-truth labels, no accuracy",
 "dataset_url":"https://huggingface.co/datasets/yuangpeng/dreambench_plus",
 "hardware":subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip(),
 "platform":platform.platform(),"torch":torch.__version__,"torchvision":torchvision.__version__,
 "threads":4,"dtype":"float32","batch_size_timing":1,"warmup":10,"repeats":50,
 "samples":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],"models":[]}
for name,fn,w in [("resnet18",resnet18,ResNet18_Weights.IMAGENET1K_V1),
                  ("mobilenet_v2",mobilenet_v2,MobileNet_V2_Weights.IMAGENET1K_V2)]:
 print("Loading",name,w.url,flush=True)
 weights_path=Path("/tmp/cnn-survey-work")/w.url.rsplit("/",1)[-1]
 assert weights_path.is_file(),f"Download original weights first: {w.url}"
 checksum=hashlib.sha256(weights_path.read_bytes()).hexdigest()
 assert checksum.startswith(weights_path.stem.rsplit("-",1)[-1])
 model=fn(weights=None).eval()
 model.load_state_dict(torch.load(weights_path,map_location="cpu",weights_only=True))
 transform=w.transforms();x=torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])
 base=model
 cl=copy.deepcopy(model).to(memory_format=torch.channels_last)
 with torch.inference_mode():
  y=base(x);yc=cl(x.contiguous(memory_format=torch.channels_last))
  # A deliberate incorrect reinterpretation after a permutation, retaining the same shape.
  wrong=x.permute(0,2,3,1).contiguous().reshape_as(x)
  yw=base(wrong)
 result={"name":name,"weights_url":w.url,"weights_sha256":checksum,"preprocessing":str(transform),
 "parameter_count":sum(p.numel() for p in model.parameters()),"input_shape":list(x.shape),
 "nchw_stride":list(x.stride()),"channels_last_stride":list(x.contiguous(memory_format=torch.channels_last).stride()),
 "correct_max_abs_logit_error":float((y-yc).abs().max()),
 "correct_mean_abs_logit_error":float((y-yc).abs().mean()),
 "top1_agreement_correct":int((y.argmax(1)==yc.argmax(1)).sum()),
 "wrong_max_abs_logit_error":float((y-yw).abs().max()),
 "top1_agreement_wrong":int((y.argmax(1)==yw.argmax(1)).sum()),"timing":{},"stage_shapes":[]}
 handles=[]
 if name=="resnet18":mods=[(n,getattr(base,n)) for n in ["conv1","layer1","layer2","layer3","layer4"]]
 else:mods=[(f"features.{n}",base.features[n]) for n in [0,3,6,13,18]]
 for n,m in mods:
  handles.append(m.register_forward_hook(lambda m,i,o,n=n:result["stage_shapes"].append({"stage":n,"shape":list(o.shape)})))
 with torch.inference_mode():base(x[:1])
 for h in handles:h.remove()
 mac=[0]
 def count_conv(m,i,o):
  mac[0]+=o.numel()*m.kernel_size[0]*m.kernel_size[1]*m.in_channels//m.groups
 handles=[m.register_forward_hook(count_conv) for m in base.modules() if isinstance(m,torch.nn.Conv2d)]
 with torch.inference_mode():base(x[:1])
 for h in handles:h.remove()
 result["conv_MACs_only"]=mac[0]
 for label,m,inp in [("nchw",base,x[:1].contiguous()),("channels_last",cl,x[:1].contiguous(memory_format=torch.channels_last))]:
  with torch.inference_mode():
   for _ in range(10):m(inp)
   times=[]
   for _ in range(50):
    t=time.perf_counter_ns();m(inp);times.append((time.perf_counter_ns()-t)/1e6)
  times.sort()
  result["timing"][label]={"median_ms":statistics.median(times),"p10_ms":times[4],"p90_ms":times[44],"samples_ms":times}
  with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as prof:m(inp)
  events=sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:12]
  result["timing"][label]["top_ops"]=[{"op":e.key,"self_cpu_ms":e.self_cpu_time_total/1000,"calls":e.count} for e in events]
 rec["models"].append(result)
 print(json.dumps(result,ensure_ascii=False),flush=True)
(OUT/"results.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2))
print("Saved",OUT/"results.json",flush=True)
