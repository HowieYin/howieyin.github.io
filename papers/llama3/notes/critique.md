# Critique

Strengths: broad end-to-end disclosure; usable dense baseline; token/data and compute scaling studied; real4D/failure logs; extensive pre/post/safety/multimodal tables.

Weaknesses: private15Tdata/manyjudges/benchmarks/productionstack; internal numerical inconsistencies; insufficient paired rare-error quantization evidence; limited full latency/energy for speech.

Trade-offs: overtraining vs future inference cost; GQA cache savings vs querycompute; FSDP retained weights vs allgather; PP throughput vs latency; FP8 outlier saturation vs underflow; safety VR vs FRR; frozen backbone vs adapter extra parameters/KV.

Open questions: equal-data equal-compute ablations, task-specific currentSOTA under fixedlatency, robust longcontext/OCR and toolreturn errors, independent safetycounts/base rates, correction of252M/40M/Table5, full speech firstaudio latency.

Personal proposals and ten-dimension assessment are in the final My Insights section of index.md.
