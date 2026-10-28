# Critique and implementation risks

Strength: clear conditional-variable insight and frozen-target exact verification.
Trade-off: serial draft depth, repeated full LM head, temporary tree KV/features/logits, path gather/synchronization,MoE expert union and TP communication.
Correctness: conditional without-replacement q,full-vector residual,ancestor mask,depth position,EOS/budget and cache lifecycle.
Reproducibility: pinv1,notcurrentmain. Variable63contains25draftnodes. Paper/code noise and precision differ.
Interpretation limits: theory does not substitute for implementation/processor/quantization consistency validation; acceptance metrics do not equal task Accuracy/F1.
Next work: per-effective-token cost traces,training rollout feedback,hardware/load-specific tree budgets,real VLM/OCR feature data,thin JNI and nativecache buffers.
