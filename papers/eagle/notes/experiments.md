# Experiment audit

All8original tables retained in HTML and PDF PNG; main text interprets every figure/table.
Datasets: MT-Bench/HumanEval/GSM8K/Alpaca. Models: Vicuna7/13/33B,LLaMA2-Chat7/13/70B,Mixtral8×7B.
Most batch1; T0/T1 separated. Speedup=walltime relative to vanilla;τ=mean target-forward acceptance;α measured on chain whileτ generally tree.
Key recomputations: Table4 same gpt-fast baseline EAGLE incremental1.8185×FP16/1.5005×INT4;6.55× versus HF includes quantization/compiler. Table5 V7τ+23.125%;Table6 relative speed+3.597%.
MT-Bench range2.13–3.07×;3.76×isHumanEval. Max-throughput ratios1.97/1.99 use different memory-limited maximum batches.

Missing: full per-op timings, all hardware/software/TP parameters, prompt/output length distribution,variance,P50/P99TTFT/TPOT,quality regression,CPU+HTP peak memory,thermal and power steady state.
No model benchmark was run locally; browser and arithmetic checks are not inference performance tests.
