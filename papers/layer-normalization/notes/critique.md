# Source-grounded critique

- Novelty: normalize current sample across hidden units, independent of minibatch composition; important for recurrent networks.
- Correctness: positive rescaling and epsilon boundary omitted; WN dataset-scale statement conflicts with Table1; appendix alpha/beta names and init swapped; Eq35 j-direction denominator typo; BN gain-Fisher simplification needs dependency assumptions.
- Experiments: controlled strongest comparators OE ours and skip-thought ours1M. Skip-thought †1.7M not same budget. BN MLP output layer normalization differs. CNN qualitative negative finding retained.
- Compute/memory: O(BTD),2D affine; reduction and input-dependent statistics cannot generally fold. Five-kernel8Ns vs idealfused2Ns explicitly analytical. No GPU/NPU or TTFT/TPOT benchmark executed.
- Generalization: RNN variable length statistic feasibility not arbitrary length extrapolation guarantee; smallbatch invariance not batch-independent SGD results.
- Missing: multi-seed uncertainty, same-budget long baseline, mean/variance/affine/placement factorial ablations, modern model loss/gradient/logits/per-op benchmarks.
