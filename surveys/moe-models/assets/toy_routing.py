"""CPU teaching verification, not a reproduction or speed benchmark."""
import json
import math
import random
from pathlib import Path

random.seed(20261116)
T, E, D, H, K = 8, 4, 6, 4, 2
x = [[random.uniform(-1, 1) for _ in range(D)] for _ in range(T)]
w1 = [[[random.uniform(-1, 1) for _ in range(D)] for _ in range(H)] for _ in range(E)]
w2 = [[[random.uniform(-1, 1) for _ in range(H)] for _ in range(D)] for _ in range(E)]
routes = [[0, 1]] * 4 + [[0, 2]] * 2 + [[0, 3]] * 2
gates = [[0.7, 0.3] for _ in range(T)]

def expert(e, value):
    hidden = [max(0, sum(a*b for a,b in zip(row,value))) for row in w1[e]]
    return [sum(a*b for a,b in zip(row,hidden)) for row in w2[e]]

reference = []
for t in range(T):
    all_outputs = [expert(e,x[t]) for e in range(E)]
    reference.append([sum(gates[t][j]*all_outputs[e][d] for j,e in enumerate(routes[t])) for d in range(D)])

grouped = [[0.0]*D for _ in range(T)]
dropped = [[0.0]*D for _ in range(T)]
counts = [0]*E
capacity = math.ceil(T*K/E)
dropped_assignments = []
for e in range(E):
    jobs = [(t,j) for t in range(T) for j in range(K) if routes[t][j] == e]
    counts[e] = len(jobs)
    for position,(t,j) in enumerate(jobs):
        value = expert(e,x[t])
        for d in range(D):
            grouped[t][d] += gates[t][j]*value[d]
            if position < capacity:
                dropped[t][d] += gates[t][j]*value[d]
        if position >= capacity:
            dropped_assignments.append([t,e])

def error(a,b):
    return max(abs(u-v) for aa,bb in zip(a,b) for u,v in zip(aa,bb))

result = {
    "identity": "actual CPU teaching run; Python float64; no latency claims",
    "seed": 20261116, "tokens": T, "experts": E, "top_k": K,
    "dimensions": {"d":D,"expert_hidden":H},
    "expert_token_counts":counts,
    "assignments":T*K,
    "capacity_at_cf1":capacity,
    "dropped_assignments":dropped_assignments,
    "grouped_vs_dense_masked_max_abs_error":error(reference,grouped),
    "dropped_vs_reference_max_abs_error":error(reference,dropped),
    "uniform_max_load_padded_rows":E*max(counts),
    "per_expert_block2_padded_rows":sum(math.ceil(n/2)*2 for n in counts),
    "expert_combinations_8choose2":math.comb(8,2),
    "expert_combinations_16choose4":math.comb(16,4),
    "freetoken_example_q_m4_bp1_bh4":4/4,
    "freetoken_server5090_q_m4":4*52.7/77.3,
    "freetoken_desktop5090_q_m4":4*49.0/53.8,
}
assert result["grouped_vs_dense_masked_max_abs_error"] < 1e-12
assert len(dropped_assignments) == 4
assert result["dropped_vs_reference_max_abs_error"] > 0
output = Path(__file__).with_name("toy_results.json")
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
