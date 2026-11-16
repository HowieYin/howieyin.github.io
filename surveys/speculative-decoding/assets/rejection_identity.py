#!/usr/bin/env python3
"""Exact finite-vocabulary teaching experiment, not an LLM throughput benchmark.

Run: python3 rejection_identity.py --output results.json
No third-party dependencies, model weights, GPU or network required.
"""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import hashlib
from itertools import product
import json
from pathlib import Path
import platform

VOCAB = ("A", "B", "C")
TARGET = {
    None: (F(5, 10), F(3, 10), F(2, 10)),
    "A": (F(1, 10), F(7, 10), F(2, 10)),
    "B": (F(6, 10), F(1, 10), F(3, 10)),
    "C": (F(2, 10), F(2, 10), F(6, 10)),
}
DRAFT = {
    None: (F(2, 10), F(5, 10), F(3, 10)),
    "A": (F(4, 10), F(4, 10), F(2, 10)),
    "B": (F(3, 10), F(4, 10), F(3, 10)),
    "C": (F(4, 10), F(3, 10), F(3, 10)),
}

def distribution(table, history):
    return table[history[-1] if history else None]

def tv(left, right):
    keys = set(left) | set(right)
    return sum((abs(left.get(k, F(0)) - right.get(k, F(0))) for k in keys), F(0)) / 2

def one_step(p, q, mode="correct"):
    accepted = tuple(min(x, y) for x, y in zip(p, q))
    rejected = 1 - sum(accepted)
    residual = tuple(max(x - y, F(0)) for x, y in zip(p, q))
    correction = p if mode == "resample_target" else tuple(x / rejected for x in residual) if rejected else (F(0),) * 3
    return tuple(x + rejected * y for x, y in zip(accepted, correction))

def serial(length):
    states = {(): F(1)}
    for _ in range(length):
        next_states = defaultdict(F)
        for history, weight in states.items():
            for token, probability in zip(VOCAB, distribution(TARGET, history)):
                next_states[history + (token,)] += weight * probability
        states = dict(next_states)
    return states

def block_outcomes(history, gamma, mode):
    """Enumerate all proposals and every accept/reject branch, dropping rejected suffixes."""
    result = defaultdict(F)
    for proposal in product(VOCAB, repeat=gamma):
        proposal_weight = F(1)
        for j, token in enumerate(proposal):
            proposal_weight *= distribution(DRAFT, history + proposal[:j])[VOCAB.index(token)]
        if not proposal_weight:
            continue
        survival = F(1)
        for j, token in enumerate(proposal):
            # Deliberate fault: later verification rows omit earlier drafted tokens.
            target_history = history if mode == "stale_history" else history + proposal[:j]
            p = distribution(TARGET, target_history)
            q = distribution(DRAFT, history + proposal[:j])
            index = VOCAB.index(token)
            acceptance = min(F(1), p[index] / q[index])
            rejected_branch = proposal_weight * survival * (1 - acceptance)
            if rejected_branch:
                residual = tuple(max(x - y, F(0)) for x, y in zip(p, q))
                correction = p if mode == "resample_target" else tuple(x / sum(residual) for x in residual)
                for replacement, probability in zip(VOCAB, correction):
                    result[proposal[:j] + (replacement,)] += rejected_branch * probability
            survival *= acceptance
        if survival:
            # Every proposed token accepted: one extra target sample is allowed.
            for bonus, probability in zip(VOCAB, distribution(TARGET, history + proposal)):
                result[proposal + (bonus,)] += proposal_weight * survival * probability
    assert sum(result.values()) == 1
    return dict(result)

def speculative(length, gamma=2, mode="correct"):
    pending = {(): F(1)}
    complete = defaultdict(F)
    while pending:
        next_pending = defaultdict(F)
        for history, weight in pending.items():
            outcomes = block_outcomes(history, min(gamma, length - len(history)), mode)
            for suffix, probability in outcomes.items():
                output = (history + suffix)[:length]
                if len(output) == length:
                    complete[output] += weight * probability
                else:
                    next_pending[output] += weight * probability
        pending = dict(next_pending)
    assert sum(complete.values()) == 1
    return dict(complete)

def strings(values):
    return [str(x) for x in values]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    simplex = [(F(a, 5), F(b, 5), F(5-a-b, 5)) for a in range(6) for b in range(6-a)]
    cases = 0
    for p, q in product(simplex, repeat=2):
        assert one_step(p, q) == p
        cases += 1
    p, q = TARGET[None], DRAFT[None]
    reference = serial(4)
    correct = speculative(4)
    bad_resample = speculative(4, mode="resample_target")
    bad_history = speculative(4, mode="stale_history")
    assert correct == reference
    assert tv(reference, bad_resample) > 0
    assert tv(reference, bad_history) > 0
    chain = {}
    for gamma in (1, 2, 3):
        observed = speculative(4, gamma=gamma)
        assert observed == reference
        chain[str(gamma)] = str(tv(reference, observed))
    wrong = one_step(p, q, mode="resample_target")
    out = {
        "identity": "this-run exact rational CPU teaching experiment; no GPU/phone/model benchmark",
        "python": platform.python_version(), "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vocabulary": VOCAB, "length": 4, "sequence_count": len(reference),
        "simplex_denominator": 5, "exhaustive_one_step_pairs": cases,
        "boundary_cases": "q=0, p=0, p=q and disjoint-support cases included",
        "initial_p": strings(p), "initial_q": strings(q),
        "accepted_mass": strings(tuple(min(x,y) for x,y in zip(p,q))),
        "rejection_probability": str(1-sum(min(x,y) for x,y in zip(p,q))),
        "correct_output": strings(one_step(p,q)),
        "wrong_resample_target_output": strings(wrong),
        "wrong_one_step_tv": str(tv(dict(zip(VOCAB,p)), dict(zip(VOCAB,wrong)))),
        "markov_tv": {"correct": str(tv(reference,correct)), "wrong_resample_target": str(tv(reference,bad_resample)), "wrong_stale_history": str(tv(reference,bad_history))},
        "correct_tv_by_draft_length": chain,
        "target_transition_probabilities": {str(k): strings(v) for k,v in TARGET.items()},
        "draft_transition_probabilities": {str(k): strings(v) for k,v in DRAFT.items()},
        "sequence_probabilities": [{"tokens": "".join(seq), "target": str(reference[seq]), "correct": str(correct[seq]), "wrong_resample_target": str(bad_resample.get(seq,F(0))), "wrong_stale_history": str(bad_history.get(seq,F(0)))} for seq in sorted(reference)],
    }
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in out.items() if k not in {"sequence_probabilities", "target_transition_probabilities", "draft_transition_probabilities"}}, indent=2))

if __name__ == "__main__":
    main()
