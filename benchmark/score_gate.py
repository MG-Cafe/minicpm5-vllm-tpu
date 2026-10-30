#!/usr/bin/env python3
"""Score logprob dumps against a reference dump with the benchmark's gate (top-1, tie-aware top-1,
mean |dlogprob| over the reference top-k, KL, cosine). Pure Python: no torch / vLLM needed.

  python3 benchmark/score_gate.py REF.json CAND1.json[.gz] [CAND2.json[.gz] ...]
  e.g. python3 benchmark/score_gate.py benchmark/ref/fp32_8192_1b_golden_logprobs.json results/gate/v6e_minicpm5_1b_gate.json.gz

When the candidate dump contains embedded 4-prompt batch-coverage sets (batch 8 / 16 / 32),
their first 4 sequences are scored against the reference's batch-4 set just like `run_benchmark.py`.
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import compare_gate  # noqa: E402


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def main():
    ref = load(sys.argv[1])
    ref4 = next((r for r in ref["results"] if r["batch_size"] == 4), None)
    for path in sys.argv[2:]:
        cand = load(path)
        ref_sets, cand_sets = list(ref["results"]), []
        inputs = cand.get("inputs") or [{}] * len(cand["results"])
        for st, cs in zip(inputs, cand["results"]):
            if "embedded_ref_batch" not in st:
                cand_sets.append(cs)
            elif ref4 is not None:
                ref_sets.append({"batch_size": cs["batch_size"], "seq_len": cs["seq_len"], "seqs": ref4["seqs"]})
                cand_sets.append({"batch_size": cs["batch_size"], "seq_len": cs["seq_len"], "seqs": cs["seqs"][:4]})
        rep = compare_gate(ref_sets, cand_sets)
        print(f"{os.path.basename(path)} vs {os.path.basename(sys.argv[1])}: gate_v2={rep['gate_v2']} "
              f"n={rep['n_seqs']} top1={rep['top1_agreement']} tie_aware={rep['top1_tie_aware']} "
              f"mean|dlp|={rep['mean_abs_dlogprob']} kl={rep['kl_top20_mean']} cos_lp={rep['cos_logprob_mean']}")
        for st in rep["sets"]:
            print(f"   {st['shape']} (n={st['n']}): top1={st['top1_agreement']} tie={st['top1_tie_aware']} "
                  f"mean|dlp|={st['mean_abs_dlogprob']} max_seq={st['max_seq_abs_dlogprob']} "
                  f"cos_lp_min={st['cos_logprob_min']}")


if __name__ == "__main__":
    main()
