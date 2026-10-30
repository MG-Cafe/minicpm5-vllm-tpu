#!/usr/bin/env python3
"""Accuracy-gate scoring (pure Python, no torch/vLLM): compare a candidate's last-position top-k
log-probabilities with a reference dump. Used by run_benchmark.py and score_gate.py."""
import math
import statistics


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def compare_gate(ref_res, cand_res):
    """Top-1 agreement + cosine similarities over the union of the two top-k id sets.

    cos_logprob: log-prob vectors over the union of top-k ids; an id missing from one side's
                 top-k is filled with that side's k-th logprob (an upper bound) -> conservative.
    cos_prob:    probability vectors over the union, missing = 0.
    """
    report = {"sets": []}
    assert len(ref_res) == len(cand_res), f"gate: {len(ref_res)} reference sets vs {len(cand_res)} candidate sets"
    all_top1, all_cos_lp, all_cos_p, all_nolp = [], [], [], []
    all_tie, all_tie_nolp, all_dlp, all_kl, all_miss = [], [], [], [], []
    for rs, cs in zip(ref_res, cand_res):
        assert (rs["batch_size"], rs["seq_len"]) == (cs["batch_size"], cs["seq_len"])
        assert len(rs["seqs"]) == len(cs["seqs"]), f"gate: set {rs['batch_size']}x{rs['seq_len']} sequence count differs"
        top1, cos_lp, cos_p, overlap, d_top1, nolp = [], [], [], [], [], []
        tie, tie_nolp, dlp, kl, miss = [], [], [], [], []
        for r, c in zip(rs["seqs"], cs["seqs"]):
            rmap = dict(zip(r["top_ids"], r["top_logprobs"]))
            cmap = dict(zip(c["top_ids"], c["top_logprobs"]))
            rfloor, cfloor = min(r["top_logprobs"]), min(c["top_logprobs"])
            union = sorted(set(rmap) | set(cmap))
            a = [rmap.get(t, rfloor) for t in union]
            b = [cmap.get(t, cfloor) for t in union]
            cos_lp.append(_cos(a, b))
            cos_p.append(_cos([math.exp(x) if t in rmap else 0.0 for t, x in zip(union, a)],
                              [math.exp(x) if t in cmap else 0.0 for t, x in zip(union, b)]))
            top1.append(1.0 if r["top_ids"][0] == c["top_ids"][0] else 0.0)
            if "sampled_nolp" in c:  # timed (no-logprobs) path vs the reference top-1
                nolp.append(1.0 if c["sampled_nolp"] == r["top_ids"][0] else 0.0)
            overlap.append(len(set(rmap) & set(cmap)) / max(1, len(rmap)))
            d_top1.append(abs(r["top_logprobs"][0] - cmap.get(r["top_ids"][0], cfloor)))
            # gate v2 (#42): tie-aware top-1 -- a flip counts only if the candidate's top-1 sits
            # > 0.1 nats below the ref top-1 under the ref distribution (ids outside the ref top-k
            # are at most the ref k-th logprob) AND the candidate does not rank the ref top-1 as
            # tied-best itself (bf16 logits are quantized to 1/16-nat steps, so near-ties in flat
            # distributions can collapse into exact ties that argmax breaks by token id).
            r1 = r["top_logprobs"][0]
            c1 = c["top_logprobs"][0]
            c_ref1 = cmap.get(r["top_ids"][0], cfloor)
            cand_tied = (c1 - c_ref1) <= 1e-6
            tie.append(1.0 if ((r1 - rmap.get(c["top_ids"][0], rfloor)) <= 0.1 or cand_tied) else 0.0)
            if "sampled_nolp" in c:
                tie_nolp.append(1.0 if ((r1 - rmap.get(c["sampled_nolp"], rfloor)) <= 0.1
                                        or (cand_tied and cmap.get(c["sampled_nolp"], cfloor) >= c1 - 1e-6))
                                else 0.0)
            # mean |dlogprob| over the ref top-k (candidate value for ids missing from its top-k is
            # filled with its k-th logprob; `missing` reports how often that happens)
            dl = [abs(rl - cmap.get(t, cfloor)) for t, rl in zip(r["top_ids"], r["top_logprobs"])]
            dlp.append(sum(dl) / len(dl))
            miss.append(sum(1 for t in r["top_ids"] if t not in cmap) / len(r["top_ids"]))
            # KL(ref || cand) over the ref top-k ids, both renormalized on that support
            pr = [math.exp(x) for x in r["top_logprobs"]]
            qc = [math.exp(cmap.get(t, cfloor)) for t in r["top_ids"]]
            zp, zq = sum(pr), sum(qc)
            kl.append(sum((p / zp) * (math.log(p / zp) - math.log(q / zq)) for p, q in zip(pr, qc)))
        all_top1 += top1
        all_cos_lp += cos_lp
        all_cos_p += cos_p
        all_nolp += nolp
        all_tie += tie
        all_tie_nolp += tie_nolp
        all_dlp += dlp
        all_kl += kl
        all_miss += miss
        report["sets"].append({
            "shape": f"({rs['batch_size']},{rs['seq_len']})", "n": len(top1),
            "top1_agreement": round(statistics.mean(top1), 4),
            "top1_agreement_nolp": round(statistics.mean(nolp), 4) if nolp else None,
            "top1_tie_aware": round(statistics.mean(tie), 4),
            "top1_tie_aware_nolp": round(statistics.mean(tie_nolp), 4) if tie_nolp else None,
            "mean_abs_dlogprob": round(statistics.mean(dlp), 5), "max_seq_abs_dlogprob": round(max(dlp), 5),
            "kl_top20_mean": round(statistics.mean(kl), 6),
            "ref_top20_missing_frac": round(statistics.mean(miss), 4),
            "cos_logprob_mean": round(statistics.mean(cos_lp), 6), "cos_logprob_min": round(min(cos_lp), 6),
            "cos_prob_mean": round(statistics.mean(cos_p), 6), "cos_prob_min": round(min(cos_p), 6),
            "topk_overlap_mean": round(statistics.mean(overlap), 4),
            "ref_top1_logprob_absdiff_max": round(max(d_top1), 5),
        })
    report["top1_agreement"] = round(statistics.mean(all_top1), 4)
    report["top1_agreement_nolp"] = round(statistics.mean(all_nolp), 4) if all_nolp else None
    report["top1_tie_aware"] = round(statistics.mean(all_tie), 4)
    report["top1_tie_aware_nolp"] = round(statistics.mean(all_tie_nolp), 4) if all_tie_nolp else None
    report["mean_abs_dlogprob"] = round(statistics.mean(all_dlp), 5)
    report["kl_top20_mean"] = round(statistics.mean(all_kl), 6)
    report["ref_top20_missing_frac"] = round(statistics.mean(all_miss), 4)
    report["cos_logprob_mean"] = round(statistics.mean(all_cos_lp), 6)
    report["cos_logprob_min"] = round(min(all_cos_lp), 6)
    report["cos_prob_mean"] = round(statistics.mean(all_cos_p), 6)
    report["n_seqs"] = len(all_top1)
    report["gate_quantized_pass"] = bool(
        report["top1_agreement"] >= 0.95 and report["cos_logprob_mean"] >= 0.99
        and (report["top1_agreement_nolp"] is None or report["top1_agreement_nolp"] >= 0.95))
    report["exact_match_top1_all"] = bool(
        report["top1_agreement"] == 1.0 and report["top1_agreement_nolp"] in (None, 1.0))
    # gate v2 verdict (coordinator #42): tie-aware top-1 >= 0.95 (logprob and timed paths),
    # cos_logprob >= 0.99 floor, mean |dlogprob| over ref top-20: <= 0.10 PASS, > 0.15 FAIL, else JUDGE
    t_ok = report["top1_tie_aware"] >= 0.95 and (report["top1_tie_aware_nolp"] is None
                                                 or report["top1_tie_aware_nolp"] >= 0.95)
    c_ok = report["cos_logprob_mean"] >= 0.99
    if not (t_ok and c_ok) or report["mean_abs_dlogprob"] > 0.15:
        report["gate_v2"] = "FAIL"
    elif report["mean_abs_dlogprob"] <= 0.10:
        report["gate_v2"] = "PASS"
    else:
        report["gate_v2"] = "JUDGE"
    return report
