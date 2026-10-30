#!/usr/bin/env python3
"""FP32 golden reference for the MiniCPM5 8,192-token accuracy gate.

Runs the HF checkpoint in full FP32 (f32 weights/activations, Precision.HIGHEST matmuls, f32 softmax) on
the gate inputs stored in a reference logprob dump and writes a dump in the same format (top-k ids and
logprobs at the last position), so run_benchmark.compare_gate can score any candidate against it.

  python3 f32_golden.py --model openbmb/MiniCPM5-1B --inputs benchmark/ref/bf16_8192_1b_ref_logprobs.json \
      --out benchmark/ref/fp32_8192_1b_golden_logprobs.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")
F32 = jnp.float32
HI = jax.lax.Precision.HIGHEST


def load_weights(snap: str):
    from safetensors.numpy import load_file

    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        try:
            part = load_file(f)
        except Exception:  # bf16 not supported by numpy loader -> torch path
            import torch
            from safetensors.torch import load_file as tload

            part = {k: v.to(torch.float32).numpy() for k, v in tload(f).items()}
        for k, v in part.items():
            sd[k] = np.asarray(v, dtype=np.float32)
    return sd


def rope_tables(S, hd, theta):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    t = np.arange(S, dtype=np.float64)
    f = np.einsum("i,j->ij", t, inv)
    e = np.concatenate([f, f], -1)
    return jnp.asarray(np.cos(e), F32), jnp.asarray(np.sin(e), F32)


def rms(x, w, eps):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rot_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)


@jax.jit
def _attn_chunk(q, k, v, q0):
    # q: (Cq, nq, hd) k/v: (S, nkv, hd)
    nq, nkv = q.shape[1], k.shape[1]
    g = nq // nkv
    S = k.shape[0]
    qg = q.reshape(q.shape[0], nkv, g, q.shape[-1])
    s = jnp.einsum("qngd,knd->ngqk", qg, k, precision=HI) / np.sqrt(q.shape[-1])
    qi = q0 + jnp.arange(q.shape[0])[:, None]
    kj = jnp.arange(S)[None, :]
    s = jnp.where((kj <= qi)[None, None], s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum("ngqk,knd->qngd", p, v, precision=HI)
    return o.reshape(q.shape[0], nq * q.shape[-1])


def forward_one(ids, W, cfg, cos, sin, k=20):
    L, nq, nkv, hd, eps = cfg["L"], cfg["nq"], cfg["nkv"], cfg["hd"], cfg["eps"]
    S = ids.shape[0]
    x = W["embed"][ids]
    c, s_ = cos[:S], sin[:S]
    for i in range(L):
        lw = W["layers"][i]
        xn = rms(x, lw["in"], eps)
        q = jnp.dot(xn, lw["q"], precision=HI).reshape(S, nq, hd)
        kk = jnp.dot(xn, lw["k"], precision=HI).reshape(S, nkv, hd)
        v = jnp.dot(xn, lw["v"], precision=HI).reshape(S, nkv, hd)
        q = q * c[:, None, :] + rot_half(q) * s_[:, None, :]
        kk = kk * c[:, None, :] + rot_half(kk) * s_[:, None, :]
        outs = [_attn_chunk(q[q0:q0 + 1024], kk, v, q0) for q0 in range(0, S, 1024)]
        a = jnp.concatenate(outs, axis=0)
        x = x + jnp.dot(a, lw["o"], precision=HI)
        xn = rms(x, lw["post"], eps)
        gte = jnp.dot(xn, lw["gate"], precision=HI)
        up = jnp.dot(xn, lw["up"], precision=HI)
        x = x + jnp.dot(jax.nn.silu(gte) * up, lw["down"], precision=HI)
    h = rms(x[-1], W["norm"], eps)
    logits = jnp.dot(h, W["lm_head"], precision=HI)
    lp = jax.nn.log_softmax(logits)
    top = jax.lax.top_k(lp, k)
    return np.asarray(top[1]).tolist(), np.asarray(top[0]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True, help="reference dump with 'inputs' (prompt_token_ids)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    snap = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])
    hc = json.load(open(os.path.join(snap, "config.json")))
    cfg = dict(L=hc["num_hidden_layers"], nq=hc["num_attention_heads"], nkv=hc["num_key_value_heads"],
               hd=hc.get("head_dim") or hc["hidden_size"] // hc["num_attention_heads"], eps=hc["rms_norm_eps"],
               theta=float(hc.get("rope_theta", 10000.0)))
    t0 = time.time()
    sd = load_weights(snap)
    put = lambda a: jnp.asarray(a, F32)  # noqa: E731
    W = {"embed": put(sd["model.embed_tokens.weight"]), "norm": put(sd["model.norm.weight"]),
         "lm_head": put(sd["lm_head.weight"].T if "lm_head.weight" in sd else sd["model.embed_tokens.weight"].T),
         "layers": []}
    for i in range(cfg["L"]):
        p = f"model.layers.{i}."
        W["layers"].append({
            "in": put(sd[p + "input_layernorm.weight"]), "post": put(sd[p + "post_attention_layernorm.weight"]),
            "q": put(sd[p + "self_attn.q_proj.weight"].T), "k": put(sd[p + "self_attn.k_proj.weight"].T),
            "v": put(sd[p + "self_attn.v_proj.weight"].T), "o": put(sd[p + "self_attn.o_proj.weight"].T),
            "gate": put(sd[p + "mlp.gate_proj.weight"].T), "up": put(sd[p + "mlp.up_proj.weight"].T),
            "down": put(sd[p + "mlp.down_proj.weight"].T),
        })
    del sd
    print(f"[golden] loaded {args.model} f32 in {time.time() - t0:.1f}s cfg={cfg}", flush=True)
    ref = json.load(open(args.inputs))
    cos, sin = rope_tables(8192 + 64, cfg["hd"], cfg["theta"])
    k = int(ref.get("k", 20))
    results = []
    for st in ref["inputs"]:
        seqs = []
        for ids in st["prompt_token_ids"]:
            t1 = time.time()
            tid, tlp = forward_one(jnp.asarray(ids, jnp.int32), W, cfg, cos, sin, k=k)
            seqs.append({"top_ids": tid, "top_logprobs": tlp})
            print(f"[golden] ({st['batch_size']},{st['seq_len']}) seq done in {time.time() - t1:.1f}s "
                  f"top5={list(zip(tid[:5], [round(v, 3) for v in tlp[:5]]))}", flush=True)
        results.append({"batch_size": st["batch_size"], "seq_len": st["seq_len"], "seqs": seqs})
    out = {"model": args.model, "label": f"fp32-golden-{os.path.basename(args.model)}", "k": k,
           "gate_seed": ref.get("gate_seed"), "reference": "fp32 HF checkpoint, Precision.HIGHEST, f32 softmax",
           "config": {"precision": "fp32", "matmul_precision": "highest"}, "inputs": ref["inputs"],
           "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[golden] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
