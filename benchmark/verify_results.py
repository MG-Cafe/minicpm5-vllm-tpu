#!/usr/bin/env python3
"""Static verifier for MiniCPM5 vLLM-TorchTPU benchmark result JSON files.

Checks:
  (1) Methodology: 5 shapes x 8,192 tokens per input (bs in {1, 4, 8, 16, 32}), 5 warmup + 50 timed iterations, prefix caching off, max_tokens=1;
  (2) Configuration completeness (engine kwargs, env, harness hash, software versions, argv);
  (3) Correctness gate (top-1 agreement, cosine similarity, and mean |dlogprob| across gate sequences);
  (4) Plausibility: monotonic percentiles, E2E >= engine, throughput consistency, and hardware roofline bounds;
  (5) Raw samples: 50 per shape, and every percentile / mean / throughput recomputes from them;
  (6) Provenance: plugin tree sha256 and harness sha256 recorded in the result equal this repo's files, and
      the vLLM / vllm-torchtpu / torch_tpu / jax / transformers / tokenizers sources were unmodified.

Usage: python3 verify_results.py RESULT.json [RESULT2.json ...]
"""

import argparse
import hashlib
import json
import os
import sys

SHAPES = [(1, 8192), (4, 8192), (8, 8192), (16, 8192), (32, 8192)]
TOP_KEYS = [
    "model", "engine", "precision", "quantization", "device", "accelerator", "chips",
    "cores_utilized", "tensor_parallel_size", "prefix_caching", "token_budget_per_call",
    "iters_per_config", "warmup_per_config", "model_load_s", "timestamp_utc", "by_config",
]
ROW_KEYS = [
    "batch_size", "seq_len", "total_tokens", "parallel_mode",
    "request_latency_ms_p50", "request_latency_ms_p90", "request_latency_ms_p95",
    "request_latency_ms_p99", "request_latency_ms_mean",
    "device_latency_ms_p50", "device_latency_ms_p90", "device_latency_ms_p95",
    "device_latency_ms_p99", "device_latency_ms_mean",
    "per_item_latency_ms_mean", "device_per_item_latency_ms_mean",
    "throughput_items_per_s", "device_throughput_items_per_s",
    "throughput_tokens_per_s", "device_throughput_tokens_per_s",
]
PEAK_TFLOPS = {
    ("v6e", "bf16"): 918.0, ("v6e", "int8"): 1836.0, ("v6e", "fp8"): 918.0,
    ("tpu7x", "bf16"): 2307.0, ("tpu7x", "fp8"): 4614.0, ("tpu7x", "int8"): 4614.0,
}
MODELS = {
    "1b": dict(L=24, H=1536, I=4608, hq=16, hkv=2, d=128, V=130560),
    "2b": dict(L=42, H=2048, I=6144, hq=16, hkv=2, d=128, V=130560),
}


class Report:
    def __init__(self, path):
        self.path = path
        self.items = []

    def add(self, level, msg):
        self.items.append((level, msg))

    def ok(self, msg):
        self.add("PASS", msg)

    def warn(self, msg):
        self.add("WARN", msg)

    def fail(self, msg):
        self.add("FAIL", msg)

    def info(self, msg):
        self.add("INFO", msg)

    @property
    def n_fail(self):
        return sum(1 for l, _ in self.items if l == "FAIL")

    @property
    def n_warn(self):
        return sum(1 for l, _ in self.items if l == "WARN")

    def render(self):
        out = [f"=== {self.path}"]
        for level, msg in self.items:
            out.append(f"  [{level}] {msg}")
        out.append(f"  => {self.n_fail} FAIL, {self.n_warn} WARN")
        return "\n".join(out)


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p, v
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield p, v
            yield from walk(v, p)


def find_keys(obj, pred):
    return [(k, v) for k, v in walk(obj) if pred(k.split(".")[-1].lower(), v)]


def model_key(d, path):
    s = (str(d.get("model", "")) + " " + os.path.basename(path)).lower()
    if "2b" in s:
        return "2b"
    if "1b" in s:
        return "1b"
    return None


def accel_key(d, path):
    s = (str(d.get("accelerator", "")) + " " + str(d.get("device", "")) + " " +
         os.path.basename(path)).lower()
    if "7x" in s or "v7" in s:
        return "tpu7x"
    if "v6e" in s:
        return "v6e"
    return None


def compute_dtype(d):
    """From the recorded precision label and the resolved quant string in the plugin status."""
    st = ((d.get("config") or {}).get("plugin_status") or [{}])[0]
    blob = (str(d.get("precision", "")) + " " + str(st.get("quant", ""))).lower()
    if "int8" in blob:
        return "int8"
    if "fp8" in blob:
        return "fp8"
    return "bf16"


def flops(m, bs, seq):
    c = MODELS[m]
    qkv = c["H"] * (c["hq"] + 2 * c["hkv"]) * c["d"]
    o = c["hq"] * c["d"] * c["H"]
    gu = c["H"] * 2 * c["I"]
    dn = c["I"] * c["H"]
    lin = 2 * (qkv + o + gu + dn) * c["L"] * bs * seq
    attn = c["L"] * bs * 2 * seq * seq * c["d"] * c["hq"]
    head = 2 * c["H"] * c["V"] * bs
    return lin + attn + head


def close(a, b, rel=0.01, absol=0.05):
    return abs(a - b) <= max(absol, rel * max(abs(a), abs(b)))


def pctl(vals, p):
    """Same definition as run_benchmark.pctl: linear interpolation between closest ranks."""
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plugin_tree_sha(repo_root):
    """Same recipe as run_all_tpu_benchmarks.sh: sha256 over `sha256sum` lines of plugin/*.py|*.txt|METADATA."""
    root = os.path.join(repo_root, "plugin")
    rels = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [x for x in dns if x != "__pycache__"]
        for fn in fns:
            if fn.endswith((".py", ".txt")) or fn == "METADATA":
                rels.append("./" + os.path.relpath(os.path.join(dp, fn), root))
    lines = []
    for rel in sorted(rels, key=lambda x: x.encode()):
        with open(os.path.join(root, rel[2:]), "rb") as f:
            lines.append(f"{hashlib.sha256(f.read()).hexdigest()}  {rel}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]


def check(path, quantized="auto"):
    rep = Report(path)
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        rep.fail(f"cannot parse JSON: {e}")
        return rep, None
    m = model_key(d, path)
    a = accel_key(d, path)
    cdt = compute_dtype(d)
    is_q = (cdt != "bf16") if quantized == "auto" else (quantized == "yes")
    rep.info(f"model={m} accel={a} compute_dtype={cdt} quantized={is_q}")
    with open(path, "rb") as f:
        rep.info(f"sha256={hashlib.sha256(f.read()).hexdigest()[:16]} timestamp={d.get('timestamp_utc')}")

    missing = [k for k in TOP_KEYS if k not in d]
    (rep.fail if missing else rep.ok)(f"top-level schema keys missing: {missing}" if missing
                                       else "top-level schema keys present")
    for k in ("config", "validation"):
        if not isinstance(d.get(k), dict):
            rep.fail(f"'{k}' dict missing")
    rows = d.get("by_config", []) or []
    shapes = [(r.get("batch_size"), r.get("seq_len")) for r in rows]
    if sorted(shapes) == sorted(SHAPES) and len(shapes) == 5:
        rep.ok("all 5 shapes present")
    else:
        rep.fail(f"unexpected shapes {shapes}")
    for r in rows:
        mk = [k for k in ROW_KEYS if k not in r]
        if mk:
            rep.fail(f"row {r.get('batch_size')}x{r.get('seq_len')} missing keys {mk}")
        if r.get("seq_len") != 8192 or r.get("total_tokens") != (r.get("batch_size", 0) * r.get("seq_len", 0)):
            rep.fail(f"row {r.get('batch_size')}x{r.get('seq_len')} seq_len != 8192 or total_tokens != batch_size * 8192")
    it, wu = d.get("iters_per_config"), d.get("warmup_per_config")
    (rep.ok if (it == 50 and wu == 5) else rep.fail)(f"iters={it} warmup={wu} (need 50/5)")
    if d.get("prefix_caching") is not False:
        rep.fail(f"top-level prefix_caching={d.get('prefix_caching')} (need false)")

    cfg = d.get("config") or {}
    val = d.get("validation") or {}
    ek = cfg.get("engine_kwargs") or {}
    miss_cfg = [k for k in ("argv", "harness_sha256", "engine_kwargs", "env", "env_overrides", "versions",
                            "sampling_params", "source_diff", "model_revision", "hf_config", "host") if not cfg.get(k)]
    (rep.fail if miss_cfg else rep.ok)(f"config fields missing: {miss_cfg}" if miss_cfg else
                                       "config complete (argv, hashes, engine kwargs, env, versions, model revision, host)")
    sp = str(cfg.get("sampling_params", ""))
    (rep.ok if "max_tokens=1" in sp and ek.get("enable_prefix_caching") is False else rep.fail)(
        f"sampling max_tokens=1 and enable_prefix_caching=False ({sp})")
    if val.get("all_lengths_and_outputs_ok") is True:
        rep.ok("every request had exactly 8,192 prompt ids and returned exactly 1 token")
    else:
        rep.fail("validation.all_lengths_and_outputs_ok is not True")
    pads = [r.get("padded_sequences") for r in d.get("by_config", []) or []]
    if all(p is not None for p in pads):
        (rep.ok if sum(pads) == 0 else rep.fail)(
            f"padding: {sum(pads)} padded prompts (0 = every prompt is BOS + 8,191 real tokens)")
    else:
        rep.fail("padded_sequences not recorded")

    gates = val.get("logprob_gates") or ([val["logprob_gate"]] if isinstance(val.get("logprob_gate"), dict) else [])
    kinds = set()
    for lg in gates:
        lbl = (str(lg.get("reference_label", "")) + " " + str(lg.get("reference", ""))).lower()
        kinds.add("fp32-golden" if ("fp32" in lbl or "golden" in lbl) else "bf16-ref")
    (rep.ok if kinds >= {"fp32-golden", "bf16-ref"} else rep.fail)(
        f"accuracy gates present for references {sorted(kinds)} (need fp32-golden + bf16-ref)")
    for lg in gates:
        rp = str(lg.get("reference", ""))
        local = os.path.join(REPO_ROOT, "benchmark", "ref", os.path.basename(rp))
        if os.path.exists(local):
            with open(local, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
            (rep.ok if lg.get("reference_sha256") == h else rep.fail)(
                f"gate reference {os.path.basename(rp)} sha256 matches the committed file")
        else:
            rep.fail(f"gate reference {os.path.basename(rp)} not found in benchmark/ref")
        shapes = sorted(int(st["shape"].strip("()").split(",")[0]) for st in lg.get("sets", []))
        (rep.ok if shapes == [1, 4, 8, 16, 32] else rep.fail)(f"gate covers batch sizes {shapes} (need 1/4/8/16/32)")
        inv = lg.get("batch_invariance") or {}
        bad = [k for k, v in inv.items() if not v.get("same_top1")]
        (rep.ok if inv and not bad else rep.fail)(
            "batch invariance: same top-1 for the 4 reference prompts at batch 4/8/16/32 "
            f"(max |dlogprob| {max([v.get('max_abs_dlogprob_vs_batch4', 0) for v in inv.values()] or [0])})"
            if inv and not bad else f"batch invariance failed or missing: {bad or 'missing'}")
    gv = val.get("gate_v2_all")
    (rep.ok if gv == "PASS" else rep.fail)(f"gate_v2_all = {gv}")
    kind = "quantized" if is_q else "non-quantized"
    for lg in gates:
        lbl = (str(lg.get("reference_label", "")) + " " + str(lg.get("reference", ""))).lower()
        rkind = "fp32-golden" if ("fp32" in lbl or "golden" in lbl) else "bf16-ref"
        tie, tie_n = lg.get("top1_tie_aware"), lg.get("top1_tie_aware_nolp")
        cl, dlp = lg.get("cos_logprob_mean"), lg.get("mean_abs_dlogprob")
        thr_t1 = 0.95 if is_q else 0.97
        for nm, v in (("top1_tie_aware", tie), ("top1_tie_aware_nolp", tie_n)):
            if v is not None:
                (rep.ok if v >= thr_t1 else rep.fail)(f"[{rkind}] {nm} {v} vs >= {thr_t1} ({kind})")
        thr_cos = 0.99 if is_q else 0.999
        (rep.ok if (cl or 0) >= thr_cos else rep.fail)(f"[{rkind}] cos_logprob_mean {cl} vs >= {thr_cos}")
        if dlp is not None:
            (rep.ok if dlp <= 0.10 else rep.fail)(f"[{rkind}] mean|dlp| {dlp} <= 0.10")

    # (5) raw samples -> statistics
    for r in rows:
        tag = f"({r.get('batch_size')},{r.get('seq_len')})"
        bad = []
        for pre, key in (("request_latency_ms", "request_latency_ms_samples"), ("device_latency_ms", "device_latency_ms_samples")):
            smp = r.get(key)
            if not isinstance(smp, list):
                bad.append(f"{key} missing")
                continue
            if len(smp) != it:
                bad.append(f"{key} has {len(smp)} samples (need {it})")
                continue
            for pp in (50, 90, 95, 99):
                if abs(pctl(smp, pp) - r[f"{pre}_p{pp}"]) > 0.002:
                    bad.append(f"{pre}_p{pp} {r[f'{pre}_p{pp}']} != {pctl(smp, pp):.3f} from samples")
            if abs(sum(smp) / len(smp) - r[f"{pre}_mean"]) > 0.002:
                bad.append(f"{pre}_mean {r[f'{pre}_mean']} != {sum(smp) / len(smp):.3f} from samples")
        smp = r.get("request_latency_ms_samples")
        if isinstance(smp, list) and smp:
            tps = r["total_tokens"] / (sum(smp) / len(smp) / 1000.0)
            if not close(r.get("throughput_tokens_per_s", -1), tps, rel=0.0005, absol=1.0):
                bad.append(f"throughput_tokens_per_s {r.get('throughput_tokens_per_s')} != {tps:.1f} from samples")
        (rep.fail if bad else rep.ok)(f"{tag} raw samples: " + ("; ".join(bad) if bad else f"{it} per metric, all statistics recompute"))

    # (5b) which kernels ran
    st = cfg.get("plugin_status") or []
    if st:
        shapes_run = st[0].get("shapes", {})
        need = [f"mk:{r.get('batch_size')}x{r.get('seq_len')}" for r in rows]
        missing = [k for k in need if k not in shapes_run]
        modes = ", ".join(k + "=" + str(shapes_run[k].get("mode")) for k in need if k in shapes_run)
        (rep.ok if not missing else rep.fail)(
            f"plugin status: every timed shape ran on the megakernel fast path ({modes}; quant={st[0].get('quant')})"
            if not missing else f"plugin status: shapes without fast-path record {missing}")
        stock = sorted({x for s_ in st for x in s_.get("stock_reasons", [])})
        (rep.ok if not stock else rep.fail)("no stock-path forward recorded" if not stock else f"stock path used: {stock}")
    else:
        rep.fail("config.plugin_status missing (which kernels ran is not recorded)")

    # (6) provenance
    notes = str(cfg.get("notes", ""))
    rec_sha = notes.split("sha256=")[1].split(";")[0].strip() if "sha256=" in notes else None
    repo_sha = plugin_tree_sha(REPO_ROOT)
    (rep.ok if rec_sha == repo_sha else rep.fail)(f"plugin tree sha256 recorded {rec_sha} vs this repo {repo_sha}")
    hp = os.path.join(REPO_ROOT, "benchmark", "run_benchmark.py")
    with open(hp, "rb") as f:
        repo_h = hashlib.sha256(f.read()).hexdigest()
    rec_h = cfg.get("harness_sha256")
    (rep.ok if rec_h == repo_h else rep.fail)(f"harness sha256 recorded {str(rec_h)[:16]} vs this repo {repo_h[:16]}")
    roots = ((cfg.get("source_diff") or {}).get("roots") or {})
    dirty = {k: v for k, v in roots.items() if v.get("modified") or v.get("added") or v.get("removed_count")}
    # The manifest (benchmark/pristine_manifest.json) hashes the library sources of the reference image
    # vllm/vllm-torchtpu:nightly-20261027-e91801a-v0.32.0. A different image/build is allowed but flagged.
    (rep.ok if roots and not dirty else rep.warn)(
        f"library sources identical to the reference image ({len(roots)} package roots checked)" if roots and not dirty
        else f"library sources differ from the reference image in {sorted(dirty) if roots else 'no source_diff'} "
             "(expected when not using vllm/vllm-torchtpu:nightly-20261027-e91801a-v0.32.0; numbers may differ)")

    peak = PEAK_TFLOPS.get((a, cdt)) if a else None
    table = []
    for r in rows:
        bs, sq = r.get("batch_size"), r.get("seq_len")
        tag = f"({bs},{sq})"
        e = [r[f"request_latency_ms_{p}"] for p in ("p50", "p90", "p95", "p99")]
        g = [r[f"device_latency_ms_{p}"] for p in ("p50", "p90", "p95", "p99")]
        em, gm = r["request_latency_ms_mean"], r["device_latency_ms_mean"]
        if e != sorted(e) or g != sorted(g):
            rep.fail(f"{tag} percentiles not monotonic: e2e={e} engine={g}")
        if e[0] < g[0] - 1e-6 or em < gm - 1e-6:
            rep.fail(f"{tag} E2E < engine (e2e p50 {e[0]} < engine p50 {g[0]})")
        tot = bs * sq
        tt = tot / (em / 1000.0)
        if not close(r.get("throughput_tokens_per_s", -1), tt, rel=0.005, absol=1.0):
            rep.fail(f"{tag} throughput_tokens_per_s inconsistent with E2E mean")
        dt = tot / (gm / 1000.0)
        if not close(r.get("device_throughput_tokens_per_s", -1), dt, rel=0.005, absol=1.0):
            rep.fail(f"{tag} device_throughput_tokens_per_s inconsistent with engine mean")
        mfu = None
        if m and peak:
            fl = flops(m, bs, sq)
            tmin = fl / (peak * 1e12) * 1e3
            mfu = tmin / g[0]
            if g[0] < tmin:
                rep.fail(f"{tag} engine p50 {g[0]} ms < roofline minimum {tmin:.2f} ms")
        table.append((tag, e[0], g[0], e[0] - g[0], r["throughput_tokens_per_s"], mfu, e[3], g[3]))

    if table:
        rep.info("shape      | p50 E2E / engine (ms) | gap (ms) | p99 E2E / eng (ms) | E2E tok/s  | MFU(eng)")
        for tag, e50, g50, gap, tps, mfu, e99, g99 in table:
            ms = f"{100 * mfu:.0f}%" if mfu else "n/a"
            rep.info(f"{tag:<10} | {e50:8.2f} / {g50:8.2f}   | {gap:8.2f} | {e99:7.2f} / {g99:7.2f}  | {tps:10.1f} | {ms:>8}")
    return rep, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--quantized", default="auto", choices=["auto", "yes", "no"])
    args = ap.parse_args()
    nfail = 0
    for p in args.paths:
        rep, _ = check(p, args.quantized)
        nfail += rep.n_fail
        print(rep.render())
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
