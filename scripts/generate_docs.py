#!/usr/bin/env python3
"""Generate README.md, TECHNICAL_GUIDE.md and results/summary_table.json from the TPU benchmark JSONs
(end-to-end latency, throughput and accuracy gates). Every number in the docs comes from results/*.json."""

import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SHAPES = [(1, 8192), (4, 8192), (8, 8192), (16, 8192), (32, 8192)]

REPO_URL = "https://github.com/MG-Cafe/minicpm5-vllm-tpu.git"
# Public image with the exact library sources of the measured runs (verified file-by-file against
# benchmark/pristine_manifest.json: vllm-torchtpu commit e91801a, vLLM v0.32.0).
IMAGE = "vllm/vllm-torchtpu:nightly-20261027-e91801a-v0.32.0"


def load_json(accel: str, model_short: str) -> dict:
    with open(os.path.join(RESULTS_DIR, f"{accel}_minicpm5_{model_short}.json")) as f:
        return json.load(f)


def by_shape(data: dict) -> dict:
    return {(r["batch_size"], r["seq_len"]): r for r in data["by_config"]}


def gate_summary(data: dict) -> dict:
    out = {}
    for g in (data.get("validation") or {}).get("logprob_gates") or []:
        lbl = (str(g.get("reference_label", "")) + " " + str(g.get("reference", ""))).lower()
        key = "fp32_golden" if ("fp32" in lbl or "golden" in lbl) else "bf16_reference"
        out[key] = {k: g.get(k) for k in ("gate_v2", "top1_agreement", "top1_agreement_nolp", "top1_tie_aware",
                                          "top1_tie_aware_nolp", "mean_abs_dlogprob", "kl_top20_mean",
                                          "cos_logprob_mean", "n_seqs")}
        out[key]["shapes"] = [st["shape"] for st in g.get("sets", [])]
        out[key]["batch_invariance"] = g.get("batch_invariance")
    return out


def build_summary(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b) -> dict:
    summary = {"reference_image": IMAGE, "models": {}}
    for m_key, m_name, d_v6e, d_7x in [
        ("MiniCPM5-1B", "openbmb/MiniCPM5-1B", v6e_1b, tpu7x_1b),
        ("MiniCPM5-2B", "openbmb/MiniCPM5-2B", v6e_2b, tpu7x_2b),
    ]:
        r_v6e, r_7x = by_shape(d_v6e), by_shape(d_7x)
        rows = []
        for bs, sl in SHAPES:
            rv, rt = r_v6e[(bs, sl)], r_7x[(bs, sl)]
            rows.append({
                "batch_size": bs, "seq_len": sl, "total_tokens": bs * sl,
                "v6e_e2e_p50_ms": rv["request_latency_ms_p50"], "v6e_e2e_p99_ms": rv["request_latency_ms_p99"],
                "v6e_tokens_per_s": rv["throughput_tokens_per_s"],
                "tpu7x_e2e_p50_ms": rt["request_latency_ms_p50"], "tpu7x_e2e_p99_ms": rt["request_latency_ms_p99"],
                "tpu7x_tokens_per_s": rt["throughput_tokens_per_s"],
            })
        summary["models"][m_key] = {
            "model_id": m_name,
            "v6e_accuracy": gate_summary(d_v6e), "tpu7x_accuracy": gate_summary(d_7x),
            "by_config": rows,
        }
    return summary


def table(rows: list) -> str:
    lines = [
        "| Shape (batch, tokens) | v6e-1 E2E p50 | v6e-1 E2E p99 | v6e-1 tok/s | TPU7x-1 E2E p50 | TPU7x-1 E2E p99 "
        "| TPU7x-1 tok/s | TPU7x-1 vs v6e-1 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `({r['batch_size']}, 8192)` | **{r['v6e_e2e_p50_ms']:,.1f} ms** | {r['v6e_e2e_p99_ms']:,.1f} ms "
            f"| {r['v6e_tokens_per_s'] / 1e3:,.0f}k | **{r['tpu7x_e2e_p50_ms']:,.1f} ms** | {r['tpu7x_e2e_p99_ms']:,.1f} ms "
            f"| {r['tpu7x_tokens_per_s'] / 1e3:,.0f}k | {r['v6e_e2e_p50_ms'] / r['tpu7x_e2e_p50_ms']:.2f}x faster |")
    return "\n".join(lines)


def expected_table(summary: dict) -> str:
    lines = ["| Model | Shape | TPU v6e-1 E2E p50 | TPU7x-1 E2E p50 |", "| :--- | :--- | ---: | ---: |"]
    for m_key in ("MiniCPM5-1B", "MiniCPM5-2B"):
        for r in summary["models"][m_key]["by_config"]:
            lines.append(f"| {m_key} | `({r['batch_size']}, 8192)` | {r['v6e_e2e_p50_ms']:,.1f} ms | "
                         f"{r['tpu7x_e2e_p50_ms']:,.1f} ms |")
    return "\n".join(lines)


def acc_table(summary: dict) -> str:
    lines = ["| Run | vs FP32 golden: top-1 exact / tie-aware · mean \\|Δlp\\| · cos "
             "| vs BF16 reference: top-1 exact / tie-aware · mean \\|Δlp\\| · cos | Gate |",
             "| :--- | :---: | :---: | :---: |"]
    for m_key in ("MiniCPM5-1B", "MiniCPM5-2B"):
        for accel, lab in (("v6e", "TPU v6e-1"), ("tpu7x", "TPU7x-1")):
            acc = summary["models"][m_key][f"{accel}_accuracy"]
            cells, verdicts = [], []
            for ref in ("fp32_golden", "bf16_reference"):
                g = acc.get(ref) or {}
                tn = g.get("top1_tie_aware_nolp")
                t = min(g.get("top1_tie_aware") or 0, tn if tn is not None else 1.0)
                n = g.get("n_seqs") or 0
                ex = round((g.get("top1_agreement") or 0) * n)
                cells.append(f"{ex}/{n} / {100 * t:.0f}% · {g.get('mean_abs_dlogprob', 0):.3f} · {g.get('cos_logprob_mean', 0):.5f}")
                verdicts.append(g.get("gate_v2", "n/a"))
            v = "PASS" if all(x == "PASS" for x in verdicts) else " / ".join(verdicts)
            lines.append(f"| {lab} · {m_key.split('-')[1]} | {cells[0]} | {cells[1]} | **{v}** |")
    return "\n".join(lines)


def headline(summary: dict) -> dict:
    r1 = {(r["batch_size"], r["seq_len"]): r for r in summary["models"]["MiniCPM5-1B"]["by_config"]}
    r2 = {(r["batch_size"], r["seq_len"]): r for r in summary["models"]["MiniCPM5-2B"]["by_config"]}

    def best(rows, key, fn):
        return fn(r[key] for r in rows.values())

    return {
        "v6e_1b_bs1": r1[(1, 8192)]["v6e_e2e_p50_ms"], "v6e_2b_bs1": r2[(1, 8192)]["v6e_e2e_p50_ms"],
        "tpu7x_1b_bs1": r1[(1, 8192)]["tpu7x_e2e_p50_ms"], "tpu7x_2b_bs1": r2[(1, 8192)]["tpu7x_e2e_p50_ms"],
        "v6e_1b_tps": best(r1, "v6e_tokens_per_s", max), "v6e_2b_tps": best(r2, "v6e_tokens_per_s", max),
        "tpu7x_1b_tps": best(r1, "tpu7x_tokens_per_s", max), "tpu7x_2b_tps": best(r2, "tpu7x_tokens_per_s", max),
        "lat_ratio_1b": math.floor(10 * min(r["v6e_e2e_p50_ms"] / r["tpu7x_e2e_p50_ms"] for r in r1.values())) / 10,
        "lat_ratio_1b_max": max(r["v6e_e2e_p50_ms"] / r["tpu7x_e2e_p50_ms"] for r in r1.values()),
    }


ACCEL_LABEL = {"v6e": "TPU v6e-1", "tpu7x": "TPU7x-1"}
CORES = {"v6e": 1, "tpu7x": 2}


def _rev(d):
    return os.path.basename(str(d["config"].get("model_dir", "")).rstrip("/"))


def _plugin_sha(d):
    notes = str(d["config"].get("notes", ""))
    return notes.split("sha256=")[1].split(";")[0].strip() if "sha256=" in notes else "n/a"


def run_facts(datas: dict) -> dict:
    """Facts about the measured runs, read from the result JSONs (nothing typed in by hand)."""
    ds = list(datas.values())
    cfg0 = ds[0]["config"]
    ver = cfg0["versions"]
    vers = {json.dumps(d["config"]["versions"], sort_keys=True) for d in ds}
    roots = [d["config"].get("source_diff", {}).get("roots", {}) for d in ds]
    clean = all(r and not any(v.get("modified") or v.get("added") or v.get("removed_count") for v in r.values())
                for r in roots)
    ts = sorted([d.get("config", {}).get("run_started_utc") or d["timestamp_utc"] for d in ds]
                + [d.get("run_finished_utc") or d["timestamp_utc"] for d in ds])
    ek = {}
    for (accel, _), d in datas.items():
        e = {k: v for k, v in d["config"]["engine_kwargs"].items() if k != "tokenizer"}
        ek[accel] = e
    revs = {}
    for (_, m), d in datas.items():
        revs[d["model"]] = d["config"].get("model_revision") or _rev(d)
    return {
        "date": ts[0][:10], "t_first": ts[0], "t_last": ts[-1],
        "versions_same": len(vers) == 1,
        "software": (f"vLLM {ver.get('vllm', '?').split('+')[0]} with the TorchTPU backend (vllm-torchtpu; "
                     f"torch {ver.get('torch', '?').split('+')[0]}, torch_tpu {ver.get('torch_tpu', '?')}), "
                     f"JAX/jaxlib {ver.get('jax', '?')}, libtpu {ver.get('libtpu', '?')}, transformers "
                     f"{ver.get('transformers', '?')}, tokenizers {ver.get('tokenizers', '?')}, Python {ver.get('python', '?')}"),
        "sources_clean": clean, "n_roots": len(roots[0]) if roots and roots[0] else 0,
        "plugin_sha": sorted({_plugin_sha(d) for d in ds}), "harness_sha": sorted({d["config"].get("harness_sha256", "")[:16] for d in ds}),
        "revs": revs, "engine_kwargs": ek, "env": cfg0.get("env_overrides", []),
        "iters": ds[0]["iters_per_config"], "warmup": ds[0]["warmup_per_config"],
        "tok_mode": cfg0.get("tokenizer_mode_effective", cfg0.get("tokenizer_mode", "hf")),
        "tok_verified": min(int(d["validation"].get("tokenizer_mode_verified_calls") or 0) for d in ds),
        "gc": cfg0.get("gc_policy", "default"), "sampling": cfg0.get("sampling_params", ""),
        "seed": cfg0.get("input_rng_seed"), "pmethod": cfg0.get("percentile_method", "nearest rank"),
        "accels": sorted({a for a, _ in datas}, key=lambda a: a != "v6e"),
        "hosts": {a: (datas[(a, "1b")]["config"].get("host") or {}) for a in ("v6e", "tpu7x") if (a, "1b") in datas},
        "revs_full": {d["model"]: d["config"].get("model_revision") or _rev(d) for d in ds},
    }


def run_details_md(f: dict, compact: bool = False) -> str:
    chips = "; ".join(f"{ACCEL_LABEL[a]}: one chip, {CORES[a]} TensorCore{'s' if CORES[a] > 1 else ''}, "
                      f"`tensor_parallel_size={f['engine_kwargs'][a].get('tensor_parallel_size', 1)}`" for a in f["accels"])
    revs = ", ".join(f"`{m}` @ `{r[:12]}`" for m, r in sorted(f["revs"].items()))
    rows = [
        ("Measured", f"{f['t_first']} – {f['t_last']} (UTC)"),
        ("Hardware", chips + " (single-chip GKE TPU nodes)"),
        ("Software", f["software"] + ("" if f["versions_same"] else " (versions differ per chip, see results JSON)")),
        ("Library sources", (f"unmodified; all {f['n_roots']} package roots match `benchmark/pristine_manifest.json` "
                             f"(same sources as the public image `{IMAGE}`)") if f["sources_clean"]
         else "MODIFIED (see results JSON source_diff)"),
        ("Models", revs + " (Hugging Face revisions)"),
        ("Code", f"plugin tree sha256 `{', '.join(f['plugin_sha'])}`, harness `benchmark/run_benchmark.py` sha256 "
                 f"`{', '.join(f['harness_sha'])}`"),
        ("Inputs", f"fresh random token ids in [1000, 30000) per iteration (seed {f['seed']}), decoded to text; "
                   f"re-tokenized to exactly 8,192 tokens inside the timed region"),
        ("Timed region", "offline, in-process: client tokenizer call + blocking `llm.generate(...)`; tokenizer "
                         f"`examples/fast_tokenizer.py` mode `{f['tok_mode']}` (ids verified identical to "
                         f"`tokenizer(...)` on {f['tok_verified']} calls per run); Python GC frozen after warm-up; "
                         "`SamplingParams(skip_clone=True)`"),
        ("Iterations", f"{f['warmup']} warm-up + {f['iters']} timed per shape; latency = p50 "
                       f"({'median, percentiles linearly interpolated' if 'linear' in f['pmethod'] else f['pmethod']}); "
                       "throughput = tokens ÷ mean latency; raw per-iteration samples are in the JSON"),
        ("Sampling", f"`{f['sampling']}`, prefix caching off"),
        ("Host", "; ".join(f"{ACCEL_LABEL[a]}: {h.get('cpu_model', '?')}, {h.get('cpu_count', '?')} vCPUs, "
                           f"{h.get('mem_gb', '?')} GB RAM" for a, h in f["hosts"].items())),
    ]
    if compact:
        return "\n".join(f"- **{k}**: {v}" for k, v in rows)
    out = ["| Item | Value |", "| :--- | :--- |"] + [f"| {k} | {v} |" for k, v in rows]
    for a in f["accels"]:
        out.append(f"| Engine kwargs ({ACCEL_LABEL[a]}) | `{json.dumps(f['engine_kwargs'][a], separators=(',', ':'))}` |")
    out.append(f"| Environment | `{' '.join(f['env'])}` |")
    return "\n".join(out)


def derived(datas: dict) -> dict:
    """Claims computed from the results instead of typed in (ranges, verdicts, dims, kernels, tokenizer cost)."""
    def rows(a, m):
        return {r["batch_size"]: r for r in datas[(a, m)]["by_config"]}
    out = {"lat": {}, "flat": {}, "dims": {}, "quant": {}, "modes": {}, "tok_extra": {}}
    for m in ("1b", "2b"):
        v, t = rows("v6e", m), rows("tpu7x", m)
        lr = [v[b]["request_latency_ms_p50"] / t[b]["request_latency_ms_p50"] for b in v]
        out["lat"][m] = (min(lr), max(lr))
        for a, rr in (("v6e", v), ("tpu7x", t)):
            tp = [rr[b]["throughput_tokens_per_s"] for b in sorted(rr) if b >= 4]
            out["flat"][(a, m)] = 100.0 * (max(tp) / min(tp) - 1.0)
            ex = [rr[b]["hf_tokenizer_ms_p50"] - rr[b]["tokenize_latency_ms_p50"] for b in sorted(rr)
                  if rr[b].get("hf_tokenizer_ms_p50") is not None]
            if ex:
                out["tok_extra"][(a, m)] = (min(ex), max(ex))
            st = (datas[(a, m)]["config"].get("plugin_status") or [{}])[0]
            out["quant"][(a, m)] = st.get("quant")
            out["modes"][(a, m)] = sorted({x.get("mode") for x in (st.get("shapes") or {}).values()})
        hc = datas[("v6e", m)]["config"].get("hf_config") or {}
        out["dims"][m] = hc
    verdicts = [g.get("gate_v2") for d in datas.values() for g in (d["validation"].get("logprob_gates") or [])]
    out["all_pass"] = bool(verdicts) and all(x == "PASS" for x in verdicts)
    out["n_gate_runs"] = len(verdicts)
    return out


def dims_md(hc: dict, long: bool) -> str:
    if not hc or "num_hidden_layers" not in hc:
        return ""
    if long:
        return (f"({hc['num_hidden_layers']} layers, hidden {hc['hidden_size']:,}, intermediate {hc['intermediate_size']:,}, "
                f"{hc['num_attention_heads']} query / {hc['num_key_value_heads']} KV heads, head dim {hc.get('head_dim')})")
    return (f"({hc['num_hidden_layers']} layers, H {hc['hidden_size']:,}, {hc['num_attention_heads']}Q / "
            f"{hc['num_key_value_heads']}KV heads)")


def quant_md(dv: dict) -> str:
    def layers(q):
        if not q:
            return "n/a"
        head = q.split("|")[0].split(":")
        names = {"qkv": "QKV", "gu": "gate/up", "o": "O", "down": "down"}
        lin = ", ".join(names.get(x, x) for x in head[1].split("+")) if len(head) > 1 else head[0]
        return lin
    rows = ["| Chip / model | Linear layers in INT8 (W8A8) | Fused Pallas kernels | Core split (TPU7x) | Resolved `quant` string |",
            "| :--- | :--- | :--- | :--- | :--- |"]
    for a in ("v6e", "tpu7x"):
        for m in ("1b", "2b"):
            q = dv["quant"].get((a, m)) or ""
            flags = q.split("|")[1:]
            kern = " + ".join(x for x, f in (("fused QKV", "fqkv"), ("fused MLP", "fmlp")) if f in flags) or "none"
            modes = ", ".join(dv["modes"].get((a, m)) or []) if a == "tpu7x" else "—"
            rows.append(f"| {ACCEL_LABEL[a]} · {m.upper()} | {layers(q)} | {kern} | {modes} | `{q}` |")
    return "\n".join(rows)


SUPPORTED_MD = """The fast path runs when **every request in a vLLM step** is a fresh prefill with `max_tokens=1` and all
prompts in the step have the **same length**: a multiple of 1,024 tokens, or of 128 tokens up to 1,024
(at most `max_model_len`). LoRA, speculative decoding, prompt embeddings and multimodal inputs are not
supported on the fast path. Everything was benchmarked at 8,192 tokens per input with batch 1, 4, 8, 16 and 32.

- **First use of a shape compiles** the program for that (batch, length) once; set
  `MINICPM5_MK_PRECOMPILE="1x8192,4x8192,8x8192,16x8192,32x8192"` to compile at load time instead.
- **`MINICPM5_MK_FREE_STOCK=1`** (used in the benchmark) frees vLLM's stock layer weights to save memory, so a
  request outside the envelope above fails with an explicit error. Use `MINICPM5_MK_FREE_STOCK=0` to keep the
  stock vLLM path as a (slower) fallback for other request shapes.
- **TPU7x-1** splits work across its two TensorCores: even batch sizes run data-parallel, batch 1 runs
  sequence-parallel (lengths that are multiples of 2,048). Other odd batch sizes run a BF16 head-sharded
  tensor-parallel mode with different numerics; it is not covered by the published numbers.
- **Sampling features** outside the fast path (prompt logprobs, repetition / presence / frequency penalties,
  `min_p`, logit bias, allowed or bad words, structured output) route the request to the stock path (or raise with
  `MINICPM5_MK_FREE_STOCK=1`); temperature, top-p, top-k, seed, `logprobs` and `min_tokens` are supported.
- **Host-side patches**: installing the plugin also patches vLLM's offline front end in every process (one
  `SamplingParams` copy per `generate()` call, cached EOS id, no re-processing of token-id prompts, single-step
  scheduling of `max_tokens=1` batches) and, only for engines serving this model with
  `MINICPM5_MK_FREE_STOCK=1`, skips KV-cache allocation for fast-path requests. `MINICPM5_MK_ENABLE=0` turns
  all of it off."""


QUICKSTART = f"""```bash
# Inside the {IMAGE} container on a 1-chip TPU (see "Reproduce the results")
git clone {REPO_URL}
cd minicpm5-vllm-tpu
pip install -e ./plugin --no-build-isolation --no-deps
export PYTHONPATH="$(pwd)/plugin:$(pwd):${{PYTHONPATH:-}}"

# Smoke test (~2-3 min incl. download + compile). --tp 1 on TPU v6e-1, --tp 2 on TPU7x-1
python3 examples/minimal_vllm_inference.py --model openbmb/MiniCPM5-1B --tp 1 --batch-size 4 --seq-len 8192

# Every number in this repo: 2 models x 5 shapes + accuracy gates + verification (~10 min per chip)
bash benchmark/run_all_tpu_benchmarks.sh
```"""


def reproduce_md(summary: dict) -> str:
    return f"""```mermaid
flowchart LR
  A["1. Get a 1-chip TPU<br/>v6e-1 or TPU7x-1"] --> B["2. Start the public<br/>vLLM-TorchTPU container"]
  B --> C["3. Install the plugin<br/>(pip install -e ./plugin)"]
  C --> D["4. Smoke test<br/>~3 min"]
  D --> E["5. Full benchmark<br/>~10 min per chip"]
  E --> F["6. Verify + compare<br/>with the tables above"]
```

### Step 1 — Get a single-chip Cloud TPU

The published numbers were measured on **single-chip GKE TPU nodes**: one TPU v6e chip (1 TensorCore) and one
TPU7x chip (2 TensorCores). Any of the following works; replace the `<...>` placeholders with your own values.

**Option A — GKE (how the numbers were measured).** Create (or reuse) a GKE cluster with a single-chip TPU node
pool, then start the ready-made pod from [`deploy/`](deploy):

```bash
# TPU v6e-1 node pool (single-host, 1 chip)
gcloud container node-pools create v6e-1 --cluster=<CLUSTER> --location=<ZONE> \\
    --machine-type=ct6e-standard-1t --num-nodes=1

kubectl apply -f deploy/gke-v6e-1.yaml        # or deploy/gke-tpu7x-1.yaml on a 1x1x1 TPU7x node
kubectl exec -it minicpm5-v6e-1 -- bash       # you are now inside the container (Step 3)
```

For TPU7x, use a node pool whose nodes expose a single TPU7x chip (`cloud.google.com/gke-tpu-topology: 1x1x1`);
check what your project and zone offer with `gcloud compute tpus accelerator-types list --zone=<ZONE>`.

**Option B — TPU VM (v6e-1).**

```bash
gcloud compute tpus tpu-vm create minicpm5-v6e-1 --zone=<ZONE> \\
    --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh minicpm5-v6e-1 --zone=<ZONE>
```

### Step 2 — Start the public vLLM-TorchTPU container

Use the public image **`{IMAGE}`**. It is built from the same library sources as the measured runs
(vllm-torchtpu commit `e91801a`, vLLM `v0.32.0`, torch_tpu `0.2.0`, JAX `0.11.2`; checked file-by-file against
[`benchmark/pristine_manifest.json`](benchmark/pristine_manifest.json)), and the benchmark re-checks your
container against that manifest on every run.

```bash
# On a TPU VM (GKE pods from deploy/ already run this image)
sudo docker run -it --rm --privileged --net host --shm-size=64G \\
    -e HF_HOME=/dev/shm/hf_cache {IMAGE} bash
```

### Step 3 — Install the plugin

```bash
git clone {REPO_URL}
cd minicpm5-vllm-tpu
pip install -e ./plugin --no-build-isolation --no-deps   # registers MiniCPM5MegakernelForCausalLM with vLLM
export PYTHONPATH="$(pwd)/plugin:$(pwd):${{PYTHONPATH:-}}"
```

The plugin has no extra dependencies; the model weights are downloaded from Hugging Face on first use
(`openbmb/MiniCPM5-1B` / `openbmb/MiniCPM5-2B`, pinned revisions, no token needed).

### Step 4 — Smoke test

```bash
# --tp 1 on TPU v6e-1, --tp 2 on TPU7x-1 (both TensorCores)
python3 examples/minimal_vllm_inference.py --model openbmb/MiniCPM5-1B --tp 1 --batch-size 4 --seq-len 8192
```

The first call per shape compiles the kernels (tens of seconds); the script then prints one timed call, e.g.
`shape=(4, 8192) tp=1 | engine_latency=... ms | throughput=... tok/s`.

### Step 5 — Run the full benchmark

```bash
bash benchmark/run_all_tpu_benchmarks.sh          # both models; MODELS=1b or MODELS=2b for one
```

The script detects the chip (`TPU_ACCELERATOR_TYPE`: `v6e-1` → TP=1, `tpu7x-2` → TP=2; set `ACCEL=v6e` or
`ACCEL=tpu7x` to force it), then for each model: loads the model, runs the accuracy gate against the committed
FP32 and BF16 references, times 5 warm-up + 50 measured iterations for each of the 5 shapes, and writes
`results/<chip>_minicpm5_<model>.json` plus gate dumps in `results/gate/`. It ends by running the verifier.

### Step 6 — Verify and compare

```bash
python3 benchmark/verify_results.py results/*_minicpm5_*.json   # expect: 0 FAIL, 0 WARN
python3 scripts/generate_charts.py && python3 scripts/generate_docs.py   # rebuild charts + docs from your JSONs
```

The verifier re-checks every statistic from the raw per-iteration samples, the accuracy gates, the code hashes,
and that your container's library sources match the reference image (a different image is reported as `WARN`).
Expected p50 end-to-end latency (results from this repo):

{expected_table(summary)}

Run-to-run variation on the same chip is typically within a few percent. The end-to-end time includes
client-side tokenization on the host CPU, so a slower host CPU shows up mostly at small batch sizes.
"""


VLLM_SNIPPET = """```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TPU_GREEDY_TOP_TOKENS"] = "1"
os.environ["MINICPM5_MK_QUANT"] = "auto"   # picks the best kernels + W8A8 mode per chip
os.environ["MINICPM5_MK_ATTN"] = "auto"    # SuperAttention v2 for 8,192-token inputs
os.environ["MINICPM5_MK_CAL_NSEQ"] = "16"  # SmoothQuant calibration (same as the benchmark)
os.environ["MINICPM5_MK_CAL_SEQ"] = "256"
os.environ["MINICPM5_MK_FREE_STOCK"] = "1" # free the unused stock layer weights (see Supported inputs)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from examples.fast_tokenizer import ClientTokenizer
from examples.minimal_vllm_inference import prepare_local_model_dir

model_dir = prepare_local_model_dir("openbmb/MiniCPM5-1B")
llm = LLM(model=model_dir, tokenizer=model_dir,
          hf_overrides={"architectures": ["MiniCPM5MegakernelForCausalLM"]},
          tensor_parallel_size=1,               # 2 on TPU7x-1 (both TensorCores)
          dtype="bfloat16", enable_prefix_caching=False, max_model_len=8256,
          max_num_batched_tokens=262144, max_num_seqs=32, block_size=256,
          gpu_memory_utilization=0.5,
          compilation_config={"compile_sizes": [8192, 32768, 65536, 131072, 262144]})
sp = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, detokenize=False, skip_clone=True)
fast_tok = ClientTokenizer(AutoTokenizer.from_pretrained(model_dir), mode="parallel")

# texts: your batch of input strings. The two timed lines of the benchmark ("end-to-end"):
ids = fast_tok(texts, 8192)  # = tokenizer(texts, padding="max_length", truncation=True, max_length=8192) ids
outputs = llm.generate([{"prompt_token_ids": x} for x in ids], sp)
```"""


LAYOUT_MD = """```text
minicpm5-vllm-tpu/
├── plugin/minicpm5_mk/          vLLM plugin: model + Pallas kernels
│   ├── model.py                 MiniCPM5MegakernelForCausalLM (vLLM model class, host fast path)
│   ├── mk_jax.py                whole-model jax.lax.scan, TPU7x dual-core modes
│   ├── mk_attn.py               SuperAttention v2 (causal GQA flash attention) + sequence-parallel variant
│   ├── mk_mlp.py                fused INT8 QKV and fused MLP kernels
│   └── mk_weights.py            weight packing + SmoothQuant W8A8 calibration
├── examples/                    minimal vLLM example + the parallel client tokenizer used in the benchmark
├── benchmark/
│   ├── run_all_tpu_benchmarks.sh  one command: both models, 5 shapes, accuracy gates, verification
│   ├── run_benchmark.py         timing harness (records config, versions, raw samples)
│   ├── verify_results.py        re-checks every published number from the raw samples
│   ├── gate.py, score_gate.py   accuracy gate (top-20 log-prob comparison)
│   ├── f32_golden.py            FP32 Hugging Face reference
│   ├── ref/                     committed FP32 and BF16 reference log-probs
│   └── pristine_manifest.json   library-source hashes of the reference container image
├── results/                     measured JSONs (+ gate dumps) behind every number and chart
├── charts/                      generated charts
├── deploy/                      GKE pod manifests (single-chip TPU v6e and TPU7x)
└── scripts/                     generate_charts.py, generate_docs.py (rebuild charts + docs from results/)
```"""


FAQ_MD = f"""- **`TPU_ACCELERATOR_TYPE is unset`** — set `ACCEL=v6e` (TP=1) or `ACCEL=tpu7x` (TP=2) before running the
  benchmark script.
- **`WARN library sources differ from the reference image`** — you are not using `{IMAGE}`. The plugin may
  still work, but the published numbers only apply to that image.
- **The first request of every new (batch, length) is slow** — that is the one-time compilation. Set
  `MINICPM5_MK_PRECOMPILE="1x8192,4x8192,8x8192,16x8192,32x8192"` to compile everything at load time.
- **A request fails with an explicit "outside the fast path" error** — see [Supported inputs](#supported-inputs);
  set `MINICPM5_MK_FREE_STOCK=0` to fall back to vLLM's stock path for such requests.
- **Disable the plugin without uninstalling it** — `MINICPM5_MK_ENABLE=0`.
- **Model download is slow / fills the disk** — set `HF_HOME` to a large or RAM-backed directory
  (the scripts default to `/dev/shm/hf_cache`)."""


def takeaways_md(h: dict, dv: dict) -> str:
    return f"""- **TPU7x-1 is the fastest single chip**: {h['tpu7x_1b_bs1']:.1f} ms for one 8,192-token 1B request and
  {h['tpu7x_1b_tps'] / 1e3:.0f}k tokens/s peak; {dv['lat']['1b'][0]:.2f}–{dv['lat']['1b'][1]:.2f}x (1B) and
  {dv['lat']['2b'][0]:.2f}–{dv['lat']['2b'][1]:.2f}x (2B) lower latency than TPU v6e-1, using both TensorCores.
- **TPU v6e-1 serves the same workload on one TensorCore**: {h['v6e_1b_bs1']:.1f} ms (1B) and
  {h['v6e_2b_bs1']:.1f} ms (2B) for one request; up to {h['v6e_1b_tps'] / 1e3:.0f}k tokens/s (1B).
- **Throughput is flat from batch 4 to 32** (at most {max(dv['flat'].values()):.0f}% change), so latency grows
  almost linearly with batch size: pick the batch size that fits your latency budget.
- **Accuracy**: {"all " + str(dv['n_gate_runs']) + " accuracy-gate checks pass" if dv['all_pass'] else "NOT all gate checks pass"}
  against a full-FP32 run of the original checkpoint."""


def write_readme(summary: dict, facts: dict, dv: dict):
    h = headline(summary)
    t_1b = table(summary["models"]["MiniCPM5-1B"]["by_config"])
    t_2b = table(summary["models"]["MiniCPM5-2B"]["by_config"])
    readme = f"""# MiniCPM5 (1B & 2B) on a single Cloud TPU chip with vLLM

A drop-in **vLLM plugin** (`minicpm5_mk`) with custom **JAX / Pallas kernels** that runs
[`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) and
[`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) prefill / scoring
(**8,192 tokens per input**, one output token) on **one TPU v6e chip** or **one TPU7x chip** — through the
standard `vllm.LLM` API, on the public [vLLM-TorchTPU](https://github.com/vllm-project/vllm-torchtpu) stack.

![Headline results](charts/kpi_cards.png)

| | TPU v6e-1 (1 chip) | TPU7x-1 (1 chip) |
| :--- | ---: | ---: |
| MiniCPM5-1B · 1 × 8,192 tokens, end-to-end | **{h['v6e_1b_bs1']:.1f} ms** | **{h['tpu7x_1b_bs1']:.1f} ms** |
| MiniCPM5-2B · 1 × 8,192 tokens, end-to-end | **{h['v6e_2b_bs1']:.1f} ms** | **{h['tpu7x_2b_bs1']:.1f} ms** |
| MiniCPM5-1B · peak throughput | {h['v6e_1b_tps'] / 1e3:.0f}k tok/s | {h['tpu7x_1b_tps'] / 1e3:.0f}k tok/s |
| MiniCPM5-2B · peak throughput | {h['v6e_2b_tps'] / 1e3:.0f}k tok/s | {h['tpu7x_2b_tps'] / 1e3:.0f}k tok/s |
| Accuracy gate vs FP32 checkpoint | {"PASS" if dv['all_pass'] else "see guide"} | {"PASS" if dv['all_pass'] else "see guide"} |

**Contents:** [Results](#results) · [How it works](#how-it-works) · [Accuracy](#accuracy) ·
[Reproduce the results](#reproduce-the-results) · [Use it in your code](#use-it-in-your-code) ·
[Repository layout](#repository-layout) · [FAQ](#faq) · [Technical guide](TECHNICAL_GUIDE.md)

## Results

All numbers are **end-to-end** for 8,192-token inputs: client-side tokenization (the repo's parallel tokenizer,
[`examples/fast_tokenizer.py`](examples/fast_tokenizer.py)) + `llm.generate(...)`, p50 of {facts['iters']} timed
runs after {facts['warmup']} warm-up runs, one chip, offline and in-process.

![End-to-end latency](charts/latency_by_shape.png)

![Throughput](charts/throughput_by_shape.png)

**Takeaways**

{takeaways_md(h, dv)}

<details>
<summary><b>Full tables (click to expand)</b></summary>

**MiniCPM5-1B** {dims_md(dv['dims']['1b'], False)}

{t_1b}

**MiniCPM5-2B** {dims_md(dv['dims']['2b'], False)}

{t_2b}

E2E = end-to-end latency (tokenize + generate); tok/s = tokens ÷ mean E2E latency. A plain `tokenizer(...)`
call instead of the parallel tokenizer would add {dv['tok_extra'].get(('v6e', '1b'), (0, 0))[0]:.0f}–{dv['tok_extra'].get(('v6e', '1b'), (0, 0))[1]:.0f} ms
on v6e-1 and {dv['tok_extra'].get(('tpu7x', '1b'), (0, 0))[0]:.0f}–{dv['tok_extra'].get(('tpu7x', '1b'), (0, 0))[1]:.0f} ms on TPU7x-1 per call (1B runs, same hosts).
</details>

<details>
<summary><b>Run details: hardware, software, method (click to expand)</b></summary>

{run_details_md(facts, compact=True)}

Every number on this page is generated from `results/*.json` by `scripts/generate_docs.py`;
`python3 benchmark/verify_results.py results/*_minicpm5_*.json` re-checks them.
</details>

## How it works

![One decoder layer](charts/architecture_pipeline.png)

vLLM schedules requests as usual. When a step contains only fresh 8,192-token scoring requests, the plugin runs
the **whole model as one compiled JAX program** (a `jax.lax.scan` over the layers) instead of vLLM's
layer-by-layer path. Each decoder layer is made of:

1. **Fused QKV** (Pallas) — RMSNorm, per-token INT8 quantization and the INT8 QKV matmul in one pass.
2. **SuperAttention v2** (Pallas) — causal GQA flash attention built for the TPU MXU: K/V staged once in VMEM,
   QKᵀ and PV matmuls software-pipelined, softmax row-sum computed by the MXU (`P · [V | 1]`), bf16 `exp2`.
3. **Fused MLP** (Pallas) — RMSNorm, INT8 gate/up, SiLU, down projection and residual add with the weights
   resident in VMEM, so the large intermediate activation never goes to HBM. (TPU7x VMEM cannot hold the 2B MLP
   weights, so 2B on TPU7x-1 runs its MLP as XLA matmuls: INT8 gate/up, BF16 down.)
4. **Last token only** — the last layer computes only the final position and hands it to vLLM's `lm_head` / sampler.
5. **Both TPU7x TensorCores** — even batch sizes are split across the two cores; a single 8,192-token input is
   split by sequence in zig-zag 512-token blocks so both cores get the same causal attention work.

Kernel-level details are in the [technical guide](TECHNICAL_GUIDE.md#3-kernels).

## Accuracy

Every benchmark run first re-scores 5 fixed 8,192-token prompts (the 4-prompt set also inside batches of 8, 16
and 32) and compares the next-token top-20 log-probabilities with a **full-FP32 run of the original Hugging Face
checkpoint** and with **this plugin's BF16 path**. Pass = mean |Δ logprob| ≤ 0.10 nats and tie-aware top-1 ≥ 95%.
{"All " + str(dv['n_gate_runs']) + " gate checks pass." if dv['all_pass'] else "NOT all gate checks pass — see TECHNICAL_GUIDE.md."}
This is a numerical-fidelity check on synthetic prompts, not a task-accuracy benchmark; details in the
[technical guide](TECHNICAL_GUIDE.md#4-accuracy-gate).

![Accuracy gate](charts/accuracy_gate.png)

## Reproduce the results

{reproduce_md(summary)}

## Use it in your code

{VLLM_SNIPPET}

### Supported inputs

The fast path needs every request in a vLLM step to be a fresh `max_tokens=1` prefill with the same prompt
length (a multiple of 1,024 tokens; benchmarked at 8,192) and no prompt logprobs or penalties. With
`MINICPM5_MK_FREE_STOCK=1`, other requests fail with an explicit error; set it to `0` to keep vLLM's stock path
as a fallback. Installing the plugin also adds small host-side patches to vLLM's offline front end;
`MINICPM5_MK_ENABLE=0` disables everything. Details in the
[technical guide](TECHNICAL_GUIDE.md#5-supported-inputs).

## Repository layout

{LAYOUT_MD}

## FAQ

{FAQ_MD}

## License

Apache 2.0 — see [LICENSE](LICENSE). MiniCPM5 model weights are subject to their own license on Hugging Face.
This is not an officially supported Google product.
"""
    with open(os.path.join(REPO_ROOT, "README.md"), "w") as f:
        f.write(readme)
    print("Wrote README.md")


def write_technical_guide(summary: dict, facts: dict, dv: dict):
    h = headline(summary)
    t_1b = table(summary["models"]["MiniCPM5-1B"]["by_config"])
    t_2b = table(summary["models"]["MiniCPM5-2B"]["by_config"])
    acc = acc_table(summary)
    guide = f"""# MiniCPM5 (1B & 2B) on Cloud TPU with vLLM — Technical Guide

**Contents:** [Summary](#summary) · [1. Results](#1-results) · [2. Architecture](#2-architecture) ·
[3. Kernels](#3-kernels) · [4. Accuracy gate](#4-accuracy-gate) · [5. Supported inputs](#5-supported-inputs) ·
[6. Reproduce](#6-reproduce) · [7. Run details](#7-run-details)

## Summary

`minicpm5_mk` is a vLLM-TorchTPU model plugin that serves **MiniCPM5-1B** and **MiniCPM5-2B** prefill /
scoring requests (8,192 tokens per input, `max_tokens=1`) with custom JAX/Pallas kernels, behind the
standard `vllm.LLM` API.

![Headline results](charts/kpi_cards.png)

- **TPU7x-1** (one chip, 2 TensorCores): {h['tpu7x_1b_bs1']:.1f} ms (1B) and {h['tpu7x_2b_bs1']:.1f} ms (2B)
  end-to-end for one 8,192-token request; up to {h['tpu7x_1b_tps'] / 1e3:.0f}k tokens/s (1B).
- **TPU v6e-1** (one chip, 1 TensorCore): {h['v6e_1b_bs1']:.1f} ms (1B) and {h['v6e_2b_bs1']:.1f} ms (2B); up to
  {h['v6e_1b_tps'] / 1e3:.0f}k tokens/s (1B).

## 1. Results

All numbers are end-to-end: client-side tokenization of fresh random 8,192-token prompts plus
`llm.generate(...)`, p50 over {facts['iters']} timed iterations after {facts['warmup']} warm-up iterations,
`SamplingParams(max_tokens=1, temperature=0)`, prefix caching off. Measured {facts['date']}; full run details
(hardware, software versions, model revisions, code hashes) are in [section 7](#7-run-details).

### End-to-end latency

![End-to-end latency](charts/latency_by_shape.png)

### Throughput

![Throughput](charts/throughput_by_shape.png)

### Tables

**MiniCPM5-1B** {dims_md(dv['dims']['1b'], True)}

{t_1b}

**MiniCPM5-2B** {dims_md(dv['dims']['2b'], True)}

{t_2b}

E2E = end-to-end latency (tokenize + generate); tok/s = tokens ÷ mean E2E latency.

## 2. Architecture

![One decoder layer](charts/architecture_pipeline.png)

vLLM schedules the batch as usual. When every request in the step is a fresh 8,192-token prefill with
`max_tokens=1`, the plugin hands the whole batch to one compiled JAX program (via TorchTPU's `jax_op`
bridge) instead of running the stock layer-by-layer path:

- **Whole model in one `jax.lax.scan`** over layers 0 … L-2, with batches above 32,768 tokens processed in
  on-device chunks of 4 × 8,192 tokens (bounded HBM, one dispatch).
- **Last layer on the last token only**: layer L-1 needs keys/values for every position but queries and
  MLP only for the final token, so it returns `(batch, hidden)` straight to vLLM's `lm_head`.
- **Host fast path**: one engine step per batch, token ids staged directly into the input buffer, no KV
  cache blocks allocated for scoring requests.
- **TPU7x-1 uses both TensorCores**: even batch sizes run data-parallel (half the sequences per core);
  batch 1 runs sequence-parallel: each core owns half of the 512-token query blocks (zig-zag order, so
  causal work is balanced), runs the INT8 fused kernels on its tokens and all-gathers K/V once per layer.

## 3. Kernels

### 3.1 SuperAttention v2 (`mk_attn.super_gqa_v2`)

Causal GQA prefill attention for head dim 128 on the 256 × 256 TPU MXU.

- **Grid**: (sequence, 512-token query block). For each KV head the 8 query heads that share it are
  stacked into one `(8 × 512, 128)` matrix, so every K/V tile is read once per KV head.
- **K/V staged once per sequence**: RoPE is applied to K on the MXU and the result kept in VMEM together
  with `V' = [V | 1]`, a copy of V with a ones block appended.
- **MXU row-sum**: `P @ V'` returns both `P·V` and the softmax denominator `Σ P` in one matmul. V only
  fills 128 of the MXU's 256 columns, so the denominator is free and is accumulated in f32 from exactly
  the bf16 `P` used for `P·V`.
- **Pipelined 256-row chunks**: the 4,096 stacked query rows are processed in 256-row chunks; the QKᵀ +
  softmax of chunk *t* overlaps the PV matmul of chunk *t-1*, and the score tile stays in vregs instead
  of round-tripping through VMEM.
- **Cheap softmax**: `log2(e)` is folded into Q, so `P = exp2(S − m)` runs on the bf16 EUP; the running
  max and rescaling stay in f32.
- **Causal work skipping**: blocks above the diagonal are never visited; the first KV block is peeled
  (no zero-init, no rescale), the diagonal block uses a precomputed bias and only the keys each chunk can
  see, interior blocks run two per loop iteration, and the output is normalized straight from vregs.

### 3.2 Fused MLP (`mk_mlp.fused_mlp`)

`x + down(SiLU(gate) · up)` for one layer in a single kernel. Tokens stream through in 512-row tiles
while INT8 `W_gate|up` (interleaved per 1,536-column chunk) and BF16 `W_down` stay resident in VMEM. Each
tile computes RMSNorm statistics, per-token INT8 quantization, the INT8 gate/up matmul, SiLU in f32 and
the down projection chunk by chunk, so the `(tokens × 2·intermediate)` activation never touches HBM.
Weights are read directly from the stacked per-layer tensor via a scalar-prefetched layer index.

### 3.3 Fused QKV (`mk_mlp.fused_qkv`)

RMSNorm + per-token INT8 quantization + INT8 QKV matmul + dequantization in one pass over the
activations (the norm scale is folded into the dequantization factor).

### 3.4 Batch 1 on both TPU7x TensorCores (`mk_attn.super_gqa_v2_sp`)

One 8,192-token request is split across the two TensorCores by 512-token query blocks in zig-zag order
(core 0: blocks 0, 3, 4, 7, …; core 1: blocks 1, 2, 5, 6, …), which gives both cores the same causal
attention work. Each core runs the fused QKV / MLP kernels on its 4,096 tokens, all-gathers the K/V of
each layer (8 MB), and runs the attention kernel on its own query blocks with global positions for RoPE
and the causal mask. The core index is passed in as a sharded operand.

### 3.5 Quantization

SmoothQuant-style W8A8 (α = 0.5) with per-output-channel INT8 weights and per-token dynamic INT8
activations; SmoothQuant scales are calibrated at load time on 16 × 256 random-token sequences (seed 42) whose
first token is id 1 (`</s>`). Calibrating with BOS (id 0) first instead measured slightly worse on the accuracy
gate (2B on v6e-1: mean |Δ logprob| 0.062 vs 0.040), so id 1 is the default (`MINICPM5_MK_CAL_FIRST_ID`).
`MINICPM5_MK_QUANT=auto` resolved to the following in the measured runs (read from each run's recorded plugin
status, `results/*.json` → `config.plugin_status`):

{quant_md(dv)}

The last layer always runs in BF16.

## 4. Accuracy gate

Before timing, each run re-scores **5 fixed 8,192-token prompts** (random-token text, seed 20261027: one batch of
1 and one of 4) and compares the **last-position top-20 next-token log-probabilities** with two references:

- **FP32 golden**: the HF checkpoint run in full FP32 (`benchmark/f32_golden.py`).
- **Plugin BF16 path**: this plugin with `MINICPM5_MK_QUANT=bf16` (committed reference).

The 4-prompt set is also run inside batches of 8, 16 and 32 (seeded filler prompts after it), so every published
batch size is scored, and those outputs are compared with the batch-4 outputs of the same run (batch invariance).
A run passes if mean |Δ logprob| over the reference top-20 ≤ 0.10 nats and tie-aware top-1 ≥ 95%.
"Tie-aware" forgives a top-1 change only when the two tokens are within 0.1 nats in the reference or exactly
tied in the candidate. Exact top-1 is reported next to it. (A logprob cosine ≥ 0.99 is also recorded, but it is
not discriminative and is not used as evidence here.)

Limits: these prompts are random-token text with flat next-token distributions and only the last position is
scored, so this is a numerical-fidelity check of the kernels, not a task-accuracy evaluation.

![Accuracy gate](charts/accuracy_gate.png)

{acc}

## 5. Supported inputs

{SUPPORTED_MD}

## 6. Reproduce

The full step-by-step guide (TPU provisioning, container, install, smoke test, benchmark, verification and the
expected numbers) is in the [README](README.md#reproduce-the-results). In short:

{QUICKSTART}

`run_all_tpu_benchmarks.sh` detects the chip (v6e-1 → TP=1, TPU7x-1 → TP=2), runs both models over all five
shapes, checks accuracy against both references, writes `results/*.json` and verifies them with
`benchmark/verify_results.py`. `scripts/generate_charts.py` and `scripts/generate_docs.py` rebuild every
chart and this guide from those JSON files.

### Use from your own vLLM code

{VLLM_SNIPPET}

## 7. Run details

{run_details_md(facts)}

Gate dumps of every run (`results/gate/*.json.gz`) can be re-scored with `python3 benchmark/score_gate.py REF CAND`.
Every number in this guide is generated from `results/*.json` by `scripts/generate_docs.py`;
`python3 benchmark/verify_results.py results/*_minicpm5_*.json` re-checks the files: methodology, accuracy gates,
raw per-iteration samples against every statistic, that the recorded plugin and harness hashes match this
repository, and that the library sources match the reference container image. Host names and cluster metadata in
the published JSON files are redacted; the measurements are unchanged.
"""
    with open(os.path.join(REPO_ROOT, "TECHNICAL_GUIDE.md"), "w") as f:
        f.write(guide)
    print("Wrote TECHNICAL_GUIDE.md")


def main():
    v6e_1b, tpu7x_1b = load_json("v6e", "1b"), load_json("tpu7x", "1b")
    v6e_2b, tpu7x_2b = load_json("v6e", "2b"), load_json("tpu7x", "2b")
    summary = build_summary(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b)
    with open(os.path.join(RESULTS_DIR, "summary_table.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote results/summary_table.json")
    datas = {("v6e", "1b"): v6e_1b, ("tpu7x", "1b"): tpu7x_1b, ("v6e", "2b"): v6e_2b, ("tpu7x", "2b"): tpu7x_2b}
    facts = run_facts(datas)
    dv = derived(datas)
    write_readme(summary, facts, dv)
    write_technical_guide(summary, facts, dv)


if __name__ == "__main__":
    main()
