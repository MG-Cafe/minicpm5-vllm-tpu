#!/usr/bin/env python3
"""MiniCPM5 x vLLM-TorchTPU 8,192-tokens-per-input forward-pass benchmark.

Methodology:
  * 5 shapes with seq_len == 8,192 per input across batch sizes {1, 4, 8, 16, 32}:
    (1,8192) (4,8192) (8,8192) (16,8192) (32,8192)
  * every iteration uses fresh random token ids in [1000, 30000) decoded to text; inside the timed
    region the text is re-tokenized to exactly seq_len ids (BOS + truncated text; padding would be
    counted per shape in `padded_sequences` and is expected to be 0)
  * SamplingParams(max_tokens=1, min_tokens=1, temperature=0, detokenize=False),
    enable_prefix_caching=False, 5 warmup + 50 timed iterations per shape
  * request_latency_ms_* ("E2E")   = client tokenizer call (--tokenizer-mode) + llm.generate(...)
  * device_latency_ms_*  ("engine") = wall time of llm.generate(...) only (NOT pure device time)

CLI Options:
  --engine-kwargs JSON (merged over the defaults below), --env KEY=VAL (set before vllm import),
  a `config` dict recording engine kwargs, env vars, versions and plugin git state,
  --shapes subset, --iters/--warmup,
  --dump-logprobs FILE     fixed seeded inputs: (1,8192) x1 and (4,8192) x4, last-position
                           top-20 logprobs (+ the prompt token ids, so compares are exact)
  --compare-logprobs REF   re-runs REF's exact prompts and prints top-1 agreement + cosine
  --profile-shape/--profile-dir/--profile-steps   XProf trace of steady-state generate() calls

Run inside a TPU pod (v6e image, or `source /workspace/ttvllm_env.sh` on TPU7x), e.g.:
  python3 benchmark/run_benchmark.py --model openbmb/MiniCPM5-1B --label v6e-minicpm5-1b-supermk \
      --out results/v6e_minicpm5_1b.json
"""

import argparse
import hashlib
import json
import os
import re
import sys

# --------------------------------------------------------------------------------------
# Argument parsing happens before importing torch/vllm so that --env can affect them.
# --------------------------------------------------------------------------------------
SWEEP_CONFIGS = [(1, 8192), (4, 8192), (8, 8192), (16, 8192), (32, 8192)]
GATE_SETS = [(1, 8192), (4, 8192)]  # (batch_size, seq_len) for the correctness gate

DEFAULT_ENGINE_KWARGS = {
    "enable_prefix_caching": False,
    "max_model_len": 8256,
    "max_num_batched_tokens": 262144,
    "max_num_seqs": 32,
    "dtype": "bfloat16",
    "seed": 0,
}

ENV_PREFIXES = (
    "VLLM_", "TPU_", "PJRT_", "JAX_", "XLA_", "LIBTPU", "TORCH_TPU", "TORCHTPU", "TORCH_",
    "ENABLE_", "REQUANTIZE_", "MODEL_IMPL", "USE_", "NEW_MODEL", "HF_", "TOKENIZERS_",
    "OMP_", "MKL_", "OPENBLAS_", "PYTHON", "MALLOC", "LD_PRELOAD", "SKIP_", "DISABLE_",
)
ENV_SKIP = re.compile(r"(^|_)(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH|AUTHORIZATION|[A-Z]*KEY)($|_)",
                      re.IGNORECASE)


def parse_shapes(s):
    if not s or s == "all":
        return list(SWEEP_CONFIGS)
    if s.strip().lower() == "none":
        return []
    out = []
    for part in s.split(","):
        part = part.strip().lower().replace("(", "").replace(")", "")
        if not part:
            continue
        bs, sl = re.split(r"[x:/]", part)
        out.append((int(bs), int(sl)))
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openbmb/MiniCPM5-1B", help="HF id or local checkpoint dir")
    ap.add_argument("--tokenizer", default=None, help="HF id/dir for the tokenizer (default: --model)")
    ap.add_argument("--revision", default="pinned",
                    help="Hugging Face revision: 'pinned' (default: the revisions in "
                         "examples/minimal_vllm_inference.PINNED_REVISIONS), a commit sha, or 'latest'")
    ap.add_argument("--label", required=True, help="free-form run label, stored as `device`/`label`")
    ap.add_argument("--out", default=None, help="result JSON path (omit to skip writing)")
    ap.add_argument("--accelerator", default=None, help="v6e or tpu7x (default: from TPU_ACCELERATOR_TYPE)")
    ap.add_argument("--precision", default="bf16", help="label only: bf16 / int8 / fp8 ...")
    ap.add_argument("--quantization-label", default=None, help="label only, e.g. compressed-tensors-int8-w8a8")
    ap.add_argument("--engine-kwargs", default="{}", help="JSON dict merged over defaults "
                    + json.dumps(DEFAULT_ENGINE_KWARGS))
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="env var set before importing vllm (repeatable); recorded in config")
    ap.add_argument("--shapes", default="all", help="e.g. 1x8192,4x8192 ; 'none' skips the sweep")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234, help="rng seed for the timed random inputs")
    ap.add_argument("--dump-logprobs", default=None, help="write gate logprobs (fixed seeded inputs) here")
    ap.add_argument("--compare-logprobs", action="append", default=None, metavar="REF",
                    help="reference dump to compare against (repeatable: one gate report per ref)")
    ap.add_argument("--gate-seed", type=int, default=20261027, help="seed for the gate inputs (new dumps)")
    ap.add_argument("--logprobs-k", type=int, default=20)
    ap.add_argument("--gate-batch-shapes", default="8,16,32",
                    help="extra gate batches: the reference's 4-prompt set runs first in a batch of each size "
                         "(filled with seeded random prompts), so every published batch size is scored")
    ap.add_argument("--profile-shape", default=None, help="e.g. 1x8192 or 1x8192,4x8192; profiles after the sweep")
    ap.add_argument("--profile-dir", default=None, help="XProf output dir (required with --profile-shape)")
    ap.add_argument("--profile-steps", type=int, default=3, help="generate() calls inside the trace")
    ap.add_argument("--notes", default="", help="free text stored in config (patches applied, etc.)")
    ap.add_argument("--source-manifest", default=None,
                    help="verifier pristine manifest for config.source_diff (default: "
                         "/tmp/pristine_manifest.json or pristine_manifest.json next to this script)")
    ap.add_argument("--no-sanitize-tokenizer-config", action="store_true")
    ap.add_argument("--tokenizer-mode", default="hf", choices=["hf", "fast", "parallel"],
                    help="client tokenization inside the timed E2E region: hf (default, standard HF call), "
                         "fast (Rust encode_batch_fast, identical ids), parallel (fast + chunked long texts, "
                         "identical ids); non-hf modes are verified against hf on the first calls")
    ap.add_argument("--gc", default="default", choices=["default", "freeze", "disable"],
                    help="client GC policy for the timed loop: freeze = gc.freeze() after warmup; "
                         "disable = gc.disable() during timed iterations (collect between shapes)")
    ap.add_argument("--skip-clone", action="store_true",
                    help="host-tuned: SamplingParams(skip_clone=True) (what vLLM's OpenAI server sets), so "
                         "the per-request deepcopy in InputProcessor is a shallow copy")
    ap.add_argument("--preload", action="append", default=[], metavar="FILE.py",
                    help="python file imported in this process before LLM() (engine-side patches; reaches "
                         "the engine only with VLLM_ENABLE_V1_MULTIPROCESSING=0 or fork). Path + sha256 + "
                         "its APPLIED list are recorded in config.preloads")
    return ap


ARGS = build_parser().parse_args()
for kv in ARGS.env:
    k, _, v = kv.partition("=")
    os.environ[k] = v

os.environ.setdefault("MINICPM5_MK_STATUS_DIR", f"/tmp/minicpm5_mk_status_{os.getpid()}")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import random  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def prepare_local_model_dir(model_id: str, sanitize: bool = True, revision: str = "pinned") -> str:
    from examples.minimal_vllm_inference import prepare_local_model_dir as _prep
    return _prep(model_id, revision=None if revision == "latest" else revision, sanitize=sanitize)


def _host_info():
    info = {"cpu_count": os.cpu_count(), "tpu_accelerator_type": os.environ.get("TPU_ACCELERATOR_TYPE")}
    try:
        with open("/proc/cpuinfo") as f:
            names = [ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")]
        info["cpu_model"] = names[0] if names else None
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo") as f:
            kb = next(int(ln.split()[1]) for ln in f if ln.startswith("MemTotal"))
        info["mem_gb"] = round(kb / 1048576, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def _hf_config_summary(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        keys = ("architectures", "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "rope_theta", "rms_norm_eps", "bos_token_id",
                "tie_word_embeddings", "torch_dtype")
        return {k: c.get(k) for k in keys}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pctl(vals, p):
    """Percentile with linear interpolation between closest ranks (numpy's default method)."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def _versions():
    from importlib import metadata
    out = {}
    for pkg in ("vllm", "vllm-torchtpu-core", "vllm-torchtpu", "torch", "torch_tpu", "jax", "jaxlib",
                "libtpu", "transformers", "tokenizers"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            pass
    out["python"] = sys.version.split()[0]
    return out


def _plugin_git_state():
    """Record plugin source state (HEAD, dirty files, diff hash) so patches are reproducible."""
    state = {}
    try:
        import vllm_torchtpu
        pkg_dir = os.path.dirname(os.path.realpath(vllm_torchtpu.__file__))
        state["plugin_path"] = pkg_dir
        root = subprocess.run(["git", "-C", pkg_dir, "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, timeout=20)
        if root.returncode == 0:
            r = root.stdout.strip()
            state["git_root"] = r
            state["git_head"] = subprocess.run(["git", "-C", r, "rev-parse", "HEAD"], capture_output=True,
                                               text=True, timeout=20).stdout.strip()
            st = subprocess.run(["git", "-C", r, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=60).stdout
            state["git_dirty_files"] = [ln[3:] for ln in st.splitlines() if ln.strip()][:200]
            diff = subprocess.run(["git", "-C", r, "diff"], capture_output=True, text=True, timeout=60).stdout
            state["git_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest() if diff else None
            state["git_diff_bytes"] = len(diff)
    except Exception as e:
        state["error"] = repr(e)
    return state


# Source-integrity diff vs the verifier's pristine manifest (ttvllm/verifier/source_integrity.py logic,
# inlined so the harness stays a single file). Records every modified/added file with its sha256.
_SI_SP = "usr/local/lib/python3.12/site-packages/"
_SI_ROOTS = ["workspace/vllm-torchtpu/", _SI_SP + "vllm/", _SI_SP + "vllm_torchtpu/", _SI_SP + "torch_tpu/",
             _SI_SP + "jax/", _SI_SP + "transformers/", _SI_SP + "tokenizers/"]
_SI_EXCLUDE = ("workspace/vllm-torchtpu/build/", "workspace/vllm-torchtpu/tests/",
               "workspace/vllm-torchtpu/tests_omni/")
_SI_BIG = 64 << 20


def _si_tracked(path):
    if "__pycache__" in path or path.endswith(".pyc") or path.startswith(_SI_EXCLUDE):
        return None
    for r in _SI_ROOTS:
        if path.startswith(r):
            return r
    return None


def _source_diff(manifest_path):
    if not manifest_path or not os.path.exists(manifest_path):
        return {"error": f"manifest not found: {manifest_path}"}
    try:
        with open(manifest_path) as f:
            man = json.load(f)["roots"]
        prefix = "/"
        out = {"manifest": os.path.abspath(manifest_path), "roots": {}}
        n_mod = 0
        for r, files in man.items():
            base = prefix + r
            if not os.path.isdir(base):
                out["roots"][r] = {"error": "missing dir"}
                continue
            seen, modified, added = set(), [], []
            for dp, dns, fns in os.walk(base):
                dns[:] = [d for d in dns if d != "__pycache__"]
                for fn in fns:
                    full = os.path.join(dp, fn)
                    rel_root = full[len(prefix):]
                    if _si_tracked(rel_root) != r:
                        continue
                    rel = rel_root[len(r):]
                    seen.add(rel)
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    if st.st_size > _SI_BIG:
                        h = f"size:{st.st_size}"
                    else:
                        with open(full, "rb") as fh:
                            h = hashlib.sha256(fh.read()).hexdigest()
                    if rel not in files:
                        added.append({"file": rel, "sha256": h})
                    elif files[rel] != h:
                        modified.append({"file": rel, "sha256": h})
            removed = sorted(set(files) - seen)
            n_mod += len(modified) + len(added) + len(removed)
            out["roots"][r] = {"modified": modified, "added": added, "removed_count": len(removed),
                               "removed_sample": removed[:20]}
        out["pristine"] = n_mod == 0
        return out
    except Exception as e:
        return {"error": repr(e)}


def _file_sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return repr(e)


def _ckpt_quant_config(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        return {"quantization_config": cfg.get("quantization_config"),
                "architectures": cfg.get("architectures"), "torch_dtype": cfg.get("torch_dtype")}
    except Exception as e:
        return {"error": repr(e)}


def _env_snapshot():
    """Full os.environ minus secrets (names containing TOKEN/SECRET/PASSWORD/KEY/CREDENTIAL/AUTH)."""
    out = {}
    for k, v in sorted(os.environ.items()):
        if ENV_SKIP.search(k):
            out[k] = "<redacted>"
            continue
        out[k] = v
    return out


def _accelerator():
    if ARGS.accelerator:
        return ARGS.accelerator
    t = os.environ.get("TPU_ACCELERATOR_TYPE", "")
    if t.startswith("v6e"):
        return "v6e"
    if "7x" in t or t.startswith("v7") or t.startswith("tpu7"):
        return "tpu7x"
    return t or "unknown"


def make_prompts(tok, bs, seq_len, rng):
    vocab_limit = min(tok.vocab_size, 30000)
    pop = range(1000, vocab_limit)
    id_lists = [rng.choices(pop, k=seq_len) for _ in range(bs)]
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is not None and hasattr(bt, "decode_batch"):
        return bt.decode_batch(id_lists, skip_special_tokens=True)
    return [tok.decode(ids, skip_special_tokens=True) for ids in id_lists]


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from examples.fast_tokenizer import ClientTokenizer  # noqa: E402  (shared with the README usage)


CLIENT_TOK = None  # set in main()


def tokenize(tok, texts, seq_len, verify=False):
    if CLIENT_TOK is not None:
        batch_ids = CLIENT_TOK(texts, seq_len, verify=verify)
    else:
        batch_ids = tok(texts, padding="max_length", truncation=True, max_length=seq_len)["input_ids"]
    return [{"prompt_token_ids": ids} for ids in batch_ids]


def run_sweep(llm, sp, tok, shapes, iters, warmup, rng, tp_used):
    import gc
    rows = []
    for bs, seq_len in shapes:
        lens_ok = True
        t_shape0 = _utc()
        padded = 0
        if CLIENT_TOK is not None:  # untimed exactness check of the client tokenizer mode
            vrng = random.Random(7919 + bs + seq_len)  # own rng: timed inputs stay identical across modes
            for _ in range(3):
                tokenize(tok, make_prompts(tok, bs, seq_len, vrng), seq_len, verify=True)
        for _ in range(warmup):
            prompts = tokenize(tok, make_prompts(tok, bs, seq_len, rng), seq_len, verify=True)
            llm.generate(prompts, sp, use_tqdm=False)
        if ARGS.gc == "freeze":
            gc.collect()
            gc.freeze()
        elif ARGS.gc == "disable":
            gc.collect()
            gc.disable()
        e2e_lats, dev_lats, tok_lats = [], [], []
        for _ in range(iters):
            texts = make_prompts(tok, bs, seq_len, rng)
            t_start = time.perf_counter()
            prompts = tokenize(tok, texts, seq_len)
            t_dev = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            t_end = time.perf_counter()
            padded += CLIENT_TOK.last_padded if CLIENT_TOK is not None else 0
            e2e_lats.append((t_end - t_start) * 1000.0)
            dev_lats.append((t_end - t_dev) * 1000.0)
            tok_lats.append((t_dev - t_start) * 1000.0)
            if any(len(p["prompt_token_ids"]) != seq_len for p in prompts) or len(outs) != bs or any(
                    len(o.outputs[0].token_ids) != 1 for o in outs):
                lens_ok = False
        if ARGS.gc == "freeze":
            gc.unfreeze()
        elif ARGS.gc == "disable":
            gc.enable()
        t_shape1 = _utc()
        # standard client for reference (NOT part of E2E): plain HF tokenizer call on fresh texts
        hf_tok = []
        if CLIENT_TOK is not None and CLIENT_TOK.mode != "hf":
            hrng = random.Random(4242 + bs + seq_len)  # own rng: the timed inputs stay unchanged
            for _ in range(10):
                texts = make_prompts(tok, bs, seq_len, hrng)
                t0 = time.perf_counter()
                tok(texts, padding="max_length", truncation=True, max_length=seq_len)["input_ids"]
                hf_tok.append((time.perf_counter() - t0) * 1000.0)
        e2e_mean = statistics.mean(e2e_lats)
        dev_mean = statistics.mean(dev_lats)
        total_tokens = bs * seq_len
        row = {
            "batch_size": bs,
            "seq_len": seq_len,
            "total_tokens": total_tokens,
            "parallel_mode": f"tp{tp_used}",
            "request_latency_ms_p50": pctl(e2e_lats, 50),
            "request_latency_ms_p90": pctl(e2e_lats, 90),
            "request_latency_ms_p95": pctl(e2e_lats, 95),
            "request_latency_ms_p99": pctl(e2e_lats, 99),
            "request_latency_ms_mean": round(e2e_mean, 3),
            "device_latency_ms_p50": pctl(dev_lats, 50),
            "device_latency_ms_p90": pctl(dev_lats, 90),
            "device_latency_ms_p95": pctl(dev_lats, 95),
            "device_latency_ms_p99": pctl(dev_lats, 99),
            "device_latency_ms_mean": round(dev_mean, 3),
            "per_item_latency_ms_mean": round(e2e_mean / bs, 4),
            "device_per_item_latency_ms_mean": round(dev_mean / bs, 4),
            "throughput_items_per_s": round(bs / (e2e_mean / 1000.0), 3),
            "device_throughput_items_per_s": round(bs / (dev_mean / 1000.0), 3),
            "throughput_tokens_per_s": round(total_tokens / (e2e_mean / 1000.0), 1),
            "device_throughput_tokens_per_s": round(total_tokens / (dev_mean / 1000.0), 1),
            # additional timing and verification fields
            "tokenize_latency_ms_p50": pctl(tok_lats, 50),
            "tokenize_latency_ms_mean": round(statistics.mean(tok_lats), 3),
            "device_latency_ms_min": round(min(dev_lats), 3),
            "lengths_and_outputs_ok": lens_ok,
            "padded_sequences": padded,
            "t_start_utc": t_shape0,
            "t_end_utc": t_shape1,
            "hf_tokenizer_ms_p50": pctl(hf_tok, 50) if hf_tok else None,
            "hf_tokenizer_ms_samples": [round(x, 3) for x in hf_tok],
            # raw per-iteration samples (ms, timed order) so every statistic above can be recomputed
            "request_latency_ms_samples": [round(x, 3) for x in e2e_lats],
            "device_latency_ms_samples": [round(x, 3) for x in dev_lats],
            "tokenize_latency_ms_samples": [round(x, 3) for x in tok_lats],
        }
        rows.append(row)
        print(
            f"[bench] ({bs}, {seq_len}) [tp{tp_used}]: "
            f"e2e p50={row['request_latency_ms_p50']}ms p99={row['request_latency_ms_p99']}ms "
            f"| engine p50={row['device_latency_ms_p50']}ms p99={row['device_latency_ms_p99']}ms "
            f"min={row['device_latency_ms_min']}ms | tok p50={row['tokenize_latency_ms_p50']}ms "
            f"| tok/s={row['throughput_tokens_per_s']} ok={lens_ok} padded={padded} "
            f"| plain-HF tokenizer p50={row['hf_tokenizer_ms_p50']}ms (not in E2E)",
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------------------
# Correctness gate: dump / compare last-position top-k logprobs on fixed seeded inputs.
# --------------------------------------------------------------------------------------
def make_gate_inputs(tok, seed):
    rng = random.Random(seed)
    sets = []
    for bs, sl in GATE_SETS:
        texts = make_prompts(tok, bs, sl, rng)
        ids = tok(texts, padding="max_length", truncation=True, max_length=sl)["input_ids"]
        sets.append({"batch_size": bs, "seq_len": sl, "prompt_token_ids": [list(map(int, x)) for x in ids]})
    return sets


def run_gate(llm, sets, k):
    """Per gate set: one generate() with logprobs=k (top-k at the last position) and one with the
    exact timed SamplingParams (no logprobs) -> `sampled_nolp`, so logprob-free fast paths
    (e.g. TPU_GREEDY_TOP_TOKENS) are covered by the gate too."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, detokenize=False, logprobs=k)
    sp_nolp = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, detokenize=False)
    results = []
    for s in sets:
        prompts = [{"prompt_token_ids": ids} for ids in s["prompt_token_ids"]]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        outs_nolp = llm.generate(prompts, sp_nolp, use_tqdm=False)
        seqs = []
        for o, onl in zip(outs, outs_nolp):
            c = o.outputs[0]
            lp = c.logprobs[0]  # dict token_id -> Logprob(logprob, rank, decoded_token)
            items = sorted(lp.items(), key=lambda kv: (kv[1].rank if kv[1].rank is not None else 1 << 30,
                                                       -kv[1].logprob))
            top = [(int(t), float(v.logprob)) for t, v in items if v.rank is not None and v.rank <= k][:k]
            if len(top) < k:  # fall back to sorting by logprob
                top = sorted(((int(t), float(v.logprob)) for t, v in lp.items()), key=lambda x: -x[1])[:k]
            seqs.append({"sampled": int(c.token_ids[0]), "sampled_nolp": int(onl.outputs[0].token_ids[0]),
                         "top_ids": [t for t, _ in top], "top_logprobs": [round(v, 6) for _, v in top]})
        results.append({"batch_size": s["batch_size"], "seq_len": s["seq_len"], "seqs": seqs})
    return results


from gate import compare_gate  # noqa: E402  (pure-Python gate scoring, shared with score_gate.py)


RUN_STARTED_UTC = None


def _plugin_status():
    d = os.environ.get("MINICPM5_MK_STATUS_DIR")
    out = []
    if d and os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("minicpm5_mk_status_rank") and fn.endswith(".json"):
                try:
                    with open(os.path.join(d, fn)) as f:
                        out.append(json.load(f))
                except Exception as e:  # noqa: BLE001
                    out.append({"file": fn, "error": repr(e)})
    return out


def main():
    global RUN_STARTED_UTC
    RUN_STARTED_UTC = _utc()
    args = ARGS
    torch.set_num_threads(1)
    from vllm import LLM, SamplingParams

    model_dir = prepare_local_model_dir(args.model, sanitize=not args.no_sanitize_tokenizer_config,
                                        revision=args.revision)
    tok_dir = prepare_local_model_dir(args.tokenizer, not args.no_sanitize_tokenizer_config) \
        if args.tokenizer else model_dir
    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = 1

    engine_kwargs = dict(DEFAULT_ENGINE_KWARGS)
    engine_kwargs.update(json.loads(args.engine_kwargs))
    engine_kwargs.setdefault("tokenizer", tok_dir)
    if args.profile_shape:
        if not args.profile_dir:
            raise SystemExit("--profile-shape requires --profile-dir")
        pc = dict(engine_kwargs.get("profiler_config") or {})
        pc.setdefault("profiler", "torch")
        pc["torch_profiler_dir"] = os.path.abspath(args.profile_dir)
        engine_kwargs["profiler_config"] = pc
    tp_used = int(engine_kwargs.get("tensor_parallel_size", 1))

    manifest = args.source_manifest
    if manifest is None:
        for cand in ("/tmp/pristine_manifest.json",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "pristine_manifest.json")):
            if os.path.exists(cand):
                manifest = cand
                break
    t_sd = time.perf_counter()
    source_diff = _source_diff(manifest)
    print(f"[bench] source diff vs manifest ({time.perf_counter() - t_sd:.1f}s): pristine="
          f"{source_diff.get('pristine')} {source_diff.get('error', '')}", flush=True)
    config = {
        "argv": sys.argv,
        "harness_sha256": _file_sha256(os.path.abspath(__file__)),
        "model_arg": args.model,
        "model_revision_requested": args.revision,
        "model_revision": os.path.basename(os.path.realpath(model_dir)).split("@")[-1],
        "model_dir": model_dir,
        "hf_config": _hf_config_summary(model_dir),
        "host": _host_info(),
        "run_started_utc": RUN_STARTED_UTC,
        "tokenizer_dir": tok_dir,
        "checkpoint_config": _ckpt_quant_config(model_dir),
        "engine_kwargs": engine_kwargs,
        "env": _env_snapshot(),
        "env_overrides": args.env,
        "versions": _versions(),
        "plugin_git": _plugin_git_state(),
        "source_diff": source_diff,
        "profiler_enabled": bool(args.profile_shape),
        "hostname": socket.gethostname(),
        "notes": args.notes,
        "sampling_params": "SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, detokenize=False"
                           + (", skip_clone=True)" if args.skip_clone else ")"),
        "input_rng_seed": args.seed,
        "percentile_method": "linear interpolation between closest ranks (numpy default); p50 = median",
        "throughput_definition": "total_tokens / mean(request_latency); request = HF-tokenizer call + llm.generate",
        "tokenizer_mode": args.tokenizer_mode,
        "gc_policy": args.gc,
        "skip_clone": bool(args.skip_clone),
        "host_tuned": bool(args.tokenizer_mode != "hf" or args.gc != "default" or args.skip_clone),
    }
    global CLIENT_TOK
    if args.tokenizer_mode != "hf":
        CLIENT_TOK = ClientTokenizer(tok, args.tokenizer_mode)
        config["tokenizer_mode_effective"] = CLIENT_TOK.mode
    preloads = []
    for pf in args.preload:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ttvllm_preload_" + re.sub(r"\W", "_", os.path.basename(pf)), os.path.abspath(pf))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        preloads.append({"file": os.path.abspath(pf), "sha256": _file_sha256(os.path.abspath(pf)),
                         "applied": list(getattr(mod, "APPLIED", []))})
    config["preloads"] = preloads
    if args.profile_shape and parse_shapes(args.shapes):
        print("[bench] WARNING: profiler_config is active for the whole run; sweep numbers from this run "
              "are not comparable (use --shapes none for profiling runs).", flush=True)

    print(f"[bench] Loading {args.model} from {model_dir} engine_kwargs={json.dumps(engine_kwargs)}", flush=True)
    t0 = time.perf_counter()
    llm = LLM(model=model_dir, **engine_kwargs)
    load_s = time.perf_counter() - t0
    print(f"[bench] Loaded in {load_s:.2f}s", flush=True)

    sp = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, detokenize=False,
                        **({"skip_clone": True} if args.skip_clone else {}))
    rng = random.Random(args.seed)
    result = {
        "model": args.model,
        "engine": "vllm-torchtpu",
        "precision": args.precision.upper(),
        "quantization": args.quantization_label,
        "device": args.label,
        "label": args.label,
        "accelerator": _accelerator(),
        "chips": 1,
        "cores_utilized": tp_used,
        "tensor_parallel_size": tp_used,
        "prefix_caching": bool(engine_kwargs.get("enable_prefix_caching", False)),
        "seq_len_per_input": 8192,
        "token_budget_per_call": 8192,
        "iters_per_config": args.iters,
        "warmup_per_config": args.warmup,
        "model_load_s": round(load_s, 3),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "validation": {},
        "by_config": [],
    }

    def write_result():
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[bench] Wrote {args.out}", flush=True)

    # ---- correctness gate first (cheap; catches broken configs before a long sweep) ----
    # --compare-logprobs is repeatable: one gate report per ref (validation.logprob_gates); the
    # first one is also validation.logprob_gate. With N refs and --dump-logprobs P, the dump for
    # ref 0 goes to P and the dump for ref i>0 to P with "__<ref basename>" inserted before .json.
    refs = list(args.compare_logprobs or [])
    jobs = []  # (ref_path or None, ref dict or None, sets)
    extra_bs = [int(x) for x in str(args.gate_batch_shapes).split(",") if x.strip()]
    if refs:
        for rp in refs:
            with open(rp) as f:
                ref = json.load(f)
            sets = [{"batch_size": s["batch_size"], "seq_len": s["seq_len"],
                     "prompt_token_ids": s["prompt_token_ids"]} for s in ref["inputs"]]
            # batch coverage: the reference 4-prompt set first, then seeded filler prompts
            base = next((s for s in sets if s["batch_size"] == 4), None)
            if base is not None:
                for b in extra_bs:
                    frng = random.Random(args.gate_seed + 1000 + b)
                    filler = tok(make_prompts(tok, b - 4, base["seq_len"], frng), padding="max_length",
                                 truncation=True, max_length=base["seq_len"])["input_ids"]
                    sets.append({"batch_size": b, "seq_len": base["seq_len"], "embedded_ref_batch": 4,
                                 "prompt_token_ids": [list(x) for x in base["prompt_token_ids"]]
                                 + [list(map(int, x)) for x in filler]})
            jobs.append((rp, ref, sets))
    elif args.dump_logprobs:
        jobs.append((None, None, make_gate_inputs(tok, args.gate_seed)))
    for ji, (rp, ref, sets) in enumerate(jobs):
        t_g = time.perf_counter()
        cand = run_gate(llm, sets, args.logprobs_k)
        print(f"[bench] gate run {ji} took {time.perf_counter() - t_g:.1f}s", flush=True)
        if args.dump_logprobs:
            dpath = args.dump_logprobs
            if ji > 0:
                stem = os.path.splitext(os.path.basename(rp))[0]
                dpath = re.sub(r"(\.json)?$", f"__{stem}.json", args.dump_logprobs, count=1)
            dump = {"model": args.model, "label": args.label, "k": args.logprobs_k,
                    "gate_seed": args.gate_seed if ref is None else ref.get("gate_seed"),
                    "reference": os.path.abspath(rp) if rp else None,
                    "config": {"engine_kwargs": engine_kwargs, "env": config["env"]},
                    "inputs": sets, "results": cand}
            os.makedirs(os.path.dirname(os.path.abspath(dpath)) or ".", exist_ok=True)
            with open(dpath, "w") as f:
                json.dump(dump, f)
            print(f"[bench] Wrote logprob dump {dpath}", flush=True)
            result["validation"].setdefault("logprob_dumps", []).append(os.path.abspath(dpath))
            if ji == 0:
                result["validation"]["logprob_dump"] = os.path.abspath(dpath)
        if ref is not None:
            # score the reference shapes directly, and the embedded 4 prompts of each coverage batch
            ref_sets, cand_sets = list(ref["results"]), []
            ref4 = next((r for r in ref["results"] if r["batch_size"] == 4), None)
            cand4 = None
            for st, cs in zip(sets, cand):
                if "embedded_ref_batch" not in st:
                    cand_sets.append(cs)
                    if cs["batch_size"] == 4:
                        cand4 = cs
                elif ref4 is not None:
                    ref_sets.append({"batch_size": cs["batch_size"], "seq_len": cs["seq_len"], "seqs": ref4["seqs"]})
                    cand_sets.append({"batch_size": cs["batch_size"], "seq_len": cs["seq_len"], "seqs": cs["seqs"][:4]})
            rep = compare_gate(ref_sets, cand_sets)
            # batch invariance: same 4 prompts, batch 4 vs batch 8/16/32, candidate vs itself
            inv = {}
            for cs in cand_sets[len(ref["results"]):]:
                dmax = 0.0
                for a, b in zip(cand4["seqs"], cs["seqs"]):
                    bm = dict(zip(b["top_ids"], b["top_logprobs"]))
                    bf = min(b["top_logprobs"])
                    dmax = max([dmax] + [abs(v - bm.get(t, bf)) for t, v in zip(a["top_ids"], a["top_logprobs"])])
                inv[f"{cs['batch_size']}x{cs['seq_len']}"] = {
                    "same_top1": all(a["top_ids"][0] == b["top_ids"][0] for a, b in zip(cand4["seqs"], cs["seqs"])),
                    "max_abs_dlogprob_vs_batch4": round(dmax, 5)}
            rep["batch_invariance"] = inv
            rep["reference"] = os.path.abspath(rp)
            rep["reference_sha256"] = _file_sha256(os.path.abspath(rp))
            rep["reference_label"] = ref.get("label")
            result["validation"].setdefault("logprob_gates", []).append(rep)
            if ji == 0:
                result["validation"]["logprob_gate"] = rep
            print(f"[gate] ref {ji}: {os.path.basename(rp)}", flush=True)
            for s in rep["sets"]:
                print(f"[gate] {s['shape']} n={s['n']} top1={s['top1_agreement']:.4f} "
                      f"top1_nolp={s['top1_agreement_nolp']} tie_aware={s['top1_tie_aware']} "
                      f"tie_aware_nolp={s['top1_tie_aware_nolp']} mean|dlp|={s['mean_abs_dlogprob']} "
                      f"kl={s['kl_top20_mean']} miss={s['ref_top20_missing_frac']} "
                      f"cos_lp={s['cos_logprob_mean']:.6f} (min {s['cos_logprob_min']:.6f}) "
                      f"cos_p={s['cos_prob_mean']:.6f} overlap={s['topk_overlap_mean']:.3f} "
                      f"|dlp_top1|max={s['ref_top1_logprob_absdiff_max']}", flush=True)
            print(f"[gate] batch invariance vs batch 4: {json.dumps(rep['batch_invariance'])}", flush=True)
            print(f"[gate] OVERALL n={rep['n_seqs']} top1={rep['top1_agreement']:.4f} "
                  f"top1_nolp={rep['top1_agreement_nolp']} tie_aware={rep['top1_tie_aware']} "
                  f"tie_aware_nolp={rep['top1_tie_aware_nolp']} mean|dlp|={rep['mean_abs_dlogprob']} "
                  f"kl={rep['kl_top20_mean']} cos_logprob={rep['cos_logprob_mean']:.6f} "
                  f"(min {rep['cos_logprob_min']:.6f}) cos_prob={rep['cos_prob_mean']:.6f} "
                  f"gate_v2={rep['gate_v2']} exact_top1={rep['exact_match_top1_all']}", flush=True)
    if len(result["validation"].get("logprob_gates", [])) > 1:
        verdicts = [g["gate_v2"] for g in result["validation"]["logprob_gates"]]
        result["validation"]["gate_v2_all"] = ("FAIL" if "FAIL" in verdicts else
                                               "JUDGE" if "JUDGE" in verdicts else "PASS")
        print(f"[gate] ALL REFS gate_v2={result['validation']['gate_v2_all']} {verdicts}", flush=True)

    # ---- timing sweep ----
    shapes = parse_shapes(args.shapes)
    if shapes:
        result["by_config"] = run_sweep(llm, sp, tok, shapes, args.iters, args.warmup, rng, tp_used)
        result["validation"]["all_lengths_and_outputs_ok"] = all(
            r["lengths_and_outputs_ok"] for r in result["by_config"])
        if CLIENT_TOK is not None:
            result["validation"]["tokenizer_mode_verified_calls"] = CLIENT_TOK.verified
        result["validation"]["padded_sequences_total"] = sum(r.get("padded_sequences", 0) for r in result["by_config"])
        result["config"]["plugin_status"] = _plugin_status()
        result["run_finished_utc"] = _utc()
        write_result()

    # ---- profiling (one start/stop session per shape -> one plugins/profile/<ts>/ run each) ----
    if args.profile_shape:
        result["profile"] = []
        for bs, seq_len in parse_shapes(args.profile_shape):
            for _ in range(max(2, args.warmup)):
                llm.generate(tokenize(tok, make_prompts(tok, bs, seq_len, rng), seq_len), sp, use_tqdm=False)
            batches = [tokenize(tok, make_prompts(tok, bs, seq_len, rng), seq_len)
                       for _ in range(args.profile_steps)]
            before = set()
            prof_root = os.path.join(os.path.abspath(args.profile_dir), "plugins", "profile")
            if os.path.isdir(prof_root):
                before = set(os.listdir(prof_root))
            llm.start_profile()
            time.sleep(1.0)
            lats = []
            for prompts in batches:
                t = time.perf_counter()
                llm.generate(prompts, sp, use_tqdm=False)
                lats.append(round((time.perf_counter() - t) * 1000.0, 3))
                time.sleep(0.05)  # visual separation between calls in the trace
            llm.stop_profile()
            time.sleep(2.0)
            new = sorted(set(os.listdir(prof_root)) - before) if os.path.isdir(prof_root) else []
            print(f"[bench] profiled ({bs},{seq_len}) x{args.profile_steps}: engine ms per call {lats} "
                  f"-> {prof_root} new runs {new}", flush=True)
            result["profile"].append({"shape": [bs, seq_len], "steps": args.profile_steps, "engine_ms": lats,
                                      "dir": os.path.abspath(args.profile_dir), "runs": new})
            time.sleep(1.5)  # distinct <ts> run dirs (second resolution)
    if args.out and (not shapes or args.profile_shape):
        write_result()


if __name__ == "__main__":
    main()
