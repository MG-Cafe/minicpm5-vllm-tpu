#!/usr/bin/env python3
"""Minimal end-to-end vLLM inference example using MiniCPM5MegakernelForCausalLM on Cloud TPU."""

import argparse
import json
import os
import time

# Same settings as benchmark/run_all_tpu_benchmarks.sh (applied in main(), so importing this module for
# prepare_local_model_dir has no side effects).
BENCH_ENV = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "TPU_GREEDY_TOP_TOKENS": "1",
    "MINICPM5_MK_QUANT": "auto",
    "MINICPM5_MK_ATTN": "auto",
    "MINICPM5_MK_CAL_NSEQ": "16",
    "MINICPM5_MK_CAL_SEQ": "256",
    "MINICPM5_MK_FREE_STOCK": "1",
}


def detect_default_tp() -> int:
    accel = os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()
    if "7x" in accel or accel.startswith("v7") or accel.startswith("tpu7"):
        return 2
    if os.path.exists("/dev/vfio/1") or os.path.exists("/dev/accel1"):
        return 2
    return 1


# Hugging Face revisions every published number was measured with (pass revision=None for the latest).
PINNED_REVISIONS = {
    "openbmb/MiniCPM5-1B": "87179e5c1f455ef22e6223592d2d61351b525bfc",
    "openbmb/MiniCPM5-2B": "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
}


def prepare_local_model_dir(model_id: str, revision: str = "pinned", sanitize: bool = True) -> str:
    """Download the HF snapshot (pinned revision by default) and return a directory vLLM can load.

    The checkpoint's tokenizer_config.json carries an `extra_special_tokens` dict that this transformers /
    vLLM version rejects; instead of editing the shared HF cache, a private directory is created with
    symlinks to the snapshot files and a sanitized tokenizer_config.json.
    """
    if os.path.isdir(model_id):
        snap_dir = model_id
    else:
        from huggingface_hub import snapshot_download

        rev = PINNED_REVISIONS.get(model_id) if revision == "pinned" else revision
        snap_dir = snapshot_download(repo_id=model_id, revision=rev)
    tok_cfg_path = os.path.join(snap_dir, "tokenizer_config.json")
    if not sanitize or not os.path.exists(tok_cfg_path):
        return snap_dir
    with open(tok_cfg_path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg.get("extra_special_tokens"), dict):
        return snap_dir
    cfg.pop("extra_special_tokens", None)
    snap_real = os.path.realpath(snap_dir)
    repo_tag = os.path.basename(os.path.dirname(os.path.dirname(snap_real)))  # models--org--name
    root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "minicpm5_mk_sanitized")
    out = os.path.join(root, f"{repo_tag}@{os.path.basename(snap_real)}")
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(snap_dir):
        dst = os.path.join(out, name)
        if name == "tokenizer_config.json" or os.path.lexists(dst):
            continue
        os.symlink(os.path.realpath(os.path.join(snap_dir, name)), dst)
    tmp = os.path.join(out, "tokenizer_config.json.tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, os.path.join(out, "tokenizer_config.json"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openbmb/MiniCPM5-1B", help="Model ID (openbmb/MiniCPM5-1B or openbmb/MiniCPM5-2B)")
    parser.add_argument("--tp", type=int, default=detect_default_tp(), help="Tensor parallel size (1 for TPU v6e-1, 2 for TPU7x-1)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of concurrent sequences (1, 4, 8, 16, or 32)")
    parser.add_argument("--seq-len", type=int, default=8192, help="Tokens per sequence (8192 max tokens per input)")
    args = parser.parse_args()
    for k, v in BENCH_ENV.items():
        os.environ.setdefault(k, v)

    from vllm import LLM, SamplingParams

    model_dir = prepare_local_model_dir(args.model)
    llm = LLM(
        model=model_dir,
        tokenizer=model_dir,
        hf_overrides={"architectures": ["MiniCPM5MegakernelForCausalLM"]},
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=8256,
        max_num_batched_tokens=262144,
        max_num_seqs=32,
        block_size=256,
        gpu_memory_utilization=0.5,
        compilation_config={"compile_sizes": [8192, 32768, 65536, 131072, 262144]},
    )

    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=0.0,
        detokenize=False,
        skip_clone=True,
    )

    from transformers import AutoTokenizer

    bos = AutoTokenizer.from_pretrained(model_dir).bos_token_id  # 0 (<s>) for MiniCPM5
    prompts = [{"prompt_token_ids": [bos] + [1024 + (i + j) % 8000 for j in range(args.seq_len - 1)]}
               for i in range(args.batch_size)]

    # Warmup (triggers one-time XLA/Pallas compilation for this shape)
    llm.generate(prompts, sampling_params, use_tqdm=False)

    # Timed single-forward-pass call
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    total_tokens = args.batch_size * args.seq_len

    print(
        f"[{args.model}] shape=({args.batch_size}, {args.seq_len}) tp={args.tp} | "
        f"engine_latency={elapsed_ms:.2f} ms | throughput={total_tokens / (elapsed_ms / 1000.0):,.1f} tok/s | "
        f"first_sampled_token_id={outputs[0].outputs[0].token_ids[0]}"
    )


if __name__ == "__main__":
    main()
