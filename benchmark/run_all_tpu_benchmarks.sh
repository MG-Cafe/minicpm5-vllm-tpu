#!/bin/bash
# End-to-end benchmark runner for MiniCPM5-1B and MiniCPM5-2B on a 1-chip Cloud TPU (v6e-1 or TPU7x-1).
# Every run is gated for accuracy against two references on the same 8,192-token prompts (5 random-token
# prompts: 1 x 8192 and 4 x 8192; the 4-prompt set is also run inside batches of 8, 16 and 32):
#   * benchmark/ref/fp32_8192_<m>_golden_logprobs.json : full-FP32 HF checkpoint (benchmark/f32_golden.py)
#   * benchmark/ref/bf16_8192_<m>_ref_logprobs.json    : this plugin's BF16 path (MINICPM5_MK_QUANT=bf16,
#                                                         SuperAttention v1 settings), committed reference
# Gate dumps of every run are written next to the results (results/gate/) so they can be re-scored.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME="${HF_HOME:-/dev/shm/hf_cache}"
export PYTHONPATH="${REPO_ROOT}/plugin:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Single-chip hosts only: TPU v6e-1 (TPU_ACCELERATOR_TYPE=v6e-1) or one TPU7x chip (tpu7x-2 = 2 TensorCores).
ACCEL="${ACCEL:-auto}"
if [ "${ACCEL}" = "auto" ]; then
  case "${TPU_ACCELERATOR_TYPE:-}" in
    v6e-1) ACCEL="v6e" ;;
    tpu7x-2) ACCEL="tpu7x" ;;
    "")
      echo "WARNING: TPU_ACCELERATOR_TYPE is unset; guessing from /dev (set ACCEL=v6e or ACCEL=tpu7x to be explicit)" >&2
      if [ -e /dev/vfio/1 ] || [ -e /dev/accel1 ]; then ACCEL="tpu7x"; else ACCEL="v6e"; fi ;;
    *) echo "ERROR: TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE} is not a single-chip v6e-1 / tpu7x-2 host" >&2; exit 1 ;;
  esac
fi

if [ "${ACCEL}" = "tpu7x" ]; then
  TP=2
  EK='{"hf_overrides":{"architectures":["MiniCPM5MegakernelForCausalLM"]},"tensor_parallel_size":2,"block_size":256,"max_num_batched_tokens":262144,"compilation_config":{"compile_sizes":[8192,32768,65536,131072,262144]},"gpu_memory_utilization":0.5}'
else
  TP=1
  EK='{"hf_overrides":{"architectures":["MiniCPM5MegakernelForCausalLM"]},"tensor_parallel_size":1,"block_size":256,"max_num_batched_tokens":262144,"compilation_config":{"compile_sizes":[8192,32768,65536,131072,262144]},"gpu_memory_utilization":0.5}'
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results}"
mkdir -p "${OUT_DIR}/gate" "${REPO_ROOT}/benchmark/ref"

MK_SHA=$(cd "${REPO_ROOT}/plugin" && find . -type f \( -name '*.py' -o -name '*.txt' -o -name 'METADATA' \) | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)

# MODELS="1b 2b" (default) or a subset, e.g. MODELS=2b
for SHORT_SEL in ${MODELS:-1b 2b}; do
  MODEL_TAG="${SHORT_SEL}:openbmb/MiniCPM5-${SHORT_SEL^^}"
  SHORT="${MODEL_TAG%%:*}"
  MODEL="${MODEL_TAG#*:}"
  OUT_JSON="${OUT_DIR}/${ACCEL}_minicpm5_${SHORT}.json"
  REF_JSON="${REPO_ROOT}/benchmark/ref/bf16_8192_${SHORT}_ref_logprobs.json"
  GOLDEN_JSON="${REPO_ROOT}/benchmark/ref/fp32_8192_${SHORT}_golden_logprobs.json"
  if [ ! -f "${REF_JSON}" ]; then
    echo "Generating BF16 8,192-token reference logprobs -> ${REF_JSON}"
    python3 "${REPO_ROOT}/benchmark/run_benchmark.py" \
      --model "${MODEL}" \
      --accelerator "${ACCEL}" \
      --label "bf16-8192-${SHORT}-ref" \
      --precision "bf16" \
      --engine-kwargs "${EK}" \
      --shapes "none" \
      --env "MINICPM5_MK_QUANT=bf16" \
      --env "MINICPM5_MK_ATTN=super:256,512,pipe" \
      --env "MINICPM5_MK_FREE_STOCK=1" \
      --env "VLLM_ENABLE_V1_MULTIPROCESSING=0" \
      --dump-logprobs "${REF_JSON}"
  fi
  if [ ! -f "${GOLDEN_JSON}" ]; then
    echo "Generating FP32 golden logprobs on the same prompts -> ${GOLDEN_JSON}"
    python3 "${REPO_ROOT}/benchmark/f32_golden.py" --model "${MODEL}" --inputs "${REF_JSON}" --out "${GOLDEN_JSON}"
  fi
  echo "================================================================================"
  echo "Running ${MODEL} on ${ACCEL} (TP=${TP}, sha=${MK_SHA}) -> ${OUT_JSON}"
  echo "================================================================================"
  python3 "${REPO_ROOT}/benchmark/run_benchmark.py" \
    --model "${MODEL}" \
    --revision pinned \
    --accelerator "${ACCEL}" \
    --label "${ACCEL}-minicpm5-${SHORT}-supermk-v2" \
    --precision "int8d" \
    --quantization-label "megakernel:auto" \
    --engine-kwargs "${EK}" \
    --notes "minicpm5_mk plugin tree sha256=${MK_SHA}; quant=auto; opt=O3; SuperAttention v2 + fused W8A8 QKV/MLP Pallas kernels (seq_len=8192 per input)" \
    --shapes "all" \
    --iters 50 \
    --warmup 5 \
    --env "MINICPM5_MK_QUANT=auto" \
    --env "MINICPM5_MK_ATTN=auto" \
    --env "MINICPM5_MK_FREE_STOCK=1" \
    --env "MINICPM5_MK_LOG_EVERY=0" \
    --env "VLLM_ENABLE_V1_MULTIPROCESSING=0" \
    --env "TPU_GREEDY_TOP_TOKENS=1" \
    --env "MINICPM5_MK_CAL_NSEQ=16" \
    --env "MINICPM5_MK_CAL_SEQ=256" \
    --tokenizer-mode parallel \
    --gc freeze \
    --skip-clone \
    --compare-logprobs "${GOLDEN_JSON}" \
    --compare-logprobs "${REF_JSON}" \
    --dump-logprobs "${OUT_DIR}/gate/${ACCEL}_minicpm5_${SHORT}_gate.json" \
    --out "${OUT_JSON}"
done

VERIFY=()
for SHORT_SEL in ${MODELS:-1b 2b}; do VERIFY+=("${OUT_DIR}/${ACCEL}_minicpm5_${SHORT_SEL}.json"); done
python3 "${REPO_ROOT}/benchmark/verify_results.py" "${VERIFY[@]}"
