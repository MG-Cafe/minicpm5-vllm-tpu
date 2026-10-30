# MiniCPM5MegakernelForCausalLM: a vLLM LlamaForCausalLM whose prefill-only scoring steps
# run the whole MiniCPM5 forward as ONE JAX/Pallas program through torch_tpu's `jax_op`
# bridge; every other step (decode, chunked/partial prefill, unequal lengths, warmup dummy
# runs) runs the stock vLLM-TorchTPU Llama path, so generation stays correct.
"""vLLM model: MiniCPM5 (Llama) with a JAX/Pallas megakernel fast path for max_tokens=1."""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Iterable

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaForCausalLM

logger = init_logger("vllm.minicpm5_mk")  # under "vllm." so vLLM's log handler/level applies

from minicpm5_mk import sampling_fast_ok  # noqa: E402

# Env knobs (all optional).
ENV_ENABLE = "MINICPM5_MK_ENABLE"            # "0" disables the fast path (pure stock model)
ENV_QUANT = "MINICPM5_MK_QUANT"              # auto (W8A8 int8d, per chip/model) | bf16 | int8 | int8d | fp8 | fp8d
ENV_ATTN = "MINICPM5_MK_ATTN"                # auto (SuperAttention v2 for S>=2048, S%512==0) | v2:bq,mc | super:... | splash | xla
ENV_REQUIRE_MT1 = "MINICPM5_MK_REQUIRE_MAX_TOKENS_1"  # "1" (default): only max_tokens==1 requests
ENV_PRECOMPILE = "MINICPM5_MK_PRECOMPILE"    # e.g. "1x8192,4x8192,8x8192,16x8192,32x8192"
ENV_FREE_STOCK = "MINICPM5_MK_FREE_STOCK"    # "1": drop stock decoder-layer weights (no fallback)
ENV_LOG_EVERY = "MINICPM5_MK_LOG_EVERY"      # log path counters every N forwards (0 = off)
ENV_MC_MODES = "MINICPM5_MK_MULTICORE_MODES"  # TP>1 engines: "dp,sp,tp" (default) or any subset
ENV_STATUS_DIR = "MINICPM5_MK_STATUS_DIR"    # optional: write run status JSON (resolved kernels, shapes used) here

_WEIGHT_SUFFIXES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
_WEIGHT_ORDER = ("w_qkv", "w_o", "w_gu", "w_down", "s_qkv", "s_o", "s_gu", "s_down",
                 "a_qkv", "a_o", "a_gu", "a_down", "wl_qkv", "wl_o", "wl_gu", "wl_down")


class _StepState:
    """Host-side batch info for the forward that follows the current _prepare_inputs."""

    batch: tuple[int, int] | None = None
    reason: str = "init"
    last_was_mk: bool = False
    tokens_staged: bool = False


_STATE = _StepState()
_OPS: dict = {}


def _get_op():
    """Single-core whole-model op."""
    op = _OPS.get("forward")
    if op is None:
        from torch_tpu._internal import pallas

        from minicpm5_mk import mk_jax

        op = _OPS["forward"] = pallas.jax_op("minicpm5_mk::forward", mk_jax.mk_forward)
    return op


def _get_multicore_op(mode: str, quant: str):
    """Whole-model op over the vLLM TP world (both TensorCores of a TPU7x chip)."""
    key = f"forward_{mode}_{quant}"
    op = _OPS.get(key)
    if op is None:
        from torch_tpu._internal import pallas

        from minicpm5_mk import mk_jax

        mesh = pallas.get_pallas_mesh(("x",))
        fn, in_specs = mk_jax.make_multicore_forward(mesh, "x", mode, quant)
        n_static = 8  # bs, seq, quant, n_q, n_kv, head_dim, eps, attn (after 21 tensors)
        qtag = quant.replace(":", "_").replace("|", "_").replace("+", "_")
        op = _OPS[key] = pallas.jax_op(
            f"minicpm5_mk::{fn.__name__}_{qtag}", fn, mesh=mesh,
            input_partition_specs=tuple(in_specs) + (None,) * n_static,
            output_partition_specs=None)
    return op


def _tp_shard(w: dict, cfg: dict, rank: int, tp: int, quant: str) -> dict:
    """Rank-major head / intermediate shard of the stacked megakernel weights (CPU)."""
    hd, n_q, n_kv, inter = cfg["head_dim"], cfg["n_q"], cfg["n_kv"], cfg["inter"]
    qd, kd = n_q * hd, n_kv * hd
    qs, ks, is_ = qd // tp, kd // tp, inter // tp

    def cols_qkv(t):
        return torch.cat([t[..., rank * qs:(rank + 1) * qs],
                          t[..., qd + rank * ks:qd + (rank + 1) * ks],
                          t[..., qd + kd + rank * ks:qd + kd + (rank + 1) * ks]], dim=-1).contiguous()

    out = {
        "w_qkv": cols_qkv(w["w_qkv"]),
        "w_o": w["w_o"][:, rank * qs:(rank + 1) * qs, :].contiguous(),
        "w_gu": w["w_gu"][..., rank * is_:(rank + 1) * is_].contiguous(),
        "w_down": w["w_down"][:, rank * is_:(rank + 1) * is_, :].contiguous(),
    }
    from minicpm5_mk.mk_jax import parse_quant

    qm = parse_quant(quant)
    for k in ("s_qkv", "s_o", "s_gu", "s_down", "a_qkv", "a_o", "a_gu", "a_down"):
        out[k] = w[k]  # replicated, or (L, 1) dummies for bf16 linears
    if w["wl_qkv"].numel() > 1:  # bf16 last layer (no leading L dim)
        out["wl_qkv"] = cols_qkv(w["wl_qkv"])
        out["wl_o"] = w["wl_o"][rank * qs:(rank + 1) * qs, :].contiguous()
        out["wl_gu"] = w["wl_gu"][..., rank * is_:(rank + 1) * is_].contiguous()
        out["wl_down"] = w["wl_down"][rank * is_:(rank + 1) * is_, :].contiguous()
    else:
        for k in ("wl_qkv", "wl_o", "wl_gu", "wl_down"):
            out[k] = w[k]
    if quant != "bf16":
        if qm["qkv"] != "bf16":
            out["s_qkv"] = cols_qkv(w["s_qkv"])
        if qm["o"] != "bf16":
            out["a_o"] = w["a_o"][:, rank * qs:(rank + 1) * qs].contiguous()
        if qm["gu"] != "bf16":
            out["s_gu"] = w["s_gu"][..., rank * is_:(rank + 1) * is_].contiguous()
        if qm["down"] != "bf16":
            out["a_down"] = w["a_down"][:, rank * is_:(rank + 1) * is_].contiguous()
    return out


def _mk_seq_ok(seq: int, max_seq: int) -> bool:
    if seq <= 0 or seq > max_seq:
        return False
    if seq <= 1024:
        return seq % 128 == 0
    return seq % 1024 == 0


def _eligible_batch(runner, start_index: int, num_reqs: int, require_mt1: bool):
    """(num_reqs, seq) iff every request of this chunk is a full, fresh prefill of the same
    length (and max_tokens == 1 when required); otherwise None."""
    qsl = runner.query_start_loc_np
    seq = int(qsl[1] - qsl[0])
    if seq <= 0 or int(qsl[num_reqs]) != seq * num_reqs:
        _STATE.reason = "ragged"
        return None
    if num_reqs > 1 and not (np.diff(qsl[: num_reqs + 1]) == seq).all():
        _STATE.reason = "ragged"
        return None
    ib = runner.input_batch
    sl = slice(start_index, start_index + num_reqs)
    if ib.num_computed_tokens_cpu[sl].any():
        _STATE.reason = "continuation"
        return None
    if not (ib.num_prompt_tokens[sl] == seq).all():
        _STATE.reason = "partial_prefill"
        return None
    reqs = runner.requests
    npl = getattr(runner, "num_prompt_logprobs", {}) or {}
    for rid in ib.req_ids[start_index:start_index + num_reqs]:
        sp = getattr(reqs.get(rid), "sampling_params", None)
        if require_mt1 and (sp is None or sp.max_tokens != 1):
            _STATE.reason = "max_tokens"
            return None
        if not sampling_fast_ok(sp) or rid in npl:
            _STATE.reason = "unsupported_sampling_feature"
            return None
    _STATE.reason = "ok"
    return (num_reqs, seq)


def _precheck_fast_batch(runner, scheduler_output, start_index: int, require_mt1: bool):
    """Fast pre-check before orig_prepare: returns (num_reqs, seq) if the entire scheduled
    batch is a single fresh uniform prefill chunk that already has a cached _prepare_inputs plan."""
    if start_index != 0:
        return None
    cache = getattr(runner, "_mk_prep_cache", None)
    if not cache:
        return None
    ib = runner.input_batch
    num_reqs = ib.num_reqs
    if num_reqs <= 0:
        return None
    seq = int(ib.num_prompt_tokens[0])
    key = (num_reqs, seq)
    if key not in cache:
        return None
    if scheduler_output.total_num_scheduled_tokens != num_reqs * seq:
        return None
    if ib.num_computed_tokens_cpu[:num_reqs].any():
        return None
    if num_reqs > 1 and not (ib.num_prompt_tokens[:num_reqs] == seq).all():
        return None
    npl = getattr(runner, "num_prompt_logprobs", {}) or {}
    for rid in ib.req_ids[:num_reqs]:
        if scheduler_output.num_scheduled_tokens.get(rid) != seq:
            return None
        sp = getattr(runner.requests.get(rid), "sampling_params", None)
        if require_mt1 and (sp is None or sp.max_tokens != 1):
            return None
        if not sampling_fast_ok(sp) or rid in npl:
            return None
    return key


def _install_runner_hook(require_mt1: bool) -> None:
    from vllm_torchtpu.runner import tpu_runner

    try:
        from vllm_torchtpu.layers.core.sequence_layout import AllSequenceLayoutPlanner
    except ImportError:
        try:
            from vllm_torchtpu.layers.common.sequence_layout import AllSequenceLayoutPlanner
        except ImportError:
            AllSequenceLayoutPlanner = None
    cls = tpu_runner.TPUModelRunner
    if getattr(cls, "_minicpm5_mk_hooked", False):
        return
    orig_prepare = cls._prepare_inputs
    orig_dummy = cls._dummy_run
    orig_update_states = cls._update_states
    orig_sample_tokens = cls.sample_tokens

    if AllSequenceLayoutPlanner is not None and not getattr(AllSequenceLayoutPlanner, "_minicpm5_mk_hooked", False):
        orig_mslhs = AllSequenceLayoutPlanner.maybe_select_logits_hidden_states

        @functools.wraps(orig_mslhs)
        def _maybe_select_logits_hidden_states(self, hidden_states, plan, logits_indices):
            if _STATE.last_was_mk and hidden_states.shape[0] == logits_indices.shape[0]:
                return hidden_states
            return orig_mslhs(self, hidden_states, plan, logits_indices)

        AllSequenceLayoutPlanner.uses_selected_logits_hidden_states = property(lambda self: True)
        AllSequenceLayoutPlanner.maybe_select_logits_hidden_states = _maybe_select_logits_hidden_states
        AllSequenceLayoutPlanner._minicpm5_mk_hooked = True

    @functools.wraps(orig_update_states)
    def _update_states(self, scheduler_output, *args, **kwargs):
        _STATE.tokens_staged = False
        try:
            cache = getattr(self, "_mk_prep_cache", None)
            new_reqs = scheduler_output.scheduled_new_reqs
            if (
                cache
                and new_reqs
                and not scheduler_output.scheduled_cached_reqs.req_ids
                and not scheduler_output.new_block_ids_to_zero
                and not scheduler_output.kv_cache_block_copies
                and not scheduler_output.scheduled_spec_decode_tokens
                and not self.lora_config
                and not self.speculative_config
                and not self.is_pooling_model
                and not self.uses_mrope
            ):
                num_reqs = len(new_reqs)
                first_toks = new_reqs[0].prompt_token_ids
                seq = len(first_toks) if first_toks is not None else 0
                key = (num_reqs, seq)
                if key in cache and scheduler_output.total_num_scheduled_tokens == num_reqs * seq:
                    ok = True
                    for r in new_reqs:
                        pt = r.prompt_token_ids
                        sp = r.sampling_params
                        if (
                            pt is None
                            or len(pt) != seq
                            or r.num_computed_tokens != 0
                            or r.prompt_embeds is not None
                            or r.mm_features
                            or sp is None
                            or (require_mt1 and sp.max_tokens != 1)
                            or not sampling_fast_ok(sp)
                        ):
                            ok = False
                            break
                    if ok:
                        import array
                        from vllm.sampling_params import SamplingType
                        from vllm.v1.worker.gpu_input_batch import CachedRequestState

                        ib = self.input_batch
                        for req_id in scheduler_output.finished_req_ids:
                            self.requests.pop(req_id, None)
                            self.num_prompt_logprobs.pop(req_id, None)
                            ib.remove_request(req_id)
                        for req_id in list(ib.req_id_to_index.keys()):
                            ib.remove_request(req_id)
                        ib.condense()
                        self._wait_input_staging_fence()
                        inp_np = self.input_ids_cpu.numpy()
                        for i, r in enumerate(new_reqs):
                            rid = r.req_id
                            sp = r.sampling_params
                            gen = None
                            if sp.sampling_type == SamplingType.RANDOM_SEED:
                                gen = torch.Generator(device=self.device)
                                gen.manual_seed(sp.seed)
                            req_state = CachedRequestState(
                                req_id=rid,
                                prompt_token_ids=r.prompt_token_ids,
                                prompt_embeds=None,
                                prompt_is_token_ids=None,
                                mm_features=None,
                                sampling_params=sp,
                                pooling_params=None,
                                generator=gen,
                                block_ids=r.block_ids,
                                num_computed_tokens=0,
                                output_token_ids=[],
                                lora_request=None,
                            )
                            self.requests[rid] = req_state
                            req_idx = ib._register_add_request(req_state)
                            if req_idx == len(ib._req_ids):
                                ib._req_ids.append(rid)
                                ib.req_output_token_ids.append(req_state.output_token_ids)
                                ib.spec_token_ids.append([])
                            else:
                                ib._req_ids[req_idx] = rid
                                ib.req_output_token_ids[req_idx] = req_state.output_token_ids
                                ib.spec_token_ids[req_idx].clear()
                            ib.req_id_to_index[rid] = req_idx
                            ib.num_prompt_tokens[req_idx] = seq
                            ib.num_tokens_no_spec[req_idx] = seq
                            ib.num_computed_tokens_cpu[req_idx] = 0
                            ib.num_accepted_tokens_cpu[req_idx] = 1
                            ib.request_lora_mapping[req_idx] = 0
                            inp_np[req_idx * seq : (req_idx + 1) * seq] = np.frombuffer(
                                array.array("i", r.prompt_token_ids), dtype=np.int32
                            )
                            if sp.sampling_type == SamplingType.GREEDY:
                                ib.temperature_cpu[req_idx] = 0.0
                                ib.greedy_reqs.add(rid)
                            else:
                                ib.temperature_cpu[req_idx] = sp.temperature
                                ib.random_reqs.add(rid)
                            ib.top_p_cpu[req_idx] = sp.top_p
                            if sp.top_p < 1:
                                ib.top_p_reqs.add(rid)
                            top_k = sp.top_k
                            if 0 < top_k < ib.vocab_size:
                                ib.top_k_reqs.add(rid)
                            else:
                                top_k = ib.vocab_size
                            ib.top_k_cpu[req_idx] = top_k
                            if gen is not None:
                                ib.generators[req_idx] = gen
                            if sp.logprobs is not None:
                                ib.num_logprobs[rid] = (
                                    ib.vocab_size if sp.logprobs == -1 else sp.logprobs
                                )
                            if sp.logprob_token_ids is not None:
                                ib.logprob_token_ids[rid] = sp.logprob_token_ids
                        ib.refresh_metadata()
                        _STATE.tokens_staged = True
                        return None
        except Exception as e:
            logger.warning("minicpm5_mk: fast _update_states fallback: %r", e)
        return orig_update_states(self, scheduler_output, *args, **kwargs)

    @functools.wraps(orig_prepare)
    def _prepare_inputs(self, scheduler_output, start_index, *args, **kwargs):
        _STATE.batch = None
        _STATE.last_was_mk = False
        try:
            fast_key = _precheck_fast_batch(self, scheduler_output, start_index, require_mt1)
            if fast_key is not None:
                num_reqs, seq = fast_key
                out, layout_plan, pos_ids, qsl = self._mk_prep_cache[fast_key]
                tot = num_reqs * seq
                if not _STATE.tokens_staged:
                    self._wait_input_staging_fence()
                    ib = self.input_batch
                    np.copyto(
                        self.input_ids_cpu.numpy()[:tot].reshape(num_reqs, seq),
                        ib.token_ids_cpu[:num_reqs, :seq],
                    )
                _STATE.tokens_staged = False
                pad_tot = layout_plan.global_padded_num_tokens
                if pad_tot > tot:
                    self.input_ids_cpu[tot:pad_tot] = 0
                self.input_ids = self.input_ids_cpu[layout_plan.token_slice].to(
                    self.device, non_blocking=True
                )
                self.position_ids = pos_ids
                self._last_sequence_layout_plan = layout_plan
                self.query_start_loc_np[: num_reqs + 1] = qsl
                self.query_start_loc_np[num_reqs + 1 :] = tot
                self.seq_lens_np[:num_reqs] = seq
                _STATE.batch = fast_key
                _STATE.reason = "ok"
                return out
        except Exception as e:
            logger.warning("minicpm5_mk: fast _prepare_inputs fallback: %r", e)

        out = orig_prepare(self, scheduler_output, start_index, *args, **kwargs)
        _STATE.batch = None
        try:
            batch = _eligible_batch(self, start_index, int(out[3]), require_mt1)
            _STATE.batch = batch
            if (
                batch is not None
                and start_index == 0
                and int(out[4]) == batch[0]
                and not self.lora_config
                and not self.speculative_config
                and not self.uses_mrope
                and not getattr(self, "_has_mamba_state", False)
            ):
                cache = getattr(self, "_mk_prep_cache", None)
                if cache is None:
                    cache = self._mk_prep_cache = {}
                if batch not in cache and _mk_seq_ok(batch[1], self.max_model_len):
                    num_reqs, _ = batch
                    cache[batch] = (
                        out,
                        self._last_sequence_layout_plan,
                        self.position_ids,
                        self.query_start_loc_np[: num_reqs + 1].copy(),
                    )
        except Exception as e:  # never break serving because of the fast-path check
            _STATE.reason = f"check_error:{e!r}"
            logger.warning("minicpm5_mk: eligibility check failed: %r", e)
        return out

    @functools.wraps(orig_sample_tokens)
    def _sample_tokens(self, grammar_output, *args, **kwargs):
        was_mk = _STATE.last_was_mk
        _STATE.last_was_mk = False
        res = orig_sample_tokens(self, grammar_output, *args, **kwargs)
        if was_mk and self.scheduler_config.async_scheduling and self._pre_async_results is not None:
            self._pre_async_results.req_id_to_index_copy.clear()
            self._pre_async_results.request_seq_lens.clear()
        return res

    @functools.wraps(orig_dummy)
    def _dummy_run(self, num_tokens, *args, **kwargs):
        _STATE.batch = None
        _STATE.last_was_mk = False
        _STATE.reason = "dummy"
        return orig_dummy(self, min(int(num_tokens), 8192), *args, **kwargs)

    orig_profile_run = getattr(cls, "profile_run", None)
    if orig_profile_run is not None:
        @functools.wraps(orig_profile_run)
        def _profile_run(self, num_tokens, *args, **kwargs):
            saved_pads = list(self.num_tokens_paddings)
            small_pads = [p for p in saved_pads if p <= 8192] or [min(saved_pads)]
            self.num_tokens_paddings = small_pads
            try:
                return orig_profile_run(self, min(int(num_tokens), small_pads[-1]), *args, **kwargs)
            finally:
                self.num_tokens_paddings = saved_pads

        cls.profile_run = _profile_run

    orig_precompile_bb = getattr(cls, "_precompile_backbone", None)
    if orig_precompile_bb is not None:
        @functools.wraps(orig_precompile_bb)
        def _precompile_backbone(self, *args, **kwargs):
            saved_pads = list(self.num_tokens_paddings)
            small_pads = [p for p in saved_pads if p <= 8192] or [min(saved_pads)]
            self.num_tokens_paddings = small_pads
            try:
                return orig_precompile_bb(self, *args, **kwargs)
            finally:
                self.num_tokens_paddings = saved_pads

        cls._precompile_backbone = _precompile_backbone

    orig_capture_model = getattr(cls, "capture_model", None)
    if orig_capture_model is not None:
        @functools.wraps(orig_capture_model)
        def _capture_model(self, *args, **kwargs):
            saved_pads = list(self.num_tokens_paddings)
            small_pads = [p for p in saved_pads if p <= 8192] or [min(saved_pads)]
            self.num_tokens_paddings = small_pads
            try:
                return orig_capture_model(self, *args, **kwargs)
            finally:
                self.num_tokens_paddings = saved_pads

        cls.capture_model = _capture_model

    orig_check_sched = getattr(cls, "_check_attention_schedule", None)
    if orig_check_sched is not None:
        @functools.wraps(orig_check_sched)
        def _check_attention_schedule(self, num_reqs, q_lens, num_decode, use_max_model_len=True):
            if len(q_lens) > 0 and _mk_seq_ok(int(q_lens[0]), self.max_model_len) and (q_lens == q_lens[0]).all():
                return
            return orig_check_sched(self, num_reqs, q_lens, num_decode, use_max_model_len)

        cls._check_attention_schedule = _check_attention_schedule

    cls._update_states = _update_states
    cls._prepare_inputs = _prepare_inputs
    cls.sample_tokens = _sample_tokens
    cls._dummy_run = _dummy_run
    cls._minicpm5_mk_hooked = True

    try:
        from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

        if not getattr(MinTokensLogitsProcessor, "_minicpm5_mk_hooked", False):
            orig_dev_tensor = MinTokensLogitsProcessor._device_tensor

            @functools.wraps(orig_dev_tensor)
            def _cached_device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
                cache = getattr(self, "_mk_dev_tensor_cache", None)
                if cache is None:
                    cache = self._mk_dev_tensor_cache = {}
                key = (tuple(data), dtype)
                t = cache.get(key)
                if t is None:
                    t = orig_dev_tensor(self, data, dtype)
                    if len(cache) < 64:
                        cache[key] = t
                return t

            MinTokensLogitsProcessor._device_tensor = _cached_device_tensor
            MinTokensLogitsProcessor._minicpm5_mk_hooked = True
    except Exception as e:  # noqa: BLE001
        logger.warning("minicpm5_mk: MinTokensLogitsProcessor hook skipped: %r", e)

    logger.info("minicpm5_mk: installed TPUModelRunner + AllSequenceLayoutPlanner + MinTokens hooks (require_max_tokens_1=%s)", require_mt1)


def _resolve_quant_pair(requested: str, num_layers: int, tp: int, hidden: int = 0, inter: int = 0) -> tuple[str, str]:
    """Return (quant_primary, quant_tp) for single/DP and TP modes respectively.

    Flags: sq = SmoothQuant (alpha 0.5) channel smoothing, lastbf16 = BF16 last layer,
    fmlp = fused RMSNorm + W8A8 gate/up + SiLU + down + residual Pallas kernel,
    fqkv = fused RMSNorm + W8A8 QKV Pallas kernel."""
    if requested != "auto":
        return requested, requested
    # fused-MLP keeps W_gu (int8) + W_down (bf16) resident in VMEM: 3 * H * I bytes.
    fmlp_ok = 3 * hidden * inter <= 56 * 1024 * 1024 if tp == 1 else 3 * hidden * inter <= 30 * 1024 * 1024
    fused = "|fqkv" + ("|fmlp" if fmlp_ok else "")
    if tp > 1:
        # On dual-core TPU7x (TP=2):
        # - DP mode (bs >= 2, even): SmoothQuant W8A8 on QKV + Gate/Up with the fused Pallas kernels
        # - TP mode (bs == 1): head/intermediate-sharded BF16 weights
        return "int8d:qkv+gu|sq|lastbf16" + fused, "bf16"
    # On single-core TPU v6e (TP=1):
    # - 1B (24 layers): SmoothQuant W8A8 on QKV + Gate/Up + BF16 last layer
    # - 2B (42 layers): SmoothQuant W8A8 on QKV + Gate/Up + O + BF16 last layer
    if num_layers <= 24:
        q = "int8d:qkv+gu|sq|lastbf16" + fused
    else:
        q = "int8d:qkv+gu+o|sq|lastbf16" + fused
    return q, q


class MiniCPM5MegakernelForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM + whole-model JAX megakernel for prefill-only (max_tokens=1) steps."""

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        hf = vllm_config.model_config.hf_config
        self._mk_enabled = os.environ.get(ENV_ENABLE, "1") != "0"
        self._mk_cfg = dict(
            num_layers=int(hf.num_hidden_layers), hidden=int(hf.hidden_size),
            inter=int(hf.intermediate_size), n_q=int(hf.num_attention_heads),
            n_kv=int(hf.num_key_value_heads),
            head_dim=int(getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads),
            eps=float(hf.rms_norm_eps), rope_theta=float(getattr(hf, "rope_theta", 10000.0)),
        )
        self._mk_bos = int(getattr(hf, "bos_token_id", 0) or 0)
        self._mk_max_seq = int(vllm_config.model_config.max_model_len)
        self._mk_attn = os.environ.get(ENV_ATTN, "auto")
        self._mk_quant = None
        self._mk_quant_tp = None
        self._mk_ready = False
        self._mk_counts = {"mk": 0, "stock": 0}
        self._mk_log_every = int(os.environ.get(ENV_LOG_EVERY, "0"))
        rope_scaling = getattr(hf, "rope_scaling", None) or getattr(hf, "rope_parameters", None)
        if isinstance(rope_scaling, dict):
            rtype = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            if "rope_theta" in rope_scaling:
                self._mk_cfg["rope_theta"] = float(rope_scaling["rope_theta"])
            if rtype not in (None, "default"):
                logger.warning("minicpm5_mk: rope_scaling=%s unsupported; fast path disabled", rope_scaling)
                self._mk_enabled = False
        elif rope_scaling:
            logger.warning("minicpm5_mk: rope_scaling=%s unsupported; fast path disabled", rope_scaling)
            self._mk_enabled = False
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            logger.warning("minicpm5_mk: pipeline parallelism unsupported; fast path disabled")
            self._mk_enabled = False
        self._mk_tp = int(vllm_config.parallel_config.tensor_parallel_size)
        self._mk_modes = tuple(m.strip() for m in os.environ.get(ENV_MC_MODES, "dp,sp,tp").split(",") if m.strip())
        if self._mk_tp > 1 and self._mk_cfg["n_kv"] % self._mk_tp and "dp" not in self._mk_modes:
            logger.warning("minicpm5_mk: n_kv %% tp != 0 and no dp mode; fast path disabled")
            self._mk_enabled = False
        if self._mk_enabled:
            _install_runner_hook(os.environ.get(ENV_REQUIRE_MT1, "1") != "0")

    # ------------------------------------------------------------------ weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if not self._mk_enabled:
            return super().load_weights(weights)
        stash: dict[str, torch.Tensor] = {}

        def tee():
            for name, t in weights:
                if (name.startswith("model.layers.") and name.endswith(_WEIGHT_SUFFIXES)) or name in (
                    "model.embed_tokens.weight", "model.norm.weight"):
                    stash[name] = t.detach().to("cpu", copy=True)
                yield name, t

        loaded = super().load_weights(tee())
        try:
            self._mk_build(stash)
        except Exception:
            if os.environ.get(ENV_FREE_STOCK, "0") == "1":
                # megakernel-only mode: never fall back silently (the KV-skip hook assumes no stock path)
                logger.exception("minicpm5_mk: megakernel build failed and MINICPM5_MK_FREE_STOCK=1")
                raise
            logger.exception("minicpm5_mk: megakernel build failed; using the stock path only")
            self._mk_ready = False
        return loaded

    def _mk_build(self, stash: dict[str, torch.Tensor]) -> None:
        from minicpm5_mk import mk_weights

        t0 = time.perf_counter()
        self._mk_quant, self._mk_quant_tp = _resolve_quant_pair(
            os.environ.get(ENV_QUANT, "auto"), self._mk_cfg["num_layers"], self._mk_tp,
            self._mk_cfg["hidden"], self._mk_cfg["inter"]
        )
        w = mk_weights.prepare_mk_weights(stash, self._mk_cfg, self._mk_quant, self._mk_max_seq,
                                          self._mk_bos, log=logger.info)
        device = self.lm_head.weight.device
        dev = {k: v.to(device) for k, v in w.items()}
        V, H = int(self.config.vocab_size), self._mk_cfg["hidden"]
        emb = getattr(self.model.embed_tokens, "weight", None)
        if emb is not None and tuple(emb.shape) == (V, H) and emb.dtype == torch.bfloat16:
            self._mk_embed = emb.data  # plain tensor view: must not register as a new Parameter
        else:
            self._mk_embed = stash["model.embed_tokens.weight"].to(torch.bfloat16).to(device)
        self._mk_fnorm = stash["model.norm.weight"].to(torch.bfloat16).to(device)
        self._mk_cos, self._mk_sin = dev.pop("cos"), dev.pop("sin")
        self._mk_wargs = None
        self._mk_wargs_tp = None
        self._mk_rank = 0
        if self._mk_tp == 1:
            self._mk_wargs = [dev[k] for k in _WEIGHT_ORDER]
            _get_op()
        else:
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()
            self._mk_rank = rank
            if "dp" in self._mk_modes or "sp" in self._mk_modes:
                self._mk_wargs = [dev[k] for k in _WEIGHT_ORDER]
                if "dp" in self._mk_modes:
                    _get_multicore_op("dp", self._mk_quant)
                if "sp" in self._mk_modes and self._mk_tp == 2:
                    # sequence-parallel batch-1 mode: rank operand (this core's index) + replicated weights
                    self._mk_rank_t = torch.tensor([rank], dtype=torch.int32).to(device)
                    _get_multicore_op("sp", self._mk_quant)
            if "tp" in self._mk_modes and self._mk_cfg["n_kv"] % self._mk_tp == 0:
                if self._mk_quant_tp == self._mk_quant:
                    w_tp = w
                else:
                    w_tp = mk_weights.prepare_mk_weights(
                        stash, self._mk_cfg, self._mk_quant_tp, self._mk_max_seq,
                        self._mk_bos, log=logger.info
                    )
                shard = _tp_shard(w_tp, self._mk_cfg, rank, self._mk_tp, self._mk_quant_tp)
                self._mk_wargs_tp = [shard[k].to(device) for k in _WEIGHT_ORDER]
                _get_multicore_op("tp", self._mk_quant_tp)
            if self._mk_wargs is None:
                del dev
            logger.info("minicpm5_mk: TP=%d rank %d multicore modes %s (quant=%s, tp quant=%s)",
                        self._mk_tp, rank, self._mk_modes, self._mk_quant, self._mk_quant_tp)
        del stash
        self._mk_ready = True
        logger.info("minicpm5_mk: megakernel ready (quant=%s quant_tp=%s attn=%s max_seq=%d) in %.1fs",
                    self._mk_quant, self._mk_quant_tp, self._mk_attn, self._mk_max_seq, time.perf_counter() - t0)
        if os.environ.get(ENV_FREE_STOCK, "0") == "1":
            self._mk_free_stock_layers()
        self._mk_status = {
            "arch": "MiniCPM5MegakernelForCausalLM", "tp": self._mk_tp, "rank": self._mk_rank,
            "quant": self._mk_quant, "quant_tp": self._mk_quant_tp, "attn": self._mk_attn,
            "multicore_modes": list(self._mk_modes) if self._mk_tp > 1 else [],
            "stock_freed": bool(getattr(self, "_mk_stock_freed", False)), "shapes": {},
        }
        self._mk_write_status()
        pre = os.environ.get(ENV_PRECOMPILE, "")
        if pre:
            for shp in pre.split(","):
                b, s = (int(v) for v in shp.strip().split("x"))
                t1 = time.perf_counter()
                ids = torch.zeros((b * s,), dtype=torch.int32).to(device)
                out = self._mk_call(ids, b, s)
                out[-1:, :1].cpu()
                logger.info("minicpm5_mk: precompiled %dx%d in %.1fs", b, s, time.perf_counter() - t1)

    def _mk_free_stock_layers(self) -> None:
        for layer in self.model.layers:
            for p in layer.parameters():
                p.data = torch.empty((0,), dtype=p.dtype, device=p.device)
        self._mk_stock_freed = True
        logger.info("minicpm5_mk: freed stock decoder-layer weights (no stock fallback)")

    def _mk_write_status(self) -> None:
        """Record which kernels / modes actually ran (one small JSON per rank; first use of each shape only)."""
        d = os.environ.get(ENV_STATUS_DIR)
        if not d or not getattr(self, "_mk_status", None):
            return
        try:
            import json

            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"minicpm5_mk_status_rank{self._mk_rank}.json")
            with open(path + ".tmp", "w") as f:
                json.dump(self._mk_status, f, indent=1)
            os.replace(path + ".tmp", path)
        except Exception as e:  # noqa: BLE001
            logger.warning("minicpm5_mk: could not write status file: %r", e)

    # ------------------------------------------------------------------ forward
    def _mk_select(self, bs: int, seq: int = 0):
        """(mode, op, weight args, quant) for this batch, or None if no fast path applies."""
        if self._mk_tp == 1:
            return "single", _get_op(), self._mk_wargs, self._mk_quant
        if self._mk_wargs is not None and "dp" in self._mk_modes and bs % self._mk_tp == 0:
            return "dp", _get_multicore_op("dp", self._mk_quant), self._mk_wargs, self._mk_quant
        if (self._mk_wargs is not None and bs == 1 and getattr(self, "_mk_rank_t", None) is not None
                and seq >= 2048 and seq % 2048 == 0):
            return "sp", _get_multicore_op("sp", self._mk_quant), self._mk_wargs, self._mk_quant
        if self._mk_wargs_tp is not None:
            return "tp", _get_multicore_op("tp", self._mk_quant_tp), self._mk_wargs_tp, self._mk_quant_tp
        return None

    def _mk_call(self, input_ids: torch.Tensor, bs: int, seq: int) -> torch.Tensor:
        c = self._mk_cfg
        mode, op, wargs, quant = self._mk_select(bs, seq)
        ids = input_ids
        if mode == "dp":  # this rank's sequences only (sequence-major token layout)
            n = bs * seq // self._mk_tp
            ids = input_ids[self._mk_rank * n:(self._mk_rank + 1) * n]
        lead = (self._mk_rank_t,) if mode == "sp" else ()
        from torch_tpu._internal.compiler_options import custom_compiler_options

        opt = os.environ.get("MINICPM5_MK_OPT_LEVEL", "O3")
        with custom_compiler_options({"xla_optimization_level": opt, "xla_tpu_autofdo": "true"}):
            out = op(*lead, ids, self._mk_embed, self._mk_cos, self._mk_sin, self._mk_fnorm,
                     *wargs, bs, seq, quant, c["n_q"], c["n_kv"],
                     c["head_dim"], c["eps"], self._mk_attn)
        _STATE.last_was_mk = True
        return out

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        batch = _STATE.batch
        _STATE.batch = None
        if (batch is not None and self._mk_ready and input_ids is not None
                and inputs_embeds is None and intermediate_tensors is None):
            bs, seq = batch
            sel = self._mk_select(bs, seq)
            if (bs * seq <= input_ids.shape[0] and _mk_seq_ok(seq, self._mk_max_seq)
                    and sel is not None):
                self._mk_counts["mk"] += 1
                key = f"mk:{bs}x{seq}"
                if key not in self._mk_counts:  # first use of a shape: log + record the path decision
                    self._mk_counts[key] = 0
                    logger.info("minicpm5_mk: fast path %s (T=%d, mode=%s, quant=%s)", key,
                                input_ids.shape[0], sel[0], sel[3])
                    if getattr(self, "_mk_status", None) is not None:
                        self._mk_status["shapes"][key] = {"mode": sel[0], "quant": sel[3]}
                        self._mk_write_status()
                self._mk_counts[key] += 1
                self._mk_maybe_log(f"mk bs={bs} seq={seq} T={input_ids.shape[0]}")
                return self._mk_call(input_ids, bs, seq)
        self._mk_counts["stock"] += 1
        rkey = f"stock:{_STATE.reason}"
        if rkey not in self._mk_counts:
            self._mk_counts[rkey] = 0
            if _STATE.reason != "dummy":
                logger.info("minicpm5_mk: stock path (reason=%s, T=%d)", _STATE.reason,
                            input_ids.shape[0] if input_ids is not None else -1)
                if getattr(self, "_mk_status", None) is not None:
                    self._mk_status.setdefault("stock_reasons", []).append(_STATE.reason)
                    self._mk_write_status()
        self._mk_counts[rkey] += 1
        if getattr(self, "_mk_stock_freed", False):
            if _STATE.reason == "dummy":
                return torch.zeros(
                    (input_ids.shape[0], self._mk_cfg["hidden"]),
                    dtype=torch.bfloat16,
                    device=input_ids.device,
                )
            raise RuntimeError(f"minicpm5_mk: stock path needed ({_STATE.reason}) but stock weights were freed")
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def _mk_maybe_log(self, msg: str) -> None:
        n = self._mk_log_every
        tot = self._mk_counts["mk"] + self._mk_counts["stock"]
        if n and tot % n == 0:
            logger.info("minicpm5_mk: %s counts=%s last_reason=%s", msg, self._mk_counts, _STATE.reason)
