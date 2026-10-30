# Out-of-tree vLLM plugin: MiniCPM5 megakernel model for vLLM-TorchTPU.
"""vLLM general plugin registering MiniCPM5MegakernelForCausalLM and host fast-path hooks."""

ARCH = "MiniCPM5MegakernelForCausalLM"

# Set in the engine-core (scheduler) process when that engine serves ARCH; read by the KV-skip hook.
ENGINE_SERVES_ARCH = {"active": False}


def sampling_fast_ok(sp) -> bool:
    """True iff a request's SamplingParams only use features the megakernel fast path implements.

    The fast path returns last-position hidden states and registers temperature / top-p / top-k /
    seed / logprobs / min_tokens; anything else (prompt logprobs, penalties, min_p, logit bias,
    allowed or bad words, structured output, custom logits processors) must take vLLM's stock path.
    """
    if sp is None:
        return False
    g = getattr
    if g(sp, "prompt_logprobs", None) is not None:
        return False
    if (g(sp, "presence_penalty", 0.0) or 0.0) != 0.0 or (g(sp, "frequency_penalty", 0.0) or 0.0) != 0.0:
        return False
    if (g(sp, "repetition_penalty", 1.0) or 1.0) != 1.0 or (g(sp, "min_p", 0.0) or 0.0) != 0.0:
        return False
    if g(sp, "logit_bias", None) or g(sp, "allowed_token_ids", None) or g(sp, "bad_words", None):
        return False
    if g(sp, "structured_outputs", None) is not None or g(sp, "guided_decoding", None) is not None:
        return False
    if g(sp, "logits_processors", None):
        return False
    return True


def register() -> None:
    """Entry point for the `vllm.general_plugins` group (runs in every vLLM process)."""
    import os

    from vllm import ModelRegistry

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "minicpm5_mk.model:MiniCPM5MegakernelForCausalLM")

    if os.environ.get("MINICPM5_MK_ENABLE", "1") == "0":
        return

    if os.environ.get("TTVLLM_OFFLINE_SKIP_CLONE", "1") == "1":
        try:
            from vllm.entrypoints.common import offline as _off
            from vllm.sampling_params import SamplingParams

            if not getattr(_off.OfflineInferenceMixin, "_ttvllm_skip_clone_hooked", False):
                _orig_params_to_seq = _off.OfflineInferenceMixin._params_to_seq

                def _params_to_seq(self, params, num_requests):
                    seq = _orig_params_to_seq(self, params, num_requests)
                    per_call = {}
                    out = []
                    for p in seq:
                        if isinstance(p, SamplingParams) and not p.skip_clone:
                            q = per_call.get(id(p))
                            if q is None:
                                q = p.clone()
                                q.skip_clone = True
                                per_call[id(p)] = q
                            out.append(q)
                        else:
                            out.append(p)
                    return out

                _off.OfflineInferenceMixin._params_to_seq = _params_to_seq
                _off.OfflineInferenceMixin._ttvllm_skip_clone_hooked = True
        except Exception:  # noqa: BLE001
            pass

    if os.environ.get("TTVLLM_CACHE_EOS_ID", "1") == "1":
        try:
            from vllm.renderers import base as _rb

            _cls = next(
                c for c in vars(_rb).values()
                if isinstance(c, type) and "get_eos_token_id" in vars(c)
            )
            if not getattr(_cls, "_ttvllm_eos_hooked", False):
                _orig_eos = _cls.get_eos_token_id
                _SENTINEL = object()

                def get_eos_token_id(self, *a, **k):
                    if a or k:
                        return _orig_eos(self, *a, **k)
                    v = self.__dict__.get("_ttvllm_eos_cache", _SENTINEL)
                    if v is _SENTINEL:
                        v = _orig_eos(self)
                        self.__dict__["_ttvllm_eos_cache"] = v
                    return v

                _cls.get_eos_token_id = get_eos_token_id
                _cls._ttvllm_eos_hooked = True

            if (
                os.environ.get("TTVLLM_FAST_VALIDATE_TOKENS", "1") == "1"
                and hasattr(_cls, "validate_token_ids")
                and not getattr(_cls, "_ttvllm_val_toks_hooked", False)
            ):
                _orig_validate_toks = _cls.validate_token_ids

                def validate_token_ids(self, token_ids, *, parameter=None):
                    if token_ids and parameter is None:
                        mt = self.__dict__.get("_ttvllm_max_tok_id", -1)
                        if mt < 0:
                            tok = self.tokenizer
                            if tok is None:
                                return None
                            mt = max(tok.max_token_id, self.model_config.get_vocab_size() - 1)
                            self.__dict__["_ttvllm_max_tok_id"] = mt
                        if 0 <= min(token_ids) and max(token_ids) <= mt:  # every id, C-speed min/max
                            return None
                    return _orig_validate_toks(self, token_ids, parameter=parameter)

                _cls.validate_token_ids = validate_token_ids
                _cls._ttvllm_val_toks_hooked = True
        except Exception:  # noqa: BLE001
            pass

    if os.environ.get("TTVLLM_FAST_PREPROCESS", "1") == "1":
        try:
            from vllm.entrypoints.common import offline as _off

            if (
                hasattr(_off.OfflineInferenceMixin, "_preprocess_cmpl_one")
                and not getattr(_off.OfflineInferenceMixin, "_ttvllm_fast_preproc_hooked", False)
            ):
                _orig_preproc_one = _off.OfflineInferenceMixin._preprocess_cmpl_one

                def _preprocess_cmpl_one(self, prompt, tokenization_kwargs=None, mm_processor_kwargs=None):
                    if (
                        tokenization_kwargs is None
                        and mm_processor_kwargs is None
                        and isinstance(prompt, dict)
                        and len(prompt) == 1
                    ):
                        toks = prompt.get("prompt_token_ids")
                        if isinstance(toks, list) and toks:
                            return {"type": "token", "prompt_token_ids": toks}
                    return _orig_preproc_one(
                        self, prompt, tokenization_kwargs, mm_processor_kwargs=mm_processor_kwargs
                    )

                _off.OfflineInferenceMixin._preprocess_cmpl_one = _preprocess_cmpl_one
                _off.OfflineInferenceMixin._ttvllm_fast_preproc_hooked = True
        except Exception:  # noqa: BLE001
            pass

    # KV-skip: max_tokens=1 scoring requests never read their KV cache on the megakernel path, so the
    # scheduler can skip allocating blocks. This is only safe when the stock path cannot run at all
    # (MINICPM5_MK_FREE_STOCK=1: an ineligible step raises instead of running stock attention without
    # KV), the engine in this process serves ARCH, and the request is a full, fresh, fast-path prefill.
    if os.environ.get("TTVLLM_SKIP_KV_ALLOC_MT1", "1") == "1":
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            from vllm.v1.engine.core import EngineCore

            if not getattr(EngineCore, "_minicpm5_arch_hooked", False):
                _orig_core_init = EngineCore.__init__

                def _core_init(self, vllm_config, *args, **kwargs):
                    try:
                        mc = vllm_config.model_config
                        archs = list(getattr(mc, "architectures", None) or []) + list(
                            getattr(mc.hf_config, "architectures", None) or [])
                        ENGINE_SERVES_ARCH["active"] = ARCH in archs
                    except Exception:  # noqa: BLE001
                        ENGINE_SERVES_ARCH["active"] = False
                    return _orig_core_init(self, vllm_config, *args, **kwargs)

                EngineCore.__init__ = _core_init
                EngineCore._minicpm5_arch_hooked = True

            if not getattr(KVCacheManager, "_ttvllm_mt1_hooked", False):
                _orig_alloc_slots = KVCacheManager.allocate_slots

                def _allocate_slots(self, request, num_new_tokens, *args, **kwargs):
                    sp = request.sampling_params
                    if (
                        ENGINE_SERVES_ARCH["active"]
                        and os.environ.get("MINICPM5_MK_ENABLE", "1") != "0"
                        and os.environ.get("MINICPM5_MK_FREE_STOCK", "0") == "1"
                        and request.num_computed_tokens == 0
                        and sp is not None
                        and sp.max_tokens == 1
                        and num_new_tokens == request.num_prompt_tokens
                        and sampling_fast_ok(sp)
                    ):
                        seq = request.num_prompt_tokens
                        if seq > 0 and ((seq <= 1024 and seq % 128 == 0) or seq % 1024 == 0):
                            if not ENGINE_SERVES_ARCH.get("logged"):
                                ENGINE_SERVES_ARCH["logged"] = True
                                print("[minicpm5_mk] KV-cache allocation skipped for fast-path max_tokens=1 "
                                      "requests (MINICPM5_MK_FREE_STOCK=1)", flush=True)
                            return self.empty_kv_cache_blocks
                    return _orig_alloc_slots(self, request, num_new_tokens, *args, **kwargs)

                KVCacheManager.allocate_slots = _allocate_slots
                KVCacheManager._ttvllm_mt1_hooked = True
        except Exception:  # noqa: BLE001
            pass

    if os.environ.get("TTVLLM_SINGLE_STEP_MT1", "1") == "1":
        try:
            from vllm.v1.engine.core import EngineCore

            if not getattr(EngineCore, "_ttvllm_single_step_hooked", False):
                _orig_step_bq = EngineCore.step_with_batch_queue

                def _step_with_batch_queue(self):
                    bq = self.batch_queue
                    if (
                        bq is not None
                        and not bq
                        and self.batch_queue_size > 1
                        and not self.scheduler.running
                        and self.scheduler.waiting
                        and len(self.scheduler.waiting) <= self.scheduler.max_num_running_reqs
                    ):
                        tot = 0
                        all_mt1 = True
                        for r in self.scheduler.waiting:
                            sp = r.sampling_params
                            if sp is None or sp.max_tokens != 1:
                                all_mt1 = False
                                break
                            tot += r.num_prompt_tokens
                        if all_mt1 and 0 < tot <= self.scheduler.max_num_scheduled_tokens:
                            saved = self.batch_queue_size
                            self.batch_queue_size = 1
                            try:
                                return _orig_step_bq(self)
                            finally:
                                self.batch_queue_size = saved
                    return _orig_step_bq(self)

                EngineCore.step_with_batch_queue = _step_with_batch_queue
                EngineCore._ttvllm_single_step_hooked = True
        except Exception:  # noqa: BLE001
            pass

