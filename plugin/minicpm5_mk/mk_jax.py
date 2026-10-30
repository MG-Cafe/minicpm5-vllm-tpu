# MiniCPM5 megakernel for vLLM-TorchTPU: trace-only JAX/Pallas forward.
#
# Ported from scripts/tpu_megakernel_minicpm5.py (standalone JAX engine). Differences:
# * Never creates concrete JAX arrays: the forward is exported through torch_tpu's
#   `jax_op` bridge and executed by TorchTPU on torch tensors.
# * Weights arrive as stacked (L, ...) arrays for all L layers; the first L-1 run in a
#   `lax.scan`, the last one runs last-token-only.
# * Returns (T_pad, H) final-normed hidden states where only the last token row of every
#   sequence is filled, so vLLM's own lm_head / sampler / logprobs stay unchanged.
"""Trace-only JAX MiniCPM5 (Llama) megakernel forward."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as sak,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as sam,
)

BF16 = jnp.bfloat16
F32 = jnp.float32


def rms_norm_unit(x: jax.Array, eps: float) -> jax.Array:
    """RMSNorm without gamma, f32 math, result in x.dtype (matches vLLM RMSNorm)."""
    xf = x.astype(F32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(var + eps)).astype(x.dtype)


def rope_perm_matrix(head_dim: int) -> jax.Array:
    """P with x @ P = [-x2, x1] (rotate_half), as an exact bf16 constant."""
    half = head_dim // 2
    p = np.zeros((head_dim, head_dim), dtype=np.float32)
    for i in range(half):
        p[i, i + half] = 1.0
        p[i + half, i] = -1.0
    return jnp.asarray(p, dtype=BF16)


def apply_rope_4d(x: jax.Array, cos: jax.Array, sin: jax.Array, perm: jax.Array) -> jax.Array:
    """x: (B, S, N, D); cos/sin: (S, D)."""
    x_rot = jnp.einsum("bsnd,de->bsne", x, perm, preferred_element_type=F32).astype(x.dtype)
    return (x * cos[None, :, None, :] + x_rot * sin[None, :, None, :]).astype(x.dtype)


# ----------------------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------------------
def attn_fused_small(qkv, cos, sin, perm, n_q: int, n_kv: int, hd: int):
    """VMEM-fused QKV split + RoPE + causal-staircase transposed (sublane) softmax GQA for S <= 1024.

    Tiles Q into bq = min(512, S) blocks so S=1024 skips the upper-right (512, 512) KV block
    and processes all g = n_q // n_kv query heads per KV group in one pass, uses hardware
    approximate reciprocal (pltpu.reciprocal(..., approx=True)), and software-pipelines PV
    matmul across KV heads at S <= 256. sm_scale is folded into w_q at load time.
    """
    B, S, total_cols = qkv.shape
    g = n_q // n_kv
    qrows = n_q * hd
    krows = n_kv * hd
    qkrows = qrows + krows
    bq = min(512, S)
    nq = S // bq
    g_chunk = next(c for c in range(g, 0, -1) if g % c == 0 and bq * c <= 4096)
    num_g_chunks = g // g_chunk
    pipe = (S <= 256) and (n_kv * num_g_chunks > 1)

    def _kern(qkv_ref, cos_ref, sin_ref, prot_ref, out_ref):
        q_all = qkv_ref[0]
        c_s = cos_ref[:]
        s_s = sin_ref[:]
        prot = prot_ref[:]
        k_ropes = []
        vs = []
        for h in range(n_kv):
            k_raw = q_all[:, qrows + h * hd: qrows + (h + 1) * hd]
            k_rot = jax.lax.dot_general(
                k_raw, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32
            ).astype(BF16)
            k_ropes.append(k_raw * c_s + k_rot * s_s)
            vs.append(q_all[:, qkrows + h * hd: qkrows + (h + 1) * hd])

        for qi in range(nq):
            q_start = qi * bq
            kv_len = (qi + 1) * bq
            c_q = c_s[q_start : q_start + bq, :]
            s_q = s_s[q_start : q_start + bq, :]
            c_gs = jnp.tile(c_q, (g_chunk, 1))
            s_gs = jnp.tile(s_q, (g_chunk, 1))
            k_idx = jax.lax.broadcasted_iota(jnp.int32, (kv_len, g_chunk * bq), 0)
            col = jax.lax.broadcasted_iota(jnp.int32, (kv_len, g_chunk * bq), 1)
            q_idx = q_start + jax.lax.rem(col, bq)
            mask_t = k_idx <= q_idx

            prev = None
            for h in range(n_kv):
                k_r = k_ropes[h][:kv_len, :]
                v_h = vs[h][:kv_len, :]
                for gc in range(num_g_chunks):
                    base_h = g * h + gc * g_chunk
                    qs = [
                        q_all[q_start : q_start + bq, (base_h + i) * hd: (base_h + i + 1) * hd]
                        for i in range(g_chunk)
                    ]
                    q_st = jnp.concatenate(qs, axis=0)
                    q_rot = jax.lax.dot_general(
                        q_st, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32
                    ).astype(BF16)
                    q_rope = q_st * c_gs + q_rot * s_gs
                    s_t = jax.lax.dot_general(
                        k_r, q_rope, (((1,), (1,)), ((), ())), preferred_element_type=F32
                    )
                    s_t = jnp.where(mask_t, s_t, -1e30)
                    s_max = jnp.max(s_t, axis=0, keepdims=True)
                    e_t = jnp.exp(s_t - s_max)
                    denom = jnp.sum(e_t, axis=0, keepdims=True)
                    inv = pltpu.reciprocal(
                        jnp.broadcast_to(denom, (8, g_chunk * bq)), approx=True
                    )[:1, :]
                    p_t = (e_t * inv).astype(BF16)
                    if pipe and prev is not None:
                        p_prev, v_prev, bh_prev = prev
                        o_st = jax.lax.dot_general(
                            p_prev, v_prev, (((0,), (0,)), ((), ())), preferred_element_type=F32
                        ).astype(BF16)
                        for i in range(g_chunk):
                            out_ref[
                                0, q_start : q_start + bq, (bh_prev + i) * hd: (bh_prev + i + 1) * hd
                            ] = o_st[i * bq: (i + 1) * bq, :]
                    prev = (p_t, v_h, base_h)
                    if not pipe:
                        o_st = jax.lax.dot_general(
                            p_t, v_h, (((0,), (0,)), ((), ())), preferred_element_type=F32
                        ).astype(BF16)
                        for i in range(g_chunk):
                            out_ref[
                                0, q_start : q_start + bq, (base_h + i) * hd: (base_h + i + 1) * hd
                            ] = o_st[i * bq: (i + 1) * bq, :]
            if pipe and prev is not None:
                p_prev, v_prev, bh_prev = prev
                o_st = jax.lax.dot_general(
                    p_prev, v_prev, (((0,), (0,)), ((), ())), preferred_element_type=F32
                ).astype(BF16)
                for i in range(g_chunk):
                    out_ref[
                        0, q_start : q_start + bq, (bh_prev + i) * hd: (bh_prev + i + 1) * hd
                    ] = o_st[i * bq: (i + 1) * bq, :]

    return pl.pallas_call(
        _kern,
        out_shape=jax.ShapeDtypeStruct((B, S, qrows), BF16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(B,),
            in_specs=[
                pl.BlockSpec((1, S, total_cols), lambda b: (b, 0, 0)),
                pl.BlockSpec((S, hd), lambda b: (0, 0)),
                pl.BlockSpec((S, hd), lambda b: (0, 0)),
                pl.BlockSpec((hd, hd), lambda b: (0, 0)),
            ],
            out_specs=pl.BlockSpec((1, S, qrows), lambda b: (b, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=100 * 1024 * 1024
        ),
    )(qkv, cos, sin, perm)


@functools.lru_cache(maxsize=None)
def _splash_fn(seq: int, n_q: int, bq: int, bkv: int, bkvc: int):
    mask = sam.CausalMask(shape=(seq, seq))
    mh = sam.MultiHeadMask(masks=[mask for _ in range(n_q)])
    bs = sak.BlockSizes(block_q=min(bq, seq), block_kv=min(bkv, seq), block_kv_compute=min(bkvc, seq))
    return jax.vmap(sak.make_splash_mha_single_device(mask=mh, block_sizes=bs))


def attn_splash(qkv, cos, sin, perm, n_q: int, n_kv: int, hd: int, blocks=(2048, 2048, 1024)):
    B, S, _ = qkv.shape
    qkv4 = qkv.reshape(B, S, n_q + 2 * n_kv, hd)
    qk = apply_rope_4d(qkv4[:, :, : n_q + n_kv, :], cos, sin, perm)
    q = qk[:, :, :n_q, :].transpose(0, 2, 1, 3)
    k = qk[:, :, n_q:, :].transpose(0, 2, 1, 3)
    v = qkv4[:, :, n_q + n_kv:, :].transpose(0, 2, 1, 3)
    out = _splash_fn(S, n_q, *blocks)(q, k, v)  # (B, n_q, S, hd)
    return out.transpose(0, 2, 1, 3).reshape(B, S, n_q * hd)


def attn_flash(qkv, cos, sin, perm, n_q: int, n_kv: int, hd: int, bq=256, bkv=512, exp_bf16=False):
    """Causal GQA flash kernel (mk_attn.gqa_flash_causal): g q-heads stacked into the MXU M dim."""
    from minicpm5_mk.mk_attn import gqa_flash_causal

    B, S, _ = qkv.shape
    g = n_q // n_kv
    qkv4 = qkv.reshape(B, S, n_q + 2 * n_kv, hd)
    qk = apply_rope_4d(qkv4[:, :, : n_q + n_kv, :], cos, sin, perm)
    q = qk[:, :, :n_q, :].reshape(B, S, n_kv, g, hd)
    k = qk[:, :, n_q:, :].transpose(0, 2, 1, 3)
    v = qkv4[:, :, n_q + n_kv:, :].transpose(0, 2, 1, 3)
    o = gqa_flash_causal(q, k, v, bq=min(bq, S), bkv=min(bkv, S), exp_bf16=exp_bf16)
    return o.reshape(B, S, n_q * hd)


def attn_xla(qkv, cos, sin, perm, n_q: int, n_kv: int, hd: int):
    B, S, _ = qkv.shape
    g = n_q // n_kv
    qkv4 = qkv.reshape(B, S, n_q + 2 * n_kv, hd)
    q = apply_rope_4d(qkv4[:, :, :n_q, :], cos, sin, perm)
    k = apply_rope_4d(qkv4[:, :, n_q: n_q + n_kv, :], cos, sin, perm)
    v = qkv4[:, :, n_q + n_kv:, :]
    q_g = q.reshape(B, S, n_kv, g, hd).transpose(0, 2, 3, 1, 4)
    k_g = k.transpose(0, 2, 1, 3)[:, :, None]
    v_g = v.transpose(0, 2, 1, 3)[:, :, None]
    sc = jnp.einsum("bngqd,bngkd->bngqk", q_g, k_g, preferred_element_type=F32)
    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))[None, None, None]
    sc = jnp.where(mask, sc, -1e30)
    p = jax.nn.softmax(sc, axis=-1).astype(BF16)
    out = jnp.einsum("bngqk,bngkd->bngqd", p, v_g, preferred_element_type=F32).astype(BF16)
    return out.transpose(0, 3, 1, 2, 4).reshape(B, S, n_q * hd)


# ----------------------------------------------------------------------------------------
# Linear layers
# ----------------------------------------------------------------------------------------
def linear(x, w, s_w, a_x, eqn: str, quant: str, dynamic_act: bool = False):
    """x @ w with optional W8A8.

    quant: "bf16" | "fp8" (TPU7x native, static act scale) | "fp8d" (fp8 dynamic per-token)
           | "int8" (v6e native, static act scale) | "int8d" (int8 dynamic per-token).
    s_w: per-output-channel weight scale (static act scale folded in), f32.
    a_x: per-input-channel inverse activation scale (norm gamma / SmoothQuant folded in), f32.
    dynamic_act: per-token dynamic activation scale (a_x carries channel smoothing only).
    """
    if quant == "bf16":
        return jnp.einsum(eqn, x, w, preferred_element_type=F32).astype(BF16)
    if quant in ("fp8", "fp8d"):
        if dynamic_act or quant == "fp8d":
            xf = x.astype(F32) * a_x
            amax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
            s_x = jnp.maximum(amax * (1.0 / 448.0), 1e-6)
            inv_s = 1.0 / s_x
            xq = jnp.clip(xf * inv_s, -448.0, 448.0).astype(jnp.float8_e4m3fn)
            out = jnp.einsum(eqn, xq, w, preferred_element_type=F32)
            s_x = s_x.reshape(s_x.shape[:-1] + (1,) * (out.ndim - x.ndim + 1))
            return (out * (s_x * s_w)).astype(BF16)
        xq = jnp.clip(x.astype(F32) * a_x, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        out = jnp.einsum(eqn, xq, w, preferred_element_type=F32)
        return (out * s_w).astype(BF16)
    if quant in ("int8", "int8d"):
        if dynamic_act or quant == "int8d":
            xf = x.astype(F32) * a_x  # a_x: per-channel smoothing (SmoothQuant), ones otherwise
            amax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
            s_x = jnp.maximum(amax / 127.0, 1e-6)
            xq = jnp.clip(jnp.round(xf / s_x), -127.0, 127.0).astype(jnp.int8)
            out = jnp.einsum(eqn, xq, w, preferred_element_type=jnp.int32)
            s_x = s_x.reshape(s_x.shape[:-1] + (1,) * (out.ndim - x.ndim + 1))  # per-token, bcast
            return (out.astype(F32) * s_x * s_w).astype(BF16)
        xq = jnp.clip(jnp.round(x.astype(F32) * a_x), -127.0, 127.0).astype(jnp.int8)
        out = jnp.einsum(eqn, xq, w, preferred_element_type=jnp.int32)
        return (out.astype(F32) * s_w).astype(BF16)
    raise ValueError(f"unsupported quant {quant}")


def linear_rmsnorm(x, w, s_w, a_x, eps: float, eqn: str, quant: str):
    """Fused RMSNorm + Linear (single-pass variance + dynamic quantization)."""
    xf = x.astype(F32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    if quant == "bf16":
        xn = (xf * rstd).astype(BF16)
        return jnp.einsum(eqn, xn, w, preferred_element_type=F32).astype(BF16)
    if quant == "int8d":
        u = xf * a_x
        amax_u = jnp.max(jnp.abs(u), axis=-1, keepdims=True)
        s_u = jnp.maximum(amax_u * (1.0 / 127.0), 1e-6)
        xq = jnp.clip(jnp.round(u / s_u), -127.0, 127.0).astype(jnp.int8)
        out = jnp.einsum(eqn, xq, w, preferred_element_type=jnp.int32)
        s_x = s_u * rstd
        s_x = s_x.reshape(s_x.shape[:-1] + (1,) * (out.ndim - x.ndim + 1))
        return (out.astype(F32) * (s_x * s_w)).astype(BF16)
    if quant == "fp8d":
        u = xf * a_x
        amax_u = jnp.max(jnp.abs(u), axis=-1, keepdims=True)
        s_u = jnp.maximum(amax_u * (1.0 / 448.0), 1e-6)
        xq = jnp.clip(u / s_u, -448.0, 448.0).astype(jnp.float8_e4m3fn)
        out = jnp.einsum(eqn, xq, w, preferred_element_type=F32)
        s_x = s_u * rstd
        s_x = s_x.reshape(s_x.shape[:-1] + (1,) * (out.ndim - x.ndim + 1))
        return (out * (s_x * s_w)).astype(BF16)
    xn = (xf * rstd).astype(x.dtype)
    return linear(xn, w, s_w, a_x, eqn, quant)


LINEARS = ("qkv", "o", "gu", "down")


def quant_flags(quant: str) -> set:
    """Flags after '|', e.g. "fp8:qkv+gu|lastbf16" -> {"lastbf16"}."""
    return set(quant.split("|")[1:])


def parse_quant(quant: str) -> dict:
    """"fp8" / "int8" / "bf16" (all linears) or "<q>:<lin>+<lin>" (only those quantized, rest bf16),
    e.g. "fp8:qkv+gu". Optional flags after '|' (see quant_flags)."""
    quant = quant.split("|")[0]
    if ":" in quant:
        q, lins = quant.split(":", 1)
        sel = set(lins.split("+"))
        return {k: (q if k in sel else "bf16") for k in LINEARS}
    return {k: quant for k in LINEARS}


def attn_super(qkv, cos, sin, perm, n_q: int, n_kv: int, hd: int, bq=256, bkv=512, sub_pipe=True, sub2=False, exp_bf16=False):
    """SuperAttention fused GQA causal flash kernel: in-VMEM MXU RoPE + g q-heads stacked + 2-phase causal loop."""
    from minicpm5_mk.mk_attn import super_gqa_fused

    return super_gqa_fused(qkv, cos, sin, perm, n_q, n_kv, hd, bq=bq, bkv=bkv, sub_pipe=sub_pipe, sub2=sub2, exp_bf16=exp_bf16)


def _attention(qkv, cos, sin, perm, n_q, n_kv, hd, attn: str):
    S = qkv.shape[1]
    if attn == "xla":
        return attn_xla(qkv, cos, sin, perm, n_q, n_kv, hd)
    if attn.startswith("v2") or (attn == "auto" and S >= 2048 and S % 512 == 0):
        # SuperAttention v2: "v2[:bq,mc]" (default bq=512, mc=256)
        from minicpm5_mk.mk_attn import super_gqa_v2

        bq, mc = 512, 256
        if ":" in attn:
            parts = attn.split(":", 1)[1].split(",")
            bq, mc = int(parts[0]), int(parts[1])
        return super_gqa_v2(qkv, cos, sin, perm, n_q, n_kv, hd, bq=bq, mc=mc)
    if attn.startswith("super") or (attn == "auto" and S >= 2048) or (attn == "auto1024" and S >= 1024):
        # "super[:bq,bkv[,pipe|nopipe|sub2|bf16]]" or "auto" (uses super:256,512,pipe for S >= 2048)
        bq = 256
        bkv = 512
        sub_pipe = True
        sub2 = False
        ebf = False
        if ":" in attn:
            parts = attn.split(":", 1)[1].split(",")
            bq, bkv = int(parts[0]), int(parts[1])
            flags = set(parts[2:])
            sub_pipe = "pipe" in flags and "nopipe" not in flags
            sub2 = "sub2" in flags
            ebf = "bf16" in flags
        return attn_super(qkv, cos, sin, perm, n_q, n_kv, hd, bq, bkv, sub_pipe, sub2, ebf)
    if attn.startswith("flash") or (attn.startswith("auto+flash") and S > 1024):
        # "flash[:bq,bkv[,bf16]]" for all S, or "auto+flash[:...]" = fused small kernel for S<=1024
        bq, bkv, ebf = 256, 512, False
        if ":" in attn:
            parts = attn.split(":", 1)[1].split(",")
            bq, bkv = int(parts[0]), int(parts[1])
            ebf = len(parts) > 2 and parts[2] == "bf16"
        return attn_flash(qkv, cos, sin, perm, n_q, n_kv, hd, bq, bkv, ebf)
    if attn == "pallas" or (attn.startswith("auto") and S <= 1024):
        return attn_fused_small(qkv, cos, sin, perm, n_q, n_kv, hd)
    if attn.startswith("splash"):
        # "splash" or "splash:bq,bkv,bkvc"
        blocks = (2048, 2048, 1024)
        if ":" in attn:
            blocks = tuple(int(v) for v in attn.split(":", 1)[1].split(","))
        return attn_splash(qkv, cos, sin, perm, n_q, n_kv, hd, blocks)
    return attn_splash(qkv, cos, sin, perm, n_q, n_kv, hd)


def _psum(x, tp_axis):
    return jax.lax.psum(x, tp_axis) if tp_axis is not None else x


def layer_step(x, lw, cos, sin, perm, *, eps, n_q, n_kv, hd, quant, attn, tp_axis=None,
               b_orig: int | None = None, s_orig: int | None = None, layer=None, stacked=None):
    """One full-sequence decoder layer. x: (B_c, S_c, H) bf16. With tp_axis the weights are this
    rank's head / intermediate shard and the O / down partial sums are psum'ed.
    With `layer` + `stacked` (full (L-1, ...) weight stacks), the fused Pallas kernels DMA layer `layer`
    straight from the stack (no per-layer XLA slice copy of the 8-30 MB weight matrices)."""
    qm = parse_quant(quant)
    W = stacked if (stacked is not None and layer is not None) else lw
    lay = layer if (stacked is not None and layer is not None) else None
    qkv = _qkv_block(x, lw, W, lay, quant, eps, tp_axis)
    if b_orig is not None and s_orig is not None and (x.shape[0] != b_orig or x.shape[1] != s_orig):
        qkv = qkv.reshape(b_orig, s_orig, -1)
    attn_out = _attention(qkv, cos, sin, perm, n_q, n_kv, hd, attn)
    if attn_out.shape[:2] != x.shape[:2]:
        attn_out = attn_out.reshape(x.shape[0], x.shape[1], -1)
    x = x + _psum(linear(attn_out, lw["w_o"], lw["s_o"], lw["a_o"], "bsd,dh->bsh", qm["o"]), tp_axis)
    return _mlp_block(x, lw, W, lay, quant, eps, tp_axis)


def _qkv_block(x, lw, W, lay, quant, eps, tp_axis):
    """RMSNorm + QKV projection: fused Pallas kernel ("fqkv") or XLA linear_rmsnorm. Returns (B, S, D)."""
    qm = parse_quant(quant)
    if "fqkv" in quant_flags(quant) and qm["qkv"] == "int8d" and tp_axis is None:
        from minicpm5_mk.mk_mlp import fused_qkv

        return fused_qkv(x.reshape(-1, x.shape[-1]), W["w_qkv"], W["s_qkv"], W["a_qkv"], eps=eps,
                         layer=lay).reshape(x.shape[0], x.shape[1], -1)
    return linear_rmsnorm(x, lw["w_qkv"], lw["s_qkv"], lw["a_qkv"], eps, "bsh,hd->bsd", qm["qkv"])


def _mlp_block(x, lw, W, lay, quant, eps, tp_axis):
    """x + MLP(RMSNorm(x)): fused Pallas kernel (weights prepared with "fmlp": W_gu (H, 2I)) or XLA."""
    qm = parse_quant(quant)
    if lw["w_gu"].ndim == 2 and tp_axis is None:
        from minicpm5_mk.mk_mlp import fused_mlp

        shp = x.shape
        q8_down = qm["down"] in ("int8", "int8d")
        y = fused_mlp(x.reshape(-1, shp[-1]), W["w_gu"], W["s_gu"], W["a_gu"], W["w_down"],
                      W["s_down"] if q8_down else None, W["a_down"] if q8_down else None, eps=eps, layer=lay)
        return y.reshape(shp)
    gu = linear_rmsnorm(x, lw["w_gu"], lw["s_gu"], lw["a_gu"], eps, "bsh,hcm->bscm", qm["gu"])
    hidden = jax.nn.silu(gu[:, :, 0, :]) * gu[:, :, 1, :]
    return x + _psum(linear(hidden, lw["w_down"], lw["s_down"], lw["a_down"], "bsm,mh->bsh", qm["down"],
                            dynamic_act=qm["down"] in ("int8", "fp8")), tp_axis)


SP_BQ = 512


def _sp_src_index(nb: int) -> np.ndarray:
    """Static gather index: position-order block gb <- row (rank * nb/2 + i) of the all-gathered locals."""
    from minicpm5_mk.mk_attn import sp_global_block

    src = np.zeros(nb, np.int32)
    for r in (0, 1):
        for i in range(nb // 2):
            src[sp_global_block(i, r)] = r * (nb // 2) + i
    return src


def core_forward_sp(ids, rank, embed, cos_tab, sin_tab, final_norm, weights, *, quant, n_q, n_kv, head_dim,
                    eps, axis, last_weights):
    """Sequence-parallel forward of ONE sequence over the 2 TensorCores of a TPU7x chip.

    ids: (S,) int32 (replicated), rank: () int32 core index. Each core owns half of the 512-token blocks
    (zig-zag, causal-work balanced) for QKV / O / MLP, all-gathers the K/V of every layer, and runs the
    sequence-parallel SuperAttention v2 kernel on its query blocks. Returns (1, H) bf16 (replicated)."""
    from minicpm5_mk.mk_attn import super_gqa_v2_sp

    S = ids.shape[0]
    nb = S // SP_BQ
    i = jnp.arange(nb // 2, dtype=jnp.int32)
    j = i % 2
    gbs = 4 * (i // 2) + j * (3 - rank) + (1 - j) * rank
    ids_l = jnp.take(ids.reshape(nb, SP_BQ), gbs, axis=0).reshape(1, S // 2)
    x = jnp.take(embed, ids_l, axis=0).astype(BF16)
    cos = cos_tab[:S].astype(BF16)
    sin = sin_tab[:S].astype(BF16)
    perm = rope_perm_matrix(head_dim)
    src = jnp.asarray(_sp_src_index(nb))
    stacked = dict(zip(LAYER_KEYS, weights))
    is_quant = quant != "bf16"
    scan_keys = [k for k in LAYER_KEYS if k.startswith("w_") or is_quant]
    static_lw = {k: stacked[k] for k in LAYER_KEYS if k not in scan_keys}
    scan_xs = {k: stacked[k] for k in scan_keys}
    qm = parse_quant(quant)
    q_dim = n_q * head_dim
    H = x.shape[-1]

    def unzigzag(a):  # (S/2, C) local blocks on every core -> (S, C) position order
        g = jax.lax.all_gather(a, axis, axis=0, tiled=True)
        return jnp.take(g.reshape(nb, SP_BQ, -1), src, axis=0).reshape(S, -1)

    def body(xc, xs):
        idx, lw_slice = xs
        lw = {**static_lw, **lw_slice}
        qkv = _qkv_block(xc, lw, stacked, idx, quant, eps, None)              # (1, S/2, D)
        kv_full = unzigzag(qkv[0, :, q_dim:])                                # (S, 2 kv_dim)
        attn_out = super_gqa_v2_sp(qkv, kv_full[None], cos, sin, perm, rank, n_q, n_kv, head_dim)
        xc = xc + linear(attn_out, lw["w_o"], lw["s_o"], lw["a_o"], "bsd,dh->bsh", qm["o"])
        return _mlp_block(xc, lw, stacked, idx, quant, eps, None), None

    n_scan = stacked["w_qkv"].shape[0]
    x, _ = jax.lax.scan(body, x, (jnp.arange(n_scan, dtype=jnp.int32), scan_xs))
    x_full = unzigzag(x[0]).reshape(1, S, H)
    dummy = jnp.zeros((1,), F32)
    lw = dict(zip(("w_qkv", "w_o", "w_gu", "w_down"), last_weights))
    lw.update({k: dummy for k in LAYER_KEYS if not k.startswith("w_")})
    x_last = last_layer_step(x_full, lw, cos, sin, perm, eps=eps, n_q=n_q, n_kv=n_kv, hd=head_dim, quant="bf16")
    return (rms_norm_unit(x_last, eps) * final_norm.astype(BF16)).astype(BF16)


def last_layer_step(x, lw, cos, sin, perm, *, eps, n_q, n_kv, hd, quant, tp_axis=None):
    """Last decoder layer, exact for the last token only. Returns (B, H)."""
    B, S, H = x.shape
    g = n_q // n_kv
    q_dim = n_q * hd
    kv_dim = n_kv * hd
    qm = parse_quant(quant)
    xn = rms_norm_unit(x, eps)
    xn_last = xn[:, -1:, :]
    w_qkv, s_qkv = lw["w_qkv"], lw["s_qkv"]
    sq, skv = (s_qkv[:q_dim], s_qkv[q_dim:]) if qm["qkv"] != "bf16" else (s_qkv, s_qkv)
    q_last = linear(xn_last, w_qkv[:, :q_dim], sq, lw["a_qkv"], "bsh,hd->bsd", qm["qkv"])
    kv_all = linear(xn, w_qkv[:, q_dim:], skv, lw["a_qkv"], "bsh,hd->bsd", qm["qkv"])
    q_last = q_last.reshape(B, 1, n_q, hd)
    k = kv_all[:, :, :kv_dim].reshape(B, S, n_kv, hd)
    v = kv_all[:, :, kv_dim:].reshape(B, S, n_kv, hd)
    q_last = apply_rope_4d(q_last, cos[-1:], sin[-1:], perm)
    k = apply_rope_4d(k, cos, sin, perm)
    q_g = q_last.reshape(B, n_kv, g, hd)
    sc = jnp.einsum("bngd,bsnd->bngs", q_g, k, preferred_element_type=F32)
    p = jax.nn.softmax(sc, axis=-1).astype(BF16)
    attn_out = jnp.einsum("bngs,bsnd->bngd", p, v, preferred_element_type=F32).astype(BF16)
    attn_out = attn_out.reshape(B, q_dim)
    x_last = x[:, -1, :] + _psum(linear(attn_out, lw["w_o"], lw["s_o"], lw["a_o"], "bd,dh->bh", qm["o"]),
                                 tp_axis)
    gu = linear_rmsnorm(x_last, lw["w_gu"], lw["s_gu"], lw["a_gu"], eps, "bh,hcm->bcm", qm["gu"])
    hidden = jax.nn.silu(gu[:, 0, :]) * gu[:, 1, :]
    return x_last + _psum(linear(hidden, lw["w_down"], lw["s_down"], lw["a_down"], "bm,mh->bh", qm["down"],
                                 dynamic_act=qm["down"] in ("int8", "fp8")), tp_axis)


LAYER_KEYS = ("w_qkv", "w_o", "w_gu", "w_down", "s_qkv", "s_o", "s_gu", "s_down",
              "a_qkv", "a_o", "a_gu", "a_down")


def _core_forward_single(ids, embed, cos_tab, sin_tab, final_norm, weights, *, quant, n_q, n_kv, head_dim,
                         eps, attn, tp_axis=None, last_weights=None):
    """Single-chunk core forward: ids (b, S) int32 -> final-normed last-token hidden (b, H) bf16."""
    b, seq = ids.shape
    tot = b * seq
    x = jnp.take(embed, ids, axis=0).astype(BF16)
    if tot >= 1024 and tot % 1024 == 0 and seq != 1024:
        x = x.reshape(tot // 1024, 1024, -1)
    cos = cos_tab[:seq].astype(BF16)
    sin = sin_tab[:seq].astype(BF16)
    perm = rope_perm_matrix(head_dim)
    stacked = dict(zip(LAYER_KEYS, weights))
    is_quant = quant != "bf16"
    scan_keys = [k for k in LAYER_KEYS if k.startswith("w_") or is_quant]
    static_lw = {k: stacked[k] for k in LAYER_KEYS if k not in scan_keys}
    scan_xs = {k: stacked[k] for k in scan_keys}

    kw = dict(eps=eps, n_q=n_q, n_kv=n_kv, hd=head_dim, quant=quant, tp_axis=tp_axis)

    if tp_axis is None and {"fmlp", "fqkv"} & quant_flags(quant):
        # Fused Pallas kernels index the (L-1, ...) weight stacks by layer in-kernel.
        n_scan = stacked["w_qkv"].shape[0]

        def body(carry, xs):
            idx, lw_slice = xs
            lw = {**static_lw, **lw_slice}
            return layer_step(carry, lw, cos, sin, perm, attn=attn, b_orig=b, s_orig=seq, layer=idx,
                              stacked=stacked, **kw), None

        x, _ = jax.lax.scan(body, x, (jnp.arange(n_scan, dtype=jnp.int32), scan_xs))
    else:
        def body(carry, lw_slice):
            lw = {**static_lw, **lw_slice}
            return layer_step(carry, lw, cos, sin, perm, attn=attn, b_orig=b, s_orig=seq, **kw), None

        x, _ = jax.lax.scan(body, x, scan_xs)
    if x.shape[:2] != (b, seq):
        x = x.reshape(b, seq, -1)
    dummy = jnp.zeros((1,), F32)
    lw = dict(zip(("w_qkv", "w_o", "w_gu", "w_down"), last_weights))
    lw.update({k: dummy for k in LAYER_KEYS if not k.startswith("w_")})
    x_last = last_layer_step(x, lw, cos, sin, perm, **dict(kw, quant="bf16"))
    return (rms_norm_unit(x_last, eps) * final_norm.astype(BF16)).astype(BF16)


def core_forward(ids, embed, cos_tab, sin_tab, final_norm, weights, *, quant, n_q, n_kv, head_dim,
                 eps, attn, tp_axis=None, last_weights=None):
    """ids: (b, S) int32 -> final-normed last-token hidden (b, H) bf16.

    weights holds the pre-sliced L-1 scan layers (no runtime HBM slice); last_weights holds the
    bf16 weights (wl_qkv, wl_o, wl_gu, wl_down) for layer L-1 (last-token-only).
    When b * S exceeds MINICPM5_MK_CHUNK_TOKENS (default 32,768 tokens = 4 x 8,192), sequences
    are processed in on-device microbatches via jax.lax.scan so peak HBM stays bounded."""
    import os
    b, seq = ids.shape
    max_chunk_tokens = int(os.environ.get("MINICPM5_MK_CHUNK_TOKENS", "32768"))
    if b > 1 and b * seq > max_chunk_tokens and tp_axis is None:
        b_chunk = max(1, max_chunk_tokens // seq)
        while b % b_chunk != 0 and b_chunk > 1:
            b_chunk -= 1
        if b_chunk < b:
            n_chunks = b // b_chunk
            ids_3d = ids.reshape(n_chunks, b_chunk, seq)

            def _chunk_step(carry, sub_ids):
                h_sub = _core_forward_single(
                    sub_ids, embed, cos_tab, sin_tab, final_norm, weights,
                    quant=quant, n_q=n_q, n_kv=n_kv, head_dim=head_dim,
                    eps=eps, attn=attn, tp_axis=tp_axis, last_weights=last_weights,
                )
                return carry, h_sub

            _, h_3d = jax.lax.scan(_chunk_step, None, ids_3d)
            return h_3d.reshape(b, -1)
    return _core_forward_single(
        ids, embed, cos_tab, sin_tab, final_norm, weights,
        quant=quant, n_q=n_q, n_kv=n_kv, head_dim=head_dim,
        eps=eps, attn=attn, tp_axis=tp_axis, last_weights=last_weights,
    )


def pad_reqs(h: jax.Array, bs: int, min_reqs: int = 8) -> jax.Array:
    """(bs, H) -> (max(min_reqs, bs), H) with h in the first bs rows, zeros after."""
    if bs >= min_reqs:
        return h
    return jnp.concatenate([h, jnp.zeros((min_reqs - bs, h.shape[-1]), h.dtype)], axis=0)


def scatter_last(h, bs: int, seq: int, t_pad: int):
    """(bs, H) -> (t_pad, H) with h[b] at row b*seq + seq - 1, zeros elsewhere."""
    H = h.shape[-1]
    out = jnp.zeros((bs, seq, H), BF16).at[:, seq - 1, :].set(h).reshape(bs * seq, H)
    if t_pad > bs * seq:
        out = jnp.concatenate([out, jnp.zeros((t_pad - bs * seq, H), BF16)], axis=0)
    return out


def mk_forward(
    input_ids: jax.Array,
    embed: jax.Array,
    cos_tab: jax.Array,
    sin_tab: jax.Array,
    final_norm: jax.Array,
    w_qkv: jax.Array,
    w_o: jax.Array,
    w_gu: jax.Array,
    w_down: jax.Array,
    s_qkv: jax.Array,
    s_o: jax.Array,
    s_gu: jax.Array,
    s_down: jax.Array,
    a_qkv: jax.Array,
    a_o: jax.Array,
    a_gu: jax.Array,
    a_down: jax.Array,
    wl_qkv: jax.Array,
    wl_o: jax.Array,
    wl_gu: jax.Array,
    wl_down: jax.Array,
    bs: int,
    seq: int,
    quant: str,
    n_q: int,
    n_kv: int,
    head_dim: int,
    eps: float,
    attn: str,
) -> jax.Array:
    """Whole-model prefill for `bs` equal-length fresh sequences of length `seq` (one core).

    input_ids: (T_pad,) with the first bs*seq tokens laid out sequence-major.
    Returns (max(8, bs), H) bf16: row b holds final_norm(h_last[b]).
    """
    ids = input_ids[: bs * seq].astype(jnp.int32).reshape(bs, seq)
    weights = (w_qkv, w_o, w_gu, w_down, s_qkv, s_o, s_gu, s_down, a_qkv, a_o, a_gu, a_down)
    h = core_forward(ids, embed, cos_tab, sin_tab, final_norm, weights, quant=quant, n_q=n_q,
                     n_kv=n_kv, head_dim=head_dim, eps=eps, attn=attn,
                     last_weights=(wl_qkv, wl_o, wl_gu, wl_down))
    return pad_reqs(h, bs)


def tp_weight_specs(axis: str, quant: str):
    """PartitionSpecs of the 12 stacked layer arrays for the TP-inside (head/intermediate) shard."""
    from jax.sharding import PartitionSpec as P

    w = (P(None, None, axis), P(None, axis, None), P(None, None, None, axis), P(None, axis, None))
    if quant == "bf16":
        return w + (P(),) * 8
    qm = parse_quant(quant)
    s_specs = {"qkv": P(None, axis), "o": P(), "gu": P(None, None, axis), "down": P()}
    a_specs = {"qkv": P(), "o": P(None, axis), "gu": P(), "down": P(None, axis)}
    s = tuple(s_specs[k] if qm[k] != "bf16" else P() for k in LINEARS)
    a = tuple(a_specs[k] if qm[k] != "bf16" else P() for k in LINEARS)
    return w + s + a


def tp_last_specs(axis: str, quant: str):
    """PartitionSpecs of the 4 bf16 last-layer arrays."""
    from jax.sharding import PartitionSpec as P

    return (P(None, axis), P(axis, None), P(None, None, axis), P(axis, None))


def make_multicore_forward(mesh, axis: str, mode: str, quant: str):
    """Builds a 2-core forward over `mesh` (vLLM TP world).

    mode "dp": weights replicated; input_ids sharded over cores (each rank passes the token ids of
               its bs/ncores sequences, sequence-major), last-token hidden states are
               all-gathered (one tiny collective per forward). Output is (max(8, bs), H).
               (lax.axis_index is avoided: TorchTPU's SPMD compile rejects partition-id.)
    mode "tp": layer weights are rank-major head / intermediate shards; O and down partial sums
               are psum'ed per layer (for bs < ncores, e.g. (1, 8192)). Output is (max(8, bs), H).
    Output is replicated.
    """
    from jax.sharding import PartitionSpec as P

    ncores = mesh.shape[axis]
    rep = P()
    if mode == "sp":
        return _make_sp_forward(mesh, axis, ncores)
    if mode == "dp":
        in_specs = (P(axis), rep, rep, rep, rep) + (rep,) * 16
    else:
        in_specs = ((rep, rep, rep, rep, rep) + tuple(tp_weight_specs(axis, quant))
                    + tuple(tp_last_specs(axis, quant)))

    def fwd(
        input_ids: jax.Array,
        embed: jax.Array,
        cos_tab: jax.Array,
        sin_tab: jax.Array,
        final_norm: jax.Array,
        w_qkv: jax.Array,
        w_o: jax.Array,
        w_gu: jax.Array,
        w_down: jax.Array,
        s_qkv: jax.Array,
        s_o: jax.Array,
        s_gu: jax.Array,
        s_down: jax.Array,
        a_qkv: jax.Array,
        a_o: jax.Array,
        a_gu: jax.Array,
        a_down: jax.Array,
        wl_qkv: jax.Array,
        wl_o: jax.Array,
        wl_gu: jax.Array,
        wl_down: jax.Array,
        bs: int,
        seq: int,
        quant: str,
        n_q: int,
        n_kv: int,
        head_dim: int,
        eps: float,
        attn: str,
    ) -> jax.Array:
        def body(ids_all, embed, cos_tab, sin_tab, final_norm, *ws):
            weights, last = ws[:12], ws[12:]
            if mode == "dp":
                per = bs // ncores
                ids_l = ids_all[: per * seq].astype(jnp.int32).reshape(per, seq)  # this core's seqs
                h_l = core_forward(ids_l, embed, cos_tab, sin_tab, final_norm, weights, quant=quant,
                                   n_q=n_q, n_kv=n_kv, head_dim=head_dim, eps=eps, attn=attn,
                                   last_weights=last)
                h = jax.lax.all_gather(h_l, axis, axis=0, tiled=True)
                return pad_reqs(h, bs)
            ids = ids_all[: bs * seq].astype(jnp.int32).reshape(bs, seq)
            h = core_forward(ids, embed, cos_tab, sin_tab, final_norm, weights, quant=quant,
                             n_q=n_q // ncores, n_kv=n_kv // ncores, head_dim=head_dim, eps=eps,
                             attn=attn, tp_axis=axis, last_weights=last)
            return pad_reqs(h, bs)

        return jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=rep, check_vma=False)(
            input_ids, embed, cos_tab, sin_tab, final_norm, w_qkv, w_o, w_gu, w_down, s_qkv, s_o, s_gu,
            s_down, a_qkv, a_o, a_gu, a_down, wl_qkv, wl_o, wl_gu, wl_down)

    fwd.__name__ = f"mk_forward_{mode}{ncores}"
    return fwd, in_specs


def _make_sp_forward(mesh, axis: str, ncores: int):
    """Sequence-parallel 1-sequence forward over the cores of `mesh` (see core_forward_sp). The first
    operand is a (ncores,) int32 iota sharded over the cores: each core reads its rank from it
    (lax.axis_index / partition-id is rejected by TorchTPU's SPMD compile)."""
    from jax.sharding import PartitionSpec as P

    assert ncores == 2, ncores
    rep = P()
    in_specs = (P(axis), rep, rep, rep, rep, rep) + (rep,) * 16

    def fwd(
        rank_ids: jax.Array,
        input_ids: jax.Array,
        embed: jax.Array,
        cos_tab: jax.Array,
        sin_tab: jax.Array,
        final_norm: jax.Array,
        w_qkv: jax.Array,
        w_o: jax.Array,
        w_gu: jax.Array,
        w_down: jax.Array,
        s_qkv: jax.Array,
        s_o: jax.Array,
        s_gu: jax.Array,
        s_down: jax.Array,
        a_qkv: jax.Array,
        a_o: jax.Array,
        a_gu: jax.Array,
        a_down: jax.Array,
        wl_qkv: jax.Array,
        wl_o: jax.Array,
        wl_gu: jax.Array,
        wl_down: jax.Array,
        bs: int,
        seq: int,
        quant: str,
        n_q: int,
        n_kv: int,
        head_dim: int,
        eps: float,
        attn: str,
    ) -> jax.Array:
        def body(rank_l, ids_all, embed, cos_tab, sin_tab, final_norm, *ws):
            weights, last = ws[:12], ws[12:]
            ids = ids_all[:seq].astype(jnp.int32)
            h = core_forward_sp(ids, rank_l[0].astype(jnp.int32), embed, cos_tab, sin_tab, final_norm, weights,
                                quant=quant, n_q=n_q, n_kv=n_kv, head_dim=head_dim, eps=eps, axis=axis,
                                last_weights=last)
            return pad_reqs(h, bs)

        return jax.shard_map(body, mesh=mesh, in_specs=in_specs, out_specs=rep, check_vma=False)(
            rank_ids, input_ids, embed, cos_tab, sin_tab, final_norm, w_qkv, w_o, w_gu, w_down, s_qkv, s_o,
            s_gu, s_down, a_qkv, a_o, a_gu, a_down, wl_qkv, wl_o, wl_gu, wl_down)

    fwd.__name__ = f"mk_forward_sp{ncores}"
    return fwd, in_specs
