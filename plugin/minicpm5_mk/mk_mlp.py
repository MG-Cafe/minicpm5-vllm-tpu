# Fused MLP Pallas kernel for the MiniCPM5 megakernel (TPU v6e / TPU7x).
"""Fused RMSNorm + dynamic W8A8 gate/up + SiLU-gate + down + residual, one Pallas kernel per layer.

x_out = x + down( silu(g) * u ),  [g | u] = int8( rmsnorm(x) * a_gu ) @ W_gu  (per-token dynamic scale)

* Weights stay resident in VMEM for the whole token sweep (single-buffered); tokens stream through in
  tm-row tiles, so the (T, 2I) gate/up activation and the (T, I) SiLU product never touch HBM.
* W_gu is stored as (H, 2I) int8 with g/u interleaved per ic-column chunk ([g_0 | u_0 | g_1 | u_1 ...]);
  s_gu (1, 2I) f32 holds the matching per-output-channel dequant scales.
* down: bf16 weights (default) or int8 with per-token dynamic quantization of the SiLU product
  (SmoothQuant channel factors a_down folded into the activation).
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.bfloat16
F32 = jnp.float32


def _is_7x() -> bool:
    import os

    try:
        kind = jax.devices()[0].device_kind.lower()
    except Exception:  # noqa: BLE001
        kind = ""
    return "7" in kind or "7" in os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()


def _vmem_limit(default_mb: int) -> int:
    return min(default_mb, 60 if _is_7x() else 110) * 1024 * 1024


def _wspec(arr, block, one):
    """BlockSpec for a weight operand: if arr has one more dim than block it is a per-layer stack
    (L, *block) and the layer is selected by the scalar-prefetched index (no XLA slice copy)."""
    if arr.ndim == len(block) + 1:
        nd = len(block)
        return pl.BlockSpec((None,) + tuple(block), lambda i, l, nd=nd: (l[0],) + (0,) * nd, **one)
    assert arr.ndim == len(block), (arr.shape, block)
    nd = len(block)
    return pl.BlockSpec(tuple(block), lambda i, l, nd=nd: (0,) * nd, **one)


def _row(a, n, layered):
    """Per-channel vector -> (L, 1, n) when layered (stack of L), else (1, n); None -> ones (1, n)."""
    if a is None:
        return jnp.ones((1, n), F32)
    if layered:
        return a.reshape(a.shape[0], 1, n)
    return a.reshape(1, n)


def mlp_chunk(I: int) -> int:
    """Intermediate column chunk used for the interleaved W_gu layout."""
    for ic in (1536, 1152, 1024, 768, 512):
        if I % ic == 0:
            return ic
    return I


def _fused_mlp_kernel(l_ref, x_ref, wgu_ref, sgu_ref, agu_ref, wd_ref, sd_ref, ad_ref, o_ref, h_sc, *,
                      ic, n_ic, eps, d8, residual):
    x = x_ref[...].astype(F32)
    var = jnp.mean(x * x, axis=1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    u = x * agu_ref[...]
    amax = jnp.max(jnp.abs(u), axis=1, keepdims=True)
    s_u = jnp.maximum(amax * (1.0 / 127.0), 1e-6)
    xq = jnp.round(u * (1.0 / s_u)).astype(jnp.int8)
    sx = s_u * rstd
    acc = None
    for c in range(n_ic):
        cols = slice(c * 2 * ic, (c + 1) * 2 * ic)
        gu = jax.lax.dot(xq, wgu_ref[:, cols], preferred_element_type=jnp.int32)
        gu = gu.astype(F32) * sx * sgu_ref[:, cols]
        g = gu[:, :ic]
        h = g * jax.nn.sigmoid(g) * gu[:, ic:]
        if d8:
            h_sc[:, c * ic:(c + 1) * ic] = h * ad_ref[:, c * ic:(c + 1) * ic]
        else:
            part = jax.lax.dot(h.astype(BF16), wd_ref[c * ic:(c + 1) * ic, :], preferred_element_type=F32)
            acc = part if acc is None else acc + part
    if d8:
        hs = h_sc[...]
        hmax = jnp.max(jnp.abs(hs), axis=1, keepdims=True)
        s_h = jnp.maximum(hmax * (1.0 / 127.0), 1e-6)
        hq = jnp.round(hs * (1.0 / s_h)).astype(jnp.int8)
        acc = jax.lax.dot(hq, wd_ref[...], preferred_element_type=jnp.int32).astype(F32) * (s_h * sd_ref[...])
    out = x + acc if residual else acc
    o_ref[...] = out.astype(o_ref.dtype)


@functools.partial(jax.named_call, name="fused_mlp")
def fused_mlp(x: jax.Array, w_gu: jax.Array, s_gu: jax.Array, a_gu: jax.Array, w_down: jax.Array,
              s_down: jax.Array | None = None, a_down: jax.Array | None = None, *, eps: float,
              residual: bool = True, tm: int = 512, layer: jax.Array | None = None,
              vmem_limit_bytes: int | None = None) -> jax.Array:
    """x: (T, H) bf16. Per layer: w_gu (H, 2I) int8 (chunk-interleaved), s_gu (2I,) f32, a_gu (H,) f32,
    w_down (I, H) bf16 or int8 (+ s_down (H,), a_down (I,)). With `layer` (int32 scalar) every weight
    argument is the full (L, ...) stack and the kernel DMAs layer `layer` directly. Returns (T, H) bf16."""
    T, H = x.shape
    lay = layer is not None
    I = w_down.shape[-2]
    ic = mlp_chunk(I)
    n_ic = I // ic
    d8 = w_down.dtype == jnp.int8
    s_gu = _row(s_gu, 2 * I, lay)
    a_gu = _row(a_gu, H, lay)
    s_down = _row(s_down if d8 else None, H, lay and d8)
    a_down = _row(a_down if d8 else None, I, lay and d8)
    tm = min(tm, T)
    assert T % tm == 0, (T, tm)
    kern = functools.partial(_fused_mlp_kernel, ic=ic, n_ic=n_ic, eps=eps, d8=d8, residual=residual)
    vmem_limit_bytes = vmem_limit_bytes or _vmem_limit(100)
    one = dict(pipeline_mode=pl.Buffered(1))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(T // tm,),
        in_specs=[
            pl.BlockSpec((tm, H), lambda i, l: (i, 0)),
            _wspec(w_gu, (H, 2 * I), one),
            _wspec(s_gu, (1, 2 * I), one),
            _wspec(a_gu, (1, H), one),
            _wspec(w_down, (I, H), one),
            _wspec(s_down, (1, H), one),
            _wspec(a_down, (1, I), one),
        ],
        out_specs=pl.BlockSpec((tm, H), lambda i, l: (i, 0)),
        scratch_shapes=[pltpu.VMEM((tm, I) if d8 else (8, 128), F32)],
    )
    lidx = (layer if lay else jnp.zeros((), jnp.int32)).astype(jnp.int32).reshape(1)
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((T, H), x.dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",),
                                             vmem_limit_bytes=vmem_limit_bytes),
    )(lidx, x, w_gu, s_gu, a_gu, w_down, s_down, a_down)


def interleave_gu(w_gu, s_gu, ic: int):
    """torch/numpy-agnostic helper used by mk_weights: (.., H, 2, I) -> (.., H, 2I) chunk-interleaved,
    s (.., 2, I) -> (.., 1, 2I)."""
    import torch

    *lead, H, two, I = w_gu.shape
    assert two == 2 and I % ic == 0
    n = I // ic
    w = w_gu.reshape(*lead, H, 2, n, ic).movedim(-3, -2).reshape(*lead, H, 2 * I)  # (.., H, n, 2, ic)
    s = s_gu.reshape(*lead, 2, n, ic).movedim(-3, -2).reshape(*lead, 1, 2 * I)
    return w.contiguous(), s.to(torch.float32).contiguous()


def _fused_qkv_kernel(l_ref, x_ref, w_ref, s_ref, a_ref, o_ref, *, eps):
    x = x_ref[...].astype(F32)
    var = jnp.mean(x * x, axis=1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    u = x * a_ref[...]
    amax = jnp.max(jnp.abs(u), axis=1, keepdims=True)
    s_u = jnp.maximum(amax * (1.0 / 127.0), 1e-6)
    xq = jnp.round(u * (1.0 / s_u)).astype(jnp.int8)
    acc = jax.lax.dot(xq, w_ref[...], preferred_element_type=jnp.int32).astype(F32)
    o_ref[...] = (acc * (s_u * rstd) * s_ref[...]).astype(o_ref.dtype)


@functools.partial(jax.named_call, name="fused_qkv")
def fused_qkv(x: jax.Array, w: jax.Array, s_w: jax.Array, a_x: jax.Array, *, eps: float, tm: int = 512,
              layer: jax.Array | None = None, vmem_limit_bytes: int | None = None) -> jax.Array:
    """Fused RMSNorm + per-token dynamic int8 quantization + int8 QKV projection + dequant.
    x: (T, H) bf16; per layer w: (H, D) int8, s_w: (D,) f32, a_x: (H,) f32 (gamma / SmoothQuant).
    With `layer`, weights are (L, ...) stacks indexed in-kernel. Returns (T, D) bf16."""
    T, H = x.shape
    lay = layer is not None
    D = w.shape[-1]
    s_w = _row(s_w, D, lay)
    a_x = _row(a_x, H, lay)
    tm = min(tm, T)
    assert T % tm == 0, (T, tm)
    one = dict(pipeline_mode=pl.Buffered(1))
    vmem_limit_bytes = vmem_limit_bytes or _vmem_limit(48)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(T // tm,),
        in_specs=[
            pl.BlockSpec((tm, H), lambda i, l: (i, 0)),
            _wspec(w, (H, D), one),
            _wspec(s_w, (1, D), one),
            _wspec(a_x, (1, H), one),
        ],
        out_specs=pl.BlockSpec((tm, D), lambda i, l: (i, 0)),
    )
    lidx = (layer if lay else jnp.zeros((), jnp.int32)).astype(jnp.int32).reshape(1)
    return pl.pallas_call(
        functools.partial(_fused_qkv_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct((T, D), x.dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",),
                                             vmem_limit_bytes=vmem_limit_bytes),
    )(lidx, x, w, s_w, a_x)

