# SuperAttention + Causal GQA Flash Attention (prefill, no KV cache) for the MiniCPM5 megakernel.
#
# Integrates the core mechanisms of SuperAttention into the MiniCPM5 megakernel:
# 1. Fused in-VMEM MXU RoPE + native 3D token-major QKV layout (zero external 4D RoPE, zero HBM transposes).
# 2. GQA query-head packing: all g = n_q / n_kv query heads per KV group are stacked into the MXU M
#    dimension ((g * bq, D)), so each K/V block is fetched once per KV head instead of g times.
# 3. VMEM-staged q_rope_sc + k_rope_sc with M_c sub-tile VPU/MXU software pipelining (matching
#    SuperAttention's _run_c_step) so live VREG pressure stays within hardware registers (zero VMEM spills).
# 4. Dynamic-bound two-phase causal KV loop (@pl.loop(0, diag_k) unmasked interior fast-path +
#    diagonal block), skipping 100% of upper-triangle KV blocks and skipping mask generation
#    (broadcasted_iota + jnp.where) on all interior blocks.
# 5. Hardware approximate reciprocal (pl.reciprocal(..., approx=True)) in the output epilogue.
"""Pallas TPU SuperAttention causal GQA kernels for MiniCPM5."""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.bfloat16
F32 = jnp.float32
NEG_INF = -1e30
LANES = 128


def _super_gqa_fused_kernel(
    qkv_ref,
    cos_ref,
    sin_ref,
    perm_ref,
    o_ref,
    k_rope_sc,
    m_sc,
    l_sc,
    acc_sc,
    *,
    bq: int,
    bkv: int,
    n_kv: int,
    g: int,
    hd: int,
    sub_pipe: bool,
    sub2: bool,
    exp_bf16: bool,
):
    qi = pl.program_id(1)
    M = g * bq
    q_dim = n_kv * g * hd
    kv_dim = n_kv * hd
    q_start = qi * bq
    diag_k = q_start // bkv
    last_k = (q_start + bq - 1) // bkv
    prot = perm_ref[...]

    # 1. Compute k_rope for all S tokens and all n_kv heads once at qi == 0 and cache in VMEM.
    @pl.when(qi == 0)
    def _init_k_rope():
        c_all = cos_ref[...]
        s_all = sin_ref[...]
        for h in range(n_kv):
            kr = qkv_ref[:, q_dim + h * hd : q_dim + (h + 1) * hd]
            k_rot = jax.lax.dot_general(
                kr, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32
            ).astype(BF16)
            k_rope_sc[:, h * hd : (h + 1) * hd] = kr * c_all + k_rot * s_all

    # Sub-tile M = g * bq only when requested (sub2), when n_kv == 1 with sub_pipe, or when M > 2560.
    n_sub = 2 if (sub2 or (sub_pipe and n_kv == 1) or M > 2560) else 1
    M_c = M // n_sub

    # 2. Load and stack the g query heads for each KV head h as (g * bq, hd), apply in-VMEM MXU RoPE,
    #    and keep in VREG chunks so the KV loop never reloads Q from VMEM.
    q_all = qkv_ref[pl.ds(q_start, bq), :q_dim]
    c_q = cos_ref[pl.ds(q_start, bq), :]
    s_q = sin_ref[pl.ds(q_start, bq), :]
    c_gs = jnp.tile(c_q, (g, 1))
    s_gs = jnp.tile(s_q, (g, 1))
    q_chunks = []
    for h in range(n_kv):
        base_h = h * g
        qs = [q_all[:, (base_h + j) * hd : (base_h + j + 1) * hd] for j in range(g)]
        q_st = jnp.concatenate(qs, axis=0)
        q_rot = jax.lax.dot_general(
            q_st, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32
        ).astype(BF16)
        q_rope = q_st * c_gs + q_rot * s_gs
        if n_sub == 1:
            q_chunks.append(q_rope)
        else:
            for sub_i in range(n_sub):
                q_chunks.append(q_rope[sub_i * M_c : (sub_i + 1) * M_c, :])

    m_sc[...] = jnp.full(m_sc.shape, NEG_INF, F32)
    l_sc[...] = jnp.zeros(l_sc.shape, F32)
    acc_sc[...] = jnp.zeros(acc_sc.shape, F32)

    def _qk_softmax_sub(h: int, sub_i: int, k_h, v_h, ki, masked: bool):
        r_off = h * M + sub_i * M_c
        rows = pl.ds(r_off, M_c)
        q_c = q_chunks[h * n_sub + sub_i]
        s = jax.lax.dot_general(q_c, k_h, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        if masked:
            if n_sub == 1:
                row_tok = q_start + jax.lax.broadcasted_iota(jnp.int32, (M_c, bkv), 0) % bq
            else:
                row_in_m = sub_i * M_c + jax.lax.broadcasted_iota(jnp.int32, (M_c, bkv), 0)
                row_tok = q_start + jax.lax.rem(row_in_m, bq)
            col_tok = ki * bkv + jax.lax.broadcasted_iota(jnp.int32, (M_c, bkv), 1)
            s = jnp.where(col_tok <= row_tok, s, NEG_INF)
        m_prev = m_sc[rows, :]
        m_cur = jnp.max(s, axis=1, keepdims=True)
        m_next = jnp.maximum(m_prev, m_cur)
        shifted = s - jnp.tile(m_next, (1, bkv // LANES))
        if exp_bf16:
            p = jnp.exp(shifted.astype(BF16))
            p_sum = jnp.sum(p.astype(F32), axis=1, keepdims=True)
            p_mm = p
        else:
            p = jnp.exp(shifted)
            p_sum = jnp.sum(p, axis=1, keepdims=True)
            p_mm = p.astype(v_h.dtype)
        alpha = jnp.exp(m_prev - m_next)
        l_sc[rows, :] = alpha * l_sc[rows, :] + p_sum
        m_sc[rows, :] = m_next
        return p_mm, v_h, alpha, rows

    def _pv_update(p_mm, v_h, alpha, rows):
        pv = jax.lax.dot(p_mm, v_h, preferred_element_type=F32)
        acc_sc[rows, :] = acc_sc[rows, :] * jnp.tile(alpha, (1, hd // LANES)) + pv

    def _step(ki, masked: bool):
        if sub_pipe and (n_kv * n_sub > 1):
            # SuperAttention _run_c_step software pipelining across (h, sub_i):
            # overlap QK+softmax of step t with PV matmul of step t-1.
            prev = None
            for h in range(n_kv):
                k_h = k_rope_sc[pl.ds(ki * bkv, bkv), h * hd : (h + 1) * hd]
                v_h = qkv_ref[pl.ds(ki * bkv, bkv), q_dim + kv_dim + h * hd : q_dim + kv_dim + (h + 1) * hd]
                for sub_i in range(n_sub):
                    cur = _qk_softmax_sub(h, sub_i, k_h, v_h, ki, masked)
                    if prev is not None:
                        _pv_update(*prev)
                    prev = cur
            _pv_update(*prev)
        else:
            for h in range(n_kv):
                k_h = k_rope_sc[pl.ds(ki * bkv, bkv), h * hd : (h + 1) * hd]
                v_h = qkv_ref[pl.ds(ki * bkv, bkv), q_dim + kv_dim + h * hd : q_dim + kv_dim + (h + 1) * hd]
                for sub_i in range(n_sub):
                    cur = _qk_softmax_sub(h, sub_i, k_h, v_h, ki, masked)
                    _pv_update(*cur)

    # Phase 1: Unmasked interior KV blocks (ki < diag_k)
    @pl.loop(0, diag_k)
    def _interior_loop(ki):
        _step(ki, False)

    # Phase 2: Diagonal / boundary KV blocks (diag_k <= ki <= last_k)
    if bkv % bq == 0:
        _step(diag_k, True)
    else:
        @pl.loop(diag_k, last_k + 1)
        def _diag_loop(ki):
            _step(ki, True)

    # 3. Epilogue with SuperAttention hardware approximate reciprocal
    inv = pltpu.reciprocal(l_sc[...], approx=True)
    o_all = (acc_sc[...] * jnp.tile(inv, (1, hd // LANES))).astype(o_ref.dtype)
    for h in range(n_kv):
        base_h = h * g
        for j in range(g):
            r0 = (h * g + j) * bq
            o_ref[:, (base_h + j) * hd : (base_h + j + 1) * hd] = o_all[r0 : r0 + bq, :]


def _super_gqa_split_kernel(
    q_ref,
    kv_ref,
    cos_ref,
    sin_ref,
    perm_ref,
    o_ref,
    k_rope_sc,
    m_sc,
    l_sc,
    acc_sc,
    *,
    bq: int,
    bkv: int,
    n_kv: int,
    g: int,
    hd: int,
    sub_pipe: bool,
    sub2: bool,
    exp_bf16: bool,
):
    qi = pl.program_id(1)
    M = g * bq
    q_dim = n_kv * g * hd
    kv_dim = n_kv * hd
    q_start = qi * bq
    diag_k = q_start // bkv
    last_k = (q_start + bq - 1) // bkv
    prot = perm_ref[...]

    @pl.when(qi == 0)
    def _init_k_rope():
        c_all = cos_ref[...]
        s_all = sin_ref[...]
        for h in range(n_kv):
            kr = kv_ref[:, h * hd : (h + 1) * hd]
            k_rot = jax.lax.dot_general(
                kr, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32
            ).astype(BF16)
            k_rope_sc[:, h * hd : (h + 1) * hd] = kr * c_all + k_rot * s_all

    n_sub = 2 if (sub2 or (sub_pipe and n_kv == 1) or M > 2560) else 1
    M_c = M // n_sub

    q_all = q_ref[:, :q_dim]
    c_q = cos_ref[pl.ds(q_start, bq), :]
    s_q = sin_ref[pl.ds(q_start, bq), :]
    c_gs = jnp.tile(c_q, (g, 1))
    s_gs = jnp.tile(s_q, (g, 1))
    q_chunks = []
    for h in range(n_kv):
        base_h = h * g
        qs = [q_all[:, (base_h + j) * hd : (base_h + j + 1) * hd] for j in range(g)]
        q_st = jnp.concatenate(qs, axis=0)
        q_rot = jax.lax.dot_general(
            q_st, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32
        ).astype(BF16)
        q_rope = q_st * c_gs + q_rot * s_gs
        if n_sub == 1:
            q_chunks.append(q_rope)
        else:
            for sub_i in range(n_sub):
                q_chunks.append(q_rope[sub_i * M_c : (sub_i + 1) * M_c, :])

    m_sc[...] = jnp.full(m_sc.shape, NEG_INF, F32)
    l_sc[...] = jnp.zeros(l_sc.shape, F32)
    acc_sc[...] = jnp.zeros(acc_sc.shape, F32)

    def _qk_softmax_sub(h: int, sub_i: int, k_h, v_h, ki, masked: bool):
        r_off = h * M + sub_i * M_c
        rows = pl.ds(r_off, M_c)
        q_c = q_chunks[h * n_sub + sub_i]
        s = jax.lax.dot_general(q_c, k_h, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        if masked:
            if n_sub == 1:
                row_tok = q_start + jax.lax.broadcasted_iota(jnp.int32, (M_c, bkv), 0) % bq
            else:
                row_in_m = sub_i * M_c + jax.lax.broadcasted_iota(jnp.int32, (M_c, bkv), 0)
                row_tok = q_start + jax.lax.rem(row_in_m, bq)
            col_tok = ki * bkv + jax.lax.broadcasted_iota(jnp.int32, (M_c, bkv), 1)
            s = jnp.where(col_tok <= row_tok, s, NEG_INF)
        m_prev = m_sc[rows, :]
        m_cur = jnp.max(s, axis=1, keepdims=True)
        m_next = jnp.maximum(m_prev, m_cur)
        shifted = s - jnp.tile(m_next, (1, bkv // LANES))
        if exp_bf16:
            p = jnp.exp(shifted.astype(BF16))
            p_sum = jnp.sum(p.astype(F32), axis=1, keepdims=True)
            p_mm = p
        else:
            p = jnp.exp(shifted)
            p_sum = jnp.sum(p, axis=1, keepdims=True)
            p_mm = p.astype(v_h.dtype)
        alpha = jnp.exp(m_prev - m_next)
        l_sc[rows, :] = alpha * l_sc[rows, :] + p_sum
        m_sc[rows, :] = m_next
        return p_mm, v_h, alpha, rows

    def _pv_update(p_mm, v_h, alpha, rows):
        pv = jax.lax.dot(p_mm, v_h, preferred_element_type=F32)
        acc_sc[rows, :] = acc_sc[rows, :] * jnp.tile(alpha, (1, hd // LANES)) + pv

    def _step(ki, masked: bool):
        if sub_pipe and (n_kv * n_sub > 1):
            prev = None
            for h in range(n_kv):
                k_h = k_rope_sc[pl.ds(ki * bkv, bkv), h * hd : (h + 1) * hd]
                v_h = kv_ref[pl.ds(ki * bkv, bkv), kv_dim + h * hd : kv_dim + (h + 1) * hd]
                for sub_i in range(n_sub):
                    cur = _qk_softmax_sub(h, sub_i, k_h, v_h, ki, masked)
                    if prev is not None:
                        _pv_update(*prev)
                    prev = cur
            _pv_update(*prev)
        else:
            for h in range(n_kv):
                k_h = k_rope_sc[pl.ds(ki * bkv, bkv), h * hd : (h + 1) * hd]
                v_h = kv_ref[pl.ds(ki * bkv, bkv), kv_dim + h * hd : kv_dim + (h + 1) * hd]
                for sub_i in range(n_sub):
                    cur = _qk_softmax_sub(h, sub_i, k_h, v_h, ki, masked)
                    _pv_update(*cur)

    @pl.loop(0, diag_k)
    def _interior_loop(ki):
        _step(ki, False)

    if bkv % bq == 0:
        _step(diag_k, True)
    else:
        @pl.loop(diag_k, last_k + 1)
        def _diag_loop(ki):
            _step(ki, True)

    inv = pltpu.reciprocal(l_sc[...], approx=True)
    o_all = (acc_sc[...] * jnp.tile(inv, (1, hd // LANES))).astype(o_ref.dtype)
    for h in range(n_kv):
        base_h = h * g
        for j in range(g):
            r0 = (h * g + j) * bq
            o_ref[:, (base_h + j) * hd : (base_h + j + 1) * hd] = o_all[r0 : r0 + bq, :]


@functools.partial(jax.named_call, name="super_gqa_fused")
def super_gqa_fused(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    perm: jax.Array,
    n_q: int,
    n_kv: int,
    hd: int,
    *,
    bq: int = 256,
    bkv: int = 512,
    sub_pipe: bool = True,
    sub2: bool = False,
    exp_bf16: bool = False,
    vmem_limit_bytes: int = 120 * 1024 * 1024,
) -> jax.Array:
    """Fused QKV split + in-VMEM MXU RoPE + SuperAttention GQA causal flash attention.

    qkv: (B, S, (n_q + 2*n_kv)*hd) bf16 (sm_scale folded into w_q).
    cos, sin: (S, hd) bf16.
    perm: (hd, hd) bf16.
    Returns: (B, S, n_q * hd) bf16.
    """
    B, S, total_cols = qkv.shape
    g = n_q // n_kv
    bq = min(bq, S)
    bkv = min(bkv, S)
    assert S % bq == 0 and S % bkv == 0 and bkv % LANES == 0 and hd % LANES == 0, (S, bq, bkv, hd)
    nq = S // bq
    M_tot = n_kv * g * bq
    q_dim = n_q * hd
    kv_dim = n_kv * hd

    try:
        dev_kind = jax.devices()[0].device_kind.lower()
    except Exception:
        dev_kind = ""
    accel_env = os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()
    is_7x = (
        "7" in dev_kind
        or "7" in accel_env
        or os.path.exists("/dev/vfio/1")
        or os.path.exists("/dev/accel1")
    )
    if S * total_cols * 2 >= 40 * 1024 * 1024 and is_7x:
        kv = qkv[:, :, q_dim:]
        kern_split = functools.partial(
            _super_gqa_split_kernel,
            bq=bq,
            bkv=bkv,
            n_kv=n_kv,
            g=g,
            hd=hd,
            sub_pipe=sub_pipe,
            sub2=sub2,
            exp_bf16=exp_bf16,
        )
        return pl.pallas_call(
            kern_split,
            out_shape=jax.ShapeDtypeStruct((B, S, q_dim), qkv.dtype),
            grid=(B, nq),
            in_specs=[
                pl.BlockSpec((None, bq, total_cols), lambda b, qi: (b, qi, 0)),
                pl.BlockSpec((None, S, 2 * kv_dim), lambda b, qi: (b, 0, 0)),
                pl.BlockSpec((S, hd), lambda b, qi: (0, 0)),
                pl.BlockSpec((S, hd), lambda b, qi: (0, 0)),
                pl.BlockSpec((hd, hd), lambda b, qi: (0, 0)),
            ],
            out_specs=pl.BlockSpec((None, bq, q_dim), lambda b, qi: (b, qi, 0)),
            scratch_shapes=[
                pltpu.VMEM((S, kv_dim), BF16),
                pltpu.VMEM((M_tot, LANES), F32),
                pltpu.VMEM((M_tot, LANES), F32),
                pltpu.VMEM((M_tot, hd), F32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                vmem_limit_bytes=min(vmem_limit_bytes, 60 * 1024 * 1024),
            ),
        )(qkv, kv, cos, sin, perm)

    kern = functools.partial(
        _super_gqa_fused_kernel,
        bq=bq,
        bkv=bkv,
        n_kv=n_kv,
        g=g,
        hd=hd,
        sub_pipe=sub_pipe,
        sub2=sub2,
        exp_bf16=exp_bf16,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((B, S, q_dim), qkv.dtype),
        grid=(B, nq),
        in_specs=[
            pl.BlockSpec((None, S, total_cols), lambda b, qi: (b, 0, 0)),
            pl.BlockSpec((S, hd), lambda b, qi: (0, 0)),
            pl.BlockSpec((S, hd), lambda b, qi: (0, 0)),
            pl.BlockSpec((hd, hd), lambda b, qi: (0, 0)),
        ],
        out_specs=pl.BlockSpec((None, bq, q_dim), lambda b, qi: (b, qi, 0)),
        scratch_shapes=[
            pltpu.VMEM((S, kv_dim), BF16),
            pltpu.VMEM((M_tot, LANES), F32),
            pltpu.VMEM((M_tot, LANES), F32),
            pltpu.VMEM((M_tot, hd), F32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=min(vmem_limit_bytes, 60 * 1024 * 1024) if is_7x else vmem_limit_bytes,
        ),
    )(qkv, cos, sin, perm)


def _flash_kernel(q_ref, k_ref, v_ref, o_ref, m_sc, l_sc, acc_sc, *, bq, bkv, g, exp_bf16):
    qi = pl.program_id(2)
    ki = pl.program_id(3)
    M = bq * g
    D = acc_sc.shape[-1]
    q_start = qi * bq
    k_start = ki * bkv
    last_k = (q_start + bq - 1) // bkv

    @pl.when(ki == 0)
    def _init():
        m_sc[...] = jnp.full(m_sc.shape, NEG_INF, F32)
        l_sc[...] = jnp.zeros(l_sc.shape, F32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, F32)

    def _step(masked: bool):
        q = q_ref[...].reshape(M, D)
        k = k_ref[...]
        v = v_ref[...]
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        if masked:
            row_tok = q_start + jax.lax.broadcasted_iota(jnp.int32, (M, bkv), 0) // g
            col_tok = k_start + jax.lax.broadcasted_iota(jnp.int32, (M, bkv), 1)
            s = jnp.where(col_tok <= row_tok, s, NEG_INF)
        m_prev = m_sc[...]
        m_cur = jnp.max(s, axis=1, keepdims=True)
        m_next = jnp.maximum(m_prev, m_cur)
        shifted = s - jnp.tile(m_next, (1, bkv // LANES))
        if exp_bf16:
            p = jnp.exp(shifted.astype(jnp.bfloat16))
            p_sum = jnp.sum(p.astype(F32), axis=1, keepdims=True)
            p_mm = p
        else:
            p = jnp.exp(shifted)
            p_sum = jnp.sum(p, axis=1, keepdims=True)
            p_mm = p.astype(v.dtype)
        alpha = jnp.exp(m_prev - m_next)
        l_sc[...] = alpha * l_sc[...] + p_sum
        m_sc[...] = m_next
        pv = jax.lax.dot(p_mm, v, preferred_element_type=F32)
        acc_sc[...] = acc_sc[...] * jnp.tile(alpha, (1, D // LANES)) + pv

    diag = k_start + bkv - 1 > q_start

    @pl.when(jnp.logical_and(ki <= last_k, diag))
    def _masked():
        _step(True)

    @pl.when(jnp.logical_and(ki <= last_k, jnp.logical_not(diag)))
    def _interior():
        _step(False)

    @pl.when(ki == last_k)
    def _finish():
        inv = pl.reciprocal(l_sc[...], approx=True)
        o = acc_sc[...] * jnp.tile(inv, (1, D // LANES))
        o_ref[...] = o.reshape(bq, g, D).astype(o_ref.dtype)


@functools.partial(jax.named_call, name="gqa_flash_causal")
def gqa_flash_causal(q, k, v, *, bq: int = 256, bkv: int = 512, exp_bf16: bool = False,
                     vmem_limit_bytes: int = 64 * 1024 * 1024):
    """q: (B, S, n_kv, g, D) (RoPE applied, sm_scale folded); k, v: (B, n_kv, S, D)."""
    B, S, n_kv, g, D = q.shape
    bq = min(bq, S)
    bkv = min(bkv, S)
    assert S % bq == 0 and S % bkv == 0 and bkv % LANES == 0 and D % LANES == 0, (S, bq, bkv, D)
    nq, nk = S // bq, S // bkv
    M = bq * g

    def kv_map(b, h, qi, ki):
        return (b, h, jnp.minimum(ki, (qi * bq + bq - 1) // bkv), 0)

    kern = functools.partial(_flash_kernel, bq=bq, bkv=bkv, g=g, exp_bf16=exp_bf16)
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(B, n_kv, nq, nk),
        in_specs=[
            pl.BlockSpec((None, bq, None, g, D), lambda b, h, qi, ki: (b, qi, h, 0, 0)),
            pl.BlockSpec((None, None, bkv, D), kv_map),
            pl.BlockSpec((None, None, bkv, D), kv_map),
        ],
        out_specs=pl.BlockSpec((None, bq, None, g, D), lambda b, h, qi, ki: (b, qi, h, 0, 0)),
        scratch_shapes=[
            pltpu.VMEM((M, LANES), F32),
            pltpu.VMEM((M, LANES), F32),
            pltpu.VMEM((M, D), F32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )(q, k, v)


# ----------------------------------------------------------------------------------------------
# SuperAttention v2 (default for S >= 2048): MXU-bound causal GQA prefill kernel.
#
# Differences vs super_gqa_fused (measured on TPU v6e-1, MiniCPM5 16Q/2KV hd=128, S=8192:
# 1.11 -> 0.75 ms per layer per sequence):
# 1. M-chunked inner loop (mc = 256 rows of the g-stacked Q block per MXU pass) with QK^T(t) /
#    PV(t-1) software pipelining across (kv_head, chunk), so the (mc, bkv) f32 score tile streams
#    through vregs instead of round-tripping a (g*bq, bkv) tile through VMEM.
# 2. Softmax row-sum on the MXU: V is staged once per sequence as V' = [V | 1] (S, 2*hd). The MXU
#    is 256 wide and V only fills 128 columns, so P @ V' returns the row-sum l in the second half at
#    no extra MXU cost, accumulated in f32 from exactly the bf16 P used for P @ V.
# 3. exp2 softmax: log2(e) is folded into Q at RoPE time, P = exp2(S - m) computed in bf16 on the
#    v6e/TPU7x bf16 EUP; running max / rescale stay in f32.
# 4. First KV block peeled (no acc / m zero-init, no rescale), additive causal bias precomputed once
#    per sequence for the diagonal block, and diagonal-block chunks only read the keys they can see.
# 5. K (RoPE'd) and V' are built once per sequence (q-block 0) in VMEM; K/V/cos/sin inputs are
#    single-buffered (pl.Buffered(1)) to fit TPU7x's 64 MiB VMEM per TensorCore.
# ----------------------------------------------------------------------------------------------
LOG2E = 1.4426950408889634


def _super_gqa_v2_kernel(q_ref, k_ref, v_ref, cos_ref, sin_ref, perm_ref, o_ref,
                         k_sc, v_sc, q_sc, m_sc, acc_sc, bias_sc, *, bq, mc, n_kv, g, hd, S):
    bkv = bq
    qi = pl.program_id(1)
    M = g * bq
    n_c = M // mc
    n_th = bq // mc
    q_start = qi * bq
    W = 2 * hd
    prot = perm_ref[...]

    @pl.when(qi == 0)
    def _init_kv():
        c_all = cos_ref[...]
        s_all = sin_ref[...]
        for h in range(n_kv):
            kr = k_ref[:, h * hd:(h + 1) * hd]
            k_rot = jax.lax.dot_general(kr, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32)
            k_sc[h] = kr * c_all + k_rot.astype(BF16) * s_all
            v_sc[h, :, :hd] = v_ref[:, h * hd:(h + 1) * hd]
            v_sc[h, :, hd:] = jnp.ones((S, hd), BF16)
        r = jax.lax.broadcasted_iota(jnp.int32, (bq, bkv), 0)
        cc = jax.lax.broadcasted_iota(jnp.int32, (bq, bkv), 1)
        bias_sc[...] = jnp.where(cc <= r, 0.0, NEG_INF).astype(F32)

    c_q = cos_ref[pl.ds(q_start, bq), :].astype(F32)
    s_q = sin_ref[pl.ds(q_start, bq), :].astype(F32)
    for h in range(n_kv):
        for j in range(g):
            col = (h * g + j) * hd
            qh = q_ref[:, col:col + hd]
            q_rot = jax.lax.dot_general(qh, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32)
            q_sc[h, j * bq:(j + 1) * bq, :] = ((qh.astype(F32) * c_q + q_rot * s_q) * LOG2E).astype(BF16)

    def qk_softmax(h, c, ki, masked, first):
        rows = pl.ds(c * mc, mc)
        t0 = (c % n_th) * mc
        kv_len = (t0 + mc) if masked else bkv
        q = q_sc[h, rows, :]
        k = k_sc[h, pl.ds(ki * bkv, kv_len), :]
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        if masked:
            s = s + bias_sc[pl.ds(t0, mc), pl.ds(0, kv_len)]
        m_cur = jnp.max(s, axis=1, keepdims=True)
        if first:
            m_new = jnp.broadcast_to(m_cur, (mc, LANES))
            alpha = None
        else:
            m_prev = m_sc[h, rows, :]
            m_new = jnp.maximum(m_prev, m_cur)
            alpha = jnp.exp2(m_prev - m_new)
        p = jnp.exp2((s - jnp.tile(m_new, (1, kv_len // LANES))).astype(BF16))
        m_sc[h, rows, :] = m_new
        v = v_sc[h, pl.ds(ki * bkv, kv_len), :]
        return p, v, alpha, h, rows

    def pv(p, v, alpha, h, rows, c, last):
        o = jax.lax.dot(p, v, preferred_element_type=F32)
        if alpha is not None:
            o = acc_sc[h, rows, :] * jnp.tile(alpha, (1, W // LANES)) + o
        if last:
            # Diagonal block is the last KV block of every q-block: normalize and write the output
            # straight from vregs (no acc store + reload). Chunk c = (head j, tokens t0 .. t0 + mc).
            j, t0 = c // n_th, (c % n_th) * mc
            col = (h * g + j) * hd
            o_ref[pl.ds(t0, mc), col:col + hd] = (o[:, :hd] * pltpu.reciprocal(o[:, hd:], approx=True)
                                                  ).astype(o_ref.dtype)
        else:
            acc_sc[h, rows, :] = o

    def run_chunks(kis, masked, first):
        """Software-pipelined (QK^T + softmax of chunk t) || (PV of chunk t-1) over the KV blocks in
        `kis` x kv heads x M-chunks."""
        prev = None
        for ki in kis:
            for h in range(n_kv):
                for c in range(n_c):
                    cur = qk_softmax(h, c, ki, masked, first) + (c, masked)
                    if prev is not None:
                        pv(*prev)
                    prev = cur
        pv(*prev)

    @pl.when(qi == 0)
    def _first_block():
        run_chunks((0,), True, True)

    @pl.when(qi > 0)
    def _rest():
        run_chunks((0,), False, True)

        @pl.loop(0, (qi - 1) // 2)
        def _interior_pairs(i):
            ki = 1 + 2 * i
            run_chunks((ki, ki + 1), False, False)

        @pl.when((qi - 1) % 2 == 1)
        def _interior_rem():
            run_chunks((qi - 1,), False, False)

        run_chunks((qi,), True, False)


@functools.partial(jax.named_call, name="super_gqa_v2")
def super_gqa_v2(qkv: jax.Array, cos: jax.Array, sin: jax.Array, perm: jax.Array, n_q: int, n_kv: int,
                 hd: int, *, bq: int = 512, mc: int = 256, vmem_limit_bytes: int | None = None) -> jax.Array:
    """SuperAttention v2 causal GQA prefill: qkv (B, S, (n_q + 2*n_kv)*hd) bf16 with sm_scale folded into
    w_q; cos/sin (S, hd) bf16; perm (hd, hd) bf16 rotate_half matrix. Returns (B, S, n_q*hd) bf16."""
    B, S, total = qkv.shape
    g = n_q // n_kv
    q_dim = n_q * hd
    kv_dim = n_kv * hd
    bq = min(bq, S)
    mc = min(mc, bq)
    M = g * bq
    assert S % bq == 0 and bq % mc == 0 and M % mc == 0 and q_dim % kv_dim == 0 and hd % LANES == 0, (
        S, bq, mc, q_dim, kv_dim)
    W = 2 * hd
    if vmem_limit_bytes is None:
        try:
            dev_kind = jax.devices()[0].device_kind.lower()
        except Exception:  # noqa: BLE001
            dev_kind = ""
        is_7x = "7" in dev_kind or "7" in os.environ.get("TPU_ACCELERATOR_TYPE", "").lower()
        vmem_limit_bytes = (60 if is_7x else 110) * 1024 * 1024
    kern = functools.partial(_super_gqa_v2_kernel, bq=bq, mc=mc, n_kv=n_kv, g=g, hd=hd, S=S)
    kb = q_dim // kv_dim
    one = dict(pipeline_mode=pl.Buffered(1))
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((B, S, q_dim), qkv.dtype),
        grid=(B, S // bq),
        in_specs=[
            pl.BlockSpec((None, bq, total), lambda b, qi: (b, qi, 0)),
            pl.BlockSpec((None, S, kv_dim), lambda b, qi: (b, 0, kb), **one),
            pl.BlockSpec((None, S, kv_dim), lambda b, qi: (b, 0, kb + 1), **one),
            pl.BlockSpec((S, hd), lambda b, qi: (0, 0), **one),
            pl.BlockSpec((S, hd), lambda b, qi: (0, 0), **one),
            pl.BlockSpec((hd, hd), lambda b, qi: (0, 0), **one),
        ],
        out_specs=pl.BlockSpec((None, bq, q_dim), lambda b, qi: (b, qi, 0)),
        scratch_shapes=[
            pltpu.VMEM((n_kv, S, hd), BF16),
            pltpu.VMEM((n_kv, S, W), BF16),
            pltpu.VMEM((n_kv, M, hd), BF16),
            pltpu.VMEM((n_kv, M, LANES), F32),
            pltpu.VMEM((n_kv, M, W), F32),
            pltpu.VMEM((bq, bq), F32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )(qkv, qkv, qkv, cos, sin, perm)


# ----------------------------------------------------------------------------------------------
# SuperAttention v2, sequence-parallel variant (TPU7x batch 1: one sequence over both TensorCores).
#
# Each core owns half of the 512-token query blocks in a zig-zag pattern that balances causal work
# (core 0: blocks 4k, 4k+3; core 1: blocks 4k+1, 4k+2) and receives the full-sequence K/V (all-gathered
# by the caller). The core rank arrives as a scalar-prefetch operand (partition-id is not available
# under TorchTPU SPMD), and the global block index drives RoPE positions and the causal KV range.
# ----------------------------------------------------------------------------------------------
def sp_global_block(i, r):
    """Global query-block index of local block i on core r (zig-zag over groups of 4 blocks)."""
    j = i % 2
    return 4 * (i // 2) + j * (3 - r) + (1 - j) * r


def _super_gqa_v2_sp_kernel(rank_ref, q_ref, k_ref, v_ref, cos_ref, sin_ref, perm_ref, o_ref,
                            k_sc, v_sc, q_sc, m_sc, acc_sc, bias_sc, *, bq, mc, n_kv, g, hd, S):
    bkv = bq
    li = pl.program_id(1)
    gb = sp_global_block(li, rank_ref[0])
    M = g * bq
    n_c = M // mc
    n_th = bq // mc
    q_start = gb * bq
    W = 2 * hd
    prot = perm_ref[...]

    @pl.when(li == 0)
    def _init_kv():
        c_all = cos_ref[...]
        s_all = sin_ref[...]
        for h in range(n_kv):
            kr = k_ref[:, h * hd:(h + 1) * hd]
            k_rot = jax.lax.dot_general(kr, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32)
            k_sc[h] = kr * c_all + k_rot.astype(BF16) * s_all
            v_sc[h, :, :hd] = v_ref[:, h * hd:(h + 1) * hd]
            v_sc[h, :, hd:] = jnp.ones((S, hd), BF16)
        r = jax.lax.broadcasted_iota(jnp.int32, (bq, bkv), 0)
        cc = jax.lax.broadcasted_iota(jnp.int32, (bq, bkv), 1)
        bias_sc[...] = jnp.where(cc <= r, 0.0, NEG_INF).astype(F32)

    c_q = cos_ref[pl.ds(q_start, bq), :].astype(F32)
    s_q = sin_ref[pl.ds(q_start, bq), :].astype(F32)
    for h in range(n_kv):
        for j in range(g):
            col = (h * g + j) * hd
            qh = q_ref[:, col:col + hd]
            q_rot = jax.lax.dot_general(qh, prot, (((1,), (0,)), ((), ())), preferred_element_type=F32)
            q_sc[h, j * bq:(j + 1) * bq, :] = ((qh.astype(F32) * c_q + q_rot * s_q) * LOG2E).astype(BF16)

    def qk_softmax(h, c, ki, masked, first):
        rows = pl.ds(c * mc, mc)
        t0 = (c % n_th) * mc
        kv_len = (t0 + mc) if masked else bkv
        q = q_sc[h, rows, :]
        k = k_sc[h, pl.ds(ki * bkv, kv_len), :]
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=F32)
        if masked:
            s = s + bias_sc[pl.ds(t0, mc), pl.ds(0, kv_len)]
        m_cur = jnp.max(s, axis=1, keepdims=True)
        if first:
            m_new = jnp.broadcast_to(m_cur, (mc, LANES))
            alpha = None
        else:
            m_prev = m_sc[h, rows, :]
            m_new = jnp.maximum(m_prev, m_cur)
            alpha = jnp.exp2(m_prev - m_new)
        p = jnp.exp2((s - jnp.tile(m_new, (1, kv_len // LANES))).astype(BF16))
        m_sc[h, rows, :] = m_new
        v = v_sc[h, pl.ds(ki * bkv, kv_len), :]
        return p, v, alpha, h, rows

    def pv(p, v, alpha, h, rows, c, last):
        o = jax.lax.dot(p, v, preferred_element_type=F32)
        if alpha is not None:
            o = acc_sc[h, rows, :] * jnp.tile(alpha, (1, W // LANES)) + o
        if last:
            j, t0 = c // n_th, (c % n_th) * mc
            col = (h * g + j) * hd
            o_ref[pl.ds(t0, mc), col:col + hd] = (o[:, :hd] * pltpu.reciprocal(o[:, hd:], approx=True)
                                                  ).astype(o_ref.dtype)
        else:
            acc_sc[h, rows, :] = o

    def run_chunks(kis, masked, first):
        prev = None
        for ki in kis:
            for h in range(n_kv):
                for c in range(n_c):
                    cur = qk_softmax(h, c, ki, masked, first) + (c, masked)
                    if prev is not None:
                        pv(*prev)
                    prev = cur
        pv(*prev)

    @pl.when(gb == 0)
    def _first_block():
        run_chunks((0,), True, True)

    @pl.when(gb > 0)
    def _rest():
        run_chunks((0,), False, True)

        @pl.loop(0, (gb - 1) // 2)
        def _interior_pairs(i):
            ki = 1 + 2 * i
            run_chunks((ki, ki + 1), False, False)

        @pl.when((gb - 1) % 2 == 1)
        def _interior_rem():
            run_chunks((gb - 1,), False, False)

        run_chunks((gb,), True, False)


@functools.partial(jax.named_call, name="super_gqa_v2_sp")
def super_gqa_v2_sp(qkv_local: jax.Array, kv_full: jax.Array, cos: jax.Array, sin: jax.Array, perm: jax.Array,
                    rank: jax.Array, n_q: int, n_kv: int, hd: int, *, bq: int = 512, mc: int = 256,
                    vmem_limit_bytes: int | None = None) -> jax.Array:
    """qkv_local: (B, S/2, (n_q + 2 n_kv) hd) this core's zig-zag query blocks (local order);
    kv_full: (B, S, 2 n_kv hd) full-sequence [K | V] in position order; rank: (1,) int32 core index.
    Returns (B, S/2, n_q hd) attention output for the local blocks."""
    B, S_l, total = qkv_local.shape
    S = kv_full.shape[1]
    g = n_q // n_kv
    q_dim = n_q * hd
    kv_dim = n_kv * hd
    M = g * bq
    assert S == 2 * S_l and S % (4 * bq) == 0 and bq % mc == 0, (S, S_l, bq, mc)
    W = 2 * hd
    if vmem_limit_bytes is None:
        vmem_limit_bytes = 60 * 1024 * 1024
    kern = functools.partial(_super_gqa_v2_sp_kernel, bq=bq, mc=mc, n_kv=n_kv, g=g, hd=hd, S=S)
    one = dict(pipeline_mode=pl.Buffered(1))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(B, S_l // bq),
        in_specs=[
            pl.BlockSpec((None, bq, total), lambda b, i, r: (b, i, 0)),
            pl.BlockSpec((None, S, kv_dim), lambda b, i, r: (b, 0, 0), **one),
            pl.BlockSpec((None, S, kv_dim), lambda b, i, r: (b, 0, 1), **one),
            pl.BlockSpec((S, hd), lambda b, i, r: (0, 0), **one),
            pl.BlockSpec((S, hd), lambda b, i, r: (0, 0), **one),
            pl.BlockSpec((hd, hd), lambda b, i, r: (0, 0), **one),
        ],
        out_specs=pl.BlockSpec((None, bq, q_dim), lambda b, i, r: (b, i, 0)),
        scratch_shapes=[
            pltpu.VMEM((n_kv, S, hd), BF16),
            pltpu.VMEM((n_kv, S, W), BF16),
            pltpu.VMEM((n_kv, M, hd), BF16),
            pltpu.VMEM((n_kv, M, LANES), F32),
            pltpu.VMEM((n_kv, M, W), F32),
            pltpu.VMEM((bq, bq), F32),
        ],
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((B, S_l, q_dim), qkv_local.dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )(rank.astype(jnp.int32).reshape(1), qkv_local, kv_full, kv_full, cos, sin, perm)
