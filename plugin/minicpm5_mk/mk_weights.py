# Host-side weight fusion / quantization / static activation calibration for the MiniCPM5
# megakernel. Ported from scripts/tpu_megakernel_minicpm5.py (_load_and_fuse_weights,
# _calibrate_fp8_act_scales, _quantize_w), using torch CPU ops on HF checkpoint tensors.
"""Prepare stacked megakernel weights from HF-named MiniCPM5 (Llama) tensors."""

from __future__ import annotations

import math
import os
import time
from typing import Dict

import numpy as np
import torch

BF16 = torch.bfloat16
F32 = torch.float32


def rope_tables(max_seq: int, head_dim: int, theta: float):
    """cos/sin (max_seq, head_dim), computed like vLLM's RotaryEmbedding (f32), stored bf16."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=F32) / head_dim))
    t = torch.arange(max_seq, dtype=F32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(BF16), emb.sin().to(BF16)


def _calibrate(layers, embed, cos_f, sin_f, cfg, quant: str, bos_id: int, n_seq=4, s_cal=128, seed=42,
               margin=None, margin_down=None, gamma_in_act=None):
    """Static per-layer activation amax from a short f32 CPU forward over random-id sequences.

    Position 0 of every calibration sequence is token MINICPM5_MK_CAL_FIRST_ID (default 1, `</s>`).
    Measured on the accuracy gate (5 x 8,192-token prompts vs FP32), BOS-first calibration (id 0) gave
    slightly worse fidelity (2B mean |dlogprob| 0.062 vs 0.040 on v6e-1), so id 1 is kept as the default.

    fp8 (e4m3) is floating point, so a generous margin (headroom against clipping on longer or
    unseen inputs) costs little precision; int8 keeps the original tight margins.
    """
    is_fp8 = quant in ("fp8", "fp8d")
    q_max = 448.0 if is_fp8 else 127.0
    if margin is None:
        margin = 1.35 if is_fp8 else 1.15
    if margin_down is None:
        margin_down = 1.50
    if gamma_in_act is None:  # which linears keep their norm gamma outside the weights
        gamma_in_act = {"qkv": quant != "bf16", "gu": quant != "bf16"}
    nh, kh, hd = cfg["n_q"], cfg["n_kv"], cfg["head_dim"]
    g = nh // kh
    half = hd // 2
    eps = cfg["eps"]
    rng = np.random.RandomState(seed)
    ids = rng.randint(1000, min(embed.shape[0], 30000), size=(n_seq, s_cal))
    ids[:, 0] = int(os.environ.get("MINICPM5_MK_CAL_FIRST_ID", "1"))  # see docstring (bos_id kept for reference)
    del bos_id
    x = embed[torch.from_numpy(ids)].to(F32)
    c = cos_f[:s_cal][None, :, None, :]
    s = sin_f[:s_cal][None, :, None, :]
    mask = torch.tril(torch.ones((s_cal, s_cal), dtype=torch.bool))[None, None, None]
    scales = []
    chan = []  # per-layer per-input-channel amax of each linear's (pre-quant) input
    chan_nob = []  # excluding position 0 (BOS sink token)
    with torch.no_grad():
        for ld in layers:
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            xe = xn * ld["in_norm"] if gamma_in_act["qkv"] else xn
            if quant == "int8":
                amax_qkv = float(xe[:, 1:].abs().max()) * margin
            else:
                amax_qkv = float(xe.abs().max()) * margin
            qkv = xe @ ld["w_qkv"]
            qkv4 = qkv.view(n_seq, s_cal, nh + 2 * kh, hd)
            q, k, v = qkv4[:, :, :nh], qkv4[:, :, nh:nh + kh], qkv4[:, :, nh + kh:]
            q = q * c + torch.cat([-q[..., half:], q[..., :half]], -1) * s
            k = k * c + torch.cat([-k[..., half:], k[..., :half]], -1) * s
            q_g = q.view(n_seq, s_cal, kh, g, hd).permute(0, 2, 3, 1, 4)
            k_g = k.permute(0, 2, 1, 3).unsqueeze(2)
            v_g = v.permute(0, 2, 1, 3).unsqueeze(2)
            sc = torch.matmul(q_g, k_g.transpose(-1, -2)).masked_fill(~mask, -1e30)
            ao = torch.matmul(torch.softmax(sc, -1), v_g).permute(0, 3, 1, 2, 4).reshape(n_seq, s_cal, nh * hd)
            if quant == "int8":
                amax_o = float(ao[:, 1:].abs().max()) * margin
            else:
                amax_o = float(ao.abs().max()) * margin
            x = x + ao @ ld["w_o"]
            xn2 = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            xe2 = xn2 * ld["post_norm"] if gamma_in_act["gu"] else xn2
            if quant == "int8":
                amax_gu = float(xe2[:, 1:].abs().max()) * margin
            else:
                amax_gu = float(xe2.abs().max()) * margin
            gu = torch.einsum("bsh,hcm->bscm", xe2, ld["w_gu"])
            hid = torch.nn.functional.silu(gu[:, :, 0]) * gu[:, :, 1]
            amax_down = float(hid.abs().max()) * margin_down
            x = x + hid @ ld["w_down"]
            scales.append({
                "qkv": max(amax_qkv / q_max, 1e-6),
                "o": max(amax_o / q_max, 1e-6),
                "gu": max(amax_gu / q_max, 1e-6),
                "down": max(amax_down / q_max, 1e-6),
            })
            chan.append({
                "qkv": xe.abs().amax(dim=(0, 1)), "o": ao.abs().amax(dim=(0, 1)),
                "gu": xe2.abs().amax(dim=(0, 1)), "down": hid.abs().amax(dim=(0, 1)),
            })
            chan_nob.append({
                "qkv": xe[:, 1:].abs().amax(dim=(0, 1)), "o": ao[:, 1:].abs().amax(dim=(0, 1)),
                "gu": xe2[:, 1:].abs().amax(dim=(0, 1)), "down": hid[:, 1:].abs().amax(dim=(0, 1)),
            })
    return scales, chan, chan_nob


def _quantize_w(w: torch.Tensor, quant: str, sx: float):
    """Per-output-channel (amax over input axis 0) 8-bit weights; returns (wq, s_w * sx)."""
    is_fp8 = quant in ("fp8", "fp8d")
    q_max = 448.0 if is_fp8 else 127.0
    amax = w.abs().amax(dim=0, keepdim=True)
    scale = torch.clamp(amax / q_max, min=1e-8)
    if is_fp8:
        wq = torch.clamp(w / scale, -q_max, q_max).to(torch.float8_e4m3fn)
    else:
        wq = torch.clamp(torch.round(w / scale), -q_max, q_max).to(torch.int8)
    return wq, (scale.squeeze(0) * sx).to(F32)


def prepare_mk_weights(sd: Dict[str, torch.Tensor], cfg: dict, quant: str, max_seq: int,
                       bos_id: int, log=print) -> Dict[str, torch.Tensor]:
    """sd: HF-named CPU tensors. cfg keys: num_layers, hidden, inter, n_q, n_kv, head_dim, eps,
    rope_theta. quant: bf16 | fp8 | int8. Returns CPU tensors in megakernel layout."""
    t0 = time.perf_counter()
    nthreads = torch.get_num_threads()
    torch.set_num_threads(max(nthreads, 16))
    try:
        L, H = cfg["num_layers"], cfg["hidden"]
        hd = cfg["head_dim"]
        from minicpm5_mk.mk_jax import parse_quant, quant_flags
        qm = parse_quant(quant)
        base_q = quant.split("|")[0].split(":", 1)[0]
        last_bf16 = "lastbf16" in quant_flags(quant)
        sm = 1.0 / math.sqrt(hd)
        layers = []
        for i in range(L):
            p = f"model.layers.{i}."
            f = lambda n: sd[p + n].to(F32)  # noqa: E731
            in_norm = f("input_layernorm.weight")
            post_norm = f("post_attention_layernorm.weight")
            wq = f("self_attn.q_proj.weight").T * sm
            wk = f("self_attn.k_proj.weight").T
            wv = f("self_attn.v_proj.weight").T
            wo = f("self_attn.o_proj.weight").T.contiguous()
            wg = f("mlp.gate_proj.weight").T
            wu = f("mlp.up_proj.weight").T
            wd = f("mlp.down_proj.weight").T.contiguous()
            # fold gammas into weight rows (exact algebra) only for bf16 linears; quantized linears
            # keep gamma in the per-input-channel activation scale so outlier channels do not
            # distort weight row quantization.
            if qm["qkv"] == "bf16":
                wq, wk, wv = in_norm[:, None] * wq, in_norm[:, None] * wk, in_norm[:, None] * wv
            if qm["gu"] == "bf16":
                wg, wu = post_norm[:, None] * wg, post_norm[:, None] * wu
            layers.append({
                "w_qkv": torch.cat([wq, wk, wv], dim=1).contiguous(),
                "w_o": wo,
                "w_gu": torch.stack([wg, wu], dim=1).contiguous(),
                "w_down": wd,
                "in_norm": in_norm,
                "post_norm": post_norm,
            })
        cos, sin = rope_tables(max_seq, hd, cfg["rope_theta"])
        out: Dict[str, torch.Tensor] = {"cos": cos, "sin": sin}
        # Pre-slice scan layers (0..L-2) vs last layer (L-1) on host so core_forward never emits
        # runtime HBM slice copies (`stacked[k][:-1]`) on every forward pass.
        ll = layers[-1]
        in_n, post_n = ll["in_norm"], ll["post_norm"]
        wqkv = ll["w_qkv"] if qm["qkv"] == "bf16" else in_n[:, None] * ll["w_qkv"]
        wgu = ll["w_gu"] if qm["gu"] == "bf16" else post_n[:, None, None] * ll["w_gu"]
        out.update({"wl_qkv": wqkv.to(BF16).contiguous(), "wl_o": ll["w_o"].to(BF16).contiguous(),
                    "wl_gu": wgu.to(BF16).contiguous(), "wl_down": ll["w_down"].to(BF16).contiguous()})
        if base_q == "bf16":
            for k in ("w_qkv", "w_o", "w_gu", "w_down"):
                out[k] = torch.stack([ld[k] for ld in layers[:-1]]).to(BF16).contiguous()
            dummy = torch.zeros((1,), dtype=F32)
            for k in ("s_qkv", "s_o", "s_gu", "s_down", "a_qkv", "a_o", "a_gu", "a_down"):
                out[k] = dummy.clone()
            log(f"[mk] bf16 weights prepared (pre-sliced L-1={L-1} scan + 1 last) in {time.perf_counter() - t0:.1f}s")
            return out
        embed = sd["model.embed_tokens.weight"].to(F32)
        import os
        s_cal = int(os.environ.get("MINICPM5_MK_CAL_SEQ", "128"))
        n_cal = int(os.environ.get("MINICPM5_MK_CAL_NSEQ", "4"))
        m = os.environ.get("MINICPM5_MK_CAL_MARGIN")
        md = os.environ.get("MINICPM5_MK_CAL_MARGIN_DOWN")
        rc, rs = rope_tables(s_cal, hd, cfg["rope_theta"])
        scales, chan, chan_nob = _calibrate(
            layers, embed, rc.to(F32), rs.to(F32), cfg, base_q, bos_id, n_seq=n_cal,
            s_cal=s_cal, margin=float(m) if m else None,
            margin_down=float(md) if md else None,
            gamma_in_act={"qkv": qm["qkv"] != "bf16", "gu": qm["gu"] != "bf16"},
            seed=int(os.environ.get("MINICPM5_MK_CAL_SEED", "42")),
        )
        use_sq = "sq" in quant_flags(quant)
        alpha = float(os.environ.get("MINICPM5_MK_SQ_ALPHA", "0.5"))
        sq_max_down = float(os.environ.get("MINICPM5_MK_SQ_MAX_DOWN", "2.5"))
        sq_max_o = float(os.environ.get("MINICPM5_MK_SQ_MAX_O", "8.0"))
        margins = {"fp8": 1.35, "fp8d": 1.35, "int8": 1.15, "int8d": 1.15}
        q_maxes = {"fp8": 448.0, "fp8d": 448.0, "int8": 127.0, "int8d": 127.0}
        acc = {k: [] for k in ("w_qkv", "w_o", "w_gu", "w_down", "s_qkv", "s_o", "s_gu", "s_down",
                               "a_qkv", "a_o", "a_gu", "a_down")}
        sq_log = []
        scan_layers = layers[:-1]
        scan_scales = scales[:-1]
        for li, (ld, sc) in enumerate(zip(scan_layers, scan_scales)):
            for name, key in (("qkv", "w_qkv"), ("o", "w_o"), ("gu", "w_gu"), ("down", "w_down")):
                w = ld[key]
                k_in = w.shape[0]
                lq = qm[name]
                if lq == "bf16":
                    acc[key].append(w.to(BF16))
                    acc["s_" + name].append(torch.zeros((1,), dtype=F32))
                    acc["a_" + name].append(torch.zeros((1,), dtype=F32))
                    continue
                dynamic = (lq in ("int8", "fp8") and name == "down") or lq in ("int8d", "fp8d")
                # activation as seen by this linear: x̂·γ for qkv/gu, x for o/down
                gam = None
                if name == "qkv":
                    gam = ld["in_norm"]
                elif name == "gu":
                    gam = ld["post_norm"]
                w_eff = w if gam is None else w  # gamma stays in the activation for int8 static
                smooth = torch.ones((k_in,), dtype=F32)
                if use_sq:
                    # SmoothQuant: X / s, W * s (rows), s_j = amax_x_j^a / amax_w_j^(1-a); exact algebra.
                    # MINICPM5_MK_SQ_STATS=all (default) takes the per-channel amax over every position;
                    # "nobos" excludes position 0 (measured worse on the accuracy gate).
                    stats = chan_nob if os.environ.get("MINICPM5_MK_SQ_STATS", "all") == "nobos" else chan
                    ax = stats[li][name].to(F32).clamp(min=1e-5)
                    aw = w_eff.reshape(k_in, -1).abs().amax(dim=1).clamp(min=1e-5)
                    smooth = (ax.pow(alpha) / aw.pow(1.0 - alpha)).clamp(1e-4, 1e4)
                    w_eff = w_eff * smooth.view(-1, *([1] * (w_eff.dim() - 1)))
                    if not dynamic:  # recalibrate the static scale on the smoothed activation
                        m_ = float(os.environ.get("MINICPM5_MK_CAL_MARGIN") or margins.get(lq, 1.15))
                        ax_s = chan_nob[li][name].to(F32) if lq == "int8" else ax
                        sc = dict(sc)
                        sc[name] = max(float((ax_s / smooth).max()) * m_ / q_maxes[lq], 1e-6)
                    if li in (0, len(scan_layers) - 1):
                        sq_log.append(f"L{li}.{name}: s∈[{float(smooth.min()):.3g},{float(smooth.max()):.3g}]")
                sx = 1.0 if dynamic else sc[name]
                wq_, sw = _quantize_w(w_eff, lq, sx)
                acc[key].append(wq_)
                acc["s_" + name].append(sw)
                a = (gam if gam is not None else torch.ones((k_in,), dtype=F32)) / (smooth * sx)
                acc["a_" + name].append(a.to(F32))
        for k, v in acc.items():
            out[k] = torch.stack(v).contiguous()
        if "fmlp" in quant_flags(quant) and qm["gu"] in ("int8", "int8d"):
            # Fused-MLP Pallas kernel layout: W_gu (L-1, H, 2I) chunk-interleaved [g_c | u_c], s_gu (L-1, 1, 2I).
            from minicpm5_mk.mk_mlp import interleave_gu, mlp_chunk

            out["w_gu"], out["s_gu"] = interleave_gu(out["w_gu"], out["s_gu"], mlp_chunk(cfg["inter"]))
        log(f"[mk] {quant} weights prepared (pre-sliced L-1={len(scan_layers)}, calibration {n_cal}x{s_cal} random ids, seed "
            f"{os.environ.get('MINICPM5_MK_CAL_SEED', '42')}) in {time.perf_counter() - t0:.1f}s; "
            f"layer0 sx={scales[0]} layerL-2 sx={scales[-2]} sq={use_sq} alpha={alpha} {' '.join(sq_log)}")
        return out
    finally:
        torch.set_num_threads(nthreads)
