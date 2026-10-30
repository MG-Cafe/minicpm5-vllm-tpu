#!/usr/bin/env python3
"""Generate the README / technical-guide charts (E2E latency, throughput, architecture, KPI cards, accuracy)
from the MiniCPM5 TPU benchmark JSON results in results/."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CHARTS_DIR = os.path.join(REPO_ROOT, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Google Cloud / Material Palette
GOOGLE_BLUE = "#1A73E8"
GOOGLE_BLUE_DARK = "#174EA6"
GOOGLE_BLUE_BG = "#E8F0FE"
GOOGLE_GREEN = "#1E8E3E"
GOOGLE_GREEN_DARK = "#0D652D"
GOOGLE_GREEN_BG = "#E6F4EA"
DARK_TEXT = "#202124"
MUTED_TEXT = "#5F6368"
GRID_COLOR = "#E8EAED"
CARD_BG = "#F8F9FA"
BORDER_COLOR = "#DADCE0"

SHAPES = [(1, 8192), (4, 8192), (8, 8192), (16, 8192), (32, 8192)]
SHAPE_LABELS = [
    "(1, 8192)\n8,192 tok",
    "(4, 8192)\n32,768 tok",
    "(8, 8192)\n65,536 tok",
    "(16, 8192)\n131,072 tok",
    "(32, 8192)\n262,144 tok",
]


def load_rows(accel: str, model_short: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{accel}_minicpm5_{model_short}.json")
    with open(path) as f:
        data = json.load(f)
    return {(r["batch_size"], r["seq_len"]): r for r in data["by_config"]}


def setup_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": BORDER_COLOR,
        "axes.linewidth": 0.9,
        "axes.labelcolor": DARK_TEXT,
        "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT,
        "text.color": DARK_TEXT,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })


def plot_latency_by_shape(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b):
    """Simple 2-panel End-to-End p50 Latency (ms) bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)
    x = np.arange(len(SHAPES))
    w = 0.34

    configs = [
        ("openbmb/MiniCPM5-1B — End-to-End Latency (p50 ms)", v6e_1b, tpu7x_1b, axes[0]),
        ("openbmb/MiniCPM5-2B — End-to-End Latency (p50 ms)", v6e_2b, tpu7x_2b, axes[1]),
    ]

    for title, r_v6e, r_7x, ax in configs:
        v6e_e2e = [r_v6e[s]["request_latency_ms_p50"] for s in SHAPES]
        t7x_e2e = [r_7x[s]["request_latency_ms_p50"] for s in SHAPES]
        ymax = max(v6e_e2e) * 1.24

        b1 = ax.bar(x - w / 2, v6e_e2e, w, label="1-Chip TPU v6e-1 (p50 E2E ms)", color=GOOGLE_BLUE, edgecolor="none", zorder=3)
        b2 = ax.bar(x + w / 2, t7x_e2e, w, label="1-Chip TPU7x-1 (p50 E2E ms)", color=GOOGLE_GREEN, edgecolor="none", zorder=3)

        for bars, color in [(b1, GOOGLE_BLUE_DARK), (b2, GOOGLE_GREEN_DARK)]:
            for bar in bars:
                h = bar.get_height()
                lbl = f"{h:.1f}" if h < 100 else f"{h:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + ymax * 0.016,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=8.6,
                    fontweight="bold",
                    color=color,
                )

        for idx in range(len(SHAPES)):
            sp = v6e_e2e[idx] / t7x_e2e[idx]
            top_h = max(v6e_e2e[idx], t7x_e2e[idx]) + ymax * 0.085
            ax.text(
                x[idx],
                top_h,
                f"{sp:.2f}x faster",
                ha="center",
                va="bottom",
                fontsize=7.9,
                fontweight="bold",
                color=GOOGLE_GREEN_DARK,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=GOOGLE_GREEN_BG, edgecolor=GOOGLE_GREEN, lw=0.8),
            )

        ax.set_title(title, fontsize=12.2, fontweight="bold", pad=12, color=DARK_TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(SHAPE_LABELS, fontsize=9.2, fontweight="bold")
        ax.set_xlabel("(batch_size, seq_len) — 8,192 Tokens per Input", fontsize=10, labelpad=8)
        ax.set_ylabel("End-to-End Latency (ms)", fontsize=10.2, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True,
               facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=10.0)
    fig.suptitle("MiniCPM5 End-to-End vLLM Latency on 1-Chip Cloud TPU (8,192 Tokens / Input)", fontsize=14.2,
                 fontweight="bold", y=1.05, color=DARK_TEXT)
    fig.tight_layout()
    out = os.path.join(CHARTS_DIR, "latency_by_shape.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_throughput_by_shape(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b):
    """Simple 2-panel Single-Chip E2E Throughput (k tokens/sec) bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300)
    x = np.arange(len(SHAPES))
    w = 0.34

    tp_configs = [
        ("openbmb/MiniCPM5-1B — Single-Chip Throughput (k tok/s)", v6e_1b, tpu7x_1b, axes[0]),
        ("openbmb/MiniCPM5-2B — Single-Chip Throughput (k tok/s)", v6e_2b, tpu7x_2b, axes[1]),
    ]
    for title, r_v6e, r_7x, ax in tp_configs:
        v6e_tps = [r_v6e[s]["throughput_tokens_per_s"] / 1000.0 for s in SHAPES]
        t7x_tps = [r_7x[s]["throughput_tokens_per_s"] / 1000.0 for s in SHAPES]
        ymax = max(max(v6e_tps), max(t7x_tps)) * 1.22

        b1 = ax.bar(x - w / 2, v6e_tps, w, label="1-Chip TPU v6e-1 (k tok/s)", color=GOOGLE_BLUE, zorder=3)
        b2 = ax.bar(x + w / 2, t7x_tps, w, label="1-Chip TPU7x-1 (k tok/s)", color=GOOGLE_GREEN, zorder=3)

        for bars, color in [(b1, GOOGLE_BLUE_DARK), (b2, GOOGLE_GREEN_DARK)]:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + ymax * 0.018,
                    f"{h:.0f}k",
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                    fontweight="bold",
                    color=color,
                )

        ax.set_title(title, fontsize=12.2, fontweight="bold", pad=12, color=DARK_TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(SHAPE_LABELS, fontsize=9.2, fontweight="bold")
        ax.set_xlabel("(batch_size, seq_len) — 8,192 Tokens per Input", fontsize=10, labelpad=8)
        ax.set_ylabel("E2E throughput (k tokens/s)", fontsize=10.2, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True,
               facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=10.0)
    fig.suptitle("MiniCPM5 End-to-End Single-Chip Throughput in vLLM (k tokens/sec)", fontsize=14.2,
                 fontweight="bold", y=1.05, color=DARK_TEXT)
    fig.tight_layout()
    out = os.path.join(CHARTS_DIR, "throughput_by_shape.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def load_gates() -> dict:
    """Accuracy gate summary per result file: {(accel, model): [gate dicts]}."""
    out = {}
    for accel in ("v6e", "tpu7x"):
        for m in ("1b", "2b"):
            path = os.path.join(RESULTS_DIR, f"{accel}_minicpm5_{m}.json")
            with open(path) as f:
                d = json.load(f)
            out[(accel, m)] = (d.get("validation") or {}).get("logprob_gates") or []
    return out


def plot_architecture_pipeline(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b):
    """2-tier infographic: per-layer kernel pipeline (top) + per-chip results / accuracy cards (bottom)."""
    fig, ax = plt.subplots(figsize=(15.5, 5.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 37)
    ax.axis("off")

    ax.text(50, 35.6, "MiniCPM5MegakernelForCausalLM — One Decoder Layer on Cloud TPU (8,192 tokens / input)",
            ha="center", va="center", fontsize=14.0, fontweight="bold", color=DARK_TEXT)

    stages = [
        (GOOGLE_BLUE, "1. Fused QKV (Pallas)",
         "• RMSNorm + per-token INT8\n• INT8 MXU QKV projection\n• one pass over activations"),
        (GOOGLE_BLUE_DARK, "2. SuperAttention v2 (Pallas)",
         "• RoPE'd K/V staged in VMEM\n• QKᵀ ‖ PV pipelined chunks\n• row-sum on MXU: P·[V | 1]\n• exp2 in bf16, causal skip"),
        ("#5F6368", "3. O-proj + Residual",
         "• BF16 matmul (INT8: 2B on v6e)\n• residual add fused (XLA)"),
        (GOOGLE_GREEN, "4. Fused MLP (Pallas)",
         "• RMSNorm + INT8 gate/up\n• SiLU·up → down + residual\n• weights resident in VMEM\n• (T, 2I) never touches HBM\n• 2B on TPU7x: XLA MLP (INT8 g/u)"),
        (GOOGLE_GREEN_DARK, "5. Last Token → lm_head",
         "• layer L-1 on last token only\n• (B, H) handed to vLLM\n• 32k-token on-device chunks"),
    ]
    width, gap, x0 = 18.2, 1.25, 1.2
    for i, (color, header, body) in enumerate(stages):
        xs = x0 + i * (width + gap)
        ax.add_patch(mpatches.FancyBboxPatch((xs, 18.6), width, 13.6, boxstyle="round,pad=0.45,rounding_size=1.1",
                                             facecolor=CARD_BG, edgecolor=color, linewidth=2.0))
        ax.add_patch(mpatches.FancyBboxPatch((xs, 29.0), width, 3.2, boxstyle="round,pad=0.45,rounding_size=1.1",
                                             facecolor=color, edgecolor=color, linewidth=1.0))
        ax.text(xs + width / 2, 30.6, header, ha="center", va="center", fontsize=9.0, fontweight="bold",
                color="#FFFFFF")
        ax.text(xs + 0.7, 27.6, body, ha="left", va="top", fontsize=8.1, color=DARK_TEXT, linespacing=1.55)
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs + width + gap + 0.05, 25.0), xytext=(xs + width + 0.35, 25.0),
                        arrowprops=dict(arrowstyle="->", color=DARK_TEXT, lw=2.0))

    def best(rows, key, fn):
        return fn(rows[s][key] for s in SHAPES)

    gates = load_gates()
    all_g = [g for gl in gates.values() for g in gl]
    def _tn(g):
        v = g.get("top1_tie_aware_nolp")
        return v if v is not None else 1.0
    tie_min = min(min(g.get("top1_tie_aware", 0), _tn(g)) for g in all_g)
    dlp_max = max(g.get("mean_abs_dlogprob", 9) for g in all_g)
    ex = [(round(g.get("top1_agreement", 0) * g.get("n_seqs", 0)), g.get("n_seqs", 0)) for g in all_g]
    ex_min = min(ex, key=lambda t: t[0] / max(1, t[1]))
    n_shapes = max(len(g.get("sets", [])) for g in all_g)
    all_pass = all(g.get("gate_v2") == "PASS" for g in all_g)

    def bmax(rows, key):
        return max(SHAPES, key=lambda s_: rows[s_][key])[0]

    t7_faster = all(t[s_]["request_latency_ms_p50"] < r[s_]["request_latency_ms_p50"]
                    for r, t in ((v6e_1b, tpu7x_1b), (v6e_2b, tpu7x_2b)) for s_ in SHAPES)
    cards = [
        (GOOGLE_BLUE, GOOGLE_BLUE_BG, "1-Chip TPU v6e-1 (1 TensorCore)",
         f"• E2E (bs=1): {v6e_1b[(1, 8192)]['request_latency_ms_p50']:.1f} ms (1B) · "
         f"{v6e_2b[(1, 8192)]['request_latency_ms_p50']:.1f} ms (2B)\n"
         f"• Peak (bs={bmax(v6e_1b, 'throughput_tokens_per_s')}): {best(v6e_1b, 'throughput_tokens_per_s', max) / 1e3:.0f}k tok/s (1B) · "
         f"{best(v6e_2b, 'throughput_tokens_per_s', max) / 1e3:.0f}k tok/s (2B)\n"
         f"• 1 TensorCore: the whole batch runs on one core"),
        (GOOGLE_GREEN, GOOGLE_GREEN_BG, f"1-Chip TPU7x-1 (2 TensorCores){' — lowest latency' if t7_faster else ''}",
         f"• E2E (bs=1): {tpu7x_1b[(1, 8192)]['request_latency_ms_p50']:.1f} ms (1B) · "
         f"{tpu7x_2b[(1, 8192)]['request_latency_ms_p50']:.1f} ms (2B)\n"
         f"• Peak (bs={bmax(tpu7x_1b, 'throughput_tokens_per_s')}): {best(tpu7x_1b, 'throughput_tokens_per_s', max) / 1e3:.0f}k tok/s (1B) · "
         f"{best(tpu7x_2b, 'throughput_tokens_per_s', max) / 1e3:.0f}k tok/s (2B)\n"
         f"• 2 TensorCores: data-parallel (bs ≥ 2) · seq-parallel (bs=1)"),
        ("#1967D2", CARD_BG, "Accuracy Gate (vs FP32 golden + BF16 ref)",
         f"• 5 prompts, last-position top-20; batch 1–32 ({n_shapes} sets)\n"
         f"• Top-1: ≥ {ex_min[0]}/{ex_min[1]} exact, tie-aware {'100' if tie_min >= 1 else '≥ ' + format(100 * tie_min, '.0f')}%\n"
         f"• Mean |Δ logprob| ≤ {dlp_max:.3f} (pass ≤ 0.10){' · all PASS' if all_pass else ' · NOT all pass'}"),
    ]
    cw = 31.4
    for i, (color, bg_col, header, body) in enumerate(cards):
        xs = 1.2 + i * (cw + 1.65)
        ax.add_patch(mpatches.FancyBboxPatch((xs, 1.0), cw, 13.6, boxstyle="round,pad=0.45,rounding_size=1.1",
                                             facecolor=bg_col, edgecolor=color, linewidth=1.8))
        ax.add_patch(mpatches.FancyBboxPatch((xs, 11.4), cw, 3.2, boxstyle="round,pad=0.45,rounding_size=1.1",
                                             facecolor=color, edgecolor=color, linewidth=1.0))
        ax.text(xs + cw / 2, 13.0, header, ha="center", va="center", fontsize=9.0, fontweight="bold",
                color="#FFFFFF")
        ax.text(xs + 0.9, 10.1, body, ha="left", va="top", fontsize=8.2, color=DARK_TEXT, linespacing=1.6)

    out = os.path.join(CHARTS_DIR, "architecture_pipeline.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_kpi_cards(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b):
    """Headline KPI cards: fastest E2E latency per chip, peak throughput, accuracy gate."""
    fig, ax = plt.subplots(figsize=(15.5, 3.3), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis("off")

    def peak(rows):
        return max(rows[s]["throughput_tokens_per_s"] for s in SHAPES)

    all_g = [g for gl in load_gates().values() for g in gl]
    n_pass = sum(g.get("gate_v2") == "PASS" for g in all_g)
    dlp_max = max(g.get("mean_abs_dlogprob", 9) for g in all_g)

    cards = [
        (GOOGLE_BLUE, GOOGLE_BLUE_BG, "TPU v6e-1 · fastest E2E (bs=1)",
         f"{v6e_1b[(1, 8192)]['request_latency_ms_p50']:.1f} ms", f"MiniCPM5-1B   ·   2B: {v6e_2b[(1, 8192)]['request_latency_ms_p50']:.1f} ms"),
        (GOOGLE_GREEN, GOOGLE_GREEN_BG, "TPU7x-1 · fastest E2E (bs=1)",
         f"{tpu7x_1b[(1, 8192)]['request_latency_ms_p50']:.1f} ms", f"MiniCPM5-1B   ·   2B: {tpu7x_2b[(1, 8192)]['request_latency_ms_p50']:.1f} ms"),
        (GOOGLE_GREEN_DARK, GOOGLE_GREEN_BG, f"Peak throughput · TPU7x-1 (bs={max(SHAPES, key=lambda s_: tpu7x_1b[s_]['throughput_tokens_per_s'])[0]})",
         f"{peak(tpu7x_1b) / 1e3:.0f}k tok/s", f"MiniCPM5-1B   ·   v6e-1: {peak(v6e_1b) / 1e3:.0f}k tok/s"),
        (GOOGLE_BLUE_DARK, GOOGLE_BLUE_BG, "Accuracy gate · all chips & models",
         f"{n_pass} / {len(all_g)} PASS", f"worst mean |Δlp| {dlp_max:.3f}  (pass ≤ 0.10)"),
    ]
    w, gap = 23.3, 1.6
    for i, (color, bg, title, big, small) in enumerate(cards):
        x0 = 0.9 + i * (w + gap)
        ax.add_patch(mpatches.FancyBboxPatch((x0, 1.0), w, 18.0, boxstyle="round,pad=0.4,rounding_size=1.4",
                                             facecolor=bg, edgecolor=color, linewidth=2.0))
        ax.add_patch(mpatches.Rectangle((x0 - 0.4, 1.0), 1.1, 18.0, facecolor=color, edgecolor="none"))
        ax.text(x0 + w / 2 + 0.45, 15.3, title, ha="center", va="center", fontsize=9.2, fontweight="bold",
                color=MUTED_TEXT)
        ax.text(x0 + w / 2 + 0.3, 9.4, big.replace("$", "\\$"), ha="center", va="center", fontsize=25,
                fontweight="bold", color=color)
        ax.text(x0 + w / 2 + 0.3, 3.9, small.replace("$", "\\$"), ha="center", va="center", fontsize=10.0,
                color=DARK_TEXT)
    out = os.path.join(CHARTS_DIR, "kpi_cards.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_accuracy():
    """Accuracy gates: mean |Δ logprob| per run vs FP32 golden and BF16 reference (pass <= 0.10),
    annotated with tie-aware top-1 agreement."""
    gates = load_gates()
    runs = [("v6e", "1b", "v6e-1 · 1B"), ("tpu7x", "1b", "TPU7x-1 · 1B"), ("v6e", "2b", "v6e-1 · 2B"),
            ("tpu7x", "2b", "TPU7x-1 · 2B")]

    def pick(gl, golden):
        for g in gl:
            lbl = (str(g.get("reference_label", "")) + str(g.get("reference", ""))).lower()
            if (("fp32" in lbl or "golden" in lbl) == golden):
                return g
        return None

    fig, ax = plt.subplots(figsize=(13.5, 4.8), dpi=300)
    x = np.arange(len(runs))
    w = 0.36
    for j, (golden, color, dark, lab) in enumerate([(True, GOOGLE_BLUE, GOOGLE_BLUE_DARK, "vs FP32 golden"),
                                                    (False, GOOGLE_GREEN, GOOGLE_GREEN_DARK, "vs BF16 reference")]):
        vals, labs = [], []
        for accel, m, _ in runs:
            g = pick(gates[(accel, m)], golden)
            vals.append(g["mean_abs_dlogprob"] if g else 0.0)
            if g:
                tn = g.get("top1_tie_aware_nolp")
                t = min(g.get("top1_tie_aware", 0), tn if tn is not None else 1.0)
                n = g.get("n_seqs", 0)
                labs.append(f"{g['mean_abs_dlogprob']:.3f}\ntop-1 {round(g.get('top1_agreement', 0) * n)}/{n}"
                            f"\ntie-aware {100 * t:.0f}%")
            else:
                labs.append("n/a")
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=color, label=f"mean |Δ logprob| {lab}", zorder=3)
        for b, v, lt in zip(bars, vals, labs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, lt, ha="center",
                    va="bottom", fontsize=7.4, fontweight="bold", color=dark)
    ax.axhline(0.10, color="#D93025", linestyle="--", linewidth=1.6, zorder=2)
    ax.text(len(runs) - 0.5, 0.102, "pass threshold 0.10", ha="right", va="bottom", fontsize=9.5, color="#D93025",
            fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[2] for r in runs], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("mean |Δ logprob| over top-20", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.135)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor=CARD_BG, edgecolor=BORDER_COLOR, fontsize=9.5, ncol=2)
    all_g = [g for gl in gates.values() for g in gl]
    verdict = "all runs PASS" if all_g and all(g.get("gate_v2") == "PASS" for g in all_g) else "NOT all runs pass"
    ax.set_title(f"Accuracy gate: last-position top-20 log-probs on 8,192-token prompts ({verdict})", fontsize=12.5,
                 fontweight="bold", pad=10)
    fig.tight_layout()
    out = os.path.join(CHARTS_DIR, "accuracy_gate.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    setup_rc()
    v6e_1b = load_rows("v6e", "1b")
    tpu7x_1b = load_rows("tpu7x", "1b")
    v6e_2b = load_rows("v6e", "2b")
    tpu7x_2b = load_rows("tpu7x", "2b")

    plot_latency_by_shape(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b)
    plot_throughput_by_shape(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b)
    plot_architecture_pipeline(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b)
    plot_kpi_cards(v6e_1b, tpu7x_1b, v6e_2b, tpu7x_2b)
    plot_accuracy()


if __name__ == "__main__":
    main()
