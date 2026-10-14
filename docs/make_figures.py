"""Genera las figuras de docs/GUIA_TRANSFORMER.md (solo numpy + matplotlib).

Uso:  python docs/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(name: str) -> None:
    plt.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close()
    print("ok", name)


# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc="#dbe7ff", ec="#2b4c8c", fs=9, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2, **kw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, color="#444", style="-|>", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}"))


def synth_logmel(n_mels=128, T=1000):
    """Espectrograma sintético: armónicos con vibrato + percusión + ruido (en dB)."""
    t = np.linspace(0, 10, T)
    f = np.arange(n_mels)[:, None]
    S = np.full((n_mels, T), -70.0)
    for k, f0 in enumerate([14, 28, 42, 56, 70]):
        centre = f0 + 3 * np.sin(2 * np.pi * 0.4 * t)[None]
        S += (45 - 7 * k) * np.exp(-((f - centre) ** 2) / 4.0) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t + k))[None]
    for tb in np.arange(0.25, 10, 0.5):
        S += 38 * np.exp(-((t - tb) ** 2) / 0.0006)[None] * np.exp(-f / 70)
    S += rng.normal(0, 2.0, S.shape)
    return np.clip(S, -80, 0)


# ── 1. Pipeline audio → log-mel ──────────────────────────────────────────────
def fig_pipeline():
    sr, dur = 16000, 0.03
    t = np.arange(int(sr * dur)) / sr
    wave = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 660 * t) + 0.05 * rng.normal(size=t.size)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
    axs[0].plot(t * 1000, wave, lw=0.8, color="#2b4c8c")
    axs[0].set(title="1) Forma de onda x[n] (30 ms)", xlabel="ms", ylabel="amplitud")
    win = np.hanning(400)
    seg = wave[:400] * win
    mag = np.abs(np.fft.rfft(seg, 400)) ** 2
    axs[1].plot(np.fft.rfftfreq(400, 1 / sr), 10 * np.log10(mag + 1e-10), lw=0.9, color="#b3402a")
    axs[1].set(title="2) |STFT|² de una ventana", xlabel="Hz", ylabel="dB")
    S = synth_logmel()
    im = axs[2].imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, 10, 0, 128])
    axs[2].set(title="3) Log-mel (128 mels × 1000 frames)", xlabel="tiempo (s)", ylabel="banda mel")
    fig.colorbar(im, ax=axs[2], label="dB")
    save("01_pipeline_logmel")


# ── 2. Escala mel y banco de filtros ─────────────────────────────────────────
def fig_mel():
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    f = np.linspace(0, 8000, 500)
    n_mels, n_fft, sr = 10, 400, 16000
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(8000), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    axs[0].plot(f, hz2mel(f), color="#2b4c8c")
    axs[0].set(title=r"$m = 2595\,\log_{10}(1 + f/700)$", xlabel="Hz", ylabel="mel")
    for i in range(1, n_mels + 1):
        l, c, r = pts[i - 1], pts[i], pts[i + 1]
        tri = np.clip(np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)), 0, None)
        axs[1].plot(freqs, tri, lw=1.1)
    axs[1].set(title="Banco de filtros triangulares (10 bandas de ejemplo)", xlabel="Hz", ylabel="peso")
    save("02_escala_mel")


# ── 3. Parcheado ─────────────────────────────────────────────────────────────
def fig_patches():
    S = synth_logmel()
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [3, 1.6]})
    axs[0].imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, 1000, 0, 128])
    for x in range(0, 1001, 16):
        axs[0].axvline(x, color="w", lw=0.3, alpha=0.6)
    for y in range(0, 129, 16):
        axs[0].axhline(y, color="w", lw=0.6, alpha=0.9)
    axs[0].set(title="Parches 16×16 sobre 128×1000 → 8 × 62 = 496 tokens (+[CLS])",
               xlabel="frames (tiempo)", ylabel="mels (frecuencia)")
    ax = axs[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Orden de aplanado (row-major)")
    for i in range(3):
        for j in range(5):
            ax.add_patch(Rectangle((j * 1.8 + 0.3, 3 - i * 1.0 - 0.9), 1.6, 0.8, fc="#dbe7ff", ec="#2b4c8c"))
            ax.text(j * 1.8 + 1.1, 3 - i * 1.0 - 0.5, str(i * 5 + j + 1), ha="center", va="center", fontsize=8)
    ax.text(5, -0.3, "n = h·W + w  (primero se recorre el tiempo)", ha="center", fontsize=8)
    save("03_parcheado")


# ── 4. Positional encoding ───────────────────────────────────────────────────
def sinusoidal(n, d):
    pos = np.arange(n)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: d // 2])
    return pe


def fig_posenc():
    pe = sinusoidal(100, 64)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
    im = axs[0].imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axs[0].set(title="PE(pos, i)  (100 posiciones × 64 dim)", xlabel="dimensión i", ylabel="posición")
    fig.colorbar(im, ax=axs[0])
    for i in [0, 4, 16, 40]:
        axs[1].plot(pe[:, i], label=f"dim {i}")
    axs[1].set(title="Frecuencias decrecientes con i", xlabel="posición"); axs[1].legend(fontsize=7)
    sim = pe @ pe.T
    im2 = axs[2].imshow(sim, cmap="viridis")
    axs[2].set(title=r"$PE_p \cdot PE_q$ depende solo de $|p-q|$")
    fig.colorbar(im2, ax=axs[2])
    save("04_positional_encoding")


# ── 5. Atención: softmax, escala √d, heatmap ─────────────────────────────────
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def fig_attention():
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.5))
    # (a) por qué dividir por sqrt(d)
    for d in [4, 64, 512]:
        q, k = rng.normal(size=(5000, d)), rng.normal(size=(5000, d))
        axs[0].hist((q * k).sum(1), bins=60, alpha=0.5, density=True, label=f"d={d}")
    axs[0].set(title=r"Var$(q\cdot k)=d$ si $q_i,k_i\sim N(0,1)$", xlabel=r"$q\cdot k$ (sin escalar)")
    axs[0].legend()
    # (b) saturación del softmax
    z = rng.normal(size=8)
    for s, c in [(1, "#2b4c8c"), (8, "#b3402a")]:
        axs[1].bar(np.arange(8) + (0.4 if s == 8 else 0), softmax(z * s), width=0.4, color=c,
                   label=f"logits × {s}")
    axs[1].set(title="Softmax saturado → gradientes ≈ 0", xlabel="token", ylabel="peso de atención")
    axs[1].legend()
    # (c) matriz A = softmax(QK^T/√d)
    n, d = 12, 16
    Q, K = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    A = softmax(Q @ K.T / np.sqrt(d))
    im = axs[2].imshow(A, cmap="Blues")
    axs[2].set(title=r"$A=\mathrm{softmax}(QK^\top/\sqrt{d_k})$", xlabel="clave j", ylabel="consulta i")
    fig.colorbar(im, ax=axs[2])
    save("05_atencion")


# ── 6. Multi-head: forma de tensores ─────────────────────────────────────────
def fig_multihead():
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.1, 1.5, 1.3, 1, "X\n(B, N, D)", fc="#fff2cc", ec="#b8860b")
    box(ax, 2.0, 1.5, 1.5, 1, "Linear\nD → 3D\n(W_qkv)")
    box(ax, 4.1, 1.5, 1.8, 1, "reshape\n(B, N, 3, h, D/h)\npermute → q,k,v")
    box(ax, 6.5, 1.5, 1.7, 1, "por cabeza:\nsoftmax(qkᵀ/√d_h)·v\n(B,h,N,D/h)")
    box(ax, 8.8, 1.5, 1.4, 1, "concat\n(B, N, D)")
    box(ax, 10.6, 1.5, 1.2, 1, "Linear\nW_O", fc="#d9ead3", ec="#38761d")
    for a, b in [(1.4, 2.0), (3.5, 4.1), (5.9, 6.5), (8.2, 8.8), (10.2, 10.6)]:
        arrow(ax, (a, 2), (b, 2))
    ax.text(6, 3.4, "Multi-Head Self-Attention  (D=256, h=8 → d_h=32)", ha="center", fontsize=11, weight="bold")
    ax.text(6, 0.6, "Cada cabeza aprende su propia relación (p. ej. armonía vs. ritmo) en un subespacio de dimensión D/h.",
            ha="center", fontsize=9, style="italic")
    save("06_multihead")


# ── 7. Arquitectura AST ──────────────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")
    box(ax, 0.2, 2.8, 1.5, 1.0, "Log-mel\n(B,1,128,1000)", fc="#fff2cc", ec="#b8860b")
    box(ax, 2.2, 2.8, 1.6, 1.0, "PatchEmbed\nConv2d 16×16\ns=16 → (B,496,D)")
    box(ax, 4.3, 2.8, 1.4, 1.0, "+ [CLS]\n(B,497,D)")
    box(ax, 6.2, 2.8, 1.3, 1.0, "+ Pos.\nEmbedding")
    for a, b in [(1.7, 2.2), (3.8, 4.3), (5.7, 6.2)]:
        arrow(ax, (a, 3.3), (b, 3.3))
    # bloque encoder (flujo de arriba abajo, residuales como puentes a la derecha)
    ax.add_patch(Rectangle((8.0, 0.4), 3.3, 5.8, fc="#f6f6f6", ec="#888", ls="--"))
    ax.text(9.65, 5.9, "Encoder block × L (=12)", ha="center", weight="bold")
    cx = 9.3
    ys = {"ln1": 5.0, "att": 4.2, "add1": 3.55, "ln2": 2.9, "mlp": 2.1, "add2": 1.45}
    box(ax, 8.2, ys["ln1"], 2.2, 0.5, "LayerNorm", fc="#eeeeee", ec="#666")
    box(ax, 8.2, ys["att"], 2.2, 0.6, "Multi-Head\nSelf-Attention", fc="#fce5cd", ec="#b45f06", fs=8)
    ax.text(cx, ys["add1"] + 0.1, "⊕", fontsize=16, color="#c00", ha="center", va="center")
    box(ax, 8.2, ys["ln2"], 2.2, 0.5, "LayerNorm", fc="#eeeeee", ec="#666")
    box(ax, 8.2, ys["mlp"], 2.2, 0.6, "MLP D→4D→D\n(GELU)", fc="#d9ead3", ec="#38761d", fs=8)
    ax.text(cx, ys["add2"] + 0.1, "⊕", fontsize=16, color="#c00", ha="center", va="center")
    arrow(ax, (cx, 5.7), (cx, 5.5)); arrow(ax, (cx, 5.0), (cx, 4.8)); arrow(ax, (cx, 4.2), (cx, 3.8))
    arrow(ax, (cx, 3.35), (cx, 3.4)); arrow(ax, (cx, 3.3), (cx, 3.4))
    arrow(ax, (cx, 3.55 - 0.15), (cx, 3.4)); 
    arrow(ax, (cx, 3.4), (cx, 3.4))
    arrow(ax, (cx, 2.9), (cx, 2.7)); arrow(ax, (cx, 2.1), (cx, 1.75))
    arrow(ax, (cx, 1.35), (cx, 0.7))
    # puentes residuales (por la derecha)
    for ytop, ybot in [(5.7, ys["add1"] + 0.1), (ys["add1"] + 0.1, ys["add2"] + 0.1)]:
        ax.plot([cx, 10.8, 10.8], [ytop, ytop, ybot], color="#c00", lw=1.2)
        arrow(ax, (10.8, ybot), (cx + 0.18, ybot), color="#c00")
    arrow(ax, (7.5, 3.3), (8.0, 3.3))
    box(ax, 11.7, 4.3, 1.2, 0.8, "LayerNorm\n+ token [CLS]", fs=8)
    box(ax, 11.7, 3.1, 1.2, 0.8, "Linear\nD→K", fc="#d9ead3", ec="#38761d")
    box(ax, 11.7, 1.9, 1.2, 0.8, "softmax\nlogits (B,K)", fc="#fff2cc", ec="#b8860b", fs=8)
    arrow(ax, (11.3, 1.55), (12.3, 4.3), rad=-0.3)
    arrow(ax, (12.3, 4.3), (12.3, 3.9)); arrow(ax, (12.3, 3.1), (12.3, 2.7))
    ax.text(6.5, 0.2, "Pre-LN: x ← x + Attn(LN(x));  x ← x + MLP(LN(x))   (flechas rojas = conexiones residuales)",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Audio Spectrogram Transformer (modelo scratch del proyecto)", fontsize=12, weight="bold")
    save("07_arquitectura_ast")


# ── 8. LayerNorm vs BatchNorm, GELU ──────────────────────────────────────────
def fig_norm_act():
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
    x = np.linspace(-4, 4, 400)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    axs[0].plot(x, gelu, label="GELU", color="#2b4c8c"); axs[0].plot(x, np.maximum(x, 0), "--", label="ReLU", color="#b3402a")
    axs[0].legend(); axs[0].set(title=r"GELU$(x)=x\,\Phi(x)$"); axs[0].axhline(0, c="k", lw=0.4)
    # tensor (B,N,D): qué ejes se normalizan
    for ax, title, hl in [(axs[1], "BatchNorm: media sobre (B,N) por canal", "col"),
                          (axs[2], "LayerNorm: media sobre D por token", "row")]:
        ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.axis("off"); ax.set_title(title)
        for i in range(8):
            for j in range(8):
                ax.add_patch(Rectangle((j, 7 - i), 1, 1, fc="#eee", ec="white"))
        if hl == "col":
            ax.add_patch(Rectangle((2, 0), 1, 8, fc="#2b4c8c", alpha=0.6))
        else:
            ax.add_patch(Rectangle((0, 4), 8, 1, fc="#b3402a", alpha=0.6))
        ax.text(4, -0.6, "eje x = D (canales)   eje y = tokens/batch", ha="center", fontsize=8)
    save("08_gelu_layernorm")


# ── 9. Loss, mixup ───────────────────────────────────────────────────────────
def fig_loss_mixup():
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
    p = np.linspace(0.01, 1, 200)
    axs[0].plot(p, -np.log(p), color="#b3402a"); axs[0].set(title=r"CE $=-\log p_y$", xlabel=r"probabilidad $p_y$")
    for a in [0.1, 0.3, 1.0, 5.0]:
        axs[1].hist(rng.beta(a, a, 20000), bins=50, histtype="step", density=True, label=fr"$\alpha$={a}")
    axs[1].set(title=r"Mixup: $\lambda\sim$Beta$(\alpha,\alpha)$", xlabel=r"$\lambda$"); axs[1].legend(fontsize=7)
    S1, S2 = synth_logmel(T=300), synth_logmel(T=300)[::-1]
    lam = 0.3
    axs[2].imshow(lam * S1 + (1 - lam) * S2, origin="lower", aspect="auto", cmap="magma")
    axs[2].set(title=r"$\tilde x=\lambda x_i+(1-\lambda)x_j$  (λ=0.3)", xlabel="frames", ylabel="mels")
    save("09_loss_mixup")


# ── 10. SpecAugment ──────────────────────────────────────────────────────────
def fig_specaug():
    S = synth_logmel(T=500)
    A = S.copy()
    A[40:40 + 20, :] = -40  # freq mask
    A[90:100, :] = -40
    A[:, 120:200] = -40  # time mask
    A[:, 330:380] = -40
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    axs[0].imshow(S, origin="lower", aspect="auto", cmap="magma"); axs[0].set_title("Original")
    axs[1].imshow(A, origin="lower", aspect="auto", cmap="magma"); axs[1].set_title("SpecAugment: 2 bandas de frecuencia + 2 de tiempo")
    for a in axs: a.set(xlabel="frames", ylabel="mels")
    save("10_specaugment")


# ── 11. Scheduler ────────────────────────────────────────────────────────────
def fig_schedule():
    epochs, warm, spe = 50, 5, 100
    total, w = epochs * spe, warm * spe
    lr0, eta = 1e-4, 1e-6
    t = np.arange(total)
    lr = np.where(t < w, lr0 * (1e-3 + (1 - 1e-3) * t / w),
                  eta + 0.5 * (lr0 - eta) * (1 + np.cos(np.pi * (t - w) / (total - w))))
    plt.figure(figsize=(7, 3))
    plt.plot(t / spe, lr, color="#2b4c8c"); plt.axvline(warm, ls="--", c="gray")
    plt.xlabel("época"); plt.ylabel("learning rate"); plt.title("Warm-up lineal (5 ép.) + decaimiento coseno")
    save("11_lr_schedule")


# ── 12. Curvas de gradiente: residual + LN ───────────────────────────────────
def fig_residual():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.set_xlim(0, 8); ax.set_ylim(0, 3.4); ax.axis("off")
    box(ax, 0.3, 1.2, 1.2, 0.9, "x", fc="#fff2cc", ec="#b8860b")
    box(ax, 2.4, 1.2, 1.4, 0.9, "LN")
    box(ax, 4.2, 1.2, 1.6, 0.9, "F (Attn/MLP)", fc="#fce5cd", ec="#b45f06")
    ax.text(6.5, 1.55, "⊕", fontsize=20, color="#c00")
    box(ax, 7.0, 1.2, 0.9, 0.9, "x+F", fc="#d9ead3", ec="#38761d")
    arrow(ax, (1.5, 1.65), (2.4, 1.65)); arrow(ax, (3.8, 1.65), (4.2, 1.65)); arrow(ax, (5.8, 1.65), (6.4, 1.65)); arrow(ax, (6.9, 1.65), (7.0, 1.65))
    arrow(ax, (0.9, 2.1), (6.6, 2.1 + 0.1), rad=-0.35, color="#c00")
    ax.text(3.7, 3.05, "atajo identidad:  ∂(x+F)/∂x = I + ∂F/∂x", ha="center", color="#c00")
    save("12_residual")


if __name__ == "__main__":
    for fn in [fig_pipeline, fig_mel, fig_patches, fig_posenc, fig_attention, fig_multihead,
               fig_architecture, fig_norm_act, fig_loss_mixup, fig_specaug, fig_schedule, fig_residual]:
        fn()
