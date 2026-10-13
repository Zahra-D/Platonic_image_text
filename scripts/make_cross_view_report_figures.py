"""Figures for docs/CROSS_VIEW_IMAGE_JEPA.md (writes docs/figures/cross_view/*.png)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from clevr_paths import CLEVR_DATA_ROOT
from data.clevr_dataset import ClevrImageFolder
from models.vqvae import VQVAE
from make_augmented_views import augment

OUT = ROOT / "docs/figures/cross_view"; OUT.mkdir(parents=True, exist_ok=True)
FEAT = ROOT / "outputs/eval_all/xret/features"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def fig_views():
    m = CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl"
    ds = ClevrImageFolder(str(Path(m).parent), (64, 96), m)
    x = torch.stack([ds[i] for i in (0, 3, 7, 11, 15)]).cuda()
    ck = torch.load(ROOT / "outputs/vqvae_training_bs128/best.pt", map_location="cpu", weights_only=False)
    vq = VQVAE(64, 512).cuda().eval(); vq.load_state_dict(ck["model"])
    g = torch.Generator(device="cuda"); g.manual_seed(3)
    with torch.no_grad():
        a = augment((x + 1) / 2, g); b = augment((x + 1) / 2, g)
        tc, ta, tb = (vq.encode_to_indices(v) for v in (x, a * 2 - 1, b * 2 - 1))
        ra = vq.decode_from_indices(ta)
    cols = ["original render", "view A (pixels)", "view B (pixels)", "view A after VQ\n(decoded tokens)", "tokens changed\nclean vs A (dark = changed)"]
    fig, ax = plt.subplots(5, 5, figsize=(10, 7.2))
    for r in range(5):
        ims = [(x[r] + 1) / 2, a[r], b[r], ((ra[r] + 1) / 2).clamp(0, 1)]
        for c, im in enumerate(ims):
            ax[r, c].imshow(im.permute(1, 2, 0).cpu().numpy())
        changed = (tc[r] != ta[r]).float().cpu().numpy()
        ax[r, 4].imshow(1 - changed, cmap="gray", vmin=0, vmax=1)
        ax[r, 4].set_xlabel(f"{changed.mean():.0%} of 384 changed", fontsize=8, color=INK2)
        for c in range(5):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([]); ax[r, c].grid(False)
            for s in ax[r, c].spines.values(): s.set_visible(False)
            if r == 0: ax[r, c].set_title(cols[c], fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "views.png", dpi=150); plt.close(fig)


def hbar(path, labels, values, colors, xlabel, title, fmt="{:.3f}", xmax=1.0):
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.1))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.01, yi, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, xmax); ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False); ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_invariance():
    labels = ["dense image diffusion (13)", "image I-JEPA, same view (12+1)", "trunk JEPA + private diffusion, same view",
              "cross-view I-JEPA (12+0.95)", "cross-view trunk JEPA + private diffusion"]
    vals = [0.021, 0.033, 0.031, 0.993, 0.993]
    hbar(OUT / "invariance.png", labels, vals, [BLUE, ORANGE, ORANGE, AQUA, AQUA],
         "R@1, view A → its own view B among 2000 scenes (L7)",
         "Lighting invariance of pooled image features")


def heldout_pc_content(lab, readout, n=8):
    from feature_variance_sources import descriptors, heldout_r2
    global _D
    if "_D" not in globals():
        a = argparse.Namespace(num_scenes=8000, image_manifest=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl",
                               image_cache=str(ROOT / "outputs/image_only_2_5m_token_cache/val_tokens.pt"))
        _D = descriptors(a)
    sp = np.random.default_rng(1).permutation(8000); fit, te = np.sort(sp[:6000]), np.sort(sp[6000:])
    Z = np.load(FEAT / f"{lab}.npz")[readout].astype(float); Z = Z - Z[fit].mean(0)
    _, s, vt = np.linalg.svd(Z[fit], full_matrices=False); P = Z @ vt[:n].T
    return [heldout_r2(P[:, [k]], _D["content"], fit, te) for k in range(n)], (s ** 2 / (s ** 2).sum())[:n]


def fig_pc_content():
    series = [("text, dense diffusion (13)", "denseT_e013_TEXT", YELLOW), ("image, dense diffusion (13)", "denseI_e013_IMAGE", BLUE),
              ("image, I-JEPA same view (12+1)", "ijepaI_s15t45_tot013_IMAGE", ORANGE), ("image, cross-view I-JEPA (12+0.95)", "xview_jepa_e13_IMAGE", AQUA)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.2; xs = np.arange(8)
    for k, (name, lab, col) in enumerate(series):
        r2, _ = heldout_pc_content(lab, "L6.mlp_out")
        ax.bar(xs + (k - 1.5) * w, np.clip(r2, 0, None), width=w - 0.02, color=col, label=name)
    ax.set_xticks(xs); ax.set_xticklabels([f"PC{i + 1}" for i in xs]); ax.set_ylim(0, 1)
    ax.set_ylabel("held-out R² from scene content"); ax.grid(axis="x", visible=False)
    ax.set_title("How much of each principal direction is scene content (L6.mlp_out)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "pc_content.png", dpi=150); plt.close(fig)


def fig_cka():
    layers = ["L7", "L6", "L6.mlp_out", "L5"]
    fam = [("dense text × dense image (13)", [0.297, 0.299, 0.181, 0.332], BLUE),
           ("text JEPA × dense image (12+1)", [0.383, 0.389, 0.183, 0.389], YELLOW),
           ("text JEPA × image JEPA (12+1)", [0.374, 0.348, 0.189, 0.335], ORANGE),
           ("text JEPA × cross-view image JEPA (12+1 / 12+0.95)", [0.417, 0.479, 0.271, 0.448], AQUA)]
    fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.2; xs = np.arange(4)
    for k, (name, v, col) in enumerate(fam):
        bars = ax.bar(xs + (k - 1.5) * w, v, width=w - 0.02, color=col, label=name)
        if k == 3:
            for b_, val in zip(bars, v):
                ax.text(b_.get_x() + b_.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(layers); ax.set_ylim(0, 0.75); ax.set_ylabel("linear CKA, text vs image (2000 scenes)")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Text–image CKA by layer at ~13 total epochs", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "cka_layers.png", dpi=150); plt.close(fig)


def fig_samples():
    S = ROOT / "outputs/cross_view_report/samples"
    rows = [("dense image diffusion, 13 epochs", "dense_e13_random.png"),
            ("frozen dense trunk + fresh private LoRA (1 epoch diffusion)", "frozen_dense_trunk_lora_random.png"),
            ("frozen same-view I-JEPA trunk + fresh private LoRA", "frozen_ijepa_trunk_lora_random.png"),
            ("frozen cross-view I-JEPA trunk + fresh private LoRA", "frozen_xview_trunk_lora_random.png"),
            ("cross-view trunk JEPA + private diffusion, trained jointly", "joint_xview_trunk_privdiff_random.png")]
    fig, ax = plt.subplots(len(rows), 1, figsize=(10, 1.75 * len(rows)))
    for a_, (title, f) in zip(ax, rows):
        a_.imshow(np.asarray(Image.open(S / f).convert("RGB"))); a_.set_title(title, loc="left", fontsize=9)
        a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
        for s in a_.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "samples.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["views", "invariance", "pc_content", "cka", "samples"]
    for name in which: globals()[f"fig_{name}"]()
    print("figures ->", OUT)
