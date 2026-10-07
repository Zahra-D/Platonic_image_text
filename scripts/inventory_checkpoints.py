"""Inventory of every trained run under outputs/ (reads checkpoint args only, via mmap).

Filters (as requested): no shared-LoRA models (train_mode == "lora"), and only runs
whose training set has more than 1M samples. Writes docs/MODEL_INVENTORY.md and
outputs/model_inventory.json.
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import clevr_paths  # noqa: F401  (expands ${CLEVR_DATA} defaults)
import os

_lines: dict[str, int | None] = {}
def line_count(path) -> int | None:
    if not path:
        return None
    path = os.path.expandvars(str(path))
    if path not in _lines:
        try:
            _lines[path] = int(subprocess.run(["wc", "-l", path], capture_output=True, text=True,
                                              check=True).stdout.split()[0])
        except Exception:
            _lines[path] = None
    return _lines[path]


def args_of(ckpt: Path) -> dict | None:
    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        try:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            return None
    a = payload.get("args") if isinstance(payload, dict) else None
    if a is None:
        return None
    return dict(a) if isinstance(a, dict) else vars(a)


def objective(a: dict) -> str:
    parts = []
    dw = a.get("diffusion_weight", 1.0)
    if a.get("lejepa_views"):
        parts.append("LeJEPA multi-crop (paper)")
    elif a.get("data2vec_hidden"):
        kind = "LeJEPA token-level (SIGReg, no teacher)" if a.get("data2vec_teacherless") else "I-JEPA/data2vec (EMA)"
        kind += f" {a.get('data2vec_mode', 'average')}"
        if a.get("jepa_trunk_only"):
            kind += ", trunk-only"
        if a.get("data2vec_projector_sigreg_weight", 0):
            kind += " + SIGReg projector"
        parts.append(kind)
    elif a.get("shared_jepa") or a.get("modulewise_jepa_mode", "none") != "none":
        parts.append("shared/modulewise JEPA")
    if dw and dw > 0:
        diff = "diffusion"
        if a.get("diffusion_private_only"):
            diff += " (private LoRA only)"
        parts.insert(0, diff)
    for flag, name in (("sigreg", "SIGReg"), ("modality_adversarial", "DANN"), ("backtranslation", "back-translation"),
                       ("modulewise_hsic", "HSIC"), ("gradient_balance", "grad-balance")):
        if a.get(flag):
            parts.append(name)
    if a.get("private_dropout", 0):
        parts.append(f"private-dropout {a['private_dropout']}")
    if a.get("cross_modal_moment_weight", 0):
        parts.append("moment matching")
    return " + ".join(parts) or "?"


def main():
    rows = []
    for d in sorted(p for p in (ROOT / "outputs").iterdir() if p.is_dir()):
        epochs = sorted(p for p in d.glob("epoch_*.pt") if not p.name.endswith("_adapter.pt"))
        if not epochs:
            continue
        idx = [int(re.search(r"epoch_(\d+)", p.name).group(1)) for p in epochs]
        a = args_of(epochs[-1])
        if a is None:
            rows.append({"run": d.name, "error": "unreadable args"}); continue
        mode = a.get("data_mode", "paired")
        tm = a.get("train_mode", "dense")
        text_n = line_count(a.get("train_text_manifest")) if a.get("train_text_manifest") else None
        main_n = line_count(a.get("train_manifest"))
        if main_n is None and a.get("train_dir"):
            main_n = line_count(Path(os.path.expandvars(a["train_dir"])) / "images.jsonl")
        size = min(x for x in (main_n, text_n) if x is not None) if (main_n or text_n) else None
        if a.get("max_train_samples"):
            size = min(size or 10**12, int(a["max_train_samples"]))
        parent = a.get("resume") or a.get("init_checkpoint")
        modality = {"text_only": "text", "image_only": "image", "unpaired": "text+image (unpaired)",
                    "paired": "text+image (paired)"}.get(mode, mode)
        reasons = []
        if tm == "lora":
            reasons.append("shared LoRA")
        if size is None or size <= 1_000_000:
            reasons.append(f"data {size:,}" if size else "data size unknown")
        rows.append({
            "run": d.name, "modality": modality, "train_mode": tm, "objective": objective(a),
            "train_samples": size, "train_manifest": a.get("train_manifest"),
            "train_text_manifest": a.get("train_text_manifest"),
            "epochs_saved": idx, "n_checkpoints": len(idx),
            "from": ("resume " if a.get("resume") else "init ") + str(Path(parent).parent.name) + "/" + Path(parent).name if parent else "scratch",
            "d_model": a.get("d_model"), "n_layers": a.get("n_layers"),
            "include": not reasons, "exclude_reason": ", ".join(reasons),
        })
    (ROOT / "outputs" / "model_inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
    inc = [r for r in rows if r.get("include")]
    exc = [r for r in rows if not r.get("include")]
    def ep(r):
        i = r["epochs_saved"]
        return f"{len(i)} (epoch_{i[0]:03d}…{i[-1]:03d})" if len(i) > 1 else f"1 (epoch_{i[0]:03d})"
    groups = ["text", "image", "text+image (unpaired)", "text+image (paired)"]
    out = ["# Model inventory", "",
           f"Generated by `scripts/inventory_checkpoints.py`. Runs with epoch checkpoints: {len(rows)}; "
           f"**included: {len(inc)}** ({sum(r['n_checkpoints'] for r in inc)} checkpoints); excluded: {len(exc)}.",
           "", "Filters: no shared-LoRA models (`train_mode: lora`); training set > 1M samples. "
           "`dense_private` models (shared dense trunk + private LoRA) are included and evaluated trunk-only.", ""]
    n = 0
    for g in groups + sorted({r["modality"] for r in inc} - set(groups)):
        sel = [r for r in inc if r["modality"] == g]
        if not sel:
            continue
        out += [f"## {g} ({len(sel)} runs)", "", "| # | run | train mode | objective | train samples | checkpoints | from |",
                "|---|---|---|---|---|---|---|"]
        for r in sel:
            n += 1
            out.append(f"| {n} | `{r['run']}` | {r['train_mode']} | {r['objective']} | {r['train_samples']:,} | {ep(r)} | {r['from']} |")
        out.append("")
    from collections import Counter
    reasons = Counter(r.get("exclude_reason", "error") for r in exc)
    out += ["## Excluded", "", "| reason | runs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in reasons.most_common()]
    out += ["", "<details><summary>excluded runs</summary>", ""] + \
           [f"- `{r['run']}` — {r.get('exclude_reason') or r.get('error')}" for r in exc] + ["", "</details>", ""]
    (ROOT / "docs" / "MODEL_INVENTORY.md").write_text("\n".join(out))
    print(f"runs {len(rows)} | included {len(inc)} runs / {sum(r['n_checkpoints'] for r in inc)} checkpoints | excluded {len(exc)}")
    for k, v in reasons.most_common():
        print(f"  excluded: {k}: {v}")


if __name__ == "__main__":
    main()
