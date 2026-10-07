#!/usr/bin/env python3
"""Regenerate docs/experiment_catalog/all_experiments.md from configs + results.

Every number in the master index comes from a file on disk: the run's config
(what it is), its train.log (validation loss), and the evaluation JSONs under
outputs/ (d', probes, binding). Nothing is transcribed by hand, so re-running
this after new runs land keeps the index honest.

    python3 scripts/build_all_experiments.py
"""
from __future__ import annotations

import copy
import glob
import json
import os
import re
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/experiment_catalog/all_experiments.md")
TEXT_TOKENS = 91.7          # mean content tokens per caption
IMAGE_TOKENS = 384          # VQ codes per image
COCO_TOKENS = 13.0


# --------------------------------------------------------------------------- configs

def load_config(path: str) -> dict | None:
    """Parse a config, tolerating the `tokenizer:`-prefixed split convention."""
    raw = open(path).read()
    try:
        if "tokenizer:" in raw:
            return yaml.safe_load("tokenizer:" + raw.split("tokenizer:", 1)[1])
        return yaml.safe_load(raw)
    except Exception:
        return None


def tokens_per_epoch(mode: str, manifest: str) -> float | None:
    manifest = str(manifest)
    both = TEXT_TOKENS + IMAGE_TOKENS
    if "platonic_text_only_v1_2m" in manifest:
        return 2_000_000 * TEXT_TOKENS
    if "train_pairs_human" in manifest:
        return 1_200_000 * (both if mode in ("paired", "unpaired") else IMAGE_TOKENS)
    if "5M_train_gpu_visible" in manifest:
        return 1_200_000 * IMAGE_TOKENS
    if "ms_coco" in manifest:
        return 118_287 * COCO_TOKENS
    if "100k_gpu_visible" in manifest:
        per = both if mode in ("paired", "unpaired") else (
            TEXT_TOKENS if mode == "text_only" else IMAGE_TOKENS)
        return 90_000 * per
    return None


def validation_loss(output_dir: str) -> str:
    log = os.path.join(ROOT, "outputs", output_dir, "train.log")
    if not os.path.exists(log):
        return "—"
    found = re.findall(r"validation epoch=\d+ .*?loss=([0-9.]+)", open(log, errors="ignore").read())
    return found[-1] if found else "—"


def collect() -> dict:
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "configs/*.yaml"))):
        config = load_config(path)
        if not isinstance(config, dict) or "train" not in config:
            continue
        train = config["train"]
        name = (train.get("output_dir") or "").rstrip("/").split("/")[-1]
        if not name:
            continue
        checkpoints = [p for p in glob.glob(os.path.join(ROOT, "outputs", name, "epoch_*.pt"))
                       if "_adapter" not in p]
        if not checkpoints:
            continue
        data = config.get("data", {}) or {}
        align = config.get("alignment", {}) or {}
        lora = config.get("lora", {}) or {}
        diff = config.get("diffusion", {}) or {}
        mode = data.get("mode", "paired")
        per = tokens_per_epoch(mode, data.get("train_manifest", ""))
        runs[name] = dict(
            name=name, config=os.path.basename(path), mode=mode,
            train_mode=lora.get("train_mode", "dense"), rank=lora.get("rank"),
            private_rank=lora.get("private_rank"), freeze_base=lora.get("freeze_base"),
            diffusion_weight=diff.get("weight", 1.0), epochs=len(checkpoints),
            planned=train.get("epochs"),
            per_epoch=per, init=(train.get("init_checkpoint") or train.get("resume")),
            align=align, diff=diff, val=validation_loss(name),
        )
    for run in runs.values():
        run["own"] = (run["per_epoch"] or 0) * run["epochs"]
    for run in runs.values():
        seen, total, cursor = set(), run["own"], run
        while cursor["init"]:
            parent = os.path.basename(os.path.dirname(cursor["init"]))
            if parent not in runs or parent in seen:
                break
            seen.add(parent)
            cursor = runs[parent]
            total += cursor["own"]
        run["total"] = total
        run["parent"] = (os.path.basename(os.path.dirname(run["init"]))
                         if run["init"] else None)
    return runs


# --------------------------------------------------------------------------- description

def objective(run: dict) -> str:
    """The JEPA objective in full, including whether an MLP predictor is used."""
    align, diff, parts = run["align"], run["diff"], []
    layers = align.get("data2vec_layers")
    top_k = align.get("data2vec_top_k")
    if align.get("data2vec_hidden"):
        mode = align.get("data2vec_mode", "average")
        if layers:
            span = f"blocks {min(layers)}-{max(layers)}"
            heads = len(layers)
        else:
            span = f"top {top_k}" if top_k else "blocks 0-7"
            heads = top_k or 8
        predictor = (f"**MLP predictor ×{heads}**" if mode == "layerwise"
                     else "**MLP predictor ×1**")
        teacher = ("**SIGReg** (no teacher, no stop-grad)"
                   if align.get("sigreg_enabled") or align.get("data2vec_sigreg_weight")
                   else "EMA teacher")
        beta = align.get("data2vec_beta")
        loss = f"SmoothL1 β={beta}" if beta else align.get("shared_jepa_loss", "SmoothL1")
        parts.append(f"hidden-state data2vec, {mode}, {span}, {predictor}, {teacher}, {loss}")
        weight = align.get("data2vec_sigreg_weight")
        if weight:
            parts.append(f"SIGReg λ={weight}")
    elif align.get("modulewise_jepa_mode", "none") != "none":
        mode = align.get("modulewise_jepa_mode")
        gated = "gated " if align.get("modulewise_jepa_gated_predictor") else ""
        # The `data2vec_*` modes are the direct-regression runs: they drop the
        # predictor entirely and regress the teacher target head-on.
        predictor = ("**no predictor (direct regression)**"
                     if mode.startswith("data2vec_")
                     else "**MLP predictor per module/layer**")
        parts.append(f"{gated}modulewise on adapter writes, {mode}, {predictor}, "
                     f"EMA teacher, {align.get('shared_jepa_loss', 'normalized_mse')}")
    if align.get("modulewise_hsic_enabled"):
        parts.append("HSIC")
    if run["train_mode"] == "lora":
        parts.append(f"Tri-LoRA {(run['rank'] or 384) - (run['private_rank'] or 128)}"
                     f"+{run['private_rank'] or 128}")
    elif run["train_mode"] == "dense_private":
        parts.append(f"private rank {run['private_rank'] or 128}, "
                     + ("frozen trunk" if run["freeze_base"] else "trunk trains"))
    if not run["diffusion_weight"]:
        parts.append("no diffusion")
    if diff.get("mask_block_2d"):
        parts.append("2D blocks")
    elif diff.get("mask_span_min"):
        parts.append(f"windows {diff['mask_span_min']}+")
    if not parts:
        return "plain diffusion"
    return "; ".join(parts)


def family(run: dict) -> int:
    align = run["align"]
    d2v = bool(align.get("data2vec_hidden"))
    modulewise = align.get("modulewise_jepa_mode", "none") != "none"
    scratch = run["parent"] is None
    if align.get("sigreg_enabled") or align.get("data2vec_sigreg_weight"):
        return 5
    if "coco" in run["name"]:
        return 8
    if "pilot" in run["name"] or "faithful" in run["name"] or "fidelity" in run["name"]:
        return 9
    if run["mode"] in ("paired", "unpaired"):
        return 7
    if run["train_mode"] == "dense_private":
        return 6
    if d2v:
        return 4 if scratch else 3
    if modulewise:
        return 2
    return 1


TITLES = {
    1: ("Baselines: diffusion only", "dense and Tri-LoRA, no JEPA"),
    2: ("Tri-LoRA with JEPA on adapter writes", "the modulewise family"),
    3: ("data2vec on hidden states, from a pretrained trunk", "the one that works"),
    4: ("data2vec on hidden states, from scratch", "the one that does not"),
    5: ("SIGReg / LeJEPA instead of an EMA teacher", "all four collapse"),
    6: ("Stage 2: dense shared route + private LoRA", "the shared/private test"),
    7: ("Multimodal: paired and unpaired", "one model, both modalities"),
    8: ("Other corpora (MS-COCO)", ""),
    9: ("Pilots and ablations (not yet evaluated)", ""),
}


# --------------------------------------------------------------------------- results

def read(path: str):
    try:
        return json.load(open(os.path.join(ROOT, path)))
    except Exception:
        return None


def best_dprime(blob, key):
    if not blob:
        return None
    metrics = blob["metrics"]
    values = [(metrics[f][key]["d"]["value"], f) for f in metrics if key in metrics[f]]
    return max(values) if values else None


def results_by_model() -> dict:
    """Map checkpoint directory -> every measurement we have for it."""
    table: dict[str, dict] = {}

    def slot(model_path: str) -> dict:
        name = os.path.basename(os.path.dirname(model_path))
        return table.setdefault(name, {})

    for folder, key in [("semantic_dprime_eval", "full"),
                        ("semantic_dprime_trunk_only", "shared"),
                        ("semantic_dprime_private_only", "private"),
                        ("semantic_dprime_unpaired", "full")]:
        for path in glob.glob(os.path.join(ROOT, "outputs", folder, "*.json")):
            if os.path.basename(path) in ("controls.json", "construction.json"):
                continue
            blob = read(os.path.relpath(path, ROOT))
            if not blob or "protocol" not in blob:
                continue
            entry = slot(blob["protocol"]["model"])
            # Several epochs of one run may have been scored; report the last.
            epoch = re.findall(r"epoch_(\d+)", blob["protocol"]["model"])
            epoch = int(epoch[0]) if epoch else -1
            if epoch < entry.get(f"epoch_{key}", -1):
                continue
            entry[f"epoch_{key}"] = epoch
            entry[f"d_sem_{key}"] = (blob["metrics"].get("L7.residual", {})
                                     .get("d_semantic", {}).get("d", {}).get("value"))
            binding = best_dprime(blob, "d_binding")
            entry[f"d_bind_{key}"] = binding[0] if binding else None

    for path in glob.glob(os.path.join(ROOT, "outputs/cross_modal*/[!c]*.json")):
        blob = read(os.path.relpath(path, ROOT))
        if not blob or "protocol" not in blob or "metrics" not in blob:
            continue
        entry = slot(blob["protocol"]["model"])
        probe = blob["metrics"].get("L7", {}).get("probe_accuracy")
        if probe is not None:
            entry.setdefault("probe_" + blob["protocol"].get("modality", "text"), probe)

    for path in glob.glob(os.path.join(ROOT, "outputs/image_binding_eval/*.json")):
        blob = read(os.path.relpath(path, ROOT))
        if not blob or "binding_accuracy" not in blob:
            continue
        slot(blob["protocol"]["model"])["image_bind"] = blob["binding_accuracy"]["value"]
    return table


# --------------------------------------------------------------------------- rendering

HEADER = ("| Run | Mod | What it is (objective, predictor, teacher, loss) | From | Ep "
          "| Own | Total | Val | d_sem full | d_bind full | d_sem shared | d_bind shared "
          "| d_sem private | d_bind private | text probe | img probe | img bind |\n"
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

MODALITY = {"text_only": "text", "image_only": "image", "unpaired": "both", "paired": "both"}


def cell(value, digits=3, plus=False, percent=False):
    if value is None:
        return "—"
    if percent:
        return f"{value*100:.1f}%"
    return f"{value:+.{digits}f}" if plus else f"{value:.{digits}f}"


def row(run: dict, found: dict) -> str:
    r = found.get(run["name"], {})
    parent = run["parent"] or "scratch"
    parent = parent.replace("text_", "").replace("image_", "")[:28]
    return ("| `{name}` | {mod} | {what} | {parent} | {ep} | {own:.2f}B | **{total:.2f}B** "
            "| {val} | {ds} | {db} | {dss} | {dbs} | {dsp} | {dbp} | {pt} | {pi} | {ib} |\n").format(
        name=run["name"], mod=MODALITY.get(run["mode"], run["mode"]), what=objective(run),
        parent=parent,
        ep=(f'{run["epochs"]}' if not run["planned"] or run["epochs"] >= run["planned"]
            else f'{run["epochs"]}/{run["planned"]} ⏳'), own=run["own"] / 1e9, total=run["total"] / 1e9,
        val=run["val"],
        ds=cell(r.get("d_sem_full"), 2, True), db=cell(r.get("d_bind_full"), 3, True),
        dss=cell(r.get("d_sem_shared"), 2, True), dbs=cell(r.get("d_bind_shared"), 3, True),
        dsp=cell(r.get("d_sem_private"), 2, True), dbp=cell(r.get("d_bind_private"), 3, True),
        pt=cell(r.get("probe_text"), percent=True), pi=cell(r.get("probe_image"), percent=True),
        ib=cell(r.get("image_bind")))


def main():
    runs = collect()
    found = results_by_model()
    groups: dict[int, list] = {}
    for run in runs.values():
        groups.setdefault(family(run), []).append(run)
    for group in groups.values():
        group.sort(key=lambda r: (r["mode"], r["name"]))

    preamble = open(os.path.join(ROOT, "docs/experiment_catalog/_all_experiments_preamble.md")).read()
    def anchor(text: str) -> str:
        """GitHub's heading slug: drop punctuation, then spaces become hyphens."""
        cleaned = re.sub(r"[^\w\s-]", "", text.lower())
        return re.sub(r"\s", "-", cleaned)

    contents = "\n".join(
        f"{n}. [{TITLES[n][0]}]"
        + f"(#{anchor(f'{n}. ' + TITLES[n][0] + f' — {len(groups[n])} runs')})"
        + (f" — {TITLES[n][1]}" if TITLES[n][1] else "")
        for n in sorted(groups))
    body = []
    for n in sorted(groups):
        body.append(f"\n## {n}. {TITLES[n][0]} — {len(groups[n])} runs\n\n" + HEADER)
        body.extend(row(run, found) for run in groups[n])

    total = sum(len(g) for g in groups.values())
    text = preamble.format(total=total, families=len(groups), contents=contents)
    open(OUT, "w").write(text + "".join(body))
    print(f"wrote {OUT}: {total} runs in {len(groups)} families")


if __name__ == "__main__":
    main()
