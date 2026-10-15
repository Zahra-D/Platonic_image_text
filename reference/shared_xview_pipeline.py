"""Watcher for the shared-trunk cross-view JEPA pipeline (R2).

1. JEPA stage: when the current R2 stage finishes and its last checkpoint is evaluated
   (shared_xview_eval.sh), continue for +7 epochs (resume) while the oracle maps still
   improve: last two evaluated epochs vs the two before, ridge text->image MRR or rotation
   image->text MRR up >= 0.01, or hard CCA R@1 up >= 0.02; at most 28 epochs in total.
2. Then: strip the private LoRA from the best epoch (oracle ridge text->image MRR), freeze
   the trunk, train fresh private text / image LoRAs with plain unpaired diffusion,
   2 epochs at a time, continuing while either modality's validation NELBO improves > 1%,
   at most 6 epochs; then score per-modality NELBO for the references.
Everything runs on GPU 0 (R2's GPU); evaluations of the LoRA stage on GPU 1.
"""
import json, os, re, subprocess, time
from pathlib import Path
import torch, yaml

ROOT = Path("/home/zd25e122/clevr_discrete_diffusion"); os.chdir(ROOT)
RUN = "mm_xview_xpara_trunk_jepa_from_ep9"; TAG = "xpr"
LORA = "mm_xpr_frozen_trunk_private_lora"
ENV = "export CLEVR_DATA=/home/zd25e122/clevr-dataset-gen_clone/output CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1"
LOG = ROOT / "outputs/eval_branches/shared_xview_pipeline.log"


def log(msg):
    with open(LOG, "a") as h:
        h.write(time.strftime("[%m-%d %H:%M] ") + msg + "\n")


def stages():
    return sorted([d for d in ROOT.glob(f"outputs/{RUN}*") if d.is_dir() and re.fullmatch(rf"{RUN}(_c\d+)?", d.name)],
                  key=lambda d: int(d.name.split("_c")[-1]) if "_c" in d.name else 1)


def complete(d):
    t = d / "train.log"
    return t.exists() and "training complete" in t.read_text()


def epochs_of(d):
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("epoch_[0-9][0-9][0-9].pt"))


def metrics(e):
    lab = f"{TAG}_e{e:03d}"
    if not (ROOT / f"outputs/eval_shared_xview/done/{lab}").exists():
        return None
    x = json.load(open(ROOT / f"outputs/eval_all/xret/results/{lab}_TEXT__{lab}_IMAGE.json"))
    h = json.load(open(ROOT / f"outputs/eval_hard_xret_variants/{lab}_TEXT__{lab}_IMAGE.json"))
    return {"ridge_t2i_MRR": x["ridge"]["text_to_image"]["MRR"], "ridge_t2i_R1": x["ridge"]["text_to_image"]["R@1"],
            "rot_i2t_MRR": x["procrustes"]["image_to_text"]["MRR"], "hard_cca": h["cca"]["R@1"]["value"]}


def launch(session, gpu, config):
    subprocess.run(["tmux", "new-session", "-d", "-s", session,
                    f"cd {ROOT} && {ENV} CUDA_VISIBLE_DEVICES={gpu} && python3 train_multimodal.py --config {config} "
                    f"> outputs/{Path(config).stem}.launch.log 2>&1"], check=True)


def write_config(src, name, edits, drop=()):
    c = yaml.safe_load(open(src))
    for path, val in edits.items():
        d = c; ks = path.split(".")
        for k in ks[:-1]: d = d.setdefault(k, {})
        d[ks[-1]] = val
    for path in drop:
        d = c; ks = path.split(".")
        for k in ks[:-1]: d = d.get(k, {})
        d.pop(ks[-1], None)
    c["train"]["output_dir"] = f"outputs/{name}"; c["wandb"]["run_name"] = name
    p = ROOT / f"configs/{name}.yaml"; p.write_text(yaml.safe_dump(c, sort_keys=False)); return str(p)


def jepa_stage():
    while True:
        st = stages(); cur = st[-1]
        if not complete(cur):
            time.sleep(120); continue
        last = epochs_of(cur)[-1]
        if metrics(last) is None:
            time.sleep(120); continue
        all_e = [e for d in st for e in epochs_of(d)]
        m = {e: metrics(e) for e in all_e}
        if any(v is None for v in m.values()):
            time.sleep(120); continue
        es = sorted(m)
        recent, before = es[-2:], es[-4:-2]
        gain = {k: sum(m[e][k] for e in recent) / 2 - sum(m[e][k] for e in before) / 2 for k in m[es[-1]]}
        log(f"JEPA epochs {es[0]}..{es[-1]} | last {m[es[-1]]} | gain(last2 - prev2) {gain}")
        improving = gain["ridge_t2i_MRR"] >= 0.01 or gain["rot_i2t_MRR"] >= 0.01 or gain["hard_cca"] >= 0.02
        if improving and last + 1 < 28:
            n = len(st) + 1; total = min(28, last + 1 + 7)
            cfg = write_config(ROOT / f"configs/{RUN}.yaml", f"{RUN}_c{n}",
                               {"train.resume": str(cur / f"epoch_{last:03d}.pt"), "train.epochs": total}, drop=("train.init_checkpoint",))
            log(f"still improving -> continue JEPA to {total} epochs ({RUN}_c{n})")
            launch(f"xpc{n}", 0, cfg)
            while not (ROOT / f"outputs/{RUN}_c{n}/train.log").exists():
                time.sleep(60)
            continue
        best = max(es, key=lambda e: m[e]["ridge_t2i_MRR"])
        log(f"JEPA plateau (or cap) at epoch {last}; best epoch by oracle ridge MRR = {best} {m[best]}")
        for d in st:
            if (d / f"epoch_{best:03d}.pt").exists():
                return d / f"epoch_{best:03d}.pt", best


def lora_stage(ck, best):
    out = ROOT / f"outputs/{LORA}"; out.mkdir(parents=True, exist_ok=True)
    p = torch.load(ck, map_location="cpu", weights_only=False)
    state = {k: v for k, v in p["model"].items() if not re.search(r"\.(text|image)_[AB]$", k)}
    torch.save({"model": state, "args": p["args"], "text_vocabulary": p["text_vocabulary"],
                "lora_modules": p.get("lora_modules")}, out / "init_trunk.pt")
    log(f"stripped private LoRA from {ck} -> {out / 'init_trunk.pt'} ({len(p['model']) - len(state)} LoRA tensors removed)")
    base_cfg = ROOT / "configs/mm_unpaired_private_dropout_gradbal_2m_40e.yaml"
    edits = {"train.init_checkpoint": str(out / "init_trunk.pt"), "train.epochs": 2, "lora.freeze_base": True,
             "lora.private_dropout": 0.0, "alignment.gradient_balance_enabled": False, "wandb.group": "shared_xview_frozen_lora"}
    cfg = write_config(base_cfg, LORA, edits, drop=("train.resume",))
    launch("xplora", 0, cfg); log(f"frozen-trunk private-LoRA diffusion started ({LORA}, from JEPA epoch {best})")
    stage, total = LORA, 2
    while True:
        d = ROOT / f"outputs/{stage}"
        while not (d / "train.log").exists() or "training complete" not in (d / "train.log").read_text():
            time.sleep(120)
        vals = [float(x) for x in re.findall(r"validation epoch=\d+ fixed-t=0.75 loss=([0-9.]+)",
                                             "".join((ROOT / f"outputs/{s}/train.log").read_text() for s in
                                                     [LORA] + [f"{LORA}_c{i}" for i in range(2, 4) if (ROOT / f"outputs/{LORA}_c{i}").exists()]))]
        log(f"LoRA validation losses so far: {vals}")
        if total < 6 and len(vals) >= 2 and (vals[-2] - vals[-1]) / vals[-2] > 0.01:
            n = total // 2 + 1; total += 2; last = epochs_of(d)[-1]
            cfg = write_config(ROOT / f"configs/{LORA}.yaml", f"{LORA}_c{n}",
                               {"train.resume": str(d / f"epoch_{last:03d}.pt"), "train.epochs": total}, drop=("train.init_checkpoint",))
            log(f"LoRA still improving -> continue to {total} epochs"); launch(f"xplora{n}", 0, cfg)
            stage = f"{LORA}_c{n}"
            while not (ROOT / f"outputs/{stage}/train.log").exists():
                time.sleep(60)
            continue
        return d / f"epoch_{epochs_of(d)[-1]:03d}.pt"


def final_eval(lora_ck, best_jepa_ck):
    B = "outputs/mm_unpaired_private_dropout_gradbal_2m_40e"
    cmd = (f"cd {ROOT} && {ENV} CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:scripts PYTHONWARNINGS=ignore && "
           f"python3 scripts/val_diffusion_per_modality.py --output outputs/eval_shared_xview/val_nelbo.json "
           f"--checkpoint frozen_xpr_lora_full={lora_ck} --checkpoint frozen_xpr_lora_trunk=TRUNK:{lora_ck} "
           f"--checkpoint xpr_jepa_full={best_jepa_ck} "
           f"--checkpoint base_e09_full={B}/epoch_008.pt --checkpoint base_e09_trunk=TRUNK:{B}/epoch_008.pt "
           f"--checkpoint base_e13_full={B}/epoch_012.pt "
           f"--checkpoint denseT_e13=outputs/text_dense_diffusion_2m_20e_continued/epoch_012.pt "
           f"--checkpoint denseI_e13=outputs/image_dense_diffusion_2_5m_40e/epoch_012.pt "
           f"> outputs/eval_shared_xview/logs/val_nelbo.log 2>&1 && "
           f"python3 generate_image_samples.py --num-samples 16 --num-steps 100 --reveal-order random "
           f"--output-dir outputs/eval_shared_xview/samples --checkpoint frozen_xpr_lora={lora_ck} "
           f"--checkpoint base_e09={B}/epoch_008.pt > outputs/eval_shared_xview/logs/samples.log 2>&1")
    subprocess.run(["bash", "-c", cmd]); log("final NELBO + samples written to outputs/eval_shared_xview/")


if __name__ == "__main__":
    log("pipeline started")
    ck, best = jepa_stage()
    lora_ck = lora_stage(ck, best)
    final_eval(lora_ck, ck)
    log("PIPELINE COMPLETE")
