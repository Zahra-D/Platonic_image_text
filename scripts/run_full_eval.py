"""Full, consistent evaluation of the inventory's eligible checkpoints (all layers).

Builds two memory-guarded eval chains from outputs/model_inventory.json and starts
them in tmux:
  chain A (GPU_A): text structure / probe suite / scene retrieval / sensitivity,
                   and the JEPA prediction check (all JEPA checkpoints)
  chain B (GPU_B): image structure / probe suite / scene retrieval / binding
                   (pairwise + strict) / sensitivity / triples d', and cross-modal
                   retrieval for every multimodal model plus matched dense pairs
dense_private checkpoints are loaded TRUNK: (shared trunk only). Labels are
<run>__e<epoch>, so results never collide with older caches. Output: outputs/eval_all/.

  python3 scripts/run_full_eval.py [--epochs last|all] [--gpu-a 1 --gpu-b 2] [--dry-run]
"""
from __future__ import annotations
import argparse, json, shlex, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Q = "outputs/eval_all"
TXTVAL = "${CLEVR_DATA}/platonic_text_only_v1_1m/val_text_only_human.jsonl"
CACHE = "outputs/image_only_2_5m_token_cache/val_tokens.pt"
REF_TEXT = "outputs/text_dense_diffusion_2m_4e_matched/epoch_003.pt"   # fixes the text-sensitivity benchmark
DENSE_T = {1: "text_dense_diffusion_2m_4e_matched/epoch_000", 2: "text_dense_diffusion_2m_4e_matched/epoch_001",
           4: "text_dense_diffusion_2m_4e_matched/epoch_003", 7: "text_dense_diffusion_2m_8e_continued/epoch_006",
           10: "text_dense_diffusion_2m_12e_continued/epoch_009", 20: "text_dense_diffusion_2m_20e_continued/epoch_019",
           40: "text_dense_diffusion_2m_40e_continued/epoch_039"}

GUARD = r'''#!/bin/bash
set -u; cd {root}
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES={gpu}; MIN_MB=${{MIN_MB:-4000}}; START_MB=${{START_MB:-9000}}
Q={q}; mkdir -p $Q/logs
log(){{ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }}
avail(){{ awk '/MemAvailable/{{print int($2/1024)}}' /proc/meminfo; }}
attempt(){{ local name=$1; shift
  local waited=0; while [ "$(avail)" -lt $START_MB ] && [ $waited -lt 1800 ]; do sleep 30; waited=$((waited+30)); done
  "$@" >> $Q/logs/$name.log 2>&1 & local pid=$! low=0
  while kill -0 $pid 2>/dev/null; do
    if [ "$(avail)" -lt $MIN_MB ]; then low=$((low+1)); else low=0; fi
    if [ $low -ge 3 ]; then kill $pid; wait $pid 2>/dev/null; return 2; fi
    sleep 5; done
  wait $pid; }}
run(){{ local name=$1; shift; log "START $name"
  attempt $name "$@"; local rc=$?
  if [ $rc -eq 2 ]; then log "ABORTED $name (low RAM), retrying once"; attempt $name "$@"; rc=$?; fi
  case $rc in 0) log "DONE $name";; 2) log "ABORTED $name again (low RAM)";; *) log "FAILED $name";; esac
  return $rc; }}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", choices=["last", "all"], default="last")
    p.add_argument("--gpu-a", default="1"); p.add_argument("--gpu-b", default="2")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    inv = [r for r in json.load(open(ROOT / "outputs" / "model_inventory.json")) if r.get("include")]
    text, image, jepa, mm = [], [], [], []
    for r in inv:
        for ep in (r["epochs_saved"] if a.epochs == "all" else r["epochs_saved"][-1:]):
            path = f"outputs/{r['run']}/epoch_{ep:03d}.pt"
            spec_path = ("TRUNK:" if r["train_mode"] == "dense_private" else "") + path
            label = f"{r['run']}__e{ep:03d}"
            mods = {"text": ["text"], "image": ["image"]}.get(r["modality"], ["text", "image"])
            if "text" in mods: text.append((label, spec_path))
            if "image" in mods: image.append((label, spec_path))
            if len(mods) == 2: mm.append(label)
            if "I-JEPA" in r["objective"] or "token-level" in r["objective"]:
                jepa.append((label, path))   # full checkpoint: the script routes trunk-only itself
    ck = lambda specs, fmt="--checkpoint {}={}": " ".join(fmt.format(l, shlex.quote(s)) for l, s in specs)

    def chunked(lines, name, prefix, specs, suffix="", fmt="--checkpoint {}={}", head=(), size=8):
        """One invocation per chunk of `size` checkpoints; a failed chunk is retried one by one."""
        for c in range(0, len(specs), size):
            part = specs[c:c + size]
            cmd = lambda sp: f"{prefix} {ck(list(head), '--checkpoint {}={}')} {ck(sp, fmt)} {suffix}".strip()
            singles = "; ".join(f"run {name}_c{c // size}_s{j} {cmd([x])}" for j, x in enumerate(part))
            lines.append(f"run {name}_c{c // size} {cmd(part)} || {{ {singles}; }}")
    rnd_t, rnd_i = ("random", f"RANDOM:{REF_TEXT}"), ("random", "RANDOM:outputs/image_dense_diffusion_2_5m_40e/epoch_000.pt")
    A = [GUARD.format(root=ROOT, gpu=a.gpu_a, q=Q)]
    chunked(A, "structure_text", "python3 evaluate_cross_modal_structure.py", text,
            f"--image-cache {CACHE} --num-scenes 4000 --seed 20260922 --output-dir $Q/structure", "--text-checkpoint {}_TEXT={}")
    chunked(A, "probes_text", f"python3 evaluate_probe_suite.py --modality text --manifest {TXTVAL} --num-scenes 6000 "
            "--noise-levels 0.0 0.25 0.5 0.75 --output-dir $Q/probe_suite/text", [rnd_t] + text)
    chunked(A, "scene_text", f"python3 evaluate_scene_retrieval.py --modality text --manifest {TXTVAL} --num-scenes 2500 "
            "--output-dir $Q/scene_retrieval/text", [rnd_t] + text)
    chunked(A, "sens_text", "python3 evaluate_text_sensitivity.py --num-worlds 2000 --output-dir $Q/sens_text",
            [rnd_t] + text, head=[("ref_text_dense_4e", REF_TEXT)])
    chunked(A, "jepa_prediction", "python3 evaluate_jepa_prediction.py --output-dir $Q/jepa_prediction", jepa)
    A.append('log "CHAIN A COMPLETE"')
    B = [GUARD.format(root=ROOT, gpu=a.gpu_b, q=Q)]
    chunked(B, "structure_image", "python3 evaluate_cross_modal_structure.py", image,
            f"--image-cache {CACHE} --num-scenes 4000 --seed 20260922 --output-dir $Q/structure", "--image-checkpoint {}_IMAGE={}")
    chunked(B, "probes_image", "python3 evaluate_probe_suite.py --modality image --num-scenes 6000 "
            "--noise-levels 0.0 0.25 0.5 0.75 --output-dir $Q/probe_suite/image", [rnd_i] + image)
    chunked(B, "scene_image", "python3 evaluate_scene_retrieval.py --modality image --num-scenes 2500 "
            "--output-dir $Q/scene_retrieval/image", [rnd_i] + image)
    chunked(B, "binding", f"python3 evaluate_image_binding.py --per-layer --seed 20260924 --image-cache {CACHE} "
            "--output-dir $Q/binding", image)
    chunked(B, "sens_image", "python3 evaluate_semantic_sensitivity.py --output-dir $Q/sens_image", [rnd_i] + image)
    chunked(B, "triples", "python3 evaluate_image_triples.py --triples outputs/image_eval_triples/triples_tokens.pt "
            "--output-dir $Q/triples", image)
    # cross-modal retrieval: each multimodal model's own text x image, plus matched separately trained dense pairs
    spec = dict(text); xt, xi, pairs = [], [], []
    for l in mm:
        xt.append((f"{l}_TEXT", spec[l])); xi.append((f"{l}_IMAGE", dict(image)[l])); pairs.append(f"{l}_TEXT|{l}_IMAGE")
    for e, tpath in DENSE_T.items():
        lt, li = f"denseT_e{e:03d}_TEXT", f"denseI_e{e:03d}_IMAGE"
        xt.append((lt, f"outputs/{tpath}.pt")); xi.append((li, f"outputs/image_dense_diffusion_2_5m_40e/epoch_{e-1:03d}.pt"))
        pairs.append(f"{lt}|{li}")
    xt.append(("random_TEXT", rnd_t[1])); xi.append(("random_IMAGE", rnd_i[1])); pairs.append("random_TEXT|random_IMAGE")
    xcmd = lambda T, I, P: ("python3 evaluate_cross_modal_retrieval.py --num-scenes 8000 --test 1000 --val 1000 "
                            f"--output-dir $Q/xret {ck(T, '--text-checkpoint {}={}')} {ck(I, '--image-checkpoint {}={}')} "
                            + " ".join(f"--pair {shlex.quote(x)}" for x in P))
    tmap, imap = dict(xt), dict(xi)
    singles = "; ".join(f"run xret_p{j} " + xcmd([(t, tmap[t])], [(i, imap[i])], [f"{t}|{i}"])
                        for j, (t, i) in enumerate(x.split("|") for x in pairs))
    B.append(f"run xret {xcmd(xt, xi, pairs)} || {{ {singles}; }}")
    B.append('log "CHAIN B COMPLETE"')
    out = ROOT / Q; out.mkdir(parents=True, exist_ok=True)
    (out / "chain_A.sh").write_text("\n".join(A) + "\n"); (out / "chain_B.sh").write_text("\n".join(B) + "\n")
    (out / "plan.json").write_text(json.dumps({"epochs": a.epochs, "text": text, "image": image, "jepa": jepa,
                                               "multimodal": mm, "dense_pairs": DENSE_T}, indent=2) + "\n")
    print(f"text models {len(text)} | image models {len(image)} | JEPA checks {len(jepa)} | "
          f"cross-modal pairs {len(pairs)} -> {Q}/chain_A.sh, chain_B.sh")
    if a.dry_run:
        return
    for name, script in (("evalallA", "chain_A.sh"), ("evalallB", "chain_B.sh")):
        subprocess.run(["tmux", "new-session", "-d", "-s", name,
                        f"cd {ROOT} && bash {Q}/{script} 2>&1 | tee -a {Q}/{script}.log"], check=True)
    print("started tmux sessions evalallA, evalallB")


if __name__ == "__main__":
    main()
