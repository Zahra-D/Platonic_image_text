"""Run the three PixMo-Cap stages, each until it converges, as a sequence of time-limited jobs.

Every call continues where the previous one stopped (state in $PIXMO_RUNS/pipeline_state.json) and exits cleanly
before --budget-min, so the same command can be submitted again and again (pixmo/slurm_pipeline.sh resubmits itself).

  stage1  dense_private unpaired diffusion (configs/pixmo/stage1_dense_private_diffusion.yaml) from scratch.
          Converged = the trainer's early stopping on the validation loss (patience 3, min delta 0.01).
  stage2  trunk-only cross-view / cross-paraphrase JEPA (configs/pixmo/stage2_xview_xpara_trunk_jepa.yaml) from
          stage-1 best.pt. Converged = the trunk's oracle-W score (pixmo/eval_oracle.py: mean image->text / text->image
          R@1 at the best readout) has not beaten its best by --stage2-min-gain for --stage2-patience epochs (at least
          --stage2-min-epochs epochs). Its best epoch, with the private LoRA removed, becomes init_trunk.pt.
  stage3  frozen stage-2 trunk + fresh private LoRA, diffusion (configs/pixmo/stage3_frozen_trunk_private_lora.yaml).
          Converged = early stopping on the validation loss (patience 2, min delta 0.02).

Every stage-1 and stage-2 epoch is evaluated (trunk oracle W) and kept in <stage>/eval/epoch_XXX.json; epoch
checkpoints are deleted after evaluation except the ones still needed (best, last).

    PIXMO_DATA=... PIXMO_RUNS=... python pixmo/pipeline.py --ngpu 4 --budget-min 80
    python pixmo/pipeline.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.setdefault("PIXMO_DATA", str(Path.home() / "platonic_pixmo_data")))
RUNS = Path(os.environ.setdefault("PIXMO_RUNS", str(REPO / "runs_pixmo")))
STAGES = {
    "stage1": ("configs/pixmo/stage1_dense_private_diffusion.yaml", RUNS / "stage1_dense_private_diffusion"),
    "stage2": ("configs/pixmo/stage2_xview_xpara_trunk_jepa.yaml", RUNS / "stage2_xview_xpara_trunk_jepa"),
    "stage3": ("configs/pixmo/stage3_frozen_trunk_private_lora.yaml", RUNS / "stage3_frozen_trunk_private_lora"),
}
NEXT = {"stage1": "stage2", "stage2": "stage3", "stage3": "done"}
PRIVATE = ("text_A", "text_B", "image_A", "image_B")


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(RUNS / "pipeline.log", "a") as handle:
        handle.write(line + "\n")


def load_state() -> dict:
    path = RUNS / "pipeline_state.json"
    return json.loads(path.read_text()) if path.exists() else {"stage": "stage1", "seconds_per_epoch": {}, "evals": {}}


def save_state(state: dict) -> None:
    (RUNS / "pipeline_state.json").write_text(json.dumps(state, indent=1))


def epochs_done(out: Path) -> int:
    """Finished epochs of a stage, from last.pt (memory-mapped, so only the small epoch field is read)."""
    last = out / "last.pt"
    return int(torch.load(last, map_location="cpu", weights_only=False, mmap=True)["epoch"]) + 1 if last.exists() else 0


def new_epoch_files(out: Path) -> list[Path]:
    """Full epoch checkpoints (epoch_XXX.pt), not the adapter-only epoch_XXX_adapter.pt written next to them."""
    files = [p for p in out.glob("epoch_*.pt") if re.fullmatch(r"epoch_\d+\.pt", p.name)]
    return sorted(files, key=lambda p: int(re.findall(r"\d+", p.stem)[0]))


def remove(checkpoint: Path) -> None:
    for path in (checkpoint, checkpoint.with_name(f"{checkpoint.stem}_adapter.pt")):
        path.unlink(missing_ok=True)


def early_stopped(out: Path) -> bool:
    log_file = out / "train.log"
    return log_file.exists() and "early stopping at epoch" in log_file.read_text()


def train(stage: str, target_epochs: int, a) -> int:
    config, out = STAGES[stage]
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={a.ngpu}",
               "train_multimodal.py", "--config", config, "--epochs", str(target_epochs),
               "--batch-size", str(a.global_batch // a.ngpu), "--output-dir", str(out)]
    if (out / "last.pt").exists():
        command += ["--resume", str(out / "last.pt"), "--init-checkpoint", ""]
    command += ["--wandb"] if a.wandb else ["--no-wandb"]
    command += a.override.split() if a.override else []
    log(f"{stage}: training to epoch {target_epochs}: {' '.join(command)}")
    with open(out / "trainer_stdout.log", "a") as handle:
        return subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT).returncode


def eval_env() -> dict:
    """Evaluate on the first GPU this job may use (not physical GPU 0 of the node)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    return {**os.environ, "CUDA_VISIBLE_DEVICES": visible}


def evaluate(stage: str, checkpoint: Path, state: dict, a) -> dict:
    out = STAGES[stage][1] / "eval" / f"{checkpoint.stem}.json"
    if not out.exists():
        command = [sys.executable, "pixmo/eval_oracle.py", "--checkpoint", str(checkpoint), "--out", str(out),
                   "--batch-size", str(a.eval_batch)]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=eval_env())
        if result.returncode != 0:
            raise RuntimeError(f"eval of {checkpoint} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        log(f"{stage} {checkpoint.stem}: {result.stdout.strip().splitlines()[-1]}")
    r = json.loads(out.read_text())
    state["evals"].setdefault(stage, {})[checkpoint.stem] = {"score": r["score"], "readout": r["best_readout"],
                                                             "oracle": r["best"]["oracle"], "identity": r["best"]["identity"]}
    return r


def strip_private(source: Path, destination: Path) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model = {k: v for k, v in payload["model"].items() if not k.endswith(PRIVATE)}
    torch.save({"model": model, "args": payload["args"], "text_vocabulary": payload["text_vocabulary"],
                "lora_modules": payload.get("lora_modules"), "epoch": payload.get("epoch")}, destination)
    log(f"wrote {destination} from {source.name}: {len(payload['model']) - len(model)} private tensors removed")


def stage2_decision(state: dict, a) -> tuple[bool, str | None]:
    scores = state["evals"].get("stage2", {})
    if not scores:
        return False, None
    order = sorted(scores, key=lambda k: int(re.findall(r"\d+", k)[0]))
    best = max(order, key=lambda k: scores[k]["score"])
    # epochs after the best one that failed to beat it by min_gain
    after = order[order.index(best) + 1:]
    stalled = len(after) >= a.stage2_patience and all(scores[k]["score"] < scores[best]["score"] + a.stage2_min_gain for k in after)
    return (len(order) >= a.stage2_min_epochs and stalled) or len(order) >= a.stage2_max_epochs, best


def step(state: dict, a, deadline: float) -> bool:
    """Advance the current stage by one trainer run (or a stage transition). False = out of time / done."""
    stage = state["stage"]
    if stage == "done":
        return False
    config, out = STAGES[stage]
    done = epochs_done(out)
    per_epoch = state["seconds_per_epoch"].get(stage)
    remaining = deadline - time.time() - a.margin_min * 60
    epochs = 1 if per_epoch is None else int(remaining // per_epoch)
    if epochs < 1 or (per_epoch is None and remaining < a.first_epoch_min * 60):
        log(f"{stage}: {remaining / 60:.0f} min left, not enough for another epoch; exiting")
        return False
    if stage == "stage2":  # convergence is judged between trainer calls, so stage 2 advances a few epochs at a time
        epochs = min(epochs, a.stage2_chunk)
    cap = a.stage2_max_epochs if stage == "stage2" else a.max_epochs.get(stage, 10_000)
    target = min(done + epochs, cap)
    began = time.time()
    code = train(stage, target, a)
    if code != 0:
        raise RuntimeError(f"{stage} trainer exited with {code}; see {out / 'trainer_stdout.log'}")
    files = new_epoch_files(out)
    if stage in ("stage1", "stage2"):
        for f in files:
            evaluate(stage, f, state, a)
    ran = epochs_done(out) - done
    if ran > 0:  # trainer start-up and evaluation included, so the estimate is what one more epoch really costs
        state["seconds_per_epoch"][stage] = (time.time() - began) / ran * 1.05
    save_state(state)

    if stage == "stage1":
        finished = early_stopped(out) or epochs_done(out) >= a.max_epochs.get("stage1", 200)
        for f in files:  # evaluated; best.pt (lowest validation loss) and last.pt are what stage 2 / resume need
            remove(f)
    elif stage == "stage2":
        finished, best = stage2_decision(state, a)
        keep = {f"{best}.pt"} if best else set()
        for f in files:
            if f.name not in keep:
                remove(f)
        if finished:
            strip_private(out / f"{best}.pt", out / "init_trunk.pt")
            state["stage2_best"] = best
    else:
        finished = early_stopped(out) or epochs_done(out) >= a.max_epochs.get("stage3", 60)
        for f in files:
            remove(f)
        if finished:
            for name, full in (("final_trunk", False), ("final_full", True)):
                result = out / "eval" / f"{name}.json"
                command = [sys.executable, "pixmo/eval_oracle.py", "--checkpoint", str(out / "best.pt"), "--out", str(result)]
                subprocess.run(command + (["--full"] if full else []), cwd=REPO, check=True,
                               env=eval_env())
    if finished:
        state["stage"] = NEXT[stage]
        log(f"{stage} converged -> {state['stage']}")
    save_state(state)
    return state["stage"] != "done"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ngpu", type=int, default=torch.cuda.device_count() or 1)
    p.add_argument("--budget-min", type=float, default=80, help="wall-clock budget of this call")
    p.add_argument("--margin-min", type=float, default=6, help="kept free for evaluation and exit")
    p.add_argument("--first-epoch-min", type=float, default=30, help="minimum time left to start a stage's first epoch")
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--stage2-min-gain", type=float, default=0.005, help="oracle-W score gain that counts as progress")
    p.add_argument("--stage2-patience", type=int, default=3)
    p.add_argument("--stage2-min-epochs", type=int, default=4)
    p.add_argument("--stage2-max-epochs", type=int, default=60)
    p.add_argument("--stage2-chunk", type=int, default=2, help="stage-2 epochs per trainer call (then evaluate/decide)")
    p.add_argument("--max-epochs", type=json.loads, default={"stage1": 200, "stage3": 60},
                   help='JSON upper bounds, e.g. \'{"stage1": 2, "stage3": 2}\' for a smoke test')
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--override", default="", help="extra trainer flags for every stage, e.g. a tiny model for a smoke test")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if a.status:
        print(state["stage"])
        return
    deadline = time.time() + a.budget_min * 60
    log(f"start: stage {state['stage']}, budget {a.budget_min} min, {a.ngpu} GPUs, data {DATA}, runs {RUNS}")
    while step(state, a, deadline):
        pass
    log(f"stop: stage {state['stage']}")


if __name__ == "__main__":
    main()
