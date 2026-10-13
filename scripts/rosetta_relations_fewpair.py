"""Unpaired / few-pair alignment with relational captions (see evaluate_unpaired_rosetta.py).

Part A (--part rel): CKA, paired orthogonal ceiling and unpaired FOSCTTM for image x
text pairs whose text side was encoded on captions with relations (_REL features).
Part B (--part few): the paper's method with p known pairs plus the unpaired halves,
against an orthogonal map and a ridge map fitted on the same p pairs alone.
Known pairs are drawn from the image half of the fit scenes (their captions come along);
the unpaired caption half stays disjoint.  Test: the 1000 held-out pairs.
"""
from __future__ import annotations
import os; os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse, json, sys, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_unpaired_rosetta import fit_unpaired, prep, retrieval, polar, lin_cka
from evaluate_cross_modal_retrieval import Ridge

FEAT = "outputs/eval_all/xret/features/"
perm = np.random.default_rng(20260922).permutation(8000)
TE = np.sort(perm[:1000]); FIT = perm[1000:]; H = len(FIT) // 2
IR, TR = np.sort(FIT[:H]), np.sort(FIT[H:]); TRVA = np.sort(FIT)
ARGS = argparse.Namespace(repeats=30, centres=30, qap_restarts=1000, rounds=100, batch=1500)


def load(spec):
    lab, r = spec.split(":")
    if lab.startswith("ORACLE"):
        sys.path.insert(0, "scripts")
        from feature_variance_sources import descriptors, zs
        from clevr_paths import CLEVR_DATA_ROOT
        a = argparse.Namespace(num_scenes=8000, image_manifest=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl",
                               image_cache="outputs/image_only_2_5m_token_cache/val_tokens.pt")
        D = descriptors(a)
        return zs(np.concatenate([D["content"], D["bound"]], 1), FIT)
    return np.load(FEAT + lab + ".npz")[r].astype(float)


def ceiling(X, Y):
    wp = polar(prep(X[TRVA], X[TRVA].mean(0)).T @ prep(Y[TRVA], Y[TRVA].mean(0)))
    return retrieval(prep(X[TE], X[TRVA].mean(0)) @ wp, prep(Y[TE], Y[TRVA].mean(0)))


def part_rel(job):
    name, xs, ys = job
    X, Y = load(xs), load(ys)
    mx, my = X[IR].mean(0), Y[TR].mean(0)
    w = fit_unpaired(prep(X[IR], mx), prep(Y[TR], my), ARGS, np.random.default_rng(1))
    u = retrieval(prep(X[TE], mx) @ w, prep(Y[TE], my))
    c = ceiling(X, Y)
    return {"name": name, "cka": lin_cka(X[TE], Y[TE]), "unpaired_FOSCTTM": u["FOSCTTM"],
            "unpaired_R1": u["text_to_image"]["R@1"], "ceiling_FOSCTTM": c["FOSCTTM"], "ceiling_R1": c["text_to_image"]["R@1"]}


def part_few(job):
    name, xs, ys, p, seed = job
    X, Y = load(xs), load(ys)
    rng = np.random.default_rng(100 + seed)
    known = np.sort(rng.choice(IR, p, replace=False)) if p else np.array([], int)
    mx, my = X[IR].mean(0), Y[TR].mean(0)
    xf, yf = prep(X[IR], mx), prep(Y[TR], my)
    xp, yp = (prep(X[known], mx), prep(Y[known], my)) if p else (None, None)
    w = fit_unpaired(xf, yf, ARGS, np.random.default_rng(1 + seed), xp, yp)
    out = {"name": name, "p": int(p), "seed": seed}
    u = retrieval(prep(X[TE], mx) @ w, prep(Y[TE], my))
    out.update(method_FOSCTTM=u["FOSCTTM"], method_R1=u["text_to_image"]["R@1"], method_R1_i2t=u["image_to_text"]["R@1"])
    if p >= 2:
        wo = polar(xp.T @ yp)
        o = retrieval(prep(X[TE], mx) @ wo, prep(Y[TE], my))
        wr = Ridge(xp).maps(yp, (1.0,))[1.0]
        r = retrieval(prep(X[TE], mx) @ wr, prep(Y[TE], my))
        out.update(pairs_only_orth_FOSCTTM=o["FOSCTTM"], pairs_only_orth_R1=o["text_to_image"]["R@1"],
                   pairs_only_ridge_FOSCTTM=r["FOSCTTM"], pairs_only_ridge_R1=r["text_to_image"]["R@1"])
    return out


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "rel":
        jobs = [("our image dense40 L7 x text dense40 L7 REL", "denseI_e040_IMAGE:L7", "denseT_e040_TEXT_REL:L7"),
                ("our image dense13 FG L6 x text dense13 L6 REL", "denseI_e013_IMAGE_FG:L6", "denseT_e013_TEXT_REL:L6"),
                ("shared+JEPA A L6.mlp_out (image x text REL)", "dropA_tot013_IMAGE:L6.mlp_out", "dropA_tot013_TEXT_REL:L6.mlp_out"),
                ("shared base L6.mlp_out (image x text REL)", "drop_e13_IMAGE:L6.mlp_out", "drop_e13_TEXT_REL:L6.mlp_out"),
                ("DINOv2 CLS x text dense40 L7 REL", "dinov2_IMAGE:cls", "denseT_e040_TEXT_REL:L7"),
                ("DINOv2 CLS x mpnet REL", "dinov2_IMAGE:cls", "mpnet_TEXT_REL:mean"),
                ("ORACLE content+where x text dense13 L7 REL", "ORACLE:x", "denseT_e013_TEXT_REL:L7"),
                ("ORACLE content+where x text dense13 L7 (no rel)", "ORACLE:x", "denseT_e013_TEXT:L7")]
        with Pool(8) as pool:
            res = list(pool.imap(part_rel, jobs))
        for r in res:
            print("%-50s CKA %.2f | unpaired FOSCTTM %.3f R@1 %.3f | ceiling FOSCTTM %.3f R@1 %.3f" % (
                r["name"], r["cka"], r["unpaired_FOSCTTM"], r["unpaired_R1"], r["ceiling_FOSCTTM"], r["ceiling_R1"]), flush=True)
        Path("outputs/eval_rosetta/relations.json").write_text(json.dumps(res, indent=2))
    else:
        settings = [("our dense40 L7 (REL)", "denseI_e040_IMAGE:L7", "denseT_e040_TEXT_REL:L7"),
                    ("shared+JEPA A L6.mlp_out (REL)", "dropA_tot013_IMAGE:L6.mlp_out", "dropA_tot013_TEXT_REL:L6.mlp_out"),
                    ("our dense13 FG L6 (REL)", "denseI_e013_IMAGE_FG:L6", "denseT_e013_TEXT_REL:L6")]
        jobs = [(n, xs, ys, p, s) for n, xs, ys in settings for p in (0, 5, 10, 20, 50, 100, 300) for s in range(3)]
        with Pool(8) as pool:
            res = list(pool.imap_unordered(part_few, jobs))
        Path("outputs/eval_rosetta/fewpair.json").write_text(json.dumps(res, indent=2))
        print("saved", len(res), flush=True)
