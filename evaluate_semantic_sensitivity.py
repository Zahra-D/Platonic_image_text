"""Semantic-vs-nuisance sensitivity on controlled CLEVR triples.

    nuisance change : anchor -> paraphrase   (camera & lights re-jittered,
                                              scene content identical)
    semantic change : paraphrase -> swap     (identical camera, two objects'
                                              colours exchanged)

    S = E[d(para, swap)] / E[d(anchor, para)]

S > 1 means the representation moves further for a change of meaning than for
a change of viewpoint. S < 1 means viewpoint dominates.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from evaluate_image_triples import encode
from sensitivity_geometry import geometry_variants
from evaluate_probe_suite import load_model_or_random


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--triples", default="outputs/image_eval_triples/triples_tokens.pt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    pay = torch.load(a.triples, map_location="cpu", weights_only=False)
    T = {s: torch.from_numpy(pay["tokens"][s].numpy().astype("int64")) for s in ("anchor", "para", "swap")}
    print(f"{T['anchor'].shape[0]} triples", flush=True)

    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        dest = out / f"{label}.json"
        if dest.exists():
            print(f"{label}: exists, skipping", flush=True); continue
        model, tok, margs = load_model_or_random(path, a.device)
        f = {s: encode(model, tok, margs, t, a.batch_size, a.device) for s, t in T.items()}
        rec = {}
        n = len(f["anchor"][next(iter(f["anchor"]))])
        for name in f["anchor"]:
            stacked = np.concatenate([f["anchor"][name], f["para"][name], f["swap"][name]])
            variants = {}
            for geometry, x in geometry_variants(stacked).items():
                anchor, para, swap = x[:n], x[n:2 * n], x[2 * n:]
                dn = 1.0 - (anchor * para).sum(1)   # nuisance: camera / light jitter
                ds = 1.0 - (para * swap).sum(1)     # semantic: two objects swap colours
                variants[geometry] = {
                    "nuisance_mean": float(dn.mean()), "semantic_mean": float(ds.mean()),
                    "S_ratio_of_means": float(ds.mean() / max(dn.mean(), 1e-12)),
                    "S_median_paired": float(np.median(ds / np.clip(dn, 1e-12, None))),
                    "fraction_semantic_larger": float((ds > dn).mean()),
                }
            # raw keys stay at the top level (earlier results), the others nested
            rec[name] = {**variants["raw"], "centred": variants["centred"], "zscore": variants["zscore"]}
        best = max(rec, key=lambda k: rec[k]["S_ratio_of_means"])
        dest.write_text(json.dumps({"protocol": {"model": str(path),
                                                 "triples": int(T["anchor"].shape[0])},
                                    "metrics": rec}, indent=2) + "\n")
        l7 = rec["L7"]
        print(f"{label}: L7 " + " | ".join(
            f"{g} S={v['S_ratio_of_means']:.3f} (d_s {v['semantic_mean']:.5f} / d_n {v['nuisance_mean']:.5f})"
            for g, v in (("raw", l7), ("centred", l7["centred"]), ("zscore", l7["zscore"]))), flush=True)
        print(f"{label}: L7 S={rec['L7']['S_ratio_of_means']:.3f} | "
              f"best {best} S={rec[best]['S_ratio_of_means']:.3f} "
              f"(sem {rec[best]['semantic_mean']:.5f} / nui {rec[best]['nuisance_mean']:.5f})", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
