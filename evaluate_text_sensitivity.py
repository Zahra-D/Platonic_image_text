"""Semantic-vs-nuisance sensitivity for text encoders.

The text counterpart of evaluate_semantic_sensitivity, built on the same
lexicon-matched benchmark as evaluate_semantic_dprime:

    nuisance change : query -> paraphrase   same scene, different wording plan
    binding change  : paraphrase -> swap    one binding exchanged, rendered by
                                            the paraphrase's own plan, so the
                                            two are word-for-word identical
    content change  : query -> twin         a different scene in the query's
                                            own template

    S_binding = E[d(paraphrase, swap)] / E[d(query, paraphrase)]
    S_content = E[d(query, twin)]      / E[d(query, paraphrase)]

S_binding is the direct analogue of the image S (there the nuisance is a
camera change and the binding change is a colour exchange at a fixed camera).
S > 1: the representation moves more for meaning than for phrasing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
import evaluate_semantic_dprime as E
from evaluate_probe_suite import load_model_or_random


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--manifest", default="/home/zd25e122/clevr-dataset-gen_clone/output/platonic_text_only_v1_1m/val_text_only_human.jsonl")
    p.add_argument("--caption-generator", default="/home/zd25e122/clevr-dataset-gen_clone/image_generation/generate_human_captions.py")
    p.add_argument("--num-worlds", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=190)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    a.lexicon_matched = True
    specs = [s.split("=", 1) for s in a.checkpoint]
    # the benchmark text is built once; every model shares the 2M-corpus vocabulary
    first_real = next(pth for _, pth in specs if not pth.startswith("RANDOM:"))
    _, tok0, _ = load_model_or_random(first_real, torch.device("cpu"))
    texts, items, stats = E.build(a, tok0)
    print(f"{len(items)} items, dropped {dict(stats)}", flush=True)
    idx = {k: np.array([it[k] for it in items]) for k in ("query", "paraphrase", "twin", "swap")}

    for label, path in specs:
        dest = out / f"{label}.json"
        if dest.exists():
            print(f"{label}: exists, skipping", flush=True); continue
        model, tok, margs = load_model_or_random(path, a.device)
        feats, worst = E.encode(model, tok, margs, texts, a.batch_size, a.device, set(range(8)),
                                check=True, ablate_private=False, ablate_shared=False)
        rec = {}
        for name, F in feats.items():
            F = F.astype(np.float64)
            F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-12, None)
            d = lambda u, v: 1.0 - (F[idx[u]] * F[idx[v]]).sum(1)
            dn, db, dc = d("query", "paraphrase"), d("paraphrase", "swap"), d("query", "twin")
            rec[name] = {"nuisance_mean": float(dn.mean()), "binding_mean": float(db.mean()),
                         "content_mean": float(dc.mean()),
                         "S_binding": float(db.mean() / max(dn.mean(), 1e-12)),
                         "S_content": float(dc.mean() / max(dn.mean(), 1e-12)),
                         "fraction_binding_larger": float((db > dn).mean()),
                         "fraction_content_larger": float((dc > dn).mean())}
        best = max(rec, key=lambda k: rec[k]["S_binding"])
        dest.write_text(json.dumps({"protocol": {"model": str(path), "items": len(items)},
                                    "metrics": rec}, indent=2) + "\n")
        l7 = rec.get("L7.residual", rec[best])
        print(f"{label}: L7 S_bind {l7['S_binding']:.3f} S_cont {l7['S_content']:.3f} | "
              f"best S_bind {rec[best]['S_binding']:.3f} @{best} "
              f"(bind {rec[best]['binding_mean']:.5f} / nui {rec[best]['nuisance_mean']:.5f})", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
