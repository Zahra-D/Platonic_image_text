"""Pretrained encoders on the retrieval scenes: DINOv2 (images) and all-mpnet-base-v2 (captions).

The 8000 retrieval scenes of evaluate_cross_modal_retrieval, same captions
(load_scenes, seed 20260922).  Images are the 96x64 renders, resized to 224x154
(16 x 11 patches of 14 px) with ImageNet normalisation; DINOv2 ViT-B/14 gives the
CLS token and the mean patch token.  Captions go through all-mpnet-base-v2 with
attention-masked mean pooling (its sentence-embedding recipe).  Saved, L2-normalised,
in the xret feature format: dinov2_IMAGE.npz {cls, mean}, mpnet_TEXT.npz {mean}.

  python3 scripts/pretrained_encoder_features.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clevr_paths import CLEVR_DATA_ROOT, CLEVR_GEN_ROOT
import evaluate_cross_modal_structure as X


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-manifest", default=CLEVR_DATA_ROOT + "/platonic_clevr_v1_5M_train_gpu_visible/val_image_only.jsonl")
    p.add_argument("--image-cache", default="outputs/image_only_2_5m_token_cache/val_tokens.pt")
    p.add_argument("--caption-generator", default=CLEVR_GEN_ROOT + "/generate_human_captions.py")
    p.add_argument("--num-scenes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--output-dir", default="outputs/eval_all/xret/features")
    a = p.parse_args()
    dev = "cuda"
    out = Path(a.output_dir)
    rows = [json.loads(l) for _, l in zip(range(a.num_scenes), open(a.image_manifest))]
    root = Path(a.image_manifest).parent
    if not (out / "dinov2_IMAGE.npz").exists():
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(dev).eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
        cls, avg = [], []
        with torch.no_grad():
            for s in range(0, len(rows), 128):
                ims = [torch.from_numpy(np.asarray(Image.open(root / r["image_path"]).convert("RGB"), np.float32) / 255.0)
                       .permute(2, 0, 1) for r in rows[s:s + 128]]
                x = F.interpolate(torch.stack(ims).to(dev), size=(154, 224), mode="bicubic", align_corners=False)
                feats = model.forward_features((x - mean) / std)
                cls.append(F.normalize(feats["x_norm_clstoken"], dim=1).cpu())
                avg.append(F.normalize(feats["x_norm_patchtokens"].mean(1), dim=1).cpu())
        np.savez(out / "dinov2_IMAGE.npz", cls=torch.cat(cls).numpy(), mean=torch.cat(avg).numpy())
        print("dinov2 done", flush=True)
        del model; torch.cuda.empty_cache()
    if not (out / "mpnet_TEXT.npz").exists():
        from transformers import AutoModel, AutoTokenizer
        ns = argparse.Namespace(num_scenes=a.num_scenes, seed=a.seed, caption_generator=a.caption_generator,
                                image_manifest=a.image_manifest, image_cache=a.image_cache)
        _, _, captions = X.load_scenes(ns)
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(dev).eval()
        embs = []
        with torch.no_grad():
            for s in range(0, len(captions), 128):
                b = tok(captions[s:s + 128], padding=True, truncation=True, max_length=384, return_tensors="pt").to(dev)
                h = model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                embs.append(F.normalize((h * m).sum(1) / m.sum(1), dim=1).cpu())
        np.savez(out / "mpnet_TEXT.npz", mean=torch.cat(embs).numpy())
        print("mpnet done", flush=True)


if __name__ == "__main__":
    main()
