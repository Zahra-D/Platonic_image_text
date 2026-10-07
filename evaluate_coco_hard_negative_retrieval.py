#!/usr/bin/env python3
"""Zero-shot COCO caption retrieval against lexically selected hard negatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate_official_data2vec_coco_retrieval import (
    FairseqRobertaTokenizer,
    build_model,
    encode,
    load_rows,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab-json", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--hard-negatives", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tfidf-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--selected-output", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def select_hard_negatives(
    queries: list[str],
    candidates: list[str],
    owners: np.ndarray,
    positive_indices: list[list[int]],
    count: int,
    chunk_size: int,
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(queries + candidates)
    query_matrix = matrix[: len(queries)]
    candidate_matrix = matrix[len(queries) :]
    hard_indices = np.empty((len(queries), count), dtype=np.int64)
    hard_scores = np.empty((len(queries), count), dtype=np.float32)
    best_positive_tfidf = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        similarity = (query_matrix[start:end] @ candidate_matrix.T).toarray()
        for local, query_index in enumerate(range(start, end)):
            positive = positive_indices[query_index]
            best_positive_tfidf[query_index] = similarity[local, positive].max()
            similarity[local, owners == query_index] = -np.inf
            selected = np.argpartition(similarity[local], -count)[-count:]
            selected = selected[np.argsort(similarity[local, selected])[::-1]]
            hard_indices[query_index] = selected
            hard_scores[query_index] = similarity[local, selected]
        print(f"selected lexical negatives for {end}/{len(queries)} queries", flush=True)
    return hard_indices, hard_scores, best_positive_tfidf, len(vectorizer.vocabulary_)


def model_hard_metrics(
    features: torch.Tensor,
    query_count: int,
    positive_indices: list[list[int]],
    hard_indices: np.ndarray,
) -> dict:
    queries = features[:query_count].float()
    candidates = features[query_count:].float()
    max_positives = max(map(len, positive_indices))
    padded_positive = torch.zeros((query_count, max_positives), dtype=torch.long)
    positive_mask = torch.zeros((query_count, max_positives), dtype=torch.bool)
    for row, indices in enumerate(positive_indices):
        padded_positive[row, : len(indices)] = torch.tensor(indices)
        positive_mask[row, : len(indices)] = True
    hard_tensor = torch.from_numpy(hard_indices)
    ranks, margins = [], []
    pair_correct = pair_ties = pair_total = 0
    for start in range(0, query_count, 512):
        end = min(start + 512, query_count)
        query = queries[start:end]
        positive_scores = torch.einsum(
            "bpd,bd->bp", candidates[padded_positive[start:end]], query
        )
        mask = positive_mask[start:end]
        positive_scores = positive_scores.masked_fill(~mask, -torch.inf)
        negative_scores = torch.einsum(
            "bkd,bd->bk", candidates[hard_tensor[start:end]], query
        )
        best_positive = positive_scores.max(dim=1).values
        ranks.append(1 + (negative_scores > best_positive[:, None]).sum(dim=1))
        comparisons = positive_scores[:, :, None] - negative_scores[:, None, :]
        valid = mask[:, :, None].expand_as(comparisons)
        pair_correct += int(((comparisons > 0) & valid).sum())
        pair_ties += int(((comparisons == 0) & valid).sum())
        pair_total += int(valid.sum())
        margins.append(best_positive - negative_scores.max(dim=1).values)
    rank = torch.cat(ranks).float()
    margin = torch.cat(margins)
    return {
        "hard_R@1": float((rank <= 1).float().mean()),
        "hard_R@5": float((rank <= 5).float().mean()),
        "hard_MRR": float((1.0 / rank).mean()),
        "all_positive_negative_pair_accuracy": (pair_correct + 0.5 * pair_ties) / pair_total,
        "mean_best_positive_minus_hardest_negative_cosine": float(margin.mean()),
        "median_best_positive_minus_hardest_negative_cosine": float(margin.median()),
        "queries": query_count,
        "hard_negatives_per_query": int(hard_indices.shape[1]),
        "mean_same_image_positives_per_query": pair_total / query_count / hard_indices.shape[1],
    }


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    manifest = Path(args.retrieval_manifest)
    rows = load_rows(manifest, args.max_images, args.seed)
    queries = [row["query"] for row in rows]
    candidates, owners, positive_indices = [], [], [[] for _ in rows]
    for owner, row in enumerate(rows):
        for positive in row["positives"]:
            positive_indices[owner].append(len(candidates))
            candidates.append(positive["caption"])
            owners.append(owner)
    owners_array = np.asarray(owners)
    hard_indices, hard_scores, best_positive_tfidf, vocabulary_size = select_hard_negatives(
        queries,
        candidates,
        owners_array,
        positive_indices,
        args.hard_negatives,
        args.tfidf_chunk_size,
    )
    selected_path = Path(args.selected_output)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w") as handle:
        for query_index, row in enumerate(rows):
            record = {
                "image_id": row["image_id"],
                "query": queries[query_index],
                "positives": row["positives"],
                "best_positive_tfidf": float(best_positive_tfidf[query_index]),
                "hard_negatives": [
                    {
                        "caption": candidates[index],
                        "different_image_id": rows[int(owners_array[index])]["image_id"],
                        "tfidf": float(score),
                    }
                    for index, score in zip(hard_indices[query_index], hard_scores[query_index])
                ],
            }
            handle.write(json.dumps(record) + "\n")

    device = torch.device(args.device)
    tokenizer = FairseqRobertaTokenizer(args.vocab_json, args.merges, args.dictionary)
    model, config, audit = build_model(Path(args.checkpoint), device, args.random_init)
    features, unknown_rate = encode(model, tokenizer, queries + candidates, args.batch_size, device)
    metrics = {}
    for name, values in features.items():
        metrics[name] = model_hard_metrics(values, len(queries), positive_indices, hard_indices)
        score = metrics[name]
        print(
            f"{name}: hard-R@1={score['hard_R@1']:.4f} hard-MRR={score['hard_MRR']:.4f} "
            f"all-pair={score['all_positive_negative_pair_accuracy']:.4f}",
            flush=True,
        )
    lexical_challenge = hard_scores[:, 0] > best_positive_tfidf
    best = max(metrics, key=lambda name: metrics[name]["hard_MRR"])
    result = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "random_init": args.random_init,
            "retrieval_manifest": str(manifest.resolve()),
            "retrieval_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "selected_hard_negatives": str(selected_path.resolve()),
            "selected_hard_negatives_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            "images": len(rows),
            "candidates": len(candidates),
            "hard_negatives_per_query": args.hard_negatives,
            "negative_definition": "highest word/unigram-bigram TF-IDF captions belonging to different COCO images",
            "positive_definition": "all held-out captions belonging to the query COCO image",
            "hard_R@1_definition": "best same-image caption beats all selected hard negatives by encoder cosine",
            "all_pair_definition": "fraction of every same-image-positive versus selected-negative comparison won; ties score 0.5",
            "tfidf_vocabulary_size": vocabulary_size,
            "fraction_where_best_lexical_negative_exceeds_best_positive_tfidf": float(lexical_challenge.mean()),
            "mean_best_positive_tfidf": float(best_positive_tfidf.mean()),
            "mean_hardest_negative_tfidf": float(hard_scores[:, 0].mean()),
            "unknown_content_token_rate": unknown_rate,
            "model_config": config.to_dict(),
            "conversion_audit": audit,
            "caveat": "Different-image captions can occasionally be valid semantic paraphrases; this is a deliberately adversarial proxy, not perfect semantic ground truth.",
        },
        "best_layer_by_hard_MRR": best,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"best layer: {best}; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
