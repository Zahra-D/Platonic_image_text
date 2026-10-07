# Paired conditional validation

> **Type:** method reference, earlier multimodal phase  
> **Script:** [alignment_evaluation.py](/home/zd25e122/clevr_discrete_diffusion/alignment_evaluation.py)  
> **Results:** stored per run in its output directory and W&B history  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Matched, shuffled, null, selection, and translation directional losses on reserved true pairs.

## How it works

Uses reserved true text-image pairs.  It compares matched context, deliberately shuffled context, and null (no-condition) context; lower token loss is better.  Directional text-to-image and image-to-text variants use the corresponding modality as clean condition.

## Where the results are

Values are logged per run because the evaluation depends on the checkpoint and its held-out split.
