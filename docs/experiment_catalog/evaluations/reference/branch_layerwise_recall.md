# Layerwise shared/private branch recall (paired text–image)

> **Type:** method reference, earlier multimodal phase  
> **Script:** [analyze_branch_layerwise_recall.py](/home/zd25e122/clevr_discrete_diffusion/analyze_branch_layerwise_recall.py)  
> **Results:** stored per run in its output directory and W&B history  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Recall@1/5/10 of paired text–image retrieval at every Transformer layer, read from the shared-only or private-only LoRA branch.

## How it works

Tri-LoRA models always run a normal full forward pass; hooks observe each branch's update. For each block, token-pooled adapter updates are reduced to `d_model` and averaged over `qkv`, attention output, and both MLP adapters. Retrieval uses only that branch-specific pooled vector. Dense models use the pooled residual output of each block as a branch-free reference.

Not to be confused with the text-only [cross-pattern semantic retrieval](../cross_pattern_semantic_retrieval.md), which uses held-out paraphrase galleries.

## Where the results are

Values are checkpoint- and layer-specific; retain the generated report next to the evaluation output.
