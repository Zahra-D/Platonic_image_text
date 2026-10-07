# Semantic clustering and probes

> **Type:** method reference, earlier multimodal phase  
> **Script:** [evaluate_shared_semantics.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_shared_semantics.py)  
> **Report:** [REPORT.md](/home/zd25e122/clevr_discrete_diffusion/outputs/shared_semantics/REPORT.md)  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Frozen shared/private probes and clustering for CLEVR semantics.

## How it works

Fits frozen probes/clusters on representation features to test whether CLEVR object count, color, shape, material, size, and relation are decodable.  These are representation-quality tests, not direct cross-modal alignment tests.
