# JEPA objective checkpoint test

> **Type:** method reference, earlier multimodal phase  
> **Script:** [evaluate_jepa_checkpoints.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_jepa_checkpoints.py)  
> **Report:** [REPORT.md](/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_evaluation_256/REPORT.md)  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Fixed-mask masked-position prediction metrics against clean targets.

## How it works

At fixed masked positions, compares the student shared readout (and, when present, its JEPA predictor) with clean stop-gradient or EMA-teacher targets.  This tests the JEPA objective itself, not cross-modal retrieval.
