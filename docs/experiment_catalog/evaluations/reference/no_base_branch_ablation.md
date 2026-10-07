# No-base branch ablation

> **Type:** method reference, earlier multimodal phase  
> **Script:** [analyze_branch_ablation_multiratio.py](/home/zd25e122/clevr_discrete_diffusion/analyze_branch_ablation_multiratio.py)  
> **Results:** stored per run in its output directory and W&B history  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Full/shared-only/private-only LoRA branch loss at fixed mask ratios.

## How it works

For a no-base Tri-LoRA checkpoint, evaluates the normal full routing, shared-only routing, and private-only routing at chosen fixed mask ratios.  This isolates which branch supplies predictive information.

## Where the results are

The original per-checkpoint tables are run-specific outputs/W&B logs; no single aggregate value is appropriate.
