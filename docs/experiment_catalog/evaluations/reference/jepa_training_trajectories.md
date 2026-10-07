# JEPA training trajectories

> **Type:** method reference, earlier multimodal phase  
> **Script:** [analyze_wandb_jepa_trajectories.py](/home/zd25e122/clevr_discrete_diffusion/analyze_wandb_jepa_trajectories.py)  
> **Report:** [REPORT.md](/home/zd25e122/clevr_discrete_diffusion/outputs/jepa_wandb_trajectory_report/REPORT.md)  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

JEPA loss/cosine trajectory and W&B history.

## How it works

Reads W&B JEPA losses and predictor/target diagnostics across training steps.  It is used to detect collapse, late divergence, or a loss that is too small relative to diffusion training.
