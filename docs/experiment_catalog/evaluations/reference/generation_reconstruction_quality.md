# Generation and reconstruction quality

> **Type:** method reference, earlier multimodal phase  
> **Script:** [evaluate_checkpoint_quality.py](/home/zd25e122/clevr_discrete_diffusion/evaluate_checkpoint_quality.py)  
> **Results:** stored per run in its output directory and W&B history  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Marginal and conditional reconstruction/generation diagnostics.

## How it works

Generates/reconstructs marginal and conditional samples.  Marginal evaluation begins with a fully masked modality; conditional evaluation keeps one modality as context and generates the other.

## Where the results are

Values and qualitative samples are kept in the run-specific output/W&B artifacts.
