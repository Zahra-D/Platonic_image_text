# Diffusion training and validation loss

> **Type:** method reference, earlier multimodal phase  
> **Script:** [train_multimodal.py](/home/zd25e122/clevr_discrete_diffusion/train_multimodal.py)  
> **Results:** stored per run in its output directory and W&B history  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

W&B train/*, val/*; weighted and unweighted masked-token cross entropy and accuracy.

## How it works

During training, `train_multimodal.py` logs masked-token cross entropy and accuracy to the W&B run for each checkpoint.  It logs both the configured time-weighted loss and the unweighted masked-token loss.

## Where the results are

Use the linked pretraining model page to reach the exact W&B history and checkpoint.  This catalog does not merge train curves from different runs into one number.
