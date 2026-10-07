# Layerwise text–image gradient conflict

> **Type:** method reference, earlier multimodal phase  
> **Script:** [analyze_layer_gradient_conflict.py](/home/zd25e122/clevr_discrete_diffusion/analyze_layer_gradient_conflict.py)  
> **Report:** [REPORT.md](/home/zd25e122/clevr_discrete_diffusion/outputs/layer_gradient_conflict_pretrained_128/REPORT.md)  
> **Menu:** [experiment catalog](../../README.md)

## What it measures

Text/image gradient cosine and conflict per layer.

## How it works

Measures the cosine between text-loss and image-loss gradients per layer.  Negative cosine indicates a conflicting update direction; layers with lower conflict are candidates for shared processing.

## Key result

Across seven multimodal checkpoints, median adjusted text–image conflict is
lowest at blocks 2–4 (+0.106 to +0.112) and rises through blocks 5–7; block 7
is the most conflicted layer in every model. This is the evidence behind
placing shared JEPA supervision at blocks 2–4.

