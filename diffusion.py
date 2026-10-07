import math

import torch
import torch.nn.functional as F


def sample_t(batch_size, device, eps=1e-3):
    """t ~ Uniform(eps, 1). eps avoids t=0, which would make the 1/t loss weight blow up."""
    return eps + (1 - eps) * torch.rand(batch_size, device=device)


def forward_process(x0, t, mask_id):
    """q(x_t | x_0) for the linear-schedule absorbing process: each of the L
    tokens is independently replaced with mask_id with probability t (per
    sequence). x0: [B, L] int64 in [0, vocab_size). t: [B] float in (0, 1]."""
    rand = torch.rand_like(x0, dtype=torch.float32)
    masked = rand < t.unsqueeze(1)
    xt = torch.where(masked, torch.full_like(x0, mask_id), x0)
    return xt, masked


def masked_diffusion_loss(logits, x0, masked, t, weight_by_t=True):
    """Cross-entropy at masked positions only. logits: [B, L, V], x0: [B, L],
    masked: [B, L] bool, t: [B].

    weight_by_t=True (default): 1/t-weighted, the actual MDLM/absorbing-
    diffusion NELBO for the linear noise schedule.
    weight_by_t=False: plain unweighted average -- matches what ~/Omni's
    Dream training does in practice (it computes p_mask but never applies
    it), kept here as an explicit ablation rather than an accidental gap."""
    b, l, v = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, v), x0.reshape(-1), reduction="none").view(b, l)
    ce = ce * masked
    weight = (1.0 / t).unsqueeze(1) if weight_by_t else torch.ones_like(t).unsqueeze(1)
    n_masked = masked.sum().clamp_min(1)
    loss = (ce * weight).sum() / n_masked
    return loss, n_masked


@torch.no_grad()
def generate(model, seq_len, mask_id, num_steps=50, batch_size=1, device="cuda", temperature=1.0,
             reveal_order="confidence"):
    """MaskGIT/LLaDA-style iterative unmasking, starting from an all-MASK
    sequence. Cosine reveal schedule. Already-revealed positions are never
    resampled, giving the same carry-over guarantee as MDLM's SUBS
    parameterization without needing the logit-scatter trick.

    Token values are always sampled (multinomial), not argmax'd -- argmax is
    deterministic given the model and the identical all-MASK start, so a
    whole batch would otherwise collapse to byte-identical outputs.

    reveal_order:
      "confidence" (default) -- each step, reveal the positions the model is
      most sure about first. Known failure mode: low-entropy regions (e.g.
      CLEVR's flat gray background) are "sure" almost immediately, so they
      get revealed first every time, regardless of whether that's actually
      useful structure to commit to early.
      "random" -- each step, reveal a uniformly random subset of the
      currently-masked positions instead, decoupling *order* from
      confidence. Token values are still sampled from the model as usual;
      only which positions get revealed changes."""
    assert reveal_order in ("confidence", "random"), reveal_order
    model.eval()
    x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)

    for step in range(num_steps):
        logits = model(x)
        probs = F.softmax(logits / temperature, dim=-1)  # [B, L, V]
        b, l, v = probs.shape
        pred = torch.multinomial(probs.view(-1, v), 1).view(b, l)  # [B, L] stochastic sample
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)  # [B, L] prob of the sampled token

        is_masked = x == mask_id
        if reveal_order == "confidence":
            rank_score = conf.masked_fill(~is_masked, -1.0)
        else:
            rank_score = torch.rand_like(conf).masked_fill(~is_masked, -1.0)

        frac_remaining = math.cos(0.5 * math.pi * (step + 1) / num_steps)
        target_num_masked = 0 if step == num_steps - 1 else max(round(frac_remaining * seq_len), 0)

        num_to_reveal = int((is_masked.sum(dim=1)[0] - target_num_masked).clamp(min=0).item())
        if num_to_reveal > 0:
            topk_idx = torch.topk(rank_score, num_to_reveal, dim=1).indices
            reveal_mask = torch.zeros_like(is_masked).scatter_(1, topk_idx, True)
            x = torch.where(reveal_mask, pred, x)

    return x
