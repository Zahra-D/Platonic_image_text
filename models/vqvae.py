import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    """64x96 RGB -> 16x24 x embed_dim latent (4x spatial downsample)."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # /2
            ResBlock(128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),  # /4
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, embed_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """16x24 x embed_dim latent -> 64x96 RGB, mirrors the encoder."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 1),
            ResBlock(128),
            ResBlock(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # x2
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # x2
            ResBlock(64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class VectorQuantizerEMA(nn.Module):
    """Codebook with EMA updates (van den Oord et al., 2017)."""

    def __init__(self, num_codes=512, embed_dim=64, decay=0.99, eps=1e-5, commitment_cost=0.25,
                 dead_code_threshold=1.0):
        super().__init__()
        self.num_codes = num_codes
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        # EMA usage below this triggers a reset (revives dead/collapsing codes so the
        # codebook doesn't converge to a handful of winners early in training).
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(num_codes, embed_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        # z: [B, C, H, W] -> [B, H, W, C]
        z = z.permute(0, 2, 3, 1).contiguous()
        b, h, w, c = z.shape
        flat = z.view(-1, c)  # [BHW, C]

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # [BHW, K]
        indices = dist.argmin(1)  # [BHW]
        one_hot = F.one_hot(indices, self.num_codes).type(flat.dtype)  # [BHW, K]

        quantized = one_hot @ self.embedding  # [BHW, C]
        quantized = quantized.view(b, h, w, c)

        if self.training:
            with torch.no_grad():
                new_cluster_size = one_hot.sum(0)
                new_embed_sum = one_hot.t() @ flat

                self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(new_embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
                )
                self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))

                dead_mask = self.cluster_size < self.dead_code_threshold
                n_dead = int(dead_mask.sum().item())
                if n_dead > 0:
                    resample_idx = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
                    revived = flat[resample_idx].to(self.embedding.dtype)
                    self.embedding[dead_mask] = revived
                    self.embed_avg[dead_mask] = revived
                    self.cluster_size[dead_mask] = self.dead_code_threshold

        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        # straight-through estimator
        quantized = z + (quantized - z).detach()

        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        indices = indices.view(b, h, w)
        return quantized, indices, commitment_loss, perplexity


class VQVAE(nn.Module):
    def __init__(self, embed_dim=64, num_codes=512, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.encoder = Encoder(embed_dim)
        self.quantizer = VectorQuantizerEMA(num_codes, embed_dim, decay, commitment_cost=commitment_cost)
        self.decoder = Decoder(embed_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, commitment_loss, perplexity = self.quantizer(z)
        recon = self.decoder(quantized)
        return {
            "recon": recon,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def encode_to_indices(self, x):
        z = self.encoder(x)
        _, indices, _, _ = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices):
        quantized = self.quantizer.embedding[indices]  # [B, H, W, C]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)
